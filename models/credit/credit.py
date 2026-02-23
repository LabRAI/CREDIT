import math
import random
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


class CREDIT(nn.Module):
    def __init__(
            self,
            target_model: nn.Module,
            dataset,
            dataset_ratio: float = 0.04,
            embedding_dim: int = 1024,
            sigma: float = 0.1,
            enable: bool = True,
    ):
        super().__init__()
        self.target = target_model.eval()
        self.dataset_ratio = dataset_ratio
        self.embedding_dim = embedding_dim
        self.sigma = float(sigma)
        self.enable = enable
        self.register_buffer("_eigs", None, persistent=False)
        self._eps = 1e-8
        self.seed = 42

        self._loader = self._preprocess_loader(
            dataset=dataset,
            batch_size=128,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        with torch.no_grad():
            W: torch.Tensor = self.target.classifier.weight.data.clone()  # [C, d]
            C = W.size(0)
            ones = torch.ones(C, 1, device=W.device, dtype=W.dtype)  # [C, 1]
            WWt = W @ W.t()  # [C, C]
            eps = 1e-6 * torch.eye(C, device=W.device, dtype=W.dtype)
            y = torch.linalg.solve(WWt + eps, ones)  # [C, 1]
            delta = 1.0 * (W.t() @ y).flatten()  # [d]
            delta = delta / delta.norm().clamp_min(1e-12)
            delta = (self.sigma * delta).unsqueeze(0)  # [1, d]
            self.register_buffer("fix_noise", delta)

    def _preprocess_loader(
            self,
            dataset,
            batch_size: int = 128,
            shuffle: bool = False,
            num_workers: int = 4,
            pin_memory: bool = True,
    ) -> DataLoader:
        n_total = len(dataset)
        n_ver = max(1, int(round(self.dataset_ratio * n_total)))
        self.n_ver = n_ver
        print(f"total: {n_total}, ver ratio: {self.dataset_ratio}, ver size: {n_ver}")
        rng = random.Random(self.seed)
        idx = list(range(n_total))
        rng.shuffle(idx)
        subset = Subset(dataset, idx[:n_ver])
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    @torch.no_grad()
    def _fix_noise(self, emb: torch.Tensor) -> torch.Tensor:
        return self.fix_noise.expand(emb.size(0), -1)  # [B, d]

    @staticmethod
    def _tangent_noise(u, sigma):
        z = torch.randn_like(u)
        z_perp = z - (z * u).sum(dim=1, keepdim=True) * u
        z_perp = z_perp / (z_perp.norm(dim=1, keepdim=True) + 1e-12)
        return sigma * z_perp

    @staticmethod
    def _rand_noise(emb, sigma):
        noise = torch.randn_like(emb) * sigma
        return noise

    def forward(self, x: torch.Tensor):
        emb, logits = self.target(x)
        if self.enable:
            # noise_t = self._tangent_noise(emb, self.sigma)
            noise_r = self._rand_noise(emb, self.sigma)
            noise_fix = self._fix_noise(emb)
            noisy_emb = emb + noise_r + noise_fix
            # noisy_emb = F.normalize(noisy_emb, dim=1)
            noisy_logits = self.target.classifier(noisy_emb)
            return noisy_emb, noisy_logits
        else:
            return emb, logits

    @staticmethod
    def _hb(p: torch.Tensor) -> torch.Tensor:
        p = torch.clamp(p, 1e-12, 1 - 1e-12)
        return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))

    @torch.no_grad()
    def _ensure_util_spectrum(
            self,
            max_batches: Optional[int],
            device: Optional[str],
    ):
        if self._eigs is not None:
            return
        self.target.eval()
        embs = []
        cnt = 0
        for batch in self._loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if device is not None:
                x = x.to(device, non_blocking=True)
            self.enable = False
            emb, _ = self.target(x)
            embs.append(emb.detach().float().cpu())
            cnt += 1
            if max_batches is not None and cnt >= max_batches:
                break
        self.enable = True
        E = torch.cat(embs, dim=0)  # [N, d]
        X = E - E.mean(dim=0, keepdim=True)
        # cov = (X^T X) / (N-1) with eigenvalues S^2 / (N-1)
        _, S, _ = torch.linalg.svd(X, full_matrices=False)
        eigs = (S ** 2) / max(1, X.shape[0] - 1)
        self._eigs = torch.clamp(eigs, min=self._eps)

    def _H_util(
            self,
            sigma: float,
            max_batches: Optional[int],
            device: Optional[str],
    ) -> float:
        """ΔH_util(sigma) = 0.5 * sum_i log(1 + sigma^2 / lambda_i)"""
        self._ensure_util_spectrum(max_batches, device)
        s2 = float(sigma) * float(sigma)
        return float(0.5 * torch.log1p(s2 / self._eigs).sum().item())

    def _H_util_task(
            self, sigma: float, max_batches: Optional[int], device: Optional[str],
            jac_batches: int = 5
    ) -> float:
        self._ensure_util_spectrum(max_batches, device)
        lam = self._eigs.clone()  # [d]
        inv_Sigma = torch.diag(1.0 / lam)  # [d, d]

        self.target.eval()
        F = None
        cnt = 0
        for batch in self._loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if device is not None:
                x = x.to(device, non_blocking=True)
            x.requires_grad_(False)

            self.enable = True
            emb, logits = self.target(x)  # emb: [N, d], logits: [N, C]
            self.enable = False
            emb = emb.detach().requires_grad_(True)

            loss_for_jac = 0.5 * (logits.detach() ** 2).sum(dim=1).mean()
            with torch.no_grad():
                head = self.target.classifier
            logits2 = head(emb)  # [N, C]
            loss2 = 0.5 * (logits2 ** 2).sum(dim=1).mean()
            G = torch.autograd.grad(loss2, emb, create_graph=False)[0]  # [N, d]

            Ft = (G.T @ G) / G.shape[0]  # [d, d]
            F = Ft if F is None else F + Ft
            cnt += 1
            if jac_batches is not None and cnt >= jac_batches:
                break

        F = F / max(cnt, 1)

        # log det(I + sigma^2 * F * inv_Sigma)
        M = torch.eye(F.shape[0], device=F.device, dtype=F.dtype) + (sigma ** 2) * (F @ inv_Sigma)
        M = 0.5 * (M + M.T)
        eigvals = torch.linalg.eigvalsh(M).clamp(min=self._eps)
        return float(0.5 * torch.log(eigvals).sum().item())

    def _H_ver(self, gamma1: float, gamma2: float, pi0: float) -> float:
        """ΔH_ver = pi0*h_b(g1) + (1-pi0)*h_b(g2)"""
        g1 = torch.tensor(gamma1, dtype=torch.float64)
        g2 = torch.tensor(gamma2, dtype=torch.float64)
        h = pi0 * self._hb(g1) + (1.0 - pi0) * self._hb(g2)
        return float(h.item())

    def _delta(
            self,
            *,
            device: Optional[str] = None,
            max_batches: Optional[int] = None,
            max_pairs_per_batch: int = 16_000,
            max_points_per_batch: int = 256,
            neighbor: str = "nn",  # "nn", "radius", "all"
            space: str = "embedding",  # "embedding" 或 "input"
            k: int = 1,  # 最近邻的个数，取最大距离（k=1 即最近邻）
            radius: Optional[float] = None,  # neighbor="radius" 时使用
            robust_quantile: Optional[float] = None,  # 例如 0.99 做稳健估计
    ) -> float:
        self.target.eval()
        global_vals = []
        seen_batches = 0

        for batch in self._loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if device is not None:
                x = x.to(device, non_blocking=True)

            self.enable = False
            with torch.no_grad():
                emb, _ = self.target(x)  # [B, d]
            self.enable = True

            if space == "embedding":
                Z = emb.detach()
                Z = F.normalize(Z, dim=-1)
            elif space == "input":
                Z = x.detach().flatten(1).float()
            else:
                raise ValueError("space must be 'embedding' or 'input'")

            B = Z.shape[0]
            if B <= 1:
                seen_batches += 1
                if max_batches is not None and seen_batches >= max_batches:
                    break
                continue

            if B > max_points_per_batch:
                idx = torch.randperm(B, device=Z.device)[:max_points_per_batch]
                Z = Z.index_select(0, idx)
                B = Z.shape[0]

            if neighbor == "all":
                if B * B <= max_pairs_per_batch:
                    D = torch.cdist(Z, Z, p=2)  # [B, B]
                    D = D + torch.eye(B, device=Z.device) * (-1e9)
                    vals = D.flatten()
                else:
                    m = max_pairs_per_batch
                    i = torch.randint(0, B, (m,), device=Z.device)
                    j = torch.randint(0, B, (m,), device=Z.device)
                    same = (i == j)
                    if same.any():
                        j[same] = (j[same] + 1) % B
                    vals = (Z[i] - Z[j]).norm(p=2, dim=1)

            elif neighbor == "nn":
                D = torch.cdist(Z, Z, p=2)  # [B, B]
                D = D + torch.eye(B, device=Z.device) * 1e9
                knn_vals, _ = torch.topk(D, k=min(k, B - 1), dim=1, largest=False)
                vals = knn_vals.reshape(-1)

            elif neighbor == "radius":
                if radius is None or radius <= 0:
                    raise ValueError("neighbor='radius' requires a positive radius")
                D = torch.cdist(Z, Z, p=2)  # [B, B]
                mask = torch.ones_like(D, dtype=torch.bool)
                mask.fill_(True)
                mask.fill_diagonal_(False)
                sel = D[mask]
                vals = sel[sel <= radius]
                if vals.numel() == 0:
                    seen_batches += 1
                    if max_batches is not None and seen_batches >= max_batches:
                        break
                    continue
            else:
                raise ValueError("neighbor must be 'nn', 'radius', or 'all'")

            if vals.numel() > 0:
                if robust_quantile is not None:
                    q = float(robust_quantile)
                    q = min(max(q, 0.0), 1.0)
                    v = torch.quantile(vals, q).item()
                else:
                    v = vals.max().item()
                global_vals.append(v)

            seen_batches += 1
            if max_batches is not None and seen_batches >= max_batches:
                break

        if len(global_vals) == 0:
            return 0.0
        return float(max(global_vals))

    @torch.no_grad()
    def optim_sigma(
            self,
            sigmas: List[float],
            gamma1_list: List[float],
            gamma2_list: List[float],
            *,
            device: Optional[str] = None,
            max_batches: Optional[int] = None,
            lambda_util: float = 1.0,
            lambda_ver: float = 1.0,
            pi0: float = 0.5,
    ) -> Tuple[float, List[float], List[float], List[float]]:
        if len(sigmas) != len(gamma1_list) or len(sigmas) != len(gamma2_list):
            raise ValueError("sigmas gamma1_list gamma2_list must have the same length")

        self._ensure_util_spectrum(max_batches, device)

        du_list, dv_list = [], []
        for s, g1, g2 in zip(sigmas, gamma1_list, gamma2_list):
            du = self._H_util(s, max_batches, device)
            dv = self._H_ver(g1, g2, pi0)
            du_list.append(du)
            dv_list.append(dv)
        du_arr = np.asarray(du_list, dtype=np.float64)
        dv_arr = np.asarray(dv_list, dtype=np.float64)
        du_scaled = (du_arr - du_arr.mean()) / (du_arr.std() + 1e-12)
        dv_scaled = (dv_arr - dv_arr.mean()) / (dv_arr.std() + 1e-12)
        objs = lambda_util * du_scaled + lambda_ver * dv_scaled
        best_index = np.argmin(objs)
        best_obj = float(objs[best_index])
        best_s = float(sigmas[best_index])
        print(f"du: {du_scaled}, dv: {dv_scaled}, obj: {objs}")
        self.sigma = float(best_s)  # apply sigma
        return best_s, lambda_util * du_scaled, lambda_ver * dv_scaled, objs

    @torch.no_grad()
    def beta(
            self,
            sigma: Optional[float] = None,
            *,
            device: Optional[str] = None,
            max_batches: Optional[int] = None,
            max_pairs_per_batch: int = 16_000,
            max_points_per_batch: int = 256,
            neighbor: str = "nn",
            space: str = "embedding",
            k: int = 1,
            radius: Optional[float] = None,
            robust_quantile: Optional[float] = None,
    ) -> float:
        s = float(self.sigma if sigma is None else sigma)
        if s <= 0:
            raise ValueError("sigma must be positive")

        delta_hat = self._delta(
            device=device,
            max_batches=max_batches,
            max_pairs_per_batch=max_pairs_per_batch,
            max_points_per_batch=max_points_per_batch,
            neighbor=neighbor,
            space=space,
            k=k,
            radius=radius,
            robust_quantile=robust_quantile,
        )

        n = self.n_ver
        beta_val = (delta_hat * delta_hat) / (2.0 * s * s + 1e-12)
        beta_val_n = beta_val * n
        print(
            f"[beta] neighbor={neighbor}, space={space}, "
            f"delta_hat={delta_hat:.6f}, sigma={s:.6f}, beta={beta_val:.6f}, n={n}, beta_val_n={beta_val_n:.6f}"
        )
        return float(beta_val)

    def tau(
            self,
            beta: float,
            *,
            Q: int,
            rho: float = 1.0,
            eta: float = 1.0,
    ) -> float:
        """
        Compute verification threshold tau.

        Formula:
            tau = beta * (1 - rho * exp(-Q * beta / (eta * d * |S|))).

        Args:
            sigma: Gaussian noise scale, if None use self.sigma.
            Q: query budget (int).
            c: constant in (0,1).
            rho: constant in (0,1).
            device: optional device for beta() computation.
            max_batches: optional number of batches for beta() estimation.

        Returns:
            tau (float)
        """
        # dataset size |S|
        n = self.n_ver
        # embedding dimension d
        d = self.embedding_dim
        # tau formula
        expo = -Q * beta / (eta * d * n + 1e-12)
        tau_val = beta * (1.0 - rho * math.exp(expo))

        print(
            f"[tau] beta={beta:.6f}, Q={Q}, d={d}, n={n}, "
            f"eta={eta}, rho={rho}, tau={tau_val:.6f}"
        )
        return float(tau_val)
