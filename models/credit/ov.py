import math
import random
from typing import Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.special import digamma
from torch.utils.data import DataLoader, Subset


class OwnershipVerifier:
    def __init__(
            self,
            defense_model: nn.Module,
            dataset,
            ver_ratio: float = 0.2,
            batch_size: int = 128,
            num_workers: int = 4,
            device: Optional[torch.device] = None,
            seed: int = 42,
    ):
        self.defense_model = defense_model
        self.dataset = dataset
        self.ver_ratio = float(ver_ratio)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = int(seed)

        self.defense_model.eval()
        for p in self.defense_model.parameters():
            p.requires_grad_(False)

        self.ver_loader = self._prepare_verification_loader()

    # ------------------------- 主流程 -------------------------

    @torch.no_grad()
    def _inference_ver_emb(self) -> np.ndarray:
        emb_ver_list = []
        for batch in self.ver_loader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(self.device, non_blocking=True)

            # self.defense_model.enable = False  # rm noise
            emb, _ = self.defense_model(images)
            # self.defense_model.enable = True

            if emb.dim() > 2:
                emb = emb.flatten(1)

            emb_ver_list.append(emb.detach().cpu().float().numpy())

        emb_ver = np.concatenate(emb_ver_list, axis=0)
        return emb_ver

    @torch.no_grad()
    def _inference_sus_emb(self, sus_models: List[nn.Module]) -> np.ndarray:
        emb_all = []

        for model in sus_models:
            model.to(self.device)
            model.eval()

            emb_sus_list = []
            for batch in self.ver_loader:
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                images = images.to(self.device, non_blocking=True)

                emb, _ = model(images)
                if emb.dim() > 2:
                    emb = emb.flatten(1)

                emb_sus_list.append(emb.detach().cpu().float().numpy())

            emb_sus = np.concatenate(emb_sus_list, axis=0)  # [n_batch, n_dim]
            emb_all.append(emb_sus)

        emb_all = np.stack(emb_all, axis=0)  # [n_models, n_batch, n_dim]
        return emb_all

    def compute_gamma(self, tau: float, C1: float, C2: float):
        n = self.n_ver

        if not hasattr(self, "mi_sus_all") or self.mi_sus_all is None:
            raise ValueError("call verify first")

        # gamma1
        gamma1 = math.exp(-2.0 * n * (tau ** 2) / (C1 ** 2))
        print(
            f"gamma1: {gamma1}, n: {n}, tau^2: {tau ** 2}, C^2: {C1 ** 2}, (-2.0 * n * (tau ** 2) / (C ** 2)): {-2.0 * n * (tau ** 2) / (C1 ** 2)}")

        # gamma2
        mi_sur_all = [mi for mi, lab in zip(self.mi_sus_all, self.sus_label) if lab == 1]
        mi_sur = float(np.mean(mi_sur_all))
        # margin = max(mi_sur - tau, 0.0)
        margin = mi_sur - tau
        gamma2 = math.exp(-2.0 * n * (margin ** 2) / (C2 ** 2))
        print(
            f"gamma2: {gamma2}, n: {n}, mi_sur: {mi_sur}, tau: {tau}, margin^2: {margin ** 2}, C^2: {C2 ** 2}, (-2.0 * n * (margin ** 2) / (C ** 2)): {-2.0 * n * (margin ** 2) / (C2 ** 2)}")

        return float(gamma1), float(gamma2)

    def verify(self, sus_models, sus_label, k: int = 5, n_perm: int = 1, tau: float = 100) -> Dict:
        emb_ver = self._inference_ver_emb()
        emb_sus_all = self._inference_sus_emb(sus_models)

        mi_sus_all = []
        for emb_sus in emb_sus_all:
            mi = self.ksg_mi_softplus(emb_ver, emb_sus, k=k, n_perm=n_perm)
            mi_sus_all.append(mi)

        mi_sus_all = np.asarray(mi_sus_all)  # [m]
        # mi_sus_all = mi_sus_all * self.n_ver  # ksg estimate single mi
        self.mi_sus_all = mi_sus_all
        self.sus_label = sus_label
        sus_label = np.asarray(sus_label).astype(int)
        pred_label = (mi_sus_all > float(tau)).astype(int)

        auc = roc_auc_score(sus_label, mi_sus_all)

        return {
            "tau": float(tau),
            "auc": float(auc),
            "mi_sus_all": mi_sus_all,
            "sus_label": sus_label,
            "pred_label": pred_label,
            "k": k,
            "n": int(emb_ver.shape[0]),
        }

    def _prepare_verification_loader(self) -> DataLoader:
        n_total = len(self.dataset)
        n_ver = max(1, int(round(self.ver_ratio * n_total)))
        self.n_ver = n_ver
        rng = random.Random(self.seed)
        idx = list(range(n_total))
        rng.shuffle(idx)
        subset = Subset(self.dataset, idx[:n_ver])
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @staticmethod
    def _zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        mu = x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True)
        sd = np.maximum(sd, eps)
        return (x - mu) / sd

    @staticmethod
    def _pairwise_chebyshev(x: np.ndarray) -> np.ndarray:
        n, d = x.shape
        x1 = x[:, None, :]
        x2 = x[None, :, :]
        dist = np.abs(x1 - x2).max(axis=2)
        return dist

    @staticmethod
    def _ksg_mi(X: np.ndarray, Y: np.ndarray, k: int = 5, eps: float = 1e-12) -> float:
        assert X.shape[0] == Y.shape[0], "shape need to be same"
        n = X.shape[0]
        if n <= k + 1:
            raise ValueError("no sufficient datapoints")

        Xn = OwnershipVerifier._zscore(X, eps=eps)
        Yn = OwnershipVerifier._zscore(Y, eps=eps)
        Z = np.concatenate([Xn, Yn], axis=1)

        dist_joint = OwnershipVerifier._pairwise_chebyshev(Z)
        np.fill_diagonal(dist_joint, np.inf)
        eps_k = np.partition(dist_joint, kth=k - 1, axis=1)[:, k - 1]

        dist_x = OwnershipVerifier._pairwise_chebyshev(Xn)
        dist_y = OwnershipVerifier._pairwise_chebyshev(Yn)
        np.fill_diagonal(dist_x, np.inf)
        np.fill_diagonal(dist_y, np.inf)

        nx = (dist_x < eps_k[:, None]).sum(axis=1)
        ny = (dist_y < eps_k[:, None]).sum(axis=1)

        k_t = torch.tensor(float(k))
        n_t = torch.tensor(float(n))
        nx_t = torch.from_numpy(nx.astype(np.float64))
        ny_t = torch.from_numpy(ny.astype(np.float64))

        mi = (
                digamma(k_t)
                - 1.0 / k_t
                + digamma(n_t)
                - (digamma(nx_t + 1.0).double().mean() + digamma(ny_t + 1.0).double().mean())
        ).item()
        return float(mi)

    def ksg_mi_softplus(self,
                        X: np.ndarray,
                        Y: np.ndarray,
                        k: int = 5,
                        eps: float = 1e-12,
                        n_perm: int = 1,
                        temperature: float = 0.02,
                        ) -> float:
        mi_raw = self._ksg_mi(X, Y, k=k, eps=eps)

        if n_perm > 0:
            perm_vals = []
            for _ in range(n_perm):
                idx = np.random.permutation(Y.shape[0])
                mi0 = self._ksg_mi(X, Y[idx], k=k, eps=eps)
                perm_vals.append(mi0)
            mi_null = float(np.mean(perm_vals))
        else:
            mi_null = 0.0

        x = mi_raw - mi_null
        # x = 1.0 / (1.0 + mi_raw - mi_null)

        mi_pos = temperature * np.log1p(np.exp(x / temperature))

        return float(mi_pos)
