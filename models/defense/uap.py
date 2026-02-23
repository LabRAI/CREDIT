import random
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset


@torch.no_grad()
def _forward_emb_logits(model, x):
    out = model(x)
    if isinstance(out, tuple) and len(out) == 2:
        emb, logits = out
    else:
        emb, logits = None, out
    return emb, logits


class UAP(nn.Module):
    def __init__(
            self,
            target_model: nn.Module,
            dataset,
            device,
            ver_dataset_ratio: float = 0.04,
            embedding_dim: int = 1024,
            ov_mode: bool = True,
            save_path: str = None,
    ):
        super().__init__()
        self.target_model = target_model
        self.ver_dataset_ratio = ver_dataset_ratio
        self.embedding_dim = embedding_dim
        self.num_classes = len(dataset.classes)
        self.seed = 42
        self.device = device
        self.ov_mode = ov_mode

        self._wm_loader, self._test_loader = self._preprocess_loader(
            dataset=dataset,
            batch_size=128,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        self._key_gen()

        if self.ov_mode:
            assert save_path is not None
            print("Inference mode")
            self._load_model(save_path)
            self._inference_downstream()
            self._inference_wm()

    def forward(self, x: torch.Tensor):
        self.target_model.eval()
        emb, logits = self.target_model(x)
        return emb, logits

    def _load_model(self, save_path):
        checkpoint = torch.load(save_path, map_location=self.device)
        self.target_model.load_state_dict(checkpoint["model"])
        self.model = self.target_model.eval()

    def _preprocess_loader(
            self,
            dataset,
            batch_size: int = 128,
            shuffle: bool = False,
            num_workers: int = 4,
            pin_memory: bool = True,
    ):
        n_total = len(dataset)
        n_ver = max(1, int(round(self.ver_dataset_ratio * n_total)))
        self.n_ver = n_ver
        print(f"total: {n_total}, ver ratio: {self.ver_dataset_ratio}, ver size: {n_ver}")

        rng = random.Random(self.seed)
        idx = list(range(n_total))
        rng.shuffle(idx)

        # watermark subset
        ver_subset = Subset(dataset, idx[:n_ver])
        # remaining subset for test
        test_subset = Subset(dataset, idx[n_ver:])

        wm_loader = DataLoader(
            ver_subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

        test_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

        return wm_loader, test_loader

    def _key_gen(
            self,
            n_fingerprints: int = 64,
            eps: float = 8 / 255,
            uap_alpha: float = 1 / 255,
            uap_epochs: int = 10,
            batch_size: int = 128,
            kmeans_iters: int = 20,
            seed: int = 42,
    ):
        torch.manual_seed(seed)
        self.target_model.eval()

        print("[UAP]Generate UAP")
        v = self._compute_uap(
            model=self.target_model,
            loader=self._wm_loader,
            eps=eps,
            alpha=uap_alpha,
            epochs=uap_epochs,
            target_fooling=0.8,
            max_points=20000,
            seed=seed,
        )  # [1, C, H, W]

        feats, imgs, labels, idxs = self._collect_features(
            model=self.target_model,
            loader=self._wm_loader,
            max_points=10000,
        )  # imgs: [M, C, H, W]
        sel_mask, _, _ = self._torch_kmeans(feats, k=n_fingerprints, iters=kmeans_iters)
        xs = imgs[sel_mask]  # [n, C, H, W]
        ys = labels[sel_mask].long()  # [n]
        idxs_sel = idxs[sel_mask].cpu()  # [n]
        x_uap = torch.clamp(xs + v, 0.0, 1.0)  # [n, C, H, W]

        fp_ds = TensorDataset(x_uap.cpu(), ys.cpu())
        fp_loader = DataLoader(
            fp_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        self.fingerprint = {
            "v": v.detach().cpu(),
            "idxs": idxs_sel,
        }
        self._wm_loader = fp_loader

        return fp_loader

    def _compute_uap(
            self,
            model,
            loader,
            eps: float,
            alpha: float,
            epochs: int,
            target_fooling: float,
            max_points: int,
            seed: int,
    ) -> torch.Tensor:
        model.eval()
        first_batch = next(iter(loader))[0]
        if not torch.is_tensor(first_batch):
            first_batch = torch.as_tensor(first_batch)
        C, H, W = first_batch.shape[1:]
        v = torch.zeros((1, C, H, W), device=self.device, dtype=first_batch.dtype)

        @torch.no_grad()
        def fooling_rate() -> float:
            total, fooled = 0, 0
            for x, y, *rest in loader:
                if not torch.is_tensor(x): x = torch.as_tensor(x)
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                _, logits0 = _forward_emb_logits(model, x)
                preds0 = logits0.argmax(1)

                x_adv = torch.clamp(x + v, 0.0, 1.0)
                _, logits1 = _forward_emb_logits(model, x_adv)
                preds1 = logits1.argmax(1)

                fooled += (preds0 != preds1).sum().item()
                total += x.size(0)
                if total >= max_points: break
            return fooled / max(1, total)

        ce = torch.nn.CrossEntropyLoss(reduction="mean")
        seen = 0
        for ep in range(epochs):
            for x, y, *rest in loader:
                if not torch.is_tensor(x): x = torch.as_tensor(x)
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                if x.dtype != torch.float32 and x.dtype != torch.float16 and x.dtype != torch.bfloat16:
                    x = x.float()
                if x.max() > 1.0:
                    x = x / 255.0

                x.requires_grad_(False)
                x_adv = (x + v).clamp(0.0, 1.0).detach().requires_grad_(True)

                out = self.target_model(x_adv)
                logits = out[1] if isinstance(out, tuple) else out

                loss = ce(logits, y)
                grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
                uap_alpha = 1.0
                with torch.no_grad():
                    v.add_(uap_alpha * grad.sign().mean(dim=0, keepdim=True))
                    v.clamp_(-eps, eps)

                seen += x.size(0)
                if seen >= max_points:
                    break

            fr = fooling_rate()
            # print(f"[UAP] epoch {ep+1}/{epochs}, fooling_rate={fr:.3f}")
            if fr >= target_fooling:
                break
            seen = 0
        return v.detach()

    @torch.no_grad()
    def _collect_features(
            self, model, loader, max_points: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feats, imgs, labels, idxs = [], [], [], []
        counted = 0
        model.eval()
        ptr = 0
        for batch in loader:
            if len(batch) == 3:
                x, y, idb = batch
            else:
                x, y = batch
                idb = torch.arange(ptr, ptr + x.size(0))
            ptr += x.size(0)

            if not torch.is_tensor(x): x = torch.as_tensor(x)
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            idb = idb.to(self.device)

            emb, _ = _forward_emb_logits(model, x)
            feat = emb if emb is not None else _
            if feat.dim() > 2:
                feat = torch.flatten(feat, 1)
            feats.append(feat.detach())
            imgs.append(x.detach())
            labels.append(y.detach())
            idxs.append(idb.detach())

            counted += x.size(0)
            if counted >= max_points: break

        feats = torch.cat(feats, 0)
        imgs = torch.cat(imgs, 0)
        labels = torch.cat(labels, 0)
        idxs = torch.cat(idxs, 0)
        return feats, imgs, labels, idxs

    @torch.no_grad()
    def _torch_kmeans(
            self, X: torch.Tensor, k: int, iters: int, g: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N, D = X.shape
        perm = torch.randperm(N, generator=g, device=X.device)
        centers = X[perm[:k]].clone()

        for _ in range(iters):
            # dist: [N, k]
            dist = (
                    X.pow(2).sum(1, keepdim=True)
                    - 2 * X @ centers.t()
                    + centers.pow(2).sum(1, keepdim=True).t()
            )
            assign = dist.argmin(dim=1)
            for c in range(k):
                mask = (assign == c)
                if mask.any():
                    centers[c] = X[mask].mean(dim=0)
                else:
                    ridx = torch.randint(0, N, (1,), generator=g, device=X.device)
                    centers[c] = X[ridx]

        sel_mask = torch.zeros(N, dtype=torch.bool, device=X.device)
        dist = (
                X.pow(2).sum(1, keepdim=True)
                - 2 * X @ centers.t()
                + centers.pow(2).sum(1, keepdim=True).t()
        )
        for c in range(k):
            mask = (assign == c)
            if mask.any():
                idx_local = torch.argmin(dist[mask, c])
                idx_global = torch.nonzero(mask, as_tuple=False)[idx_local]
                sel_mask[idx_global] = True
        return sel_mask, centers, assign

    def _inference_downstream(self):
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in self._test_loader:
                x, y = x.to(self.device), y.to(self.device)
                _, logits = self.model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        print(f"[UAP]eval test acc: {correct / total}")

    def _inference_wm(self):
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in self._wm_loader:
                x, y = x.to(self.device), y.to(self.device)
                _, logits = self.model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        print(f"[UAP]eval wm acc: {correct / total}")

    def _finetune_with_backdoor(self, model, epochs, lr, weight_decay):
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total, loss_sum, acc_sum = 0, 0.0, 0.0
            for x, y in self._wm_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                _, logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                bs = y.size(0)
                total += bs
                loss_sum += loss.item() * bs
                acc_sum += (logits.argmax(1) == y).float().sum().item()
            print(
                f"[UAP]train epoch: {epoch}, loss: {loss_sum / total}, wm acc: {acc_sum / total}")

        model.eval()
        total, loss_sum, acc_sum = 0, 0.0, 0.0
        for x, y in self._test_loader:
            x, y = x.to(self.device), y.to(self.device)
            _, logits = model(x)
            loss = criterion(logits, y)
            bs = y.size(0)
            total += bs
            loss_sum += loss.item() * bs
            acc_sum += (logits.argmax(1) == y).float().sum().item()
        print(f"[UAP]test loss: {loss_sum / total}, test acc: {acc_sum / total}")

        return model.eval()

    def train_defense_model(self, epochs, lr=1e-3, weight_decay=1e-5):
        model = self._finetune_with_backdoor(self.target_model, epochs, lr, weight_decay)
        self.model = model
        return self.model

    def eval_downstream(self, criterion):
        model = self.target_model.eval()
        total, loss_sum, acc_sum = 0, 0.0, 0.0
        for x, y in self._test_loader:
            x, y = x.to(self.device), y.to(self.device)
            _, logits = model(x)
            loss = criterion(logits, y)
            bs = y.size(0)
            total += bs
            loss_sum += loss.item() * bs
            acc_sum += (logits.argmax(1) == y).float().sum().item()
        loss = loss_sum / total
        acc = acc_sum / total
        print(f"[UAP]Downstream Loss: {loss:.6f}, acc: {acc:.6f}")
        return loss, acc

    def eval_wm(self, criterion):
        model = self.target_model.eval()
        total, loss_sum, acc_sum = 0, 0.0, 0.0
        for x, y in self._wm_loader:
            x, y = x.to(self.device), y.to(self.device)
            _, logits = model(x)
            loss = criterion(logits, y)
            bs = y.size(0)
            total += bs
            loss_sum += loss.item() * bs
            acc_sum += (logits.argmax(1) == y).float().sum().item()
        loss = loss_sum / total
        acc = acc_sum / total
        print(f"[UAP]WM Loss: {loss:.6f}, acc: {acc:.6f}")
        return loss, acc
