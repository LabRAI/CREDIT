import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader as GeoDataLoader


class RandFeatRandLabelDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, n_classes, seed_):
        self.base_ds = base_ds
        self.n_classes = int(n_classes)
        self.seed = int(seed_)

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        data = self.base_ds[idx]
        if hasattr(data, "clone"):
            data = data.clone()
        if hasattr(data, "x") and data.x is not None:
            x = data.x
            if not torch.is_floating_point(x):
                x = x.float()
            g = torch.Generator(device=x.device if x.is_cuda else "cpu")
            g.manual_seed(self.seed + idx)
            x_rand = torch.randn(x.size(), device=x.device, dtype=x.dtype, generator=g)
            data.x = x_rand
        else:
            raise ValueError("graph sample has no x features, cannot randomize")

        r = random.Random(self.seed + 12345 + idx)
        label = r.randint(0, self.n_classes - 1)
        data.y = torch.tensor(label, dtype=torch.long)

        return data


class SurviveWM(nn.Module):
    def __init__(
            self,
            target_model: nn.Module,
            dataset,
            device,
            ver_dataset_ratio: float = 0.1,
            embedding_dim: int = 128,
    ):
        super().__init__()
        self.target_model = target_model
        self.ver_dataset_ratio = ver_dataset_ratio
        self.embedding_dim = embedding_dim
        self.num_classes = dataset.dataset.num_classes
        self.seed = 42
        self.device = device
        self.wm_feat_len = 2
        self.wm_target_label = 0

        self._wm_loader, self._test_loader = self._preprocess_loader(
            dataset=dataset,
            batch_size=64,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        self._key_gen()

    def forward(self, x: torch.Tensor):
        self.target_model.train()
        emb, logits = self.target_model(x)
        return emb, logits

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

        wm_loader = GeoDataLoader(
            ver_subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

        test_loader = GeoDataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

        return wm_loader, test_loader

    def _key_gen(self):
        base_dataset = self._wm_loader.dataset
        num_classes = self.num_classes
        seed = self.seed

        new_wm_dataset = RandFeatRandLabelDataset(base_dataset, num_classes, seed)

        _new_wm_loader = GeoDataLoader(
            new_wm_dataset,
            batch_size=self._wm_loader.batch_size,
            shuffle=False,
            num_workers=self._wm_loader.num_workers,
            pin_memory=getattr(self._wm_loader, "pin_memory", True),
            drop_last=False,
        )

        self._wm_loader = _new_wm_loader
        print(f"[KeyGen] randomized all node features and assigned random labels for {len(new_wm_dataset)} graphs")

    @staticmethod
    def _snn_loss(x, y, T: float = 0.5):
        """
        x: [B, D]
        y: [B]
        """
        x = F.normalize(x, p=2, dim=1)
        dist_matrix = torch.cdist(x, x, p=2) ** 2
        eye = torch.eye(len(x), device=x.device).bool()
        sim = torch.exp(-dist_matrix / T)
        mask_same = y.unsqueeze(1) == y.unsqueeze(0)
        sim = sim.masked_fill(eye, 0)
        denom = sim.sum(1)
        nom = (sim * mask_same.float()).sum(1)
        loss = -torch.log(nom / (denom + 1e-10) + 1e-10).mean()
        return loss

    def train_defense_model(self, epochs, lr=1e-3, weight_decay=1e-5, snnl_weight: float = 1.0,
                            ce_weight: float = 1.0, ):
        self.target_model.train()
        optimizer = optim.SGD(self.target_model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total, loss_sum, acc_sum = 0, 0.0, 0.0
            for batch in self._wm_loader:
                batch = batch.to(self.device)
                emb, logits = self.target_model(batch)
                loss_ce = criterion(logits, batch.y)
                loss_snn = self._snn_loss(emb, batch.y)
                loss = ce_weight * loss_ce + snnl_weight * loss_snn
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                bs = batch.num_graphs
                total += bs
                loss_sum += loss.item() * bs
                acc_sum += (logits.argmax(1) == batch.y).float().sum().item()
            print(
                f"[SurviveWM]train epoch: {epoch}, loss: {loss_sum / total}, wm acc: {acc_sum / total}")

        self.target_model.eval()
        total, loss_sum, acc_sum = 0, 0.0, 0.0
        for batch in self._wm_loader:
            batch = batch.to(self.device)
            _, logits = self.target_model(batch)
            loss = criterion(logits, batch.y)
            bs = batch.num_graphs
            total += bs
            loss_sum += loss.item() * bs
            acc_sum += (logits.argmax(1) == batch.y).float().sum().item()
        print(f"[SurviveWM]test loss: {loss_sum / total}, test acc: {acc_sum / total}")

        return self.target_model.eval()

    def eval_downstream(self):
        model = self.target_model.eval()
        total, acc_sum = 0, 0.0
        for x, y in self._test_loader:
            x, y = x.to(self.device), y.to(self.device)
            _, logits = model(x)
            bs = y.size(0)
            total += bs
            acc_sum += (logits.argmax(1) == y).float().sum().item()
        acc = acc_sum / total
        print(f"[SurviveWM]Downstream acc: {acc:.6f}")
        return acc

    def eval_wm(self):
        model = self.target_model.eval()
        total, acc_sum = 0, 0.0
        for x, y in self._wm_loader:
            x, y = x.to(self.device), y.to(self.device)
            _, logits = model(x)
            bs = y.size(0)
            total += bs
            acc_sum += (logits.argmax(1) == y).float().sum().item()
        acc = acc_sum / total
        print(f"[SurviveWM]WM acc: {acc:.6f}")
        return acc
