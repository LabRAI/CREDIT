import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


class LabelOverrideSubset(Subset):
    def __init__(self, dataset, indices, labels):
        super().__init__(dataset, indices)
        self.override_labels = labels

    def __getitem__(self, i):
        x, _ = super().__getitem__(i)
        return x, self.override_labels[i]


class IPGuard(nn.Module):
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

        if self.ov_mode:
            assert save_path is not None
            print("Inference mode")
            self._load_model_and_wm(save_path)
            self._inference_downstream()
            self._inference_wm()
        else:
            self._wm_loader, self.wm_indices, self.wm_labels = self._key_gen()

    def forward(self, x: torch.Tensor):
        self.target_model.eval()
        emb, logits = self.target_model(x)
        return emb, logits

    def _load_model_and_wm(self, save_path):
        checkpoint = torch.load(save_path, map_location=self.device)
        self.target_model.load_state_dict(checkpoint["model"])
        self.model = self.target_model.eval()
        self.wm_indices = checkpoint["wm_indices"]
        self.wm_labels = checkpoint["wm_labels"]
        wm_subset = LabelOverrideSubset(self._wm_loader.dataset.dataset, self.wm_indices,
                                        self.wm_labels)  # get base datasets
        _new_wm_loader = DataLoader(
            wm_subset,
            batch_size=128,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )
        self._wm_loader = _new_wm_loader
        print(f"Load watermark: {len(self.wm_indices)}")

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

    def _key_gen(self, k: float = 10.0):
        device = self.device
        self.target_model.eval()

        wm_indices, wm_labels = [], []

        with torch.no_grad():
            for idx in tqdm(range(len(self._wm_loader.dataset)), desc="KeyGen - scan boundary"):
                x, y = self._wm_loader.dataset[idx]
                x_in = x.unsqueeze(0).to(device)

                _, logits = self.target_model(x_in)
                top2 = torch.topk(logits, k=2, dim=1).values
                margin = (top2[:, 0] - top2[:, 1]).abs().item()

                if margin <= k:
                    rng = random.Random((self.seed << 20) ^ idx)
                    off = rng.randrange(1, self.num_classes)
                    wm_y = (int(y) + off) % self.num_classes

                    wm_indices.append(idx)
                    wm_labels.append(wm_y)

        print("KeyGen - scan boundary size:", len(wm_indices))

        if not wm_indices:
            raise RuntimeError(f"increase k (current k={k})")

        wm_subset = LabelOverrideSubset(self._wm_loader.dataset, wm_indices, wm_labels)
        _new_wm_loader = DataLoader(
            wm_subset,
            batch_size=128,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )

        self._wm_loader = _new_wm_loader

        return _new_wm_loader, wm_indices, wm_labels

    def _inference_downstream(self):
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in self._test_loader:
                x, y = x.to(self.device), y.to(self.device)
                _, logits = self.model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        print(f"[IPGuard]eval test acc: {correct / total}")

    def _inference_wm(self):
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in self._wm_loader:
                x, y = x.to(self.device), y.to(self.device)
                _, logits = self.model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        print(f"[IPGuard]eval wm acc: {correct / total}")

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
                f"[IPGuard]train epoch: {epoch}, loss: {loss_sum / total}, wm acc: {acc_sum / total}")

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
        print(f"[IPGuard]test loss: {loss_sum / total}, test acc: {acc_sum / total}")

        return model.eval()

    def train_defense_model(self, epochs, lr=1e-3, weight_decay=1e-5):
        model = self._finetune_with_backdoor(self.target_model, epochs, lr, weight_decay)
        self.model = model
        return self.model, self.wm_indices, self.wm_labels

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
        print(f"[IPGuard]Downstream Loss: {loss:.6f}, acc: {acc:.6f}")
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
        print(f"[IPGuard]WM Loss: {loss:.6f}, acc: {acc:.6f}")
        return loss, acc
