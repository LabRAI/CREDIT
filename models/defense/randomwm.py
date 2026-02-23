import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader as GeoDataLoader


class LabelOverrideDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, labels):
        self.base_dataset = base_dataset
        self.labels = labels

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        data = data.clone()
        data.y = torch.tensor([self.labels[idx]], dtype=torch.long)
        return data


class RandomWM(nn.Module):
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
        rng = random.Random(self.seed)

        wm_dataset = self._wm_loader.dataset

        random_labels = []
        for _ in range(len(wm_dataset)):
            rand_label = rng.randint(0, self.num_classes - 1)
            random_labels.append(rand_label)

        new_wm_dataset = LabelOverrideDataset(wm_dataset, random_labels)

        _new_wm_loader = GeoDataLoader(
            new_wm_dataset,
            batch_size=self._wm_loader.batch_size,
            shuffle=False,
            num_workers=self._wm_loader.num_workers,
            pin_memory=self._wm_loader.pin_memory,
            drop_last=False,
        )

        self._wm_loader = _new_wm_loader

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
                f"[RandomWM]train epoch: {epoch}, loss: {loss_sum / total}, wm acc: {acc_sum / total}")

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
        print(f"[RandomWM]test loss: {loss_sum / total}, test acc: {acc_sum / total}")

        return model.eval()

    def train_defense_model(self, epochs, lr=1e-3, weight_decay=1e-5):
        model = self._finetune_with_backdoor(self.target_model, epochs, lr, weight_decay)
        self.model = model
        return self.model

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
        print(f"[RandomWM]Downstream acc: {acc:.6f}")
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
        print(f"[RandomWM]WM acc: {acc:.6f}")
        return acc
