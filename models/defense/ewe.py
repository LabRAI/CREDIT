import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm


class EWE(nn.Module):
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
        # EWE
        self.target_label = torch.zeros([1], dtype=torch.long).to(device)
        self.device = device
        self.ov_mode = ov_mode

        self._wm_loader, self._test_loader = self._preprocess_loader(
            dataset=dataset,
            batch_size=128,
            shuffle=True,
            num_workers=4,
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

    def _key_gen(self):
        torch.manual_seed(self.seed)
        trigger, trigger_label = [], []
        print("[EWE] Generating keys...")
        for idx, (x, y) in enumerate(tqdm(self._wm_loader, desc="Generating triggers")):
            for xi, yi in zip(x, y):
                trigger.append(xi)
                trigger_label.append(self.target_label)

        trigger_tensor = torch.stack(trigger)  # [N, C, H, W]
        trigger_label_tensor = torch.tensor(trigger_label, dtype=torch.long)  # [N]
        mk_dataset = TensorDataset(trigger_tensor, trigger_label_tensor)

        trigger_loader = DataLoader(
            mk_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        self._wm_loader = trigger_loader

    def _inference_downstream(self):
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in self._test_loader:
                x, y = x.to(self.device), y.to(self.device)
                _, logits = self.model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        print(f"[EWE]eval test acc: {correct / total}")

    def _inference_wm(self):
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in self._wm_loader:
                x, y = x.to(self.device), y.to(self.device)
                _, logits = self.model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        print(f"[EWE]eval wm acc: {correct / total}")

    def _finetune_with_backdoor(self, model, epochs, lr, weight_decay):
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        snnl_comp = SNNLComputer(model, device=self.device)

        for epoch in range(epochs):
            total, loss_sum, acc_sum = 0, 0.0, 0.0
            for x, y in self._wm_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                _, logits = model(x)
                loss_ce = criterion(logits, y)
                # loss_snnl
                loss_snnl = snnl_comp.compute(x, y)
                loss = loss_ce + 1.0 * loss_snnl
                loss.backward()
                optimizer.step()
                bs = y.size(0)
                total += bs
                loss_sum += loss.item() * bs
                acc_sum += (logits.argmax(1) == y).float().sum().item()
            print(
                f"[EWE]train epoch: {epoch}, snnl loss: {loss_snnl}, loss: {loss_sum / total}, acc: {acc_sum / total}")

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
        print(f"[EWE]test loss: {loss_sum / total}, acc: {acc_sum / total}")

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
        print(f"[EWE]Downstream Loss: {loss:.6f}, acc: {acc:.6f}")
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
        print(f"[EWE]WM Loss: {loss:.6f}, acc: {acc:.6f}")
        return loss, acc


class SNNLComputer:
    def __init__(self, model: nn.Module, device=None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self._feat = None

        last_linear = None
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is None:
            raise RuntimeError("no nn.Linear")

        def pre_hook(mod, inputs):
            x = inputs[0]
            if x.dim() > 2:
                x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
            self._feat = x

        self._handle = last_linear.register_forward_pre_hook(pre_hook)

    def cleanup(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @torch.enable_grad()
    def compute(self, x: torch.Tensor, w):
        x = x.to(self.device)

        if isinstance(w, int):
            w = torch.full((x.size(0),), w, dtype=torch.long, device=self.device)
        else:
            w = torch.as_tensor(w, dtype=torch.long, device=self.device)

        _ = self.model(x)
        if self._feat is None:
            raise RuntimeError("no embedding")

        feat = self._feat
        inv_t = 1.0

        loss = calculate_snnl_torch(feat, w, inv_t)
        return loss


def calculate_snnl_torch(x, y, t, metric='euclidean'):
    x = F.relu(x)
    same_label_mask = y.eq(y.unsqueeze(1)).squeeze()
    if metric == 'euclidean':
        dist = pairwise_euclid_distance(x.contiguous().view([x.shape[0], -1]))
    elif metric == 'cosine':
        dist = cosine_distance_torch(x.contiguous().view([x.shape[0], -1]))
    else:
        raise NotImplementedError()
    exp = torch.clamp((-(dist / t)).exp() - torch.eye(x.shape[0]).cuda(), 0, 1)
    prob = (exp / (0.00001 + exp.sum(1).unsqueeze(1))) * same_label_mask
    loss = - (0.00001 + prob.mean(1)).log().mean()
    return loss


def pairwise_euclid_distance(A):
    sqr_norm_A = A.pow(2).sum(1).unsqueeze(0)
    sqr_norm_B = A.pow(2).sum(1).unsqueeze(1)
    inner_prod = torch.matmul(A, A.t())
    tile_1 = sqr_norm_A.repeat([A.shape[0], 1])
    tile_2 = sqr_norm_B.repeat([1, A.shape[0]])
    return tile_1 + tile_2 - 2 * inner_prod


def cosine_distance_torch(x1, eps=1e-8):
    x2 = x1
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
