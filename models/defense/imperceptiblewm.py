import copy
import math
import random
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.utils import to_dense_adj, dense_to_sparse


class ImperceptibleWM(nn.Module):
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

        # params
        self.wm_target_label = 0
        self.eps = 0.03
        self.frac_nodes = 0.1
        self.last_feat_len = 2
        self.enable_trigger_nodes = False
        self.num_triggers = 0
        self.connect_top_k = 3

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
        if n_total == 0:
            raise ValueError("Empty dataset passed into ImperceptibleWM.")
        if n_total >= 2:
            n_ver = int(math.ceil(self.ver_dataset_ratio * n_total))
            n_ver = max(1, min(n_total - 1, n_ver))
        else:
            n_ver = 1

        self.n_ver = n_ver
        print(f"total: {n_total}, ver ratio: {self.ver_dataset_ratio}, ver size: {n_ver}, test size: {n_total - n_ver}")

        rng = random.Random(self.seed)
        idx = list(range(n_total))
        rng.shuffle(idx)

        ver_subset = Subset(dataset, idx[:n_ver])
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
            shuffle=(shuffle and len(test_subset) > 0),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        return wm_loader, test_loader

    def _key_gen(self):
        print(
            "[KeyGen] Scheme B enabled. No model or CUDA ops inside Dataset. Online perturbations will be applied per batch.")

    @torch.no_grad()
    def _select_nodes_for_perturb(self, num_nodes: int, device: torch.device, seed_offset: int):
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + seed_offset)
        m = max(1, ceil(self.frac_nodes * num_nodes))
        perm = torch.randperm(num_nodes, generator=g)[:m]
        return perm.to(device)

    def _imperceptible_step_single(self, data):
        new_data = copy.deepcopy(data).to(self.device)
        if new_data.x is None:
            return new_data

        x = new_data.x.detach().clone().to(self.device).requires_grad_(True)

        was_training = self.target_model.training
        self.target_model.eval()

        with torch.enable_grad():
            tmp = copy.deepcopy(new_data)
            tmp.x = x

            emb, logits = self.target_model(tmp)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

            target = torch.tensor([self.wm_target_label], device=self.device, dtype=torch.long)
            loss = F.cross_entropy(logits, target)

            grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False, allow_unused=False)[0]

        if was_training:
            self.target_model.train()

        C = x.size(-1)
        r = min(self.last_feat_len, C) if self.last_feat_len > 0 else 0
        sel_nodes = self._select_nodes_for_perturb(x.size(0), x.device, seed_offset=int(new_data.num_nodes))

        x_new = x.detach().clone()
        if r > 0:
            x_new[sel_nodes, C - r:C] = x_new[sel_nodes, C - r:C] + self.eps * torch.sign(grad[sel_nodes, C - r:C])
        else:
            x_new[sel_nodes] = x_new[sel_nodes] + self.eps * torch.sign(grad[sel_nodes])

        new_data.x = x_new.detach()

        if self.enable_trigger_nodes and self.num_triggers > 0:
            new_data = self._add_tiny_trigger_nodes(new_data, self.num_triggers, self.connect_top_k)

        return new_data

    @staticmethod
    def _graph_degree_scores(edge_index, num_nodes, device):
        deg = torch.zeros(num_nodes, device=device)
        deg.index_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=device))
        return deg

    @torch.no_grad()
    def _add_tiny_trigger_nodes(self, data, num_triggers: int, connect_top_k: int):
        if num_triggers <= 0:
            return data

        new_data = copy.deepcopy(data).to(self.device)
        x = new_data.x
        N, C = x.size()
        device = x.device

        mean_feat = x.mean(dim=0, keepdim=True)  # [1, C]
        trig_feats = mean_feat.repeat(num_triggers, 1)  # [T, C]

        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + int(N))
        jitter = 0.01 * torch.randn(trig_feats.size(), generator=g, device=device, dtype=x.dtype)
        trig_feats = trig_feats + jitter

        deg = self._graph_degree_scores(new_data.edge_index, N, device)
        topk = torch.topk(deg, k=min(connect_top_k, N)).indices.tolist()

        adj = to_dense_adj(new_data.edge_index, max_num_nodes=N)[0]
        total_nodes = N + num_triggers
        new_adj = torch.zeros((total_nodes, total_nodes), device=device, dtype=adj.dtype)
        new_adj[:N, :N] = adj

        trig_ids = list(range(N, N + num_triggers))
        for t in trig_ids:
            for v in topk:
                new_adj[v, t] = 1
                new_adj[t, v] = 1

        new_data.x = torch.cat([x, trig_feats], dim=0)
        new_edge_index, _ = dense_to_sparse(new_adj)
        new_data.edge_index = new_edge_index
        return new_data

    def perturb_batch(self, batch: Batch) -> Batch:
        data_list = batch.to_data_list()
        new_list = [self._imperceptible_step_single(d) for d in data_list]
        new_batch = Batch.from_data_list(new_list)
        return new_batch

    def train_defense_model(self, epochs, lr=1e-3, weight_decay=1e-5, ce_weight: float = 1.0):
        model = self.target_model.to(self.device)
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total, loss_sum, acc_sum = 0, 0.0, 0.0
            for batch in self._wm_loader:
                batch = batch.to(self.device)

                batch_pert = self.perturb_batch(batch)

                _, logits = model(batch_pert)
                y = batch_pert.y.view(-1).long()

                loss_ce = criterion(logits, y)
                loss = ce_weight * loss_ce

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bs = batch_pert.num_graphs
                total += bs
                loss_sum += loss.item() * bs
                acc_sum += (logits.argmax(1) == y).float().sum().item()
            print(
                f"[ImperceptibleWM][train] epoch: {epoch}, loss: {loss_sum / total:.6f}, wm acc: {acc_sum / total:.6f}")

        model.eval()
        total, loss_sum, acc_sum = 0, 0.0, 0.0
        with torch.no_grad():
            for batch in self._wm_loader:
                batch = batch.to(self.device)
                batch_pert = self.perturb_batch(batch)
                _, logits = model(batch_pert)
                y = batch_pert.y.view(-1).long()
                loss = criterion(logits, y)
                bs = batch_pert.num_graphs
                total += bs
                loss_sum += loss.item() * bs
                acc_sum += (logits.argmax(1) == y).float().sum().item()
        print(f"[ImperceptibleWM][wm eval] loss: {loss_sum / total:.6f}, acc: {acc_sum / total:.6f}")

        return model.eval()

    def eval_downstream(self):
        model = self.target_model.eval()
        total, acc_sum = 0, 0.0
        with torch.no_grad():
            for batch in self._test_loader:
                batch = batch.to(self.device)
                _, logits = model(batch)
                y = batch.y.view(-1).long()
                bs = batch.num_graphs
                total += bs
                acc_sum += (logits.argmax(1) == y).float().sum().item()
        acc = acc_sum / max(1, total)
        print(f"[ImperceptibleWM]Downstream acc: {acc:.6f}")
        return acc

    def eval_wm(self):
        model = self.target_model.eval()
        total, acc_sum = 0, 0.0
        with torch.no_grad():
            for batch in self._wm_loader:
                batch = batch.to(self.device)
                batch_pert = self.perturb_batch(batch)
                _, logits = model(batch_pert)
                y = batch_pert.y.view(-1).long()
                bs = batch_pert.num_graphs
                total += bs
                acc_sum += (logits.argmax(1) == y).float().sum().item()
        acc = acc_sum / max(1, total)
        print(f"[ImperceptibleWM]WM acc: {acc:.6f}")
        return acc
