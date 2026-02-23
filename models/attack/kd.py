from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from utils.datasets import seed_all


class KnowledgeDistillation:
    def __init__(
            self,
            train_dataset: Dataset,
            test_dataset: Dataset,
            target_model: nn.Module,
            surrogate_model: nn.Module,
            device: Optional[str] = None,
            batch_size: int = 128,
            seed: int = 42,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.teacher = target_model.to(self.device).eval()
        self.student = surrogate_model.to(self.device)
        self.batch_size = batch_size

        seed_all(seed)

        self.all_indices = np.arange(len(self.train_dataset))
        np.random.shuffle(self.all_indices)

        self.buf_x: List[torch.Tensor] = []
        self.buf_te: List[torch.Tensor] = []
        self.buf_tl: List[torch.Tensor] = []

    @torch.no_grad()
    def query(
            self,
            budget_ratio: float = 0.1,
            queries_per_image: int = 1,
            transform=None,
    ):
        assert 0 < budget_ratio <= 1
        budget = int(budget_ratio * len(self.train_dataset))
        chosen = self.all_indices[:budget]
        bs = self.batch_size

        for s in range(0, len(chosen), bs):
            e = min(s + bs, len(chosen))
            idxs = chosen[s:e]

            imgs = []
            for i in idxs:
                x_i = self.train_dataset[i][0]
                x_i = transform(x_i) if transform is not None else x_i
                if not torch.is_tensor(x_i):
                    x_i = torch.as_tensor(x_i)
                imgs.append(x_i)
            x = torch.stack(imgs, 0).to(self.device)

            embeds, logits = [], []
            for _ in range(queries_per_image):
                emb_t, logit_t = self.teacher(x)
                embeds.append(emb_t)
                logits.append(logit_t)
            emb_mean = torch.stack(embeds, 0).mean(0)
            logit_mean = torch.stack(logits, 0).mean(0)

            self.buf_x.append(x.detach().cpu())
            self.buf_te.append(emb_mean.detach().cpu())
            self.buf_tl.append(logit_mean.detach().cpu())

    def get_transferset(self):
        out = []
        for bx, be, bl in zip(self.buf_x, self.buf_te, self.buf_tl):
            for i in range(bx.size(0)):
                out.append({"input": bx[i], "teacher_embed": be[i], "teacher_logits": bl[i]})
        return out

    def _build_loader(self, shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
        assert len(self.buf_x) > 0, "call query first"
        X = torch.cat(self.buf_x, 0)
        T_e = torch.cat(self.buf_te, 0)
        T_l = torch.cat(self.buf_tl, 0)
        ds = TensorDataset(X, T_e, T_l)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, drop_last=False), DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

    @torch.no_grad()
    def query_multi_sample(
            self,
            budget_ratio: float = 0.1,
            queries_per_image: int = 1,
            transform=None,
            ema_tau: float | None = 0.7,  # None: no EMA
            clear_buffer: bool = True,
    ):
        assert 0 < budget_ratio <= 1
        assert queries_per_image >= 1

        if clear_buffer:
            self.buf_x, self.buf_te, self.buf_tl = [], [], []

        budget = int(budget_ratio * len(self.train_dataset))
        chosen = self.all_indices[:budget]
        bs = self.batch_size

        self.teacher.eval()

        for s in range(0, len(chosen), bs):
            e = min(s + bs, len(chosen))
            idxs = chosen[s:e]

            imgs = []
            for i in idxs:
                x_i = self.train_dataset[i][0]
                x_i = transform(x_i) if transform is not None else x_i
                if not torch.is_tensor(x_i):
                    x_i = torch.as_tensor(x_i)
                imgs.append(x_i)
            x = torch.stack(imgs, 0).to(self.device, non_blocking=True)

            emb_list, log_list = [], []
            with torch.inference_mode():
                for _ in range(queries_per_image):
                    emb_t, logit_t = self.teacher(x)
                    emb_list.append(emb_t)
                    log_list.append(logit_t)

            emb_stack = torch.stack(emb_list, dim=0)
            log_stack = torch.stack(log_list, dim=0)

            if ema_tau is None:
                emb_agg = emb_stack.mean(dim=0)
                log_agg = log_stack.mean(dim=0)
            else:
                emb_agg = emb_stack[0]
                log_agg = log_stack[0]
                for i in range(1, emb_stack.size(0)):
                    emb_agg = ema_tau * emb_agg + (1 - ema_tau) * emb_stack[i]
                    log_agg = ema_tau * log_agg + (1 - ema_tau) * log_stack[i]

            self.buf_x.append(x.detach().cpu())
            self.buf_te.append(emb_agg.detach().cpu())
            self.buf_tl.append(log_agg.detach().cpu())

    def train_surrogate_model(
            self,
            epochs: int = 20,
            optimizer=None,
            scheduler=None,
            temperature: float = 4.0,
            alpha_pred: float = 1.0,
            beta_emb: float = 1.0,
            wandb=None,
    ) -> nn.Module:
        train_loader, test_loader = self._build_loader(shuffle=True)
        self.student.train()
        opt = optimizer
        sch = scheduler
        kd_kl = nn.KLDivLoss(reduction="batchmean")
        mse = nn.MSELoss()

        best_acc = 0.0
        best_model = None
        for ep in range(1, epochs + 1):
            tot, tot_kd, tot_mse, n = 0.0, 0.0, 0.0, 0
            self.student.train()
            for x, t_emb, t_log in train_loader:
                x = x.to(self.device, non_blocking=True)
                t_emb = t_emb.to(self.device, non_blocking=True)
                t_log = t_log.to(self.device, non_blocking=True)

                s_emb, s_log = self.student(x)

                T = temperature
                loss_pred = kd_kl(F.log_softmax(s_log / T, dim=1), F.softmax(t_log / T, dim=1)) * (T * T)

                # s_e = F.normalize(s_emb.view(s_emb.size(0), -1), dim=1)
                # t_e = F.normalize(t_emb.view(t_emb.size(0), -1), dim=1)
                # loss_emb = mse(s_e, t_e)
                loss_emb = mse(s_emb, t_emb)

                loss = alpha_pred * loss_pred + beta_emb * loss_emb

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sch.step()

                b = x.size(0)
                tot += loss.item() * b
                tot_kd += loss_pred.item() * b
                tot_mse += loss_emb.item() * b
                n += b

            self.student.eval()
            total, acc_sum = 0, 0.0
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                _, logits = self.student(x)
                bs = y.size(0)
                total += bs
                acc_sum += (logits.argmax(1) == y).float().sum().item()
            val_acc = acc_sum / total

            if val_acc > best_acc:
                best_acc = val_acc
                best_model = self.student

            if wandb is not None:
                wandb.log({
                    "epoch": ep,
                    "total_loss": tot / n,
                    "kd_loss": tot_kd / n,
                    "emb_loss": tot_mse / n,
                    "val_acc": val_acc,
                    "lr": optimizer.param_groups[0]["lr"],
                })

        return best_model
