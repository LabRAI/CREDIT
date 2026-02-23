from typing import Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


class OwnershipVerifier_WM:
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

        self.ver_loader = self.defense_model._wm_loader

    @torch.no_grad()
    def _inference_ver_label(self) -> np.ndarray:
        label_list = []
        for batch in self.ver_loader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(self.device, non_blocking=True)

            _, logits = self.defense_model(images)

            preds = torch.argmax(logits, dim=1)

            label_list.append(preds.detach().cpu().numpy())

        labels = np.concatenate(label_list, axis=0)
        return labels

    @torch.no_grad()
    def _inference_sus_label(self, sus_models: List[nn.Module]) -> np.ndarray:
        """
        Inference suspicious models on the verification set,
        return predicted labels with shape [n_models, n_batch].
        """
        label_all = []

        for model in sus_models:
            model.to(self.device)
            model.eval()

            label_sus_list = []
            for batch in self.ver_loader:
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                images = images.to(self.device, non_blocking=True)

                _, logits = model(images)
                preds = torch.argmax(logits, dim=1)

                label_sus_list.append(preds.detach().cpu().numpy())

            labels = np.concatenate(label_sus_list, axis=0)  # [n_batch]
            label_all.append(labels)

        label_all = np.stack(label_all, axis=0)  # [n_models, n_batch]
        return label_all

    def verify(self, sus_models, sus_label) -> Dict:
        """
        Verify by label agreement.

        Args:
            sus_models: List[nn.Module], suspicious models
            sus_label:  List[int] or np.ndarray of shape [n_models], 1=surrogate, 0=independent
        """
        # 1) infer labels
        ver_labels = self._inference_ver_label()  # [n_ver]
        sus_labels_all = self._inference_sus_label(sus_models)  # [n_models, n_ver]

        # 2) accuracy per suspicious model
        #    broadcast ver_labels to [n_models, n_ver] then mean over axis=1
        matches = (sus_labels_all == ver_labels[None, :])  # [n_models, n_ver]
        acc_sus_all = matches.mean(axis=1)  # [n_models]

        # 3) AUC using accuracy as the score
        sus_label = np.asarray(sus_label, dtype=int).reshape(-1)  # [n_models]
        if sus_label.shape[0] != acc_sus_all.shape[0]:
            raise ValueError(f"Length mismatch: sus_label={sus_label.shape[0]} vs acc={acc_sus_all.shape[0]}")

        # roc_auc_score
        unique_classes = np.unique(sus_label)
        if unique_classes.size < 2:
            auc = float('nan')
        else:
            auc = float(roc_auc_score(sus_label, acc_sus_all))

        return {
            "auc": auc,
            "acc_sus_all": acc_sus_all,  # [n_models]
            "sus_label": sus_label,  # [n_models]
        }
