import os
import random
import tarfile
from abc import ABC, abstractmethod
from typing import List
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader as GeoDataLoader
from torchvision import datasets, transforms as T
from torchvision.datasets.utils import download_url


def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _stratified_split_indices(dataset: datasets.ImageFolder, train_ratio: float = 0.8, seed: int = 42) -> Tuple[
    List[int], List[int]]:
    seed_all(seed)
    class_to_indices = {cls_idx: [] for cls_idx in dataset.class_to_idx.values()}
    for idx, (_, y) in enumerate(dataset.samples):
        class_to_indices[y].append(idx)
    train_indices, test_indices = [], []
    for cls, idxs in class_to_indices.items():
        random.shuffle(idxs)
        n_train = int(len(idxs) * train_ratio)
        train_indices.extend(idxs[:n_train])
        test_indices.extend(idxs[n_train:])
    return train_indices, test_indices


class CustomDataset(ABC):
    def __init__(
            self,
            root: str = "./data/datasets",
            batch_size: int = 128,
            num_workers: int = 8,
            download: bool = False,
            train_ratio: float = 0.8,
            seed: int = 42,
    ):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        self.train_ratio = train_ratio
        self.seed = seed

        self.train_dataset = None
        self.test_dataset = None

        self.transform_train = None
        self.transform_test = None

        ensure_dir(self.root)

    @abstractmethod
    def _download(self):
        ...

    @abstractmethod
    def _preprocess(self):
        ...

    @abstractmethod
    def _load(self):
        ...

    def get_loader(self) -> Tuple[DataLoader, DataLoader]:
        if self.train_dataset is None or self.test_dataset is None:
            self._download()
            self._preprocess()
            self._load()

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        return train_loader, test_loader


class CIFAR10Dataset(CustomDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.transform_train = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)),
        ])
        self.transform_test = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)),
        ])

    def _download(self):
        # torchvision 自带
        if self.download:
            datasets.CIFAR10(self.root, train=True, download=True)
            datasets.CIFAR10(self.root, train=False, download=True)

    def _preprocess(self):
        pass

    def _load(self):
        self.train_dataset = datasets.CIFAR10(
            root=self.root, train=True, transform=self.transform_train, download=False
        )
        self.test_dataset = datasets.CIFAR10(
            root=self.root, train=False, transform=self.transform_test, download=False
        )


class CIFAR100Dataset(CustomDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.transform_train = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.5071, 0.4865, 0.4409),
                        (0.2673, 0.2564, 0.2762)),
        ])
        self.transform_test = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5071, 0.4865, 0.4409),
                        (0.2673, 0.2564, 0.2762)),
        ])

    def _download(self):
        if self.download:
            datasets.CIFAR100(self.root, train=True, download=True)
            datasets.CIFAR100(self.root, train=False, download=True)

    def _preprocess(self):
        pass

    def _load(self):
        self.train_dataset = datasets.CIFAR100(
            root=self.root, train=True, transform=self.transform_train, download=False
        )
        self.test_dataset = datasets.CIFAR100(
            root=self.root, train=False, transform=self.transform_test, download=False
        )


class Caltech256Dataset(CustomDataset):
    URLS = [
        "https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar?download=1"
    ]
    ARCHIVE_NAME = "256_ObjectCategories.tar"
    EXTRACTED_DIRNAME = "256_ObjectCategories"
    DATASET_DIR = "caltech256"

    def __init__(self, img_size: int = 224, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_root = os.path.join(self.root, self.DATASET_DIR)
        self.archive_path = os.path.join(self.dataset_root, self.ARCHIVE_NAME)
        self.extracted_path = os.path.join(self.dataset_root, self.EXTRACTED_DIRNAME)

        self.transform_train = T.Compose([
            T.Resize((256, 256)),
            T.RandomResizedCrop(img_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225)),
        ])
        self.transform_test = T.Compose([
            T.Resize((img_size, img_size)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225)),
        ])

    def _download(self):
        ensure_dir(self.dataset_root)
        if os.path.isdir(self.extracted_path) and len(os.listdir(self.extracted_path)) > 0:
            return

        if not self.download:
            return

        for url in self.URLS:
            try:
                print(f"Downloading Caltech256 from {url} ...")
                download_url(url, self.archive_path)
                break
            except Exception as e:
                print(f"Failed to download from {url}: {e}")
        else:
            raise RuntimeError("All Caltech256 URLs failed, please download manually.")

        print("Extracting Caltech256 archive...")
        with tarfile.open(self.archive_path, "r") as tar:
            tar.extractall(self.dataset_root)
        print("Extraction done.")

    def _preprocess(self):
        pass

    def _load(self):
        if not os.path.isdir(self.extracted_path):
            raise FileNotFoundError(
                f"Caltech256 not found at {self.extracted_path}. "
                f"Set download=True to auto download, or place files manually."
            )

        full_dataset = datasets.ImageFolder(root=self.extracted_path, transform=self.transform_train)
        test_dataset = datasets.ImageFolder(root=self.extracted_path, transform=self.transform_test)

        classes = [c for c in full_dataset.classes if c != "257.clutter"]
        class_to_idx = {c: i for i, c in enumerate(classes)}

        new_samples, new_targets = [], []
        for p, old_idx in full_dataset.samples:
            cls_name = full_dataset.classes[old_idx]
            if cls_name == "257.clutter":
                continue
            new_idx = class_to_idx[cls_name]
            new_samples.append((p, new_idx))
            new_targets.append(new_idx)

        full_dataset.samples = list(new_samples)
        full_dataset.targets = list(new_targets)
        full_dataset.classes = list(classes)
        full_dataset.class_to_idx = dict(class_to_idx)

        test_dataset.samples = list(new_samples)
        test_dataset.targets = list(new_targets)
        test_dataset.classes = list(classes)
        test_dataset.class_to_idx = dict(class_to_idx)
        # ----------------------------------------------------------------------

        self.num_classes = len(classes)

        train_idx, test_idx = _stratified_split_indices(
            full_dataset, train_ratio=self.train_ratio, seed=self.seed
        )

        self.train_dataset = Subset(full_dataset, train_idx)
        self.test_dataset = Subset(test_dataset, test_idx)


class CUB200Dataset(CustomDataset):
    DATASET_DIR = "CUB_200_2011"

    def __init__(self, img_size: int = 224, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_root = os.path.join(self.root, self.DATASET_DIR)

        self.transform_train = T.Compose([
            T.Resize((256, 256)),
            T.RandomResizedCrop(img_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225)),
        ])
        self.transform_test = T.Compose([
            T.Resize((img_size, img_size)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225)),
        ])

    def _download(self):
        pass

    def _preprocess(self):
        pass

    def _load(self):
        if hasattr(datasets, "CUB200"):
            try:
                self.train_dataset = datasets.CUB200(
                    root=self.root, train=True, transform=self.transform_train, download=self.download
                )
                self.test_dataset = datasets.CUB200(
                    root=self.root, train=False, transform=self.transform_test, download=self.download
                )
                return
            except Exception as e:
                print("Falling back to ImageFolder for CUB-200-2011:", e)

        train_dir = os.path.join(self.dataset_root, "train")
        test_dir = os.path.join(self.dataset_root, "test")
        if not (os.path.isdir(train_dir) and os.path.isdir(test_dir)):
            raise FileNotFoundError(
                f"CUB-200-2011 not found for ImageFolder fallback. "
                f"Expected directories: {train_dir} and {test_dir}."
            )
        self.train_dataset = datasets.ImageFolder(root=train_dir, transform=self.transform_train)
        self.test_dataset = datasets.ImageFolder(root=test_dir, transform=self.transform_test)


class CustomGraphDataset(ABC):
    def __init__(
            self,
            root: str = "./data/graph_datasets",
            batch_size: int = 128,
            num_workers: int = 8,
            train_ratio: float = 0.8,
            seed: int = 42,
    ):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.seed = seed

        self.train_dataset = None  # type: Optional[torch.utils.data.Dataset]
        self.test_dataset = None  # type: Optional[torch.utils.data.Dataset]

        ensure_dir(self.root)

    @abstractmethod
    def _load(self) -> None:
        """
        Must set self.train_dataset and self.test_dataset.
        """
        ...

    def get_loader(self) -> Tuple[GeoDataLoader, GeoDataLoader]:
        if self.train_dataset is None or self.test_dataset is None:
            self._load()

        train_loader = GeoDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = GeoDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return train_loader, test_loader


class ENZYMESDataset(CustomGraphDataset):
    """
    ENZYMES from TU collection.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "ENZYMES"
        self.num_classes = 6

    def _load(self) -> None:
        ds = TUDataset(root=os.path.join(self.root, self.name), name=self.name)
        n_total = len(ds)
        n_train = int(n_total * self.train_ratio)
        n_test = n_total - n_train
        g = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.test_dataset = random_split(ds, [n_train, n_test], generator=g)


class COLLABDataset(CustomGraphDataset):
    """
    COLLAB from TU collection.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "COLLAB"
        self.num_classes = 3

    def _load(self) -> None:
        ds = TUDataset(root=os.path.join(self.root, self.name), name=self.name)
        n_total = len(ds)
        n_train = int(n_total * self.train_ratio)
        n_test = n_total - n_train
        g = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.test_dataset = random_split(ds, [n_train, n_test], generator=g)


class PROTEINSDataset(CustomGraphDataset):
    """
    PROTEINS from TU collection.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "PROTEINS"
        self.num_classes = 3

    def _load(self) -> None:
        ds = TUDataset(root=os.path.join(self.root, self.name), name=self.name)
        n_total = len(ds)
        n_train = int(n_total * self.train_ratio)
        n_test = n_total - n_train
        g = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.test_dataset = random_split(ds, [n_train, n_test], generator=g)


def get_dataset(
        name: str,
        root: str = "./data/datasets",
        batch_size: int = 128,
        num_workers: int = 4,
        download: bool = True,
        train_ratio: float = 0.8,
        seed: int = 42,
):
    name = name.lower()
    if name == "cifar10":
        ds = CIFAR10Dataset(root=root, batch_size=batch_size, num_workers=num_workers, download=download,
                            train_ratio=train_ratio, seed=seed)
    elif name == "cifar100":
        ds = CIFAR100Dataset(root=root, batch_size=batch_size, num_workers=num_workers, download=download,
                             train_ratio=train_ratio, seed=seed)
    elif name == "caltech256":
        ds = Caltech256Dataset(root=root, batch_size=batch_size, num_workers=num_workers, download=download,
                               train_ratio=train_ratio, seed=seed)
    elif name == "cub200":
        ds = CUB200Dataset(root=root, batch_size=batch_size, num_workers=num_workers, download=download,
                           train_ratio=train_ratio, seed=seed)
    else:
        raise ValueError(f"Unknown dataset name: {name}")
    return ds


def get_graph_dataset(
        name: str,
        root: str = "./data/graph_datasets",
        batch_size: int = 128,
        num_workers: int = 8,
        train_ratio: float = 0.8,
        seed: int = 42,
) -> "CustomGraphDataset":
    """
    Factory for graph datasets. Returns a CustomGraphDataset subclass instance.
    """
    key = name.lower()
    if key in {"enzymes"}:
        ds = ENZYMESDataset(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            train_ratio=train_ratio,
            seed=seed,
        )
    elif key in {"collab"}:
        ds = COLLABDataset(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            train_ratio=train_ratio,
            seed=seed,
        )
    elif key in {"proteins"}:
        ds = PROTEINSDataset(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            train_ratio=train_ratio,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown graph dataset name: {name}")
    return ds


def inspect_graph_dataset(ds: "CustomGraphDataset"):
    print("Train size:", len(ds.train_dataset))
    print("Test size:", len(ds.test_dataset))

    raw_ds = ds.train_dataset.dataset
    print(raw_ds)
    print("Num classes:", raw_ds.num_classes)
    print("Node feature dim:", raw_ds.num_node_features)

    data = raw_ds[0]
    print("First graph:", data)
    print("Num nodes:", data.num_nodes)
    print("Num edges:", data.num_edges)
    print("Graph label:", data.y.item())
