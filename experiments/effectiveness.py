import itertools
import os
from pathlib import Path

import numpy as np
import ray
import torch
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from models.attack.kd import KnowledgeDistillation
from models.credit.credit import CREDIT
from models.nn import ResNet, VGG, GoogLeNet, DenseNet
from utils.datasets import CIFAR10Dataset, CIFAR100Dataset
from utils.metrics import compute_downstream_perf, save_metrics_table


def _train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total, loss_sum, acc_sum = 0, 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        _, logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        bs = y.size(0)
        total += bs
        loss_sum += loss.item() * bs
        acc_sum += (logits.argmax(1) == y).float().sum().item()
    return loss_sum / total, acc_sum / total


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    model.eval()
    total, loss_sum, acc_sum = 0, 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, logits = model(x)
        loss = criterion(logits, y)
        bs = y.size(0)
        total += bs
        loss_sum += loss.item() * bs
        acc_sum += (logits.argmax(1) == y).float().sum().item()
    return loss_sum / total, acc_sum / total


@torch.no_grad()
def _run_inference(model, dataloader, device):
    model.eval()
    model.to(device)

    preds, labels = [], []
    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        emb, logits = model(x)
        p = torch.argmax(logits, dim=1)
        preds.append(p.detach().cpu())
        labels.append(y.detach().cpu())

    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(labels).numpy()

    return y_true, y_pred


@ray.remote(num_gpus=1)
def worker_cv(dataset_name, backbone_name, optimizer_name, scheduler_name, cfg):
    """
    optimizer_name: "sgd" | "adam" | "adamw" | "rmsprop"
    scheduler_name: "cosine" | "step" | "none"
    """
    import time
    from pathlib import Path
    import torch
    import torch.nn as nn
    from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
    import wandb

    download = False
    num_workers = 8
    batch_size = 128

    if dataset_name == "cifar10":
        dataset = CIFAR10Dataset(batch_size=batch_size,
                                 download=download, num_workers=num_workers)
        num_classes, small_input = 10, True
    elif dataset_name == "cifar100":
        dataset = CIFAR100Dataset(batch_size=batch_size,
                                  download=download, num_workers=num_workers)
        num_classes, small_input = 100, True
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    train_loader, test_loader = dataset.get_loader()

    if backbone_name == "resnet":
        model = ResNet(num_classes=num_classes, small_input=small_input)
    elif backbone_name == "vgg":
        model = VGG(num_classes=num_classes, small_input=small_input)
    elif backbone_name == "densenet":
        model = DenseNet(num_classes=num_classes, small_input=small_input)
    elif backbone_name == "googlenet":
        model = GoogLeNet(num_classes=num_classes, small_input=small_input)
    else:
        raise ValueError(f"Unknown backbone_name: {backbone_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(cfg.get("label_smoothing", 0.0))
    ).to(device)

    lr = float(cfg.get("lr", 0.1))
    weight_decay = float(cfg.get("weight_decay", 5e-4))
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(cfg.get("nesterov", True)),
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=tuple(cfg.get("betas", (0.9, 0.999))),
            eps=float(cfg.get("eps", 1e-8)),
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=tuple(cfg.get("betas", (0.9, 0.999))),
            eps=float(cfg.get("eps", 1e-8)),
            weight_decay=weight_decay,
        )
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            alpha=float(cfg.get("rmsprop_alpha", 0.99)),
            eps=float(cfg.get("eps", 1e-8)),
            momentum=float(cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
            centered=bool(cfg.get("rmsprop_centered", False)),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    epochs = int(cfg.get("epochs", 20))
    scheduler_name = scheduler_name.lower()
    if scheduler_name == "cosine":
        T_max = int(cfg.get("cosine_T_max", epochs))
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_name == "step":
        step_size = int(cfg.get("step_size", max(1, epochs // 3)))
        gamma = float(cfg.get("step_gamma", 0.1))
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    # wandb
    run = wandb.init(
        project="certified",
        group=cfg["wandb_group"],
        name=f"{dataset_name}_{backbone_name}_{optimizer_name}_{scheduler_name}",
        tags=[dataset_name, backbone_name, optimizer_name, scheduler_name],
        config={"dataset": dataset_name, "backbone": backbone_name,
                "optimizer": optimizer_name, "scheduler": scheduler_name, **cfg},
        settings=wandb.Settings(init_timeout=180, start_method="thread"),
        reinit=True,
    )

    best_acc, best_epoch = 0.0, -1
    save_root = Path(cfg["save_root"])
    save_root.mkdir(parents=True, exist_ok=True)
    save_path = save_root / f"{dataset_name}_{backbone_name}_{optimizer_name}_{scheduler_name}.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = _evaluate(model, test_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()
        dt = time.time() - t0

        wandb.log({
            "epoch": epoch,
            "time_per_epoch_sec": dt,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss, "val_acc": va_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if va_acc > best_acc:
            best_acc, best_epoch = va_acc, epoch
            torch.save(
                dict(
                    model=model.state_dict(),
                    acc=best_acc,
                    epoch=best_epoch,
                    dataset=dataset_name,
                    backbone=backbone_name,
                    optimizer=optimizer_name,
                    scheduler=scheduler_name,
                    hparams=dict(lr=lr, weight_decay=weight_decay, epochs=epochs),
                    cfg=cfg,
                ),
                save_path,
            )

    wandb.summary["best_val_acc"] = best_acc
    wandb.summary["best_epoch"] = best_epoch
    wandb.finish()
    return dataset_name, backbone_name, optimizer_name, scheduler_name, best_acc, best_epoch


@ray.remote(num_gpus=1)
def worker_cv_credit_kd(dataset_name, backbone_name, optimizer_name, scheduler_name, cfg):
    download = False
    num_workers = 8
    batch_size = 128

    if dataset_name == "cifar10":
        dataset = CIFAR10Dataset(batch_size=batch_size,
                                 download=download, num_workers=num_workers)
        num_classes, small_input = 10, True
    elif dataset_name == "cifar100":
        dataset = CIFAR100Dataset(batch_size=batch_size,
                                  download=download, num_workers=num_workers)
        num_classes, small_input = 100, True
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    train_loader, test_loader = dataset.get_loader()

    if backbone_name == "resnet":
        model = ResNet(num_classes=num_classes, small_input=small_input, pretrained=False)
    elif backbone_name == "vgg":
        model = VGG(num_classes=num_classes, small_input=small_input, pretrained=False)
    elif backbone_name == "densenet":
        model = DenseNet(num_classes=num_classes, small_input=small_input, pretrained=False)
    elif backbone_name == "googlenet":
        model = GoogLeNet(num_classes=num_classes, small_input=small_input, pretrained=False)
    else:
        raise ValueError(f"Unknown backbone_name: {backbone_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(cfg.get("label_smoothing", 0.0))
    ).to(device)

    lr = float(cfg.get("lr", 0.1))
    weight_decay = float(cfg.get("weight_decay", 5e-4))
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(cfg.get("nesterov", True)),
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=tuple(cfg.get("betas", (0.9, 0.999))),
            eps=float(cfg.get("eps", 1e-8)),
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=tuple(cfg.get("betas", (0.9, 0.999))),
            eps=float(cfg.get("eps", 1e-8)),
            weight_decay=weight_decay,
        )
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            alpha=float(cfg.get("rmsprop_alpha", 0.99)),
            eps=float(cfg.get("eps", 1e-8)),
            momentum=float(cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
            centered=bool(cfg.get("rmsprop_centered", False)),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    epochs = int(cfg.get("epochs", 20))
    scheduler_name = scheduler_name.lower()
    if scheduler_name == "cosine":
        T_max = int(cfg.get("cosine_T_max", epochs))
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_name == "step":
        step_size = int(cfg.get("step_size", max(1, epochs // 3)))
        gamma = float(cfg.get("step_gamma", 0.1))
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    # wandb
    run = wandb.init(
        project="certified",
        group=cfg["wandb_group"],
        name=f"{dataset_name}_{backbone_name}_{optimizer_name}_{scheduler_name}",
        tags=[dataset_name, backbone_name, optimizer_name, scheduler_name],
        config={"dataset": dataset_name, "backbone": backbone_name,
                "optimizer": optimizer_name, "scheduler": scheduler_name, **cfg},
        settings=wandb.Settings(init_timeout=180, start_method="thread"),
        reinit=True,
    )

    teacher = ResNet(num_classes=num_classes, small_input=small_input).to(device)
    state = torch.load(f"./data/models/target/{dataset_name}_resnet_sgd_cosine.pt", map_location=device)
    teacher.load_state_dict(state["model"], strict=False)
    teacher.eval()

    sigma = cfg["sigma"]
    temperature = 4.0
    alpha_pred = cfg["alpha_pred"]
    beta_emb = cfg["beta_emb"]
    seed = 42

    dataset_ratio = cfg["ver_dataset_ratio"]
    budget_ratio = cfg["query_budget_ratio"]

    teacher = CREDIT(teacher, test_loader.dataset, dataset_ratio=dataset_ratio, sigma=sigma, enable=True)

    kd = KnowledgeDistillation(
        train_dataset=train_loader.dataset,
        test_dataset=test_loader.dataset,
        target_model=teacher,
        surrogate_model=model,
        device=device,
        batch_size=batch_size,
        seed=seed,
    )

    kd.query(budget_ratio=budget_ratio, queries_per_image=1)

    surrogate = kd.train_surrogate_model(
        epochs=epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        temperature=temperature,
        alpha_pred=alpha_pred,
        beta_emb=beta_emb,
        wandb=wandb
    )

    save_root = Path(cfg["save_root"])
    save_root.mkdir(parents=True, exist_ok=True)
    save_path = save_root / f"{dataset_name}_{backbone_name}_{optimizer_name}_{scheduler_name}.pt"

    torch.save(
        dict(
            model=surrogate.state_dict(),
            dataset=dataset_name,
            backbone=backbone_name,
            optimizer=optimizer_name,
            scheduler=scheduler_name,
            hparams=dict(lr=lr, weight_decay=weight_decay, epochs=epochs),
            cfg=cfg,
        ),
        save_path,
    )
    wandb.finish()
    return dataset_name, backbone_name, optimizer_name, scheduler_name


def train_cv_target():
    ray.init(ignore_reinit_error=True)

    config = {
        "epochs": 10,
        "wandb_group": "cv_target",
        "save_root": "./data/models/target"
    }
    datasets = ["cifar10"]  # , "cifar100"
    backbones = ["resnet"]  # "resnet", "vgg", "googlenet", "densenet"
    optimizers = ["sgd"]
    schedulers = ["cosine"]

    futures = []
    for ds, bk, o, s in itertools.product(datasets, backbones, optimizers, schedulers):
        futures.append(worker_cv.remote(ds, bk, o, s, config))

    results = ray.get(futures)
    print("Training finished.")
    for ds, bk, o, s, acc, ep in results:
        print(f"{ds}-{bk}-{o}-{s}: best_acc={acc:.4f} @ epoch {ep}")


def train_cv_ind():
    ray.init(ignore_reinit_error=True)

    config = {
        "epochs": 20,
        "wandb_group": "cv_independent",
        "save_root": "./data/models/independent",
        "lr": 0.001,  # 0.001, 0.0005 for rmsprop
        "weight_decay": 1.0e-5,
    }
    datasets = ["cifar10"]  # , "cifar100"
    backbones = ["vgg", "googlenet", "densenet"]  # "vgg",
    optimizers = ["adam", "adamw", "rmsprop"]  # "adam", "adamw",
    schedulers = ["cosine", "step"]  #

    futures = []
    for ds, bk, o, s in itertools.product(datasets, backbones, optimizers, schedulers):
        futures.append(worker_cv.remote(ds, bk, o, s, config))

    results = ray.get(futures)
    print("Training finished.")
    for ds, bk, o, s, acc, ep in results:
        print(f"{ds}-{bk}-{o}-{s}: best_acc={acc:.4f} @ epoch {ep}")


def train_cv_surrogate_credit():
    # train surrogate model
    ray.init(ignore_reinit_error=True)

    config = {
        "epochs": 40,  # 40 for both
        "wandb_group": "cv_surrogates_credit",
        "save_root": "./data/models/surrogate/credit",
        "lr": 0.001,  # 0.001 for cifar-10; 0.001 for cifar-100
        "weight_decay": 1.0e-5,  # 1.0e-5 for cifar-10; 5.0e-4 for cifar-100
        "sigma": 0.27,
        "alpha_pred": 0.01,  # 0.02 for cifar-10; 1.0 for cifar-100
        "beta_emb": 10.0,  # 10.0 for cifar-10; 1.0 for cifar-100
        "ver_dataset_ratio": 0.1,
        "query_budget_ratio": 0.8,  # 0.6
    }

    datasets = ["cifar10"]
    backbones = ["vgg", "googlenet", "densenet"]
    optimizers = ["adam"]  # "adamw", "rmsprop"
    schedulers = ["cosine"]  # , "step"

    futures = []
    for ds, bk, o, s in itertools.product(datasets, backbones, optimizers, schedulers):
        futures.append(worker_cv_credit_kd.remote(ds, bk, o, s, config))

    results = ray.get(futures)
    print("Training finished.")
    print(results)


def get_model(backbone_name: str, dataset_name: str, optimizer_name: str, scheduler_name: str,
              model_dir: str = "./data/models/target"):
    # load dataset
    if dataset_name == "cifar10":
        num_classes, small_input = 10, True
    elif dataset_name == "cifar100":
        num_classes, small_input = 100, True
    else:
        raise ValueError(dataset_name)
    # backbone
    if backbone_name == "resnet":
        model = ResNet(num_classes=num_classes, small_input=small_input)
    elif backbone_name == "vgg":
        model = VGG(num_classes=num_classes, small_input=small_input)
    elif backbone_name == "densenet":
        model = DenseNet(num_classes=num_classes, small_input=small_input)
    elif backbone_name == "googlenet":
        model = GoogLeNet(num_classes=num_classes, small_input=small_input)
    else:
        raise ValueError(backbone_name)

    # load model
    ckpt_name = f"{dataset_name}_{backbone_name}_{optimizer_name}_{scheduler_name}.pt"
    ckpt_path = os.path.join(model_dir, ckpt_name)

    if not os.path.isfile(ckpt_path):
        print(f"[WARN] Checkpoint not found: {ckpt_path}, skip this combination.")
        return None

    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = state.get("model", state)
    model.load_state_dict(state_dict, strict=False)

    return model


def get_model_baseline(backbone_name: str, dataset_name: str, optimizer_name: str, scheduler_name: str,
                       baseline_name: str,
                       model_dir: str = "./data/models/target/baseline"):
    # load dataset
    if dataset_name == "cifar10":
        num_classes, small_input = 10, True
    elif dataset_name == "cifar100":
        num_classes, small_input = 100, True
    else:
        raise ValueError(dataset_name)
    # backbone
    if backbone_name == "resnet":
        model = ResNet(num_classes=num_classes, small_input=small_input)
    elif backbone_name == "vgg":
        model = VGG(num_classes=num_classes, small_input=small_input)
    elif backbone_name == "densenet":
        model = DenseNet(num_classes=num_classes, small_input=small_input)
    elif backbone_name == "googlenet":
        model = GoogLeNet(num_classes=num_classes, small_input=small_input)
    else:
        raise ValueError(backbone_name)

    # load model
    ckpt_name = f"{dataset_name}_{backbone_name}_{optimizer_name}_{scheduler_name}_{baseline_name}.pt"
    ckpt_path = os.path.join(model_dir, ckpt_name)

    if not os.path.isfile(ckpt_path):
        print(f"[WARN] Checkpoint not found: {ckpt_path}, skip this combination.")
        return None

    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = state.get("model", state)
    model.load_state_dict(state_dict, strict=False)

    return model


def get_model_dataloader(backbone_name: str, dataset_name: str, optimizer_name: str, scheduler_name: str,
                         model_dir: str = "./data/models/target"):
    # load dataset
    if dataset_name == "cifar10":
        dataset = CIFAR10Dataset()
        num_classes, small_input = 10, True
    elif dataset_name == "cifar100":
        dataset = CIFAR100Dataset()
        num_classes, small_input = 100, True
    else:
        raise ValueError(dataset_name)

    _, test_loader = dataset.get_loader()

    # backbone
    if backbone_name == "resnet":
        model = ResNet(num_classes=num_classes, small_input=small_input)
    elif backbone_name == "vgg":
        model = VGG(num_classes=num_classes, small_input=small_input)
    elif backbone_name == "densenet":
        model = DenseNet(num_classes=num_classes, small_input=small_input)
    elif backbone_name == "googlenet":
        model = GoogLeNet(num_classes=num_classes, small_input=small_input)
    else:
        raise ValueError(backbone_name)

    # load model
    ckpt_name = f"{dataset_name}_{backbone_name}_{optimizer_name}_{scheduler_name}.pt"
    ckpt_path = os.path.join(model_dir, ckpt_name)

    if not os.path.isfile(ckpt_path):
        print(f"[WARN] Checkpoint not found: {ckpt_path}, skip this combination.")
        return None, None

    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = state.get("model", state)
    model.load_state_dict(state_dict, strict=False)

    return model, test_loader


def eval_downstream_perf(verbose=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_records = []
    missing = []

    # model_dir = "./data/models/independent"
    # model_dir = "./data/models/surrogate/credit"
    model_dir = "./data/models/target"

    datasets = ["cifar10", "cifar100"]  # , "cifar100"
    backbones = ["resnet", "vgg", "densenet", "googlenet"]
    optimizers = ["sgd"]  # , "adamw", "rmsprop"
    schedulers = ["cosine"]  # , "step"

    for backbone, dataset, optimizer, scheduler in itertools.product(backbones, datasets, optimizers, schedulers):
        model, test_loader = get_model_dataloader(backbone, dataset, optimizer, scheduler, model_dir=model_dir)

        if model is None:
            miss_instance = {
                "backbone": backbone,
                "dataset": dataset,
                "optimizer": optimizer,
                "scheduler": scheduler
            }
            missing.append(miss_instance)
            if verbose:
                print(miss_instance)
            continue

        y_true, y_pred = _run_inference(model, test_loader, device)
        perf = compute_downstream_perf(y_true, y_pred)

        record = {
            "backbone": backbone,
            "dataset": dataset,
            "optimizer": optimizer,
            "scheduler": scheduler,
            **perf,
        }

        all_records.append(record)

        if verbose:
            print(record)

    if all_records:
        save_metrics_table(all_records, out_name='downstream_perf_vanilla')

    return all_records, missing


def eval_downstream_perf_credit(verbose=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_records = []
    missing = []

    model_dir = "./data/models/target"
    datasets = ["cifar10", "cifar100"]
    backbones = ["resnet", "vgg", "densenet", "googlenet"]
    optimizers = ["sgd"]
    schedulers = ["cosine"]

    sigma = 0.4
    ver_dataset_ratio = 0.1

    for backbone, dataset, optimizer, scheduler in itertools.product(backbones, datasets, optimizers, schedulers):
        model, test_loader = get_model_dataloader(backbone, dataset, optimizer, scheduler, model_dir=model_dir)

        if model is None:
            miss_instance = {
                "backbone": backbone,
                "dataset": dataset,
                "optimizer": optimizer,
                "scheduler": scheduler
            }
            missing.append(miss_instance)
            if verbose:
                print(miss_instance)
            continue

        credit = CREDIT(target_model=model, dataset=test_loader.dataset, dataset_ratio=ver_dataset_ratio, sigma=sigma)
        y_true, y_pred = _run_inference(credit, test_loader, device)
        perf = compute_downstream_perf(y_true, y_pred)

        record = {
            "backbone": backbone,
            "dataset": dataset,
            "optimizer": optimizer,
            "scheduler": scheduler,
            **perf,
            "sigma": credit.sigma,
            "dataset_ratio": credit.dataset_ratio
        }

        all_records.append(record)

        if verbose:
            print(record)

    if all_records:
        save_metrics_table(all_records, out_name='effect_credit')

    return all_records, missing


def eval_ov_credit(backbone_baseline_model_name):
    from models.nn.resnet import ResNet
    from models.credit.credit import CREDIT
    from models.credit.ov import OwnershipVerifier
    from utils.datasets import get_dataset

    device = torch.device("cuda:2")

    # CREDIT params
    sigma = 0.27

    # verification params
    ver_dataset_ratio = 0.1  # verification set ratio
    k = 5  # ksg estimator
    batch_size = 128
    num_workers = 8
    seed = 42

    # suspicious models
    datasets = ["cifar10"]  # , "cifar100"
    backbones = ["vgg", "googlenet", "densenet"]
    optimizers = ["adam"]  # , "adamw", "rmsprop"
    schedulers = ["cosine"]  # , "step"

    # load dataset
    ds = get_dataset(name="cifar10")
    train_loader, test_loader = ds.get_loader()

    # 1. load defense model
    target_model_path = f"./data/models/target/cifar10_{backbone_baseline_model_name}_sgd_cosine.pt"
    if backbone_baseline_model_name == "resnet":
        target_model = ResNet(num_classes=10, small_input=True)
    elif backbone_baseline_model_name == "vgg":
        target_model = VGG(num_classes=10, small_input=True)
    elif backbone_baseline_model_name == "densenet":
        target_model = DenseNet(num_classes=10, small_input=True)
    elif backbone_baseline_model_name == "googlenet":
        target_model = GoogLeNet(num_classes=10, small_input=True)
    else:
        raise ValueError(backbone_baseline_model_name)
    target_model.to(device)
    state = torch.load(target_model_path, map_location="cpu")
    state_dict = state.get("model", state)
    target_model.load_state_dict(state_dict, strict=False)
    defense_model = CREDIT(
        target_model=target_model,
        dataset=train_loader.dataset,
        dataset_ratio=ver_dataset_ratio,
        sigma=sigma
    )

    # 2. load surrogate models
    surrogate_models = []
    sur_model_dir = "./data/models/surrogate/credit"
    for backbone, dataset, optimizer, scheduler in itertools.product(backbones, datasets, optimizers, schedulers):
        sur_model = get_model(backbone, dataset, optimizer, scheduler, model_dir=sur_model_dir)
        surrogate_models.append(sur_model)

    # 3. load independent models
    independent_models = []
    ind_model_dir = "./data/models/independent"
    for backbone, dataset, optimizer, scheduler in itertools.product(backbones, datasets, optimizers, schedulers):
        ind_model = get_model(backbone, dataset, optimizer, scheduler, model_dir=ind_model_dir)
        independent_models.append(ind_model)

    # combine
    suspicious_models = []
    suspicious_label = []
    for s in surrogate_models:
        suspicious_models.append(s)
        suspicious_label.append(1)
    for i in independent_models:
        suspicious_models.append(i)
        suspicious_label.append(0)

    # 4. init ov pipeline
    ov = OwnershipVerifier(
        defense_model=defense_model,
        dataset=test_loader.dataset,
        ver_ratio=ver_dataset_ratio,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        seed=seed,
    )

    # 5. ownership verification
    ver_res = ov.verify(suspicious_models, suspicious_label, k=k)

    res = {
        "baseline_name": "CREDIT",
        "baseline_model_name": backbone_baseline_model_name,
        **ver_res
    }

    print(res)

    return res


def run_eval_ov_credit():
    from utils.metrics import save_metrics_table
    baseline_model_name = ["resnet", "vgg", "densenet", "googlenet"]

    records = []
    for blm in baseline_model_name:
        res = eval_ov_credit(blm)
        records.append(res)

    save_metrics_table(records, out_name="auc_credit")


@ray.remote(num_gpus=1)
def worker_cv_baseline(dataset_name, backbone_name, optimizer_name, scheduler_name, baseline_name, cfg):
    from models.defense import Backdooring, EWE, IPGuard, UAP
    download = False
    pretrained = False
    num_workers = 8
    batch_size = 128

    if dataset_name == "cifar10":
        dataset = CIFAR10Dataset(batch_size=batch_size,
                                 download=download, num_workers=num_workers)
        num_classes, small_input = 10, True
    elif dataset_name == "cifar100":
        dataset = CIFAR100Dataset(batch_size=batch_size,
                                  download=download, num_workers=num_workers)
        num_classes, small_input = 100, True
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    train_loader, test_loader = dataset.get_loader()

    if backbone_name == "resnet":
        model = ResNet(num_classes=num_classes, small_input=small_input, pretrained=pretrained)
    elif backbone_name == "vgg":
        model = VGG(num_classes=num_classes, small_input=small_input, pretrained=pretrained)
    elif backbone_name == "densenet":
        model = DenseNet(num_classes=num_classes, small_input=small_input, pretrained=pretrained)
    elif backbone_name == "googlenet":
        model = GoogLeNet(num_classes=num_classes, small_input=small_input, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown backbone_name: {backbone_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    state = torch.load(f"./data/models/target/{dataset_name}_{backbone_name}_{optimizer_name}_{scheduler_name}.pt",
                       map_location=device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    model.to(device)

    baseline_name = baseline_name.lower()
    if baseline_name == "backdooring":
        defense = Backdooring(model, dataset=test_loader.dataset, device=device,
                              ver_dataset_ratio=cfg["ver_dataset_ratio"],
                              embedding_dim=cfg["embedding_dim"], ov_mode=False)
    elif baseline_name == "ewe":
        defense = EWE(model, dataset=test_loader.dataset, device=device,
                      ver_dataset_ratio=cfg["ver_dataset_ratio"],
                      embedding_dim=cfg["embedding_dim"], ov_mode=False)
    elif baseline_name == "ipguard":
        defense = IPGuard(model, dataset=test_loader.dataset, device=device,
                          ver_dataset_ratio=cfg["ver_dataset_ratio"],
                          embedding_dim=cfg["embedding_dim"], ov_mode=False)
    elif baseline_name == "uap":
        defense = UAP(model, dataset=test_loader.dataset, device=device,
                      ver_dataset_ratio=cfg["ver_dataset_ratio"],
                      embedding_dim=cfg["embedding_dim"], ov_mode=False)
    else:
        raise ValueError(f"Unsupported defense: {baseline_name}")

    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(cfg.get("label_smoothing", 0.0))
    ).to(device)

    lr = float(cfg.get("lr", 0.1))
    weight_decay = float(cfg.get("weight_decay", 5e-4))
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(cfg.get("nesterov", True)),
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=tuple(cfg.get("betas", (0.9, 0.999))),
            eps=float(cfg.get("eps", 1e-8)),
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=tuple(cfg.get("betas", (0.9, 0.999))),
            eps=float(cfg.get("eps", 1e-8)),
            weight_decay=weight_decay,
        )
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            alpha=float(cfg.get("rmsprop_alpha", 0.99)),
            eps=float(cfg.get("eps", 1e-8)),
            momentum=float(cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
            centered=bool(cfg.get("rmsprop_centered", False)),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    epochs = int(cfg.get("epochs", 20))
    scheduler_name = scheduler_name.lower()
    if scheduler_name == "cosine":
        T_max = int(cfg.get("cosine_T_max", epochs))
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_name == "step":
        step_size = int(cfg.get("step_size", max(1, epochs // 3)))
        gamma = float(cfg.get("step_gamma", 0.1))
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    save_model_name = f"{dataset_name}_{backbone_name}_{optimizer_name}_{scheduler_name}_{baseline_name}.pt"

    # wandb
    run = wandb.init(
        project="certified",
        group=cfg["wandb_group"],
        name=save_model_name,
        tags=[dataset_name, backbone_name, optimizer_name, scheduler_name, baseline_name],
        config={"dataset": dataset_name, "defense": baseline_name, **cfg},
        settings=wandb.Settings(init_timeout=180, start_method="thread"),
        reinit=True,
    )

    loss, acc = defense.eval_defense_model(criterion=criterion)
    print(f"Pre-Finetune Loss: {loss}, Accuracy: {acc}")
    defense.train_defense_model(criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                                epochs=cfg["finetune_epochs"], wandb=wandb)
    loss, acc = defense.eval_defense_model(criterion=criterion)
    print(f"After-Finetune Loss: {loss}, Accuracy: {acc}")
    wandb.log({
        "eval_loss": loss,
        "eval_accuracy": acc,
    })
    model = defense.target_model

    save_root = Path(cfg["save_root"])
    save_root.mkdir(parents=True, exist_ok=True)
    save_path = save_root / save_model_name

    torch.save(
        dict(
            model=model.state_dict(),
            dataset_name=dataset_name,
            backbone_name=backbone_name,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            baseline_name=baseline_name,
            cfg=cfg,
        ),
        save_path,
    )

    wandb.finish()
    return dataset_name, backbone_name, optimizer_name, scheduler_name, baseline_name


def train_cv_baseline_single(dataset_name, backbone_name, baseline_name, cfg):
    from models.defense import Backdooring, EWE, IPGuard, UAP
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset_name = "cifar100"
    # backbone_name = "vgg"
    # baseline_name = "backdooring"

    pretrained = False
    # cfg = {"epochs": 20,
    #        "ver_dataset_ratio": 0.1}  # 10 for backdooring, 5 for UAP
    ver_dataset_ratio = cfg["ver_dataset_ratio"]  # 0.1 for backdooring, 0.01 for others

    if dataset_name == "cifar10":
        dataset = CIFAR10Dataset(batch_size=128,
                                 download=False, num_workers=8)
        num_classes, small_input = 10, True
    elif dataset_name == "cifar100":
        dataset = CIFAR100Dataset(batch_size=128, download=False, num_workers=8)
        num_classes, small_input = 100, True
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if backbone_name == "resnet":
        model = ResNet(num_classes=num_classes, small_input=small_input, pretrained=pretrained)
    elif backbone_name == "vgg":
        model = VGG(num_classes=num_classes, small_input=small_input, pretrained=pretrained)
    elif backbone_name == "densenet":
        model = DenseNet(num_classes=num_classes, small_input=small_input, pretrained=pretrained)
    elif backbone_name == "googlenet":
        model = GoogLeNet(num_classes=num_classes, small_input=small_input, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown backbone_name: {backbone_name}")

    train_loader, test_loader = dataset.get_loader()

    model.to(device)
    state = torch.load(f"./data/models/target/{dataset_name}_{backbone_name}_sgd_cosine.pt", map_location=device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    # criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    lr = float(cfg.get("lr", 0.1))
    weight_decay = float(cfg.get("weight_decay", 5e-4))
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=float(cfg.get("momentum", 0.9)),
        weight_decay=weight_decay,
        nesterov=bool(cfg.get("nesterov", True)),
    )

    # scheduler
    epochs = int(cfg.get("epochs", 20))
    T_max = int(cfg.get("cosine_T_max", epochs))
    scheduler = CosineAnnealingLR(optimizer, T_max=T_max)

    if baseline_name == "backdooring":
        defense = Backdooring(model, test_loader.dataset, device, ver_dataset_ratio=ver_dataset_ratio, ov_mode=False)
    elif baseline_name == "ewe":
        defense = EWE(model, test_loader.dataset, device, ver_dataset_ratio=ver_dataset_ratio, ov_mode=False)
    elif baseline_name == "ipguard":
        defense = IPGuard(model, test_loader.dataset, device, ver_dataset_ratio=ver_dataset_ratio, ov_mode=False)
    elif baseline_name == "uap":
        defense = UAP(model, test_loader.dataset, device, ver_dataset_ratio=ver_dataset_ratio, ov_mode=False)
    else:
        raise ValueError(f"Unsupported baseline_name: {baseline_name}")

    defense.eval_downstream(criterion=criterion)
    defense.eval_wm(criterion=criterion)
    defense.train_defense_model(epochs)

    n_trials = 3
    test_acc_list = []
    wm_acc_list = []
    for _ in range(n_trials):
        defense.train_defense_model(1)
        downstream_loss, downstream_acc = defense.eval_downstream(criterion=criterion)
        wm_loss, wm_acc = defense.eval_wm(criterion=criterion)
        test_acc_list.append(downstream_acc)
        wm_acc_list.append(wm_acc)

    test_acc_mean = np.mean(test_acc_list)
    test_acc_std = np.std(test_acc_list)
    wm_acc_mean = np.mean(wm_acc_list)
    wm_acc_std = np.std(wm_acc_list)

    save_path = f"./data/models/target/{dataset_name}_{backbone_name}_sgd_cosine_{baseline_name}.pt"

    if baseline_name == "ipguard":
        torch.save(
            dict(model=defense.target_model.state_dict(),
                 wm_indices=defense.wm_indices,
                 wm_labels=defense.wm_labels),
            save_path,
        )
    else:
        torch.save(
            dict(model=defense.target_model.state_dict()),
            save_path,
        )

    res = {
        "dataset": dataset_name,
        "backbone_name": backbone_name,
        "baseline_name": baseline_name,
        "epochs": epochs,
        "ver_dataset_ratio": ver_dataset_ratio,
        # "downstream_loss": downstream_loss,
        "test_acc_mean": test_acc_mean,
        "test_acc_std": test_acc_std,
        # "wm_loss": wm_loss,
        "wm_acc_mean": wm_acc_mean,
        "wm_acc_std": wm_acc_std,
    }
    return res


def train_cv_baseline():
    dataset_names = ["cifar10", "cifar100"]  # "cifar10",
    backbone_names = ["resnet", "vgg", "densenet", "googlenet"]
    baseline_names = ["uap"]  # "backdooring", "ewe", "ipguard", "uap"

    # backdooring
    # cfg = {"epochs": 10,
    #        "ver_dataset_ratio": 0.1}
    # ewe
    # cfg = {"epochs": 20,
    #        "ver_dataset_ratio": 0.01}
    # ipguard
    # cfg = {"epochs": 20,
    #        "ver_dataset_ratio": 0.05}
    # uap
    cfg = {"epochs": 5,
           "ver_dataset_ratio": 0.01}

    all_records = []
    for dataset_name, backbone_name, baseline_name in itertools.product(dataset_names, backbone_names, baseline_names):
        res = train_cv_baseline_single(dataset_name, backbone_name, baseline_name, cfg)
        all_records.append(res)

    save_metrics_table(all_records, out_name='effect_baselines_uap')


@ray.remote(num_gpus=1)
def worker_cv_baseline_kd(dataset_name, backbone_name, optimizer_name, scheduler_name, baseline_name, cfg):
    from models.defense import Backdooring, EWE, IPGuard, UAP

    download = False
    num_workers = 8
    batch_size = 128

    if dataset_name == "cifar10":
        dataset = CIFAR10Dataset(batch_size=batch_size,
                                 download=download, num_workers=num_workers)
        num_classes, small_input = 10, True
    elif dataset_name == "cifar100":
        dataset = CIFAR100Dataset(batch_size=batch_size,
                                  download=download, num_workers=num_workers)
        num_classes, small_input = 100, True
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    train_loader, test_loader = dataset.get_loader()

    if backbone_name == "resnet":
        model = ResNet(num_classes=num_classes, small_input=small_input, pretrained=False)
    elif backbone_name == "vgg":
        model = VGG(num_classes=num_classes, small_input=small_input, pretrained=False)
    elif backbone_name == "densenet":
        model = DenseNet(num_classes=num_classes, small_input=small_input, pretrained=False)
    elif backbone_name == "googlenet":
        model = GoogLeNet(num_classes=num_classes, small_input=small_input, pretrained=False)
    else:
        raise ValueError(f"Unknown backbone_name: {backbone_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(cfg.get("label_smoothing", 0.0))
    ).to(device)

    lr = float(cfg.get("lr", 0.1))
    weight_decay = float(cfg.get("weight_decay", 5e-4))
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(cfg.get("nesterov", True)),
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=tuple(cfg.get("betas", (0.9, 0.999))),
            eps=float(cfg.get("eps", 1e-8)),
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=tuple(cfg.get("betas", (0.9, 0.999))),
            eps=float(cfg.get("eps", 1e-8)),
            weight_decay=weight_decay,
        )
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            alpha=float(cfg.get("rmsprop_alpha", 0.99)),
            eps=float(cfg.get("eps", 1e-8)),
            momentum=float(cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
            centered=bool(cfg.get("rmsprop_centered", False)),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    epochs = int(cfg.get("epochs", 20))
    scheduler_name = scheduler_name.lower()
    if scheduler_name == "cosine":
        T_max = int(cfg.get("cosine_T_max", epochs))
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_name == "step":
        step_size = int(cfg.get("step_size", max(1, epochs // 3)))
        gamma = float(cfg.get("step_gamma", 0.1))
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    # wandb
    run = wandb.init(
        project="certified",
        group=cfg["wandb_group"],
        name=f"{dataset_name}_{backbone_name}_{optimizer_name}_{scheduler_name}",
        tags=[dataset_name, backbone_name, optimizer_name, scheduler_name],
        config={"dataset": dataset_name, "backbone": backbone_name,
                "optimizer": optimizer_name, "scheduler": scheduler_name, **cfg},
        settings=wandb.Settings(init_timeout=180, start_method="thread"),
        reinit=True,
    )

    teacher = ResNet(num_classes=num_classes, small_input=small_input).to(device)
    model_path = f"./data/models/target/{dataset_name}_resnet_sgd_cosine_{baseline_name}.pt"
    state = torch.load(model_path, map_location=device)
    teacher.load_state_dict(state["model"], strict=False)
    teacher.eval()

    temperature = 4.0
    alpha_pred = cfg["alpha_pred"]
    beta_emb = cfg["beta_emb"]
    seed = 42

    dataset_ratio = cfg["ver_dataset_ratio"]
    budget_ratio = cfg["query_budget_ratio"]

    baseline_name = baseline_name.lower()
    if baseline_name == "backdooring":
        defense_model = Backdooring(teacher, dataset=test_loader.dataset, device=device,
                                    ver_dataset_ratio=cfg["ver_dataset_ratio"],
                                    embedding_dim=cfg["embedding_dim"], ov_mode=True, save_path=model_path)
    elif baseline_name == "ewe":
        defense_model = EWE(teacher, dataset=test_loader.dataset, device=device,
                            ver_dataset_ratio=cfg["ver_dataset_ratio"],
                            embedding_dim=cfg["embedding_dim"], ov_mode=True, save_path=model_path)
    elif baseline_name == "ipguard":
        defense_model = IPGuard(teacher, dataset=test_loader.dataset, device=device,
                                ver_dataset_ratio=cfg["ver_dataset_ratio"],
                                embedding_dim=cfg["embedding_dim"], ov_mode=True, save_path=model_path)
    elif baseline_name == "uap":
        defense_model = UAP(teacher, dataset=test_loader.dataset, device=device,
                            ver_dataset_ratio=cfg["ver_dataset_ratio"],
                            embedding_dim=cfg["embedding_dim"], ov_mode=True, save_path=model_path)
    else:
        raise ValueError(f"Unsupported defense: {baseline_name}")

    kd = KnowledgeDistillation(
        train_dataset=train_loader.dataset,
        test_dataset=test_loader.dataset,
        target_model=defense_model,
        surrogate_model=model,
        device=device,
        batch_size=batch_size,
        seed=seed,
    )

    kd.query(budget_ratio=budget_ratio, queries_per_image=1)

    surrogate = kd.train_surrogate_model(
        epochs=epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        temperature=temperature,
        alpha_pred=alpha_pred,
        beta_emb=beta_emb,
        wandb=wandb
    )

    save_root = Path(cfg["save_root"])
    save_root.mkdir(parents=True, exist_ok=True)
    save_path = save_root / f"{dataset_name}_{backbone_name}_{optimizer_name}_{scheduler_name}_{baseline_name}.pt"

    torch.save(
        dict(
            model=surrogate.state_dict(),
            dataset=dataset_name,
            backbone=backbone_name,
            optimizer=optimizer_name,
            scheduler=scheduler_name,
            hparams=dict(lr=lr, weight_decay=weight_decay, epochs=epochs),
            cfg=cfg,
        ),
        save_path,
    )
    wandb.finish()
    return dataset_name, backbone_name, optimizer_name, scheduler_name, baseline_name


def train_cv_baseline_surrogate():
    # train surrogate model
    ray.init(ignore_reinit_error=True)

    config = {
        "epochs": 40,  # 40 for both
        "wandb_group": "cv_surrogates_baseline",
        "save_root": "./data/models/surrogate/baselines",
        "embedding_dim": 1024,
        "lr": 0.001,  # 0.001 for cifar-10; 0.001 for cifar-100
        "weight_decay": 1.0e-5,  # 1.0e-5 for cifar-10; 5.0e-4 for cifar-100
        "alpha_pred": 1.0,  # 0.02 for cifar-10; 1.0 for cifar-100
        "beta_emb": 1.0,  # 10.0 for cifar-10; 1.0 for cifar-100
        "ver_dataset_ratio": 0.1,
        "query_budget_ratio": 0.8,  # 0.6
    }

    datasets = ["cifar10"]  # , "cifar100"
    backbones = ["vgg", "googlenet", "densenet"]  # , "googlenet", "densenet"
    optimizers = ["adam", "adamw", "rmsprop"]  # "adamw", "rmsprop"
    schedulers = ["cosine"]  # , "step"
    baseline_names = ["ipguard", "uap"]  # , "ewe", "ipguard", "uap"

    futures = []
    for ds, bk, o, s, bl in itertools.product(datasets, backbones, optimizers, schedulers, baseline_names):
        futures.append(worker_cv_baseline_kd.remote(ds, bk, o, s, bl, config))

    results = ray.get(futures)
    print("Training finished.")
    print(results)


def eval_ov_baseline(baseline_name, baseline_model_name):
    from models.nn.resnet import ResNet
    from models.defense import Backdooring, EWE, IPGuard, UAP
    from models.defense.ov import OwnershipVerifier_WM
    from utils.datasets import get_dataset

    device = torch.device("cuda:2")

    # verification params
    cfg = {
        "ver_dataset_ratio": 0.1,
        "embedding_dim": 1024,
    }

    # suspicious models
    dataset_name = "cifar10"  # , "cifar100"
    backbones = ["vgg", "googlenet", "densenet"]
    optimizers = ["adam", "adamw", "rmsprop"]  # , "adamw", "rmsprop"
    schedulers = ["cosine"]  # , "step"

    # load dataset
    ds = get_dataset(name=dataset_name)
    train_loader, test_loader = ds.get_loader()

    # 1. load defense model
    target_model_path = f"./data/models/target/{dataset_name}_{baseline_model_name}_sgd_cosine_{baseline_name}.pt"
    if baseline_model_name == "resnet":
        target_model = ResNet(num_classes=10, small_input=True)
    elif baseline_model_name == "vgg":
        target_model = VGG(num_classes=10, small_input=True)
    elif baseline_model_name == "densenet":
        target_model = DenseNet(num_classes=10, small_input=True)
    elif baseline_model_name == "googlenet":
        target_model = GoogLeNet(num_classes=10, small_input=True)
    else:
        raise ValueError(baseline_model_name)
    target_model.to(device)
    state = torch.load(target_model_path, map_location=device)
    state_dict = state.get("model", state)
    target_model.load_state_dict(state_dict, strict=False)

    baseline_name = baseline_name.lower()
    if baseline_name == "backdooring":
        defense_model = Backdooring(target_model, dataset=test_loader.dataset, device=device,
                                    ver_dataset_ratio=cfg["ver_dataset_ratio"],
                                    embedding_dim=cfg["embedding_dim"], ov_mode=True, save_path=target_model_path)
    elif baseline_name == "ewe":
        defense_model = EWE(target_model, dataset=test_loader.dataset, device=device,
                            ver_dataset_ratio=cfg["ver_dataset_ratio"],
                            embedding_dim=cfg["embedding_dim"], ov_mode=True, save_path=target_model_path)
    elif baseline_name == "ipguard":
        defense_model = IPGuard(target_model, dataset=test_loader.dataset, device=device,
                                ver_dataset_ratio=cfg["ver_dataset_ratio"],
                                embedding_dim=cfg["embedding_dim"], ov_mode=True, save_path=target_model_path)
    elif baseline_name == "uap":
        defense_model = UAP(target_model, dataset=test_loader.dataset, device=device,
                            ver_dataset_ratio=cfg["ver_dataset_ratio"],
                            embedding_dim=cfg["embedding_dim"], ov_mode=True, save_path=target_model_path)
    else:
        raise ValueError(f"Unsupported defense: {baseline_name}")

    # 2. load surrogate models
    surrogate_models = []
    sur_model_dir = "./data/models/surrogate/baselines"
    for backbone, optimizer, scheduler in itertools.product(backbones, optimizers, schedulers):
        sur_model = get_model_baseline(backbone, dataset_name, optimizer, scheduler, baseline_name,
                                       model_dir=sur_model_dir)
        surrogate_models.append(sur_model)

    # 3. load independent models
    independent_models = []
    ind_model_dir = "./data/models/independent"
    for backbone, optimizer, scheduler in itertools.product(backbones, optimizers, schedulers):
        ind_model = get_model(backbone, dataset_name, optimizer, scheduler, model_dir=ind_model_dir)
        independent_models.append(ind_model)

    # combine
    suspicious_models = []
    suspicious_label = []
    for s in surrogate_models:
        suspicious_models.append(s)
        suspicious_label.append(1)
    for i in independent_models:
        suspicious_models.append(i)
        suspicious_label.append(0)

    # 4. init ov pipeline
    ov = OwnershipVerifier_WM(
        defense_model=defense_model,
        dataset=test_loader.dataset,
        ver_ratio=cfg["ver_dataset_ratio"],
        device=device,
    )

    # 5. ownership verification
    ver_res = ov.verify(suspicious_models, suspicious_label)

    res = {
        "baseline_name": baseline_name,
        "baseline_model_name": baseline_model_name,
        **ver_res
    }

    print(res)

    return res


def run_eval_ov_baseline():
    from utils.metrics import save_metrics_table
    baseline_name = ["backdooring", "ewe", "ipguard", "uap"]
    baseline_model_name = ["resnet", "vgg", "densenet", "googlenet"]

    records = []
    for bl, blm in itertools.product(baseline_name, baseline_model_name):
        res = eval_ov_baseline(bl, blm)
        records.append(res)

    save_metrics_table(records, out_name="auc_baselines")


def test_baseline():
    from models.defense import Backdooring

    cfg = {
        "ver_dataset_ratio": 0.1,
        "embedding_dim": 128,
    }
    dataset = CIFAR10Dataset(batch_size=128,
                             download=False, num_workers=8)
    num_classes, small_input = 10, True

    train_loader, test_loader = dataset.get_loader()

    model = ResNet(num_classes=num_classes, small_input=small_input, pretrained=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    state = torch.load(f"./data/models/target/cifar10_resnet_sgd_cosine.pt",
                       map_location=device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    model.to(device)

    defense = Backdooring(model, dataset=test_loader.dataset, device=device,
                          ver_dataset_ratio=cfg["ver_dataset_ratio"],
                          embedding_dim=cfg["embedding_dim"], ov_mode=False)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(cfg.get("label_smoothing", 0.0))
    ).to(device)

    lr = float(cfg.get("lr", 0.1))
    weight_decay = float(cfg.get("weight_decay", 5e-4))

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=float(cfg.get("momentum", 0.9)),
        weight_decay=weight_decay,
        nesterov=bool(cfg.get("nesterov", True)),
    )
    epochs = int(cfg.get("epochs", 20))
    T_max = int(cfg.get("cosine_T_max", epochs))
    scheduler = CosineAnnealingLR(optimizer, T_max=T_max)

    loss, acc = defense.eval_defense_model(criterion=criterion)
    print(f"Pre-Finetune Loss: {loss}, Acc: {acc}.")
    defense.train_defense_model(criterion, optimizer=optimizer, scheduler=scheduler, epochs=epochs)
    loss, acc = defense.eval_defense_model(criterion=criterion)
    print(f"After-Finetune Loss: {loss}, Acc: {acc}.")
