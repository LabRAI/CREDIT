import itertools
from time import time

import numpy as np
from torch import optim
from tqdm import tqdm


def defense_time_credit():
    import torch
    from utils.datasets import CIFAR10Dataset
    from models.credit.credit import CREDIT
    from models.nn import ResNet, VGG
    from time import time

    device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone_name = "resnet"
    ver_dataset_ratio = 0.1

    # CREDIT
    sigma = 0.4
    beta = 2650
    Q = 1000

    dataset = CIFAR10Dataset(batch_size=128,
                             download=False, num_workers=8)
    train_loader, test_loader = dataset.get_loader()

    if backbone_name == "resnet":
        model = ResNet(num_classes=10, small_input=True, pretrained=False)
    elif backbone_name == "vgg":
        model = VGG(num_classes=10, small_input=True, pretrained=False)
    else:
        raise ValueError(backbone_name)

    model.to(device)

    credit = CREDIT(model, test_loader.dataset, dataset_ratio=ver_dataset_ratio, sigma=sigma)

    n_trial = 3

    time_list = []
    for i in range(n_trial):
        t0 = time()
        # credit.beta(device=device)
        credit.tau(beta=beta, Q=Q)
        # sigma_star = credit.optim_sigma(
        #     sigmas=[0.0001],
        #     gamma1_list=[0.001],
        #     gamma2_list=[0.001],
        #     device=device,
        #     max_batches=1,
        #     lambda_util=1.0,
        #     lambda_ver=1.0,
        #     pi0=0.5,
        # )
        t1 = time()
        delta_t = t1 - t0
        time_list.append(delta_t)

    time_list = np.array(time_list)
    print(f"[CREDIT]Backbone: {backbone_name}, Defense time: {np.mean(time_list)} +- {np.std(time_list)}")


def defense_time_baseline(backbone_name="resnet", baseline_name="backdooring"):
    import torch
    from models.defense import Backdooring, EWE, IPGuard, UAP
    from utils.datasets import CIFAR10Dataset
    from models.nn import ResNet, VGG
    from time import time

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ver_dataset_ratio = 0.1

    dataset = CIFAR10Dataset(batch_size=128,
                             download=False, num_workers=8)
    train_loader, test_loader = dataset.get_loader()

    if backbone_name == "resnet":
        model = ResNet(num_classes=10, small_input=True, pretrained=False)
    elif backbone_name == "vgg":
        model = VGG(num_classes=10, small_input=True, pretrained=False)
    else:
        raise ValueError(backbone_name)

    model.to(device)

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

    n_trial = 3
    time_list = []
    for i in range(n_trial):
        t0 = time()
        defense._key_gen()
        t1 = time()
        delta_t = t1 - t0
        time_list.append(delta_t)

    time_list = np.array(time_list)
    print(f"backbone: {backbone_name}, baseline: {baseline_name}, time: {np.mean(time_list)} +- {np.std(time_list)}")
    return {
        "backbone_name": backbone_name,
        "baseline_name": baseline_name,
        "n_trial": n_trial,
        "defense_time_mean": np.mean(time_list),
        "defense_time_std": np.std(time_list),
    }


def run_defense_time_baseline():
    from utils.metrics import save_metrics_table
    backbone_names = ["resnet", "vgg"]
    baseline_names = ["backdooring", "ewe", "ipguard", "uap"]

    records = []
    for bb, bl in itertools.product(backbone_names, baseline_names):
        res = defense_time_baseline(bb, bl)
        records.append(res)

    save_metrics_table(records, "defense_time_baseline")


def train_epoch(model, train_loader, epochs=10, device="cuda"):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total, loss_sum, acc_sum = 0, 0.0, 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
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


def verification_time_credit():
    import torch
    from utils.datasets import CIFAR10Dataset
    from models.credit.credit import CREDIT
    from models.credit.ov import OwnershipVerifier
    from models.nn import ResNet, VGG

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # verification params
    ver_ratio = 0.1
    batch_size = 128
    num_workers = 8
    k = 3
    seed = 42
    rho = 2.0
    eta = 100.0

    backbone_name = "vgg"
    ver_dataset_ratio = 0.1

    # CREDIT
    sigma = 0.4
    beta = 2650
    Q = 1000

    dataset = CIFAR10Dataset(batch_size=128,
                             download=False, num_workers=8)
    train_loader, test_loader = dataset.get_loader()

    if backbone_name == "resnet":
        model = ResNet(num_classes=10, small_input=True, pretrained=True)
    elif backbone_name == "vgg":
        model = VGG(num_classes=10, small_input=True, pretrained=True)
    else:
        raise ValueError(backbone_name)

    model.to(device)

    credit = CREDIT(model, test_loader.dataset, dataset_ratio=ver_dataset_ratio, sigma=sigma)

    # 2. load surrogate models
    surrogate_models = [model]

    # 3. load independent models
    independent_models = [model]

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
        defense_model=credit,
        dataset=test_loader.dataset,
        ver_ratio=ver_ratio,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        seed=seed,
    )

    n_trial = 3
    time_list = []
    for i in range(n_trial):
        t0 = time()
        beta = credit.beta(sigma=sigma, device=device)
        tau = credit.tau(beta=beta, Q=Q, rho=rho, eta=eta)
        result = ov.verify(suspicious_models, suspicious_label, k=k, n_perm=1, tau=tau)
        t1 = time()
        delta_t = t1 - t0
        time_list.append(delta_t)
    print(f"[CREDIT]Backbone: {backbone_name}, Verification time: {np.mean(time_list)} +- {np.std(time_list)}")


def verification_time_baseline(backbone_name, baseline_name):
    import torch
    from utils.datasets import CIFAR10Dataset
    from models.defense import Backdooring, EWE, IPGuard, UAP
    from models.defense.ov import OwnershipVerifier_WM
    from models.nn import ResNet, VGG
    import copy

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # verification params
    ver_dataset_ratio = 0.1
    batch_size = 128
    num_workers = 8
    seed = 42

    dataset = CIFAR10Dataset(batch_size=128, download=False, num_workers=8)
    train_loader, test_loader = dataset.get_loader()

    if backbone_name == "resnet":
        model = ResNet(num_classes=10, small_input=True, pretrained=True)
    elif backbone_name == "vgg":
        model = VGG(num_classes=10, small_input=True, pretrained=True)
    else:
        raise ValueError(backbone_name)

    model.to(device)

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

    # 2. load surrogate models
    surrogate_model = copy.deepcopy(model).to(device)
    surrogate_models = [surrogate_model]

    # 3. load independent models
    independent_model = copy.deepcopy(model).to(device)
    independent_models = [independent_model]

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
        defense_model=defense,
        dataset=test_loader.dataset,
        ver_ratio=ver_dataset_ratio,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        seed=seed,
    )

    n_trial = 3
    time_list = []
    for i in range(n_trial):
        t0 = time()
        train_epoch(surrogate_models[0], train_loader, epochs=5, device=device)
        train_epoch(independent_models[0], train_loader, epochs=5, device=device)
        ov.verify(suspicious_models, suspicious_label)
        t1 = time()
        delta_t = t1 - t0
        time_list.append(delta_t)
    print(f"[Baseline]Backbone: {backbone_name}, Verification time: {np.mean(time_list)} +- {np.std(time_list)}")
    return {
        "backbone_name": backbone_name,
        "baseline_name": baseline_name,
        "n_trial": n_trial,
        "verification_time_mean": np.mean(time_list),
        "verification_time_std": np.std(time_list),
    }


def run_verification_time_baseline():
    from utils.metrics import save_metrics_table
    backbone_names = ["resnet", "vgg"]
    baseline_names = ["backdooring", "ewe", "ipguard", "uap"]

    records = []
    for bb, bl in itertools.product(backbone_names, baseline_names):
        res = verification_time_baseline(bb, bl)
        records.append(res)

    save_metrics_table(records, "verification_time_baseline")


def count_parameters():
    from models.nn import ResNet, VGG, DenseNet, GoogLeNet

    backbone_name = "googlenet"

    if backbone_name == "resnet":
        model = ResNet(num_classes=10, small_input=True, pretrained=False)  # 25m
    elif backbone_name == "vgg":
        model = VGG(num_classes=10, small_input=True, pretrained=False)  # 15260746
    elif backbone_name == "densenet":
        model = DenseNet(num_classes=10, small_input=True, pretrained=False)  # 6958474
    elif backbone_name == "googlenet":
        model = GoogLeNet(num_classes=10, small_input=True, pretrained=False)  # 5602346
    else:
        raise ValueError(backbone_name)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Backbone: {backbone_name}, Total parameters: {total}, Trainable parameters: {trainable}")
