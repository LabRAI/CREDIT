import itertools

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from utils.datasets import get_graph_dataset


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        _, out = model(data)  # forward (embedding, logits)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.num_graphs

    avg_loss = total_loss / len(loader.dataset)
    acc = correct / total
    return avg_loss, acc


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for data in loader:
        data = data.to(device)
        _, out = model(data)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.num_graphs
    return correct / total


def train_vanilla_one(dataset_name, backbone_name):
    from models.nn import GAT, GCN, GraphSAGE, SSGC
    from utils.datasets import inspect_graph_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding_dim = 128

    # dataset
    gds = get_graph_dataset(dataset_name, batch_size=64, train_ratio=0.8, seed=42)
    train_loader, test_loader = gds.get_loader()
    num_classes = gds.train_dataset.dataset.num_classes
    inspect_graph_dataset(gds)

    # backbone model
    if backbone_name == "GAT":
        model = GAT(num_classes=num_classes, embedding_dim=embedding_dim).to(device)
    elif backbone_name == "GCN":
        model = GCN(num_classes=num_classes, embedding_dim=embedding_dim).to(device)
    elif backbone_name == "GraphSAGE":
        model = GraphSAGE(num_classes=num_classes, embedding_dim=embedding_dim).to(device)
    elif backbone_name == "SSGC":
        model = SSGC(num_classes=num_classes, embedding_dim=embedding_dim).to(device)
    else:
        raise ValueError(f"{backbone_name} is not supported.")

    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    n_trial = 3
    test_acc_list = []
    for _ in range(n_trial):
        for epoch in range(1, 41):
            loss_trian, acc_train = train_one_epoch(model, train_loader, optimizer, device)
            acc = eval_model(model, test_loader, device)
            print(f"Epoch {epoch:02d}: Loss {loss_trian:.4f}, Train Acc  {acc_train:.4f}, Test Acc {acc:.4f}")

        test_acc = eval_model(model, test_loader, device)
        test_acc_list.append(test_acc)
    test_acc_mean = np.mean(test_acc_list)
    test_acc_std = np.std(test_acc_list)

    return {
        "dataset_name": dataset_name,
        "backbone_name": backbone_name,
        "test_acc_mean": test_acc_mean,
        "test_acc_std": test_acc_std,
    }


def train_vanilla():
    from utils.metrics import save_metrics_table
    dataset_names = ["ENZYMES", "PROTEINS"]
    backbone_names = ["GCN", "GAT", "GraphSAGE", "SSGC"]

    records = []
    for dn, bn in itertools.product(dataset_names, backbone_names):
        res = train_vanilla_one(dn, bn)
        records.append(res)

    save_metrics_table(records, out_name="graph_vanilla_trials")


def train_baseline_once(dataset_name, backbone_name, defense_name):
    from models.nn import GAT, GCN, GraphSAGE, SSGC
    from utils.datasets import inspect_graph_dataset
    from models.defense import RandomWM, BackdoorWM, SurviveWM, ImperceptibleWM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 20
    ver_dataset_ratio = 0.5
    embedding_dim = 128

    # dataset
    gds = get_graph_dataset(dataset_name, batch_size=64, train_ratio=0.8, seed=42)
    train_loader, test_loader = gds.get_loader()
    num_classes = gds.train_dataset.dataset.num_classes
    inspect_graph_dataset(gds)

    # backbone model
    if backbone_name == "GAT":
        model = GAT(num_classes=num_classes, embedding_dim=embedding_dim).to(device)
    elif backbone_name == "GCN":
        model = GCN(num_classes=num_classes, embedding_dim=embedding_dim).to(device)
    elif backbone_name == "GraphSAGE":
        model = GraphSAGE(num_classes=num_classes, embedding_dim=embedding_dim).to(device)
    elif backbone_name == "SSGC":
        model = SSGC(num_classes=num_classes, embedding_dim=embedding_dim).to(device)
    else:
        raise ValueError(f"{backbone_name} is not supported.")

    # defense mechanism
    if defense_name == "RandomWM":
        defense = RandomWM(model, test_loader.dataset, device, ver_dataset_ratio=0.6, embedding_dim=embedding_dim)
    elif defense_name == "BackdoorWM":
        defense = BackdoorWM(model, test_loader.dataset, device, ver_dataset_ratio=0.6, embedding_dim=embedding_dim)
    elif defense_name == "SurviveWM":
        defense = SurviveWM(model, test_loader.dataset, device, ver_dataset_ratio, embedding_dim)
    elif defense_name == "ImperceptibleWM":
        defense = ImperceptibleWM(model, test_loader.dataset, device, ver_dataset_ratio, embedding_dim)
    else:
        raise ValueError("Unknown defense name: {}".format(defense_name))

    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    n_trial = 3
    wm_acc_list = []
    test_acc_list = []
    for _ in range(n_trial):
        if defense_name in ["SurviveWM", "ImperceptibleWM"]:
            defense.train_defense_model(epochs)
        else:
            for epoch in range(1, epochs + 1):
                loss_trian, acc_train = train_one_epoch(defense, train_loader, optimizer, device)
                acc = eval_model(defense, test_loader, device)
                print(f"Epoch {epoch:02d}: Loss {loss_trian:.4f}, Train Acc  {acc_train:.4f}, Test Acc {acc:.4f}")

        wm_acc = eval_model(defense, defense._wm_loader, device)
        wm_acc_list.append(wm_acc)
        print(f"WM acc: {wm_acc:.4f}")
        test_acc = eval_model(defense, test_loader, device)
        test_acc_list.append(test_acc)
        print(f"Test Acc: {test_acc:.4f}")

    wm_acc_mean = np.mean(wm_acc_list)
    wm_acc_std = np.std(wm_acc_list)
    test_acc_mean = np.mean(test_acc_list)
    test_acc_std = np.std(test_acc_list)
    return {
        "dataset_name": dataset_name,
        "backbone_name": backbone_name,
        "defense_name": defense_name,
        "ver_dataset_ratio": ver_dataset_ratio,
        "wm_acc_mean": wm_acc_mean,
        "wm_acc_std": wm_acc_std,
        "test_acc_mean": test_acc_mean,
        "test_acc_std": test_acc_std,
    }


def train_baseline():
    from utils.metrics import save_metrics_table
    dataset_names = ["ENZYMES", "PROTEINS"]
    backbone_names = ["GCN", "GAT", "GraphSAGE", "SSGC"]
    baseline_names = ["RandomWM", "BackdoorWM"]  # , "SurviveWM", "ImperceptibleWM"

    records = []
    for dn, bn, bl in itertools.product(dataset_names, backbone_names, baseline_names):
        res = train_baseline_once(dn, bn, bl)
        records.append(res)

    save_metrics_table(records, out_name="graph_baselines_trials")


def train_credit_once(dataset_name, backbone_name):
    from models.nn import GAT, GCN, GraphSAGE, SSGC
    from utils.datasets import inspect_graph_dataset
    from models.credit.credit import CREDIT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 40
    ver_dataset_ratio = 0.5
    embedding_dim = 128
    sigma = 0.4

    # dataset
    gds = get_graph_dataset(dataset_name, batch_size=64, train_ratio=0.8, seed=42)
    train_loader, test_loader = gds.get_loader()
    num_classes = gds.train_dataset.dataset.num_classes
    inspect_graph_dataset(gds)

    # backbone model
    if backbone_name == "GAT":
        model = GAT(num_classes=num_classes, embedding_dim=embedding_dim).to(device)
    elif backbone_name == "GCN":
        model = GCN(num_classes=num_classes, embedding_dim=embedding_dim).to(device)
    elif backbone_name == "GraphSAGE":
        model = GraphSAGE(num_classes=num_classes, embedding_dim=embedding_dim).to(device)
    elif backbone_name == "SSGC":
        model = SSGC(num_classes=num_classes, embedding_dim=embedding_dim).to(device)
    else:
        raise ValueError(f"{backbone_name} is not supported.")

    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    credit = CREDIT(model, test_loader.dataset, dataset_ratio=ver_dataset_ratio, embedding_dim=embedding_dim,
                    sigma=sigma, enable=True)

    n_trials = 3
    test_acc_list = []
    for _ in range(n_trials):
        for epoch in range(1, epochs + 1):
            loss_trian, acc_train = train_one_epoch(credit, train_loader, optimizer, device)
            acc = eval_model(credit, test_loader, device)
            print(f"Epoch {epoch:02d}: Loss {loss_trian:.4f}, Train Acc  {acc_train:.4f}, Test Acc {acc:.4f}")

        test_acc = eval_model(credit, test_loader, device)
        test_acc_list.append(test_acc)
        print(f"Test Acc: {test_acc:.4f}")

    test_acc_mean = np.mean(test_acc_list)
    test_acc_std = np.std(test_acc_list)

    return {
        "dataset_name": dataset_name,
        "backbone_name": backbone_name,
        "test_acc_mean": test_acc_mean,
        "test_acc_std": test_acc_std,
    }


def train_credit():
    from utils.metrics import save_metrics_table
    dataset_names = ["ENZYMES", "PROTEINS"]
    backbone_names = ["GCN", "GAT", "GraphSAGE", "SSGC"]

    records = []
    for dn, bn in itertools.product(dataset_names, backbone_names):
        res = train_credit_once(dn, bn)
        records.append(res)

    save_metrics_table(records, out_name="graph_credit_trials")
