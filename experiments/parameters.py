import itertools
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from models.credit.credit import CREDIT
from models.credit.ov import OwnershipVerifier
from models.nn import ResNet, VGG, DenseNet, GoogLeNet
from utils.datasets import CIFAR10Dataset
from utils.datasets import get_dataset


def get_model(backbone_name: str, dataset_name: str, optimizer_name: str, scheduler_name: str,
              model_dir: str = "./data/models/target_model"):
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


def param_beta():
    device = "cuda:2"
    model_path = "./data/models/target/cifar10_resnet_sgd_cosine.pt"

    sigma = 0.4
    ver_dataset_ratio = 0.1

    target_model = ResNet(num_classes=10, small_input=True).to(device)
    state = torch.load(model_path, map_location=device)
    target_model.load_state_dict(state["model"], strict=False)
    train_loader, test_loader = CIFAR10Dataset().get_loader()
    credit = CREDIT(target_model, test_loader.dataset, dataset_ratio=ver_dataset_ratio, sigma=sigma)

    beta = credit.beta(device=device)
    print(f"beta = {beta}")


def param_tau():
    device = "cuda:2"
    model_path = "./data/models/target/cifar10_resnet_sgd_cosine.pt"

    sigma = 0.4
    ver_dataset_ratio = 0.1
    beta = 2650
    Q = 1000

    target_model = ResNet(num_classes=10, small_input=True).to(device)
    state = torch.load(model_path, map_location=device)
    target_model.load_state_dict(state["model"], strict=False)
    train_loader, test_loader = CIFAR10Dataset().get_loader()
    credit = CREDIT(target_model, test_loader.dataset, dataset_ratio=ver_dataset_ratio, sigma=sigma)

    tau = credit.tau(beta=beta, Q=Q)
    print(f"tau = {tau}")


def param_sigma():
    device = "cuda:2"
    model_path = "./data/models/target/cifar10_resnet_sgd_cosine.pt"

    sigma = 1.0
    ver_dataset_ratio = 0.1

    target_model = ResNet(num_classes=10, small_input=True).to(device)
    state = torch.load(model_path, map_location=device)
    target_model.load_state_dict(state["model"], strict=False)
    train_loader, test_loader = CIFAR10Dataset().get_loader()
    credit = CREDIT(target_model, train_loader.dataset, dataset_ratio=ver_dataset_ratio, sigma=sigma)
    sigma_star = credit.optim_sigma(
        sigmas=[0.0001, 0.01, 0.02, 1.0],
        gamma1_list=[0.001, 0.001, 0.001, 0.002],
        gamma2_list=[0.001, 0.001, 0.001, 0.002],
        device=device,
        max_batches=10,
        lambda_util=1.0,
        lambda_ver=1.0,
        pi0=0.5,
    )
    print(f"sigma_star = {sigma_star:.4f}")


def test_gamma():
    device = torch.device("cuda:2")

    # CREDIT params
    sigma = 0.4

    # verification params
    ver_ratio = 0.1
    batch_size = 128
    num_workers = 8
    k = 1
    seed = 42
    rho = 1.0
    eta = 0.8

    # suspicious models
    datasets = ["cifar10"]  # , "cifar100"
    dataset_name = "cifar10"
    backbones = ["vgg", "googlenet", "densenet"]
    optimizers = ["adam"]  # , "adamw", "rmsprop"
    schedulers = ["cosine"]  # , "step"

    # load dataset
    ds = get_dataset(name=dataset_name)
    train_loader, test_loader = ds.get_loader()

    Q = int(len(train_loader.dataset) * 0.6)

    # 1. load defense model
    target_model_path = f"./data/models/target/{dataset_name}_resnet_sgd_cosine.pt"
    target_model = ResNet(num_classes=10, small_input=True).to(device)
    state = torch.load(target_model_path, map_location="cpu")
    state_dict = state.get("model", state)
    target_model.load_state_dict(state_dict, strict=False)
    credit = CREDIT(
        target_model=target_model,
        dataset=test_loader.dataset,
        dataset_ratio=ver_ratio,
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
        defense_model=credit,
        dataset=test_loader.dataset,
        ver_ratio=ver_ratio,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        seed=seed,
    )

    # 5. ownership verification
    result = ov.verify(suspicious_models, suspicious_label, k=k, n_perm=1, tau=0.5)
    print(result)

    # sigma_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # sigma_finer_list = [0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45]
    sigma_finer_list = np.arange(0.1, 0.6 + 1e-9, 0.01)
    gamma1_list = []
    gamma2_list = []
    results = []
    C1 = 4
    C2 = 4
    for sigma in sigma_finer_list:
        beta = credit.beta(sigma=sigma, device=device)
        print("credit beta: ", beta)

        tau = credit.tau(beta=beta, Q=Q, rho=rho, eta=eta)
        print("credit tau: ", tau)

        gamma1, gamma2 = ov.compute_gamma(tau, C1, C2)
        print(f"gamma1: {gamma1}, gamma2: {gamma2}")

        gamma1_list.append(gamma1)
        gamma2_list.append(gamma2)

        results.append({
            "sigma": sigma,
            "beta": beta,
            "tau": tau,
            "C1": C1,
            "C2": C2,
            "gamma1": gamma1,
            "gamma2": gamma2,
            "max_gamma:": max(gamma1, gamma2),
        })

    optim_sigma, du_list, dv_list, objs = credit.optim_sigma(sigma_finer_list, gamma1_list, gamma2_list,
                                                             lambda_util=1.0, lambda_ver=5.0, device=device)

    if len(du_list) == len(results) == len(dv_list) == len(objs):
        for i in range(len(results)):
            results[i]["du"] = float(du_list[i])
            results[i]["dv"] = float(dv_list[i])
            results[i]["obj"] = float(objs[i])
    else:
        raise ValueError("optim sigma and du list must be the same length")

    for r in results:
        r["optim_sigma"] = float(optim_sigma)

    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"./results/sigma_beta_tau_gamma_{timestamp}.csv", index=False)
    print("optim_sigma: ", optim_sigma)

    return result


def test_ov():
    device = torch.device("cuda:2")

    # CREDIT params
    sigma = 0.4

    # verification params
    ver_ratio = 0.1
    batch_size = 128
    num_workers = 8
    k = 3
    seed = 42
    rho = 2.0
    eta = 100.0

    # suspicious models
    datasets = ["cifar10"]  # , "cifar100"
    backbones = ["vgg", "googlenet", "densenet"]
    optimizers = ["adam"]  # , "adamw", "rmsprop"
    schedulers = ["cosine"]  # , "step"

    # load dataset
    ds = get_dataset(name="cifar10")
    train_loader, test_loader = ds.get_loader()

    Q = int(len(train_loader.dataset) * 0.6)

    # 1. load defense model
    target_model_path = "./data/models/target_model/cifar10_resnet_sgd_cosine.pt"
    target_model = ResNet(num_classes=10, small_input=True).to(device)
    state = torch.load(target_model_path, map_location="cpu")
    state_dict = state.get("model", state)
    target_model.load_state_dict(state_dict, strict=False)
    credit = CREDIT(
        target_model=target_model,
        dataset=test_loader.dataset,
        dataset_ratio=ver_ratio,
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
        defense_model=credit,
        dataset=test_loader.dataset,
        ver_ratio=ver_ratio,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        seed=seed,
    )

    beta = credit.beta(sigma=sigma, device=device)
    print("credit beta: ", beta)

    tau = credit.tau(beta=beta, Q=Q, rho=rho, eta=eta)
    print("credit tau: ", tau)

    # 5. ownership verification
    result = ov.verify(suspicious_models, suspicious_label, k=k, n_perm=1, tau=tau)
    print(result)

    ov.compute_gamma(tau, C=10.0)

    return result


def plot_sigma_gamma():
    """
    figure 1
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import numpy as np

    rcParams['font.family'] = 'Times New Roman'
    rcParams['font.size'] = 20

    x_labels = [0.35,
                0.36,
                0.37,
                0.38,
                0.39,
                0.4,
                0.41,
                0.42,
                0.43,
                0.44, ]

    cited_inference_avg = np.array([0.1136, 0.1124, 0.1213, 0.1091, 0.1122, 0.1119, 0.1317])
    grove_inference_avg = np.array([0.1853, 0.1789, 0.1954, 0.2388, 0.3355, 0.4080, 1.0642])

    gamma1 = np.array([1.2255931209649804e-11,
                       2.5109103295846747e-09,
                       4.6728699567362625e-07,
                       2.4496975541896696e-06,
                       5.00542477697712e-05,
                       0.0001364037435077574,
                       0.0007503544434366763,
                       0.002288379429028542,
                       0.003405170218036428,
                       0.023113313154914022,
                       ])
    gamma2 = np.array([0.252457202811743,
                       0.049157403073097385,
                       0.0036733006354561465,
                       0.0012124228762576922,
                       9.755709726027929e-05,
                       3.515340580295286e-05,
                       4.6728006918128006e-06,
                       9.774648649786729e-07,
                       5.276807955880224e-07,
                       1.4957298331881612e-08,
                       ])

    cited_inference_std = np.array([0.1585, 0.1569, 0.1692, 0.1509, 0.1532, 0.1540, 0.1621])
    grove_inference_std = np.array([0.1935, 0.1754, 0.1735, 0.1834, 0.1834, 0.1476, 0.1588])

    fig = plt.figure(figsize=(4, 3.5))

    plt.plot(x_labels, gamma1, label='Gamma1', marker='s', linewidth=3, markersize=11,
             color=(123 / 255.0, 141 / 255.0, 191 / 255.0))
    plt.plot(x_labels, gamma2, label='Gamma2', marker='v', linewidth=3,
             markersize=11, color=(248 / 255.0, 120 / 255.0, 80 / 255.0))

    plt.gca().set_facecolor('#EEF0F2')
    plt.grid(True, linestyle='--', color='gray', alpha=0.5)
    plt.xlabel('Sigma', fontsize=23)
    plt.ylabel('Gamma (log)', fontsize=23)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', which='major', labelsize=21)
    ax.tick_params(axis='x', which='major', labelsize=18)
    plt.yscale('log')

    plt.legend(fontsize=16)

    save_path = './imgs/plot_sigma_gamma.pdf'
    plt.savefig(save_path, dpi=600, format='pdf', bbox_inches='tight')
    print(f"Saved: {save_path}")


def plot_sigma_beta_tau():
    """
    figure 2
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import numpy as np

    rcParams['font.family'] = 'Times New Roman'
    rcParams['font.size'] = 20

    x_labels = [0.35,
                0.36,
                0.37,
                0.38,
                0.39,
                0.4,
                0.41,
                0.42,
                0.43,
                0.44, ]

    cited_inference_avg = np.array([0.1136, 0.1124, 0.1213, 0.1091, 0.1122, 0.1119, 0.1317])
    grove_inference_avg = np.array([0.1853, 0.1789, 0.1954, 0.2388, 0.3355, 0.4080, 1.0642])

    beta = [3.6153450042489683,
            3.399895243818289,
            3.1418937846348767,
            3.045908709191115,
            2.8448217030169225,
            2.7679985188807636,
            2.6212160832792364,
            2.510656252953804,
            2.4676214435695907,
            2.2217213797364397,
            ]

    tau = [
        0.4483303346037226,
        0.39802131067013163,
        0.34148290986546853,
        0.3214908519793047,
        0.2814591009049726,
        0.26683165281292975,
        0.23991606652029104,
        0.22054317282288224,
        0.21321276415857277,
        0.17360521882511454,
    ]

    cited_inference_std = np.array([0.1585, 0.1569, 0.1692, 0.1509, 0.1532, 0.1540, 0.1621])
    grove_inference_std = np.array([0.1935, 0.1754, 0.1735, 0.1834, 0.1834, 0.1476, 0.1588])

    fig = plt.figure(figsize=(4, 3.5))

    plt.plot(x_labels, beta, label='Beta', marker='s', linewidth=3, markersize=11,
             color=(123 / 255.0, 141 / 255.0, 191 / 255.0))
    plt.plot(x_labels, tau, label='Tau', marker='v', linewidth=3,
             markersize=11, color=(248 / 255.0, 120 / 255.0, 80 / 255.0))

    plt.gca().set_facecolor('#EEF0F2')
    plt.grid(True, linestyle='--', color='gray', alpha=0.5)
    plt.xlabel('Sigma', fontsize=23)
    plt.ylabel('Mutual Information', fontsize=18)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', which='major', labelsize=21)
    ax.tick_params(axis='x', which='major', labelsize=18)
    # plt.yscale('log')

    plt.legend(fontsize=16)

    save_path = './imgs/plot_sigma_beta_tau.pdf'
    plt.savefig(save_path, dpi=600, format='pdf', bbox_inches='tight')
    print(f"Saved: {save_path}")


def plot_sigma_util_ver():
    """
    figure 2
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import numpy as np

    rcParams['font.family'] = 'Times New Roman'
    rcParams['font.size'] = 20

    sigma = [0.35,
             0.36,
             0.37,
             0.38,
             0.39,
             0.4,
             0.41,
             0.42,
             0.43,
             0.44, ]

    util_perf = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

    ver_robust = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    fig, ax1 = plt.subplots(figsize=(4, 3.5))

    color1 = (123 / 255.0, 141 / 255.0, 191 / 255.0)
    ax1.plot(sigma, util_perf, label='Utility', marker='s', linewidth=3,
             markersize=11, color=color1)
    ax1.set_xlabel('Sigma', fontsize=23)
    ax1.set_ylabel('Utility', fontsize=18, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=21)
    ax1.tick_params(axis='x', which='major', labelsize=18)
    ax1.set_facecolor('#EEF0F2')
    ax1.grid(True, linestyle='--', color='gray', alpha=0.5)

    ax2 = ax1.twinx()
    color2 = (248 / 255.0, 120 / 255.0, 80 / 255.0)
    ax2.plot(sigma, ver_robust, label='Robustness', marker='v', linewidth=3,
             markersize=11, color=color2)
    ax2.set_ylabel('Robustness', fontsize=18, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=21)

    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=16, loc='upper left')

    save_path = './imgs/plot_sigma_util_ver.pdf'
    plt.savefig(save_path, dpi=600, format='pdf', bbox_inches='tight')
    print(f"Saved: {save_path}")


def plot_sigma_selection():
    """
    sigma selection
    figure 4
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import numpy as np

    rcParams['font.family'] = 'Times New Roman'
    rcParams['font.size'] = 20

    sigma = [0.35,
             0.36,
             0.37,
             0.38,
             0.39,
             0.4,
             0.41,
             0.42,
             0.43,
             0.44, ]

    du = np.array([-0.659868242400251,
                   -0.41438885126845126,
                   -0.1756174619608238,
                   0.05681072408632602,
                   0.28322187970641927,
                   0.5039207189938361,
                   0.7191819138082994,
                   0.9292758442617243,
                   1.1344385564835597,
                   1.3349018048554475,
                   ])
    dv = np.array([1.0328748164732482,
                   0.09805259557592408,
                   -0.6595231054599755,
                   -0.7268092440343997,
                   -0.6917282811110529,
                   -0.6139044363497193,
                   -0.6957319285907565,
                   -0.6248568811536958,
                   -0.48084821312934084,
                   -0.105684687645567,
                   ])
    obj = np.array([4.504505839965989,
                    0.0758741266111691,
                    -3.4732329892607012,
                    -3.5772354960856725,
                    -3.175419525848845,
                    -2.5656014627547603,
                    -2.759477729145483,
                    -2.1950085615067545,
                    -1.2698025091631446,
                    0.8064783666276125,
                    ])

    fig = plt.figure(figsize=(4, 3.5))

    plt.plot(sigma, du, label='delta_u', marker='s', linewidth=3, )
    plt.plot(sigma, dv, label='delta_v', marker='v', linewidth=3, )
    plt.plot(sigma, obj, label='Obj', marker='s', linewidth=3, markersize=11,
             color=(123 / 255.0, 141 / 255.0, 191 / 255.0))

    plt.gca().set_facecolor('#EEF0F2')
    plt.grid(True, linestyle='--', color='gray', alpha=0.5)
    plt.xlabel('Sigma', fontsize=23)
    plt.ylabel('Obj', fontsize=23)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', which='major', labelsize=21)
    ax.tick_params(axis='x', which='major', labelsize=18)

    # y log scale
    # plt.yscale('log')

    plt.legend(fontsize=12)

    save_path = './imgs/plot_sigma_selection.pdf'
    plt.savefig(save_path, dpi=600, format='pdf', bbox_inches='tight')
    print(f"Saved: {save_path}")


def plot_sigma_gamma_range():
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib import font_manager as fm, rcParams

    path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
    fm.fontManager.addfont(path)
    rcParams['font.family'] = 'Times New Roman'
    rcParams['mathtext.fontset'] = 'stix'  # latex font
    rcParams['font.size'] = 20

    sigma = np.arange(0.23, 0.61, 0.01)
    gamma1 = np.array([2.1818886354527504e-79,
                       1.842528692669838e-77,
                       9.04358629917419e-57,
                       6.460168632731255e-67,
                       1.0629627277596738e-19,
                       2.376120104962765e-19,
                       2.022168954157279e-18,
                       3.925687897107083e-17,
                       5.474420744091655e-10,
                       4.6663202654751754e-11,
                       2.653821663680442e-06,
                       4.8706781702858135e-05,
                       0.0001175159023418723,
                       0.001384726240772122,
                       0.004132360708004411,
                       0.0025903381221489313,
                       0.011669012826848742,
                       0.0628328438477332,
                       0.07353575272846705,
                       0.06196263422577981,
                       0.2673752808383246,
                       0.27322259869136273,
                       0.3649361896225071,
                       0.25116546411640944,
                       0.35514393530775723,
                       0.5243196984895233,
                       0.3843557002899686,
                       0.6557847838812122,
                       0.7136857674519002,
                       0.6689993307734887,
                       0.6903030654519138,
                       0.762576818002348,
                       0.7053187608325613,
                       0.7945121833896808,
                       0.900003871243982,
                       0.8284867787445396,
                       0.8564552533444688,
                       0.8886649223263479,
                       ])
    gamma2 = np.array([0.9207856740930832,
                       0.9853594481324475,
                       0.037618922346117675,
                       0.0257901721880402,
                       2.0116583908495096e-19,
                       8.985235272875237e-20,
                       9.753159166287557e-21,
                       3.707640222336312e-22,
                       1.6978637788933552e-32,
                       1.3402277106306725e-30,
                       1.195829621566105e-40,
                       2.501923003950162e-44,
                       1.3959892212770042e-45,
                       1.4231665937999484e-49,
                       1.2017470795842168e-51,
                       9.863397991931921e-51,
                       7.301853509820763e-54,
                       3.091645260498207e-58,
                       1.022649373562352e-58,
                       3.404300771928183e-58,
                       1.6886570867690957e-63,
                       1.345078895389703e-63,
                       5.121336648039318e-65,
                       3.222941736370788e-63,
                       7.1039603638074225e-65,
                       3.742936276435167e-67,
                       2.7078357139177006e-65,
                       8.108441684389349e-69,
                       1.4419456271114795e-69,
                       5.492733080225399e-69,
                       2.916375825917464e-69,
                       3.1585767002292245e-70,
                       1.8577024483784438e-69,
                       1.117087334538817e-70,
                       2.131678573395512e-72,
                       3.4818886707390984e-71,
                       1.2507727469245403e-71,
                       3.469797435109541e-72,
                       ])

    print("sigma: {}, gamma1: {}, gamma2: {}".format(len(sigma),
                                                     len(gamma2),
                                                     len(gamma2)))
    assert len(sigma) == len(gamma2) == len(gamma2)

    sigma_tol_0 = 0.25
    sigma_tol_1 = 0.45

    gmax = np.maximum(gamma1, gamma2)

    idx0 = int(np.argmin(np.abs(sigma - sigma_tol_0)))
    idx1 = int(np.argmin(np.abs(sigma - sigma_tol_1)))

    color_blue = (123 / 255.0, 141 / 255.0, 191 / 255.0)
    color_orange = (248 / 255.0, 120 / 255.0, 80 / 255.0)

    def annotate_edge(idx, side):
        s = float(sigma[idx])
        g1 = float(gamma1[idx])
        g2 = float(gamma2[idx])
        if g1 >= g2:
            gmax_val = g1
            winner = r"\boldsymbol{\gamma}_{1}"
            color = color_blue
        else:
            gmax_val = g2
            winner = r"\boldsymbol{\gamma}_{2}"
            color = color_orange

        plt.scatter([s], [gmax_val], s=50, zorder=4, color=color, edgecolors="black")

        plt.axhline(gmax_val, linestyle=":", linewidth=1.2, color=color, alpha=0.7)

        plt.annotate(
            rf"$\mathbf{{{winner} = {gmax_val:.3f}}}$",
            xy=(s, gmax_val),
            xytext=(10, -20),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", lw=4, color=color),
            ha="left", va="top", color=color, fontsize=30, fontweight="bold"
        )

    def log10_labels(y, pos):
        return f"{int(np.log10(y))}"

    plt.figure(figsize=(10, 6))
    plt.axvspan(sigma_tol_0, sigma_tol_1, alpha=0.15)
    plt.plot(sigma, gamma1, label=r"$\gamma_1$", linewidth=4, color=color_blue)
    plt.plot(sigma, gamma2, label=r"$\gamma_2$", linewidth=4, color=color_orange)
    # plt.plot(sigma, gmax, label="max(gamma1, gamma2)", linewidth=2, linestyle="--")
    plt.axvline(sigma_tol_0, linestyle=":", linewidth=2.5)
    plt.axvline(sigma_tol_1, linestyle=":", linewidth=2.5)

    annotate_edge(idx0, side="left")
    annotate_edge(idx1, side="right")

    plt.yscale("log")
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(log10_labels))
    plt.xlabel(r"$\sigma$", fontsize=40)
    plt.ylabel(r"$\log_{10}(\gamma)$", fontsize=40)
    plt.tick_params(axis="both", which="major", labelsize=34)

    plt.legend(ncol=2, fontsize=36, loc="lower right")
    plt.tight_layout()
    plt.savefig("./imgs/sigma_gamma_range.pdf", dpi=600, format="pdf", bbox_inches="tight")
    plt.show()


def plot_sigma_optimal():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm, rcParams

    path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
    fm.fontManager.addfont(path)
    rcParams['font.family'] = 'Times New Roman'
    rcParams['mathtext.fontset'] = 'stix'  # latex font
    rcParams['font.size'] = 20

    sigma = np.arange(0.23, 0.61, 0.01)

    H_util = np.array([
        -0.6472157352595429,
        - 0.561019583096656,
        - 0.4781668573229797,
        - 0.39840648907253323,
        - 0.3215170387243894,
        - 0.24729837892160694,
        - 0.17557377381649747,
        - 0.10617844322165743,
        - 0.03896787959103571,
        0.026190468961133938,
        0.08942135715086685,
        0.15083186610940952,
        0.21052543828879128,
        0.2685946001033902,
        0.3251261600431003,
        0.38019964923938127,
        0.4338857620313085,
        0.48625571256927413,
        0.5373703187773355,
        0.5872898787682335,
        0.6360668934849578,
        0.6837538638704982,
        0.7303975729433605,
        0.7760442839107337,
        0.8207340222440059,
        0.8645057737919322,
        0.9073985244032671,
        0.949444502379648,
        0.9906774954566624,
        1.0311281725019974,
        1.0708256429493899,
        1.1097979766099433,
        1.1480706442381774,
        1.1856675571546618,
        1.2226136663025997,
        1.2589292443233433,
        1.2946382025374625,
        1.3297576947184093,

    ])
    H_ver = np.array([
        0.7108570255560481,
        - 2.9896689622255934,
        - 1.4404439255426036,
        8.394133164127284,
        - 4.3994587091364155,
        - 4.3994587091364155,
        - 4.3994587091364155,
        - 4.3994587091364155,
        - 4.399458484057403,
        - 4.399458688313522,
        - 4.398780753760311,
        - 4.3896320605727155,
        - 4.377660377550619,
        - 4.205668625049615,
        - 3.9046466433222573,
        - 4.066920107810886,
        - 3.226620938113098,
        - 0.06726899828676866,
        0.4495509684429733,
        - 0.11079722189439645,
        6.318425430484149,
        6.425619039194994,
        7.713463575282558,
        6.004274700887564,
        7.609490912841435,
        8.373527181397344,
        7.89714426964945,
        7.484319852770009,
        6.654153043357961,
        7.31990772979075,
        7.024063443208054,
        5.7177815232540805,
        6.79207360027287,
        4.9761941285585465,
        1.6011029890097894,
        4.059903149517288,
        3.193600008298062,
        2.048238096068298,
    ])

    weighted_util = 4 * H_util
    weighted_ver = 1 * H_ver
    Objective = weighted_util + weighted_ver

    idx_star = int(np.argmin(Objective))
    sigma_star = float(sigma[idx_star])
    obj_star = float(Objective[idx_star])

    color_blue = (123 / 255.0, 141 / 255.0, 191 / 255.0)
    color_orange = (248 / 255.0, 120 / 255.0, 80 / 255.0)
    color_red = (223 / 255.0, 113 / 255.0, 182 / 255.0)
    color_green = (87 / 255.0, 184 / 255.0, 147 / 255.0)

    plt.figure(figsize=(10, 6))
    plt.plot(sigma, weighted_util, linewidth=4, label=r"$H_{util}$", linestyle="--", color=color_orange)
    plt.plot(sigma, weighted_ver, linewidth=4, label=r"$H_{ver}$", linestyle="--", color=color_green)
    plt.plot(sigma, Objective, linewidth=5.5, label="Objective", color=color_blue)
    # (87 / 255.0, 184 / 255.0, 147 / 255.0)

    plt.axvline(sigma_star, linestyle=":", linewidth=2, color=color_blue)
    plt.axhline(obj_star, linestyle=":", linewidth=2, color=color_blue)
    plt.scatter([sigma_star], [obj_star], s=50, zorder=3, color=color_blue)
    plt.annotate(
        rf"$\sigma^*={sigma_star:.2f}$",
        xy=(sigma_star, obj_star),
        xytext=(10, -10),
        textcoords="offset points",
        # arrowprops=dict(arrowstyle="->", lw=3),
        fontsize=34,
    )

    plt.xlabel(r"$\sigma$", fontsize=40)
    plt.ylabel("Entropy", fontsize=40)
    plt.tick_params(axis="both", which="major", labelsize=34)
    plt.legend(ncol=1, fontsize=26, loc="lower right", labelspacing=0.3)
    plt.tight_layout()
    plt.savefig("./imgs/sigma_optimal.pdf", dpi=600, format='pdf', bbox_inches="tight")
    plt.show()


def plot_sigma_band():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm, rcParams

    path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
    fm.fontManager.addfont(path)
    rcParams['font.family'] = 'Times New Roman'
    rcParams['mathtext.fontset'] = 'stix'  # latex font
    rcParams['font.size'] = 20

    sigma = np.arange(0.23, 0.61, 0.01)

    beta = np.array([
        6.05378086316457,
        6.014272177789191,
        5.536484688447494,
        5.784309620550986,
        4.172302235888089,
        4.1522183316991335,
        4.097380016010973,
        4.017828357735239,
        3.4655041597893774,
        3.564640801554625,
        3.0410475523003853,
        2.8468310862878385,
        2.779810803732254,
        2.5621717724158652,
        2.4458546233231355,
        2.497464720057445,
        2.318298483638485,
        2.053688052149906,
        2.023305426349797,
        2.0563196175559586,
        1.7010013522668979,
        1.6938744445306853,
        1.588773447928191,
        1.7211282638383523,
        1.5995401230198225,
        1.419138972908229,
        1.5676379774712077,
        1.2742714460974476,
        1.204162821243132,
        1.2587559772990211,
        1.2331762410163136,
        1.139422244349504,
        1.2146683504682392,
        1.0931594668458289,
        0.8977004339036911,
        1.039091641556207,
        0.9894164073936608,
        0.9237879092348023,
    ])

    tau = np.array([
        1.203740940562386,
        1.1889084514558643,
        1.0160523391653267,
        1.104201234590903,
        0.5911890170251611,
        0.5857210791319519,
        0.5709105779285861,
        0.5497374281208167,
        0.4130449317931504,
        0.4362390648584255,
        0.32049349222234713,
        0.28184665735668324,
        0.26905667647479486,
        0.22947335913299385,
        0.20955011687535147,
        0.21828361116124556,
        0.18869697245974962,
        0.14878917570883382,
        0.14449867983765277,
        0.14916363496229587,
        0.1027269024638202,
        0.1018810457824957,
        0.08980123667901213,
        0.10513394646650914,
        0.09100470845440259,
        0.07186953007122254,
        0.08746139044445495,
        0.05809802869066689,
        0.05194708962963542,
        0.056707827974133575,
        0.05445178068929585,
        0.04656625650866834,
        0.052847360224012374,
        0.04289773499475539,
        0.02903187410896858,
        0.03879736076654395,
        0.03520831790323091,
        0.030729143142898895,
    ])

    print("sigma: {}, beta: {}, tau: {}".format(len(sigma),
                                                len(beta),
                                                len(tau)))
    assert len(sigma) == len(beta) == len(tau)

    plt.figure(figsize=(10, 6))

    plt.plot(sigma, tau, label=r"$\tau$", linewidth=2.5)
    plt.plot(sigma, beta, label=r"$\beta$", linewidth=2.5)

    plt.fill_between(sigma, beta, 0, alpha=0.25, label=r"$\tau - 0$")

    plt.fill_between(sigma, tau, beta, alpha=0.25, label=r"$\beta − \tau$")

    color_band_orange = (209 / 255, 197 / 255, 181 / 255)
    color_band_blue = (202 / 255, 219 / 255, 235 / 255)
    plt.xlabel(r"$\sigma$", fontsize=40)
    plt.ylabel("MI", fontsize=40)
    plt.tick_params(axis="both", which="major", labelsize=34)
    leg = plt.legend(ncol=1, fontsize=26, loc="upper right", labelspacing=0.3)
    leg.legend_handles[2].set_color(color_band_blue)
    leg.legend_handles[3].set_color(color_band_orange)
    plt.tight_layout()

    sigma_star = 0.27
    tau_star = np.interp(sigma_star, sigma, tau)
    beta_star = np.interp(sigma_star, sigma, beta)

    plt.axvline(x=sigma_star, color="black", linestyle="--", linewidth=2)

    plt.text(sigma_star + 0.02, plt.ylim()[1] * 0.85, r"$\sigma^*$",
             ha="center", va="bottom", fontsize=34)

    plt.annotate(rf"$\tau\ = {tau_star:.2f}$",
                 xy=(sigma_star, tau_star),
                 xytext=(sigma_star + 0.05, tau_star + 0.2),
                 arrowprops=dict(arrowstyle="->", linewidth=2),
                 fontsize=30)

    plt.annotate(rf"$\beta - \tau \ = {beta_star - tau_star:.2f}$",
                 xy=(sigma_star, beta_star),
                 xytext=(sigma_star + 0.05, beta_star - 0.3),
                 arrowprops=dict(arrowstyle="->", linewidth=2),
                 fontsize=30)

    plt.savefig("./imgs/sigma_band.pdf", dpi=600, format="pdf", bbox_inches="tight")
    plt.show()
