import yaml


def load_cfg(cfg_path):
    if isinstance(cfg_path, str):
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)
    assert isinstance(cfg_path, dict)
    return cfg_path


def get_hparams(cfg_path, dataset, backbone):
    """
    merge priority:
    defaults < per_dataset < per_backbone < per_combo
    """
    cfg = load_cfg(cfg_path)

    OCFG = cfg.get("optim", {})
    O_DEFAULTS = OCFG.get("defaults", {})
    O_DS = OCFG.get("per_dataset", {})
    O_BK = OCFG.get("per_backbone", {})
    O_COMBO = OCFG.get("per_combo", {})

    hp = dict(O_DEFAULTS)  # defaults
    hp.update(O_DS.get(dataset, {}))  # dataset-specific
    hp.update(O_BK.get(backbone, {}))  # backbone-specific
    hp.update(O_COMBO.get(f"{dataset}.{backbone}", {}))  # combo-specific
    return hp
