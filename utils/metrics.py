import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def _to_pct2(x: float) -> float:
    return float(f"{x * 100.0:.2f}")


def _acc(y_true, y_pred) -> float:
    return _to_pct2(accuracy_score(y_true, y_pred))


def _f1_macro(y_true, y_pred) -> float:
    return _to_pct2(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _recall_macro(y_true, y_pred) -> float:
    return _to_pct2(recall_score(y_true, y_pred, average="macro", zero_division=0))


def _precision_macro(y_true, y_pred) -> float:
    return _to_pct2(precision_score(y_true, y_pred, average="macro", zero_division=0))


def compute_downstream_perf(y_true, y_pred) -> dict:
    """
    dict：
    {
      "accuracy": 00.00,
      "f1_macro": 00.00,
      "recall_macro": 00.00,
      "precision_macro": 00.00
    }
    """
    return {
        "accuracy": _acc(y_true, y_pred),
        "f1_macro": _f1_macro(y_true, y_pred),
        "recall_macro": _recall_macro(y_true, y_pred),
        "precision_macro": _precision_macro(y_true, y_pred),
    }


def save_metrics_table(records: list[dict], out_name: str, results_dir: str = "./results"):
    """
    ./results/out_name_YYYYmmdd_HHMMSS.csv
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(results_dir) / f"{out_name}_{ts}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame.from_records(records)
    df.to_csv(out_path, index=False)
    return str(out_path)
