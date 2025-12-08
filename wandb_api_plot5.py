#!/usr/bin/env python
"""
For each dataset, compare DyGFormer:
- original time encoder + GeLU based FFN
- NWI time encoder + GeLU based FFN
- original time encoder + SwiGLU based FFN

For each (dataset, time_encoder, act_fn) config:
- Filter runs with:
    model_name == "DyGFormer"
    act_fn      in {gelu, swiglu}
    negative_sample_strategy in {historical, random, inductive}
- Aggregate ALL these runs (3 strategies * seeds) into
  a single smooth curve (mean + std).
- For each metric in METRICS, and for each dataset,
  plot one figure with three curves (mean ± std).
"""

import os
from typing import Dict

import numpy as np
import pandas as pd
import wandb

import matplotlib

matplotlib.use("Agg")  # force Agg backend
import matplotlib.pyplot as plt


# ===================== CONSTANTS =====================

WANDB_ENTITY = "dttutty"
WANDB_PROJECT = "DyGFormer-LinkPrediction"

META_CSV = "wandb_analysis/wandb_runs_meta.csv"

# x-axis: you can switch to "_step" if you prefer step-wise curves
X_AXIS = "epoch"

# All metrics you showed from history
METRICS = [
    # "train/average_precision",
    # "test_new_node/average_precision",
    # "train/roc_auc",
    # "val/loss",
    # "test/roc_auc",
    # "val/roc_auc",
    # "test/average_precision",
    "val/average_precision",
    # "train/batch_loss",
    # "train/loss_epoch",
    # "test_new_node/roc_auc",
    # "val_new_node/average_precision",
    # "val_new_node/roc_auc",
    # "val/new_node_loss",
]

METRIC_LABELS = {
    "val/average_precision": "Validation Average Precision",
    "test/average_precision": "Test Average Precision",
    "val/roc_auc": "Validation ROC-AUC",
    "test/roc_auc": "Test ROC-AUC",
}

MODEL_NAME = "DyGFormer"

# we now care about both GeLU and SwiGLU
ACT_FNS = ["gelu", "swiglu"]

NEG_STRATEGIES = ["historical", "random", "inductive"]

ONLY_FINISHED = True
SMOOTHING_WINDOW = 5

def smooth_curve(y: np.ndarray, window: int) -> np.ndarray:
    if window is None or window <= 1:
        return y

    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(y)           

    if mask.sum() == 0:
        return y

    kernel = np.ones(window, dtype=float)

    y_filled = np.where(mask, y, 0.0)
    num = np.convolve(y_filled, kernel, mode="same")

    denom = np.convolve(mask.astype(float), kernel, mode="same")

    out = num / np.maximum(denom, 1e-8)
    out[denom == 0] = np.nan

    return out


# (time_encoder, act_fn, legend_label, key_for_dict)
CONFIGS = [
    (
        "original",
        "gelu",
        "Original Time Encoder + GeLU based FFN",
        "original_gelu",
    ),
    (
        "nwi",
        "gelu",
        "NWI Time Encoder + GeLU based FFN",
        "nwi_gelu",
    ),
    (
        "original",
        "swiglu",
        "Original Time Encoder + SwiGLU based FFN",
        "original_swiglu",
    ),
]

# Root dir; each metric will have its own sub-folder under this
OUTPUT_ROOT = "wandb_analysis/plots_avg15_all_metrics"


# ===================== UTILS =====================


def extract_wandb_run_name(run_id: str) -> str:
    """Convert local run-20251207_054805-xxxx to W&B short id xxxx."""
    if "-" not in run_id:
        return run_id
    return run_id.split("-")[-1]


def fetch_curve_for_run(
    api: wandb.Api,
    entity: str,
    project: str,
    run_id: str,
    x_axis: str,
    metric: str,
) -> pd.DataFrame | None:
    """Fetch (x_axis, metric) curve for a single run via W&B API."""
    run_name = extract_wandb_run_name(run_id)
    path = f"{entity}/{project}/{run_name}"

    try:
        run = api.run(path)
    except Exception as e:
        print(f"[WARN] Failed to get run {path}: {e}")
        return None

    try:
        df = run.history(keys=[x_axis, metric])
    except Exception as e:
        print(f"[WARN] Failed to fetch history for {path}: {e}")
        return None

    if x_axis not in df.columns or metric not in df.columns:
        print(f"[WARN] Run {path} missing x={x_axis} or metric={metric}")
        return None

    df = df[[x_axis, metric]].dropna()
    if df.empty:
        print(f"[WARN] Empty history for run {path}")
        return None

    return df


def aggregate_curves(
    curves: Dict[str, pd.DataFrame],
    x_axis: str,
    metric: str,
) -> pd.DataFrame:
    """
    Aggregate multiple seed curves: align on x-axis and compute mean/std.
    """
    merged = None

    for label, df in curves.items():
        df = df.copy()
        df = df.sort_values(by=x_axis)
        df = df.drop_duplicates(subset=[x_axis], keep="last")
        df = df.rename(columns={metric: label})

        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on=x_axis, how="outer")

    if merged is None:
        return pd.DataFrame(columns=[x_axis, "mean", "std"])

    merged = merged.sort_values(by=x_axis)
    value_cols = [c for c in merged.columns if c != x_axis]
    values = merged[value_cols].to_numpy(dtype=float)

    mean = np.nanmean(values, axis=1)
    std = np.nanstd(values, axis=1)

    return pd.DataFrame({x_axis: merged[x_axis].to_numpy(), "mean": mean, "std": std})



def plot_dataset_comparison(
    dataset_name: str,
    metric: str,
    cfg_aggs: Dict[str, pd.DataFrame],
    x_axis: str,
    output_path: str,
):
    has_any = any((df is not None and not df.empty) for df in cfg_aggs.values())
    if not has_any:
        print(f"[WARN] dataset={dataset_name}: nothing to plot for metric={metric}")
        return

    plt.figure(figsize=(8, 5))

    for enc, act, legend_label, key in CONFIGS:
        agg = cfg_aggs.get(key)
        if agg is None or agg.empty:
            continue

        x = agg[x_axis].to_numpy()
        mean = agg["mean"].to_numpy()
        std = agg["std"].to_numpy()

        mean_smooth = smooth_curve(mean, SMOOTHING_WINDOW)
        std_smooth = smooth_curve(std, SMOOTHING_WINDOW)

        plt.plot(x, mean_smooth, linewidth=2.0, label=legend_label)
        plt.fill_between(
            x,
            mean_smooth - std_smooth,
            mean_smooth + std_smooth,
            alpha=0.15,
        )

    plt.xlabel(x_axis)
    plt.ylabel(METRIC_LABELS.get(metric, metric))
    plt.title(f"{dataset_name}")
    plt.legend(fontsize=9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved plot to {output_path}")



# ===================== MAIN =====================


def main():
    try:
        wandb.login()
    except Exception as e:
        print(f"[WARN] wandb.login() failed: {e}")

    if not os.path.exists(META_CSV):
        raise FileNotFoundError(f"Meta CSV not found: {META_CSV}")

    meta = pd.read_csv(META_CSV)
    if "run_id" not in meta.columns:
        raise ValueError("meta CSV must contain a 'run_id' column.")

    df = meta.copy()

    if ONLY_FINISHED and "finished" in df.columns:
        df = df[df["finished"] == True]

    # basic filters: DyGFormer + desired activations + desired strategies
    if "model_name" in df.columns:
        df = df[df["model_name"] == MODEL_NAME]
    if "act_fn" in df.columns:
        df = df[df["act_fn"].isin(ACT_FNS)]
    if "negative_sample_strategy" in df.columns:
        df = df[df["negative_sample_strategy"].isin(NEG_STRATEGIES)]

    if df.empty:
        print("[WARN] No runs left after filtering.")
        return

    if "dataset_name" not in df.columns or "time_encoder" not in df.columns:
        raise ValueError("meta CSV must contain 'dataset_name' and 'time_encoder'.")

    api = wandb.Api()

    # For each metric, build one folder and one figure per dataset
    for metric in METRICS:
        metric_safe = metric.replace("/", "-")
        metric_output_dir = os.path.join(OUTPUT_ROOT, metric_safe)

        for dataset, sub_df in df.groupby("dataset_name"):
            print(
                f"[INFO] metric={metric}, dataset={dataset}, "
                f"total runs={len(sub_df)}"
            )

            cfg_aggs: Dict[str, pd.DataFrame] = {}

            for enc, act, legend_label, key in CONFIGS:
                cfg_sub = sub_df[
                    (sub_df["time_encoder"] == enc) & (sub_df["act_fn"] == act)
                ]
                if cfg_sub.empty:
                    print(
                        f"[WARN] dataset={dataset}, time_encoder={enc}, "
                        f"act_fn={act}: no runs."
                    )
                    cfg_aggs[key] = pd.DataFrame(columns=[X_AXIS, "mean", "std"])
                    continue

                curves: Dict[str, pd.DataFrame] = {}

                for _, row in cfg_sub.iterrows():
                    run_id = str(row["run_id"])

                    parts = [enc, act]
                    if "negative_sample_strategy" in row and not pd.isna(
                        row["negative_sample_strategy"]
                    ):
                        parts.append(str(row["negative_sample_strategy"]))
                    if "seed" in row and not pd.isna(row["seed"]):
                        parts.append(f"seed={row['seed']}")
                    label = "|".join(parts)

                    df_curve = fetch_curve_for_run(
                        api=api,
                        entity=WANDB_ENTITY,
                        project=WANDB_PROJECT,
                        run_id=run_id,
                        x_axis=X_AXIS,
                        metric=metric,
                    )
                    if df_curve is None:
                        continue

                    curves[label] = df_curve

                agg_df = aggregate_curves(curves, x_axis=X_AXIS, metric=metric)
                cfg_aggs[key] = agg_df

            out_name = f"dataset-{dataset}_{metric_safe}_three_configs.png"
            output_path = os.path.join(metric_output_dir, out_name)

            plot_dataset_comparison(
                dataset_name=dataset,
                metric=metric,
                cfg_aggs=cfg_aggs,
                x_axis=X_AXIS,
                output_path=output_path,
            )


if __name__ == "__main__":
    main()
