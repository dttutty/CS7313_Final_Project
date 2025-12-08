#!/usr/bin/env python
"""
Use W&B API to fetch curves and plot aggregated metrics.

No command line arguments: all configs are in the CONSTANTS section below.
"""

import os
from typing import Dict

import pandas as pd
import numpy as np
import wandb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===================== CONSTANTS =====================

WANDB_ENTITY = "dttutty"
WANDB_PROJECT = "DyGFormer-LinkPrediction"

META_CSV = "wandb_analysis/wandb_runs_meta.csv"

X_AXIS = "epoch"

METRICS = [
    "val/average_precision",
    "val_new_node/average_precision",
    "val/roc_auc",
    "val_new_node/roc_auc",
]

GROUP_BY = [
    "dataset_name",
    "model_name",
    "time_encoder",
    "act_fn",
    "negative_sample_strategy",
]

ONLY_FINISHED = True
OUTPUT_DIR = "wandb_analysis/plots"


# ===================== FUNCTIONS =====================


def extract_wandb_run_name(run_id: str) -> str:
    """
    Convert local run_id like 'run-20251207_054805-tw9wjod9'
    or directory name into W&B API run name.
    """
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
    """
    Fetch (x_axis, metric) curve for a single run via W&B API.

    Returns DataFrame or None if failed.
    """
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
        print(f"[WARN] Run {path} missing required columns")
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
    Aggregate multiple runs by merging curves on x-axis and computing mean/std.
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

    out = pd.DataFrame({x_axis: merged[x_axis].to_numpy(), "mean": mean, "std": std})
    return out


def plot_group_curves(
    group_name: str,
    curves: Dict[str, pd.DataFrame],
    agg: pd.DataFrame,
    x_axis: str,
    metric: str,
    output_path: str,
):
    """
    Plot all runs + aggregated statistics.
    """
    if not curves:
        print(f"[WARN] Group {group_name} metric {metric}: no curves to plot.")
        return

    plt.figure(figsize=(8, 5))

    for label, df in curves.items():
        df = df.sort_values(by=x_axis)
        plt.plot(df[x_axis], df[metric], alpha=0.35, linewidth=1.0, label=label)

    if not agg.empty:
        plt.plot(agg[x_axis], agg["mean"], linewidth=2.0, label="mean")
        plt.fill_between(
            agg[x_axis],
            agg["mean"] - agg["std"],
            agg["mean"] + agg["std"],
            alpha=0.2,
        )

    plt.xlabel(x_axis)
    plt.ylabel(metric)
    plt.title(f"{group_name} | {metric}")
    plt.legend(fontsize=7)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved plot to {output_path}")


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

    if df.empty:
        print("[WARN] No runs to process after filtering.")
        return

    group_by_cols = [c for c in GROUP_BY if c in df.columns]

    if not group_by_cols:
        groups = [("all_runs", df)]
    else:
        groups = []
        for key, sub in df.groupby(group_by_cols):
            if not isinstance(key, tuple):
                key = (key,)
            group_name = ", ".join(f"{col}={val}" for col, val in zip(group_by_cols, key))
            groups.append((group_name, sub))

    api = wandb.Api()

    for metric in METRICS:
        for group_name, sub_df in groups:
            print(
                f"[INFO] Processing metric={metric}, group={group_name}, "
                f"runs={len(sub_df)}"
            )

            curves: Dict[str, pd.DataFrame] = {}

            for _, row in sub_df.iterrows():
                run_id = str(row["run_id"])

                parts = []
                for col in ["model_name", "time_encoder", "act_fn", "negative_sample_strategy", "seed"]:
                    if col in row and pd.notna(row[col]):
                        parts.append(str(row[col]))
                label = "|".join(parts) if parts else run_id

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

            agg = aggregate_curves(curves, x_axis=X_AXIS, metric=metric)

            safe_group_name = (
                group_name.replace(" ", "_")
                .replace("/", "_")
                .replace("=", "-")
                .replace(",", "_")
            )
            metric_safe = metric.replace("/", "-")
            output_path = os.path.join(
                OUTPUT_DIR,
                f"{safe_group_name}_{metric_safe}.png",
            )

            plot_group_curves(
                group_name=group_name,
                curves=curves,
                agg=agg,
                x_axis=X_AXIS,
                metric=metric,
                output_path=output_path,
            )


if __name__ == "__main__":
    main()
