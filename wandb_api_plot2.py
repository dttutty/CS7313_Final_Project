#!/usr/bin/env python
"""
Fetch curves from W&B API and aggregate seed runs:
1. For each (dataset, model, time_encoder, act_fn, negative_sample_strategy),
   aggregate all seed curves (mean + std).
2. For each (dataset, model, time_encoder, act_fn),
   plot historical / random / inductive aggregated curves in the same figure.

All configs are defined in the CONSTANTS section.
"""

import os
from typing import Dict

import numpy as np
import pandas as pd
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
]

BASE_GROUP_KEYS = [
    "dataset_name",
    "model_name",
    "time_encoder",
    "act_fn",
]

NEG_STRATEGIES = ["historical", "random", "inductive"]

ONLY_FINISHED = True
OUTPUT_DIR = "wandb_analysis/plots"


# ===================== COMMON UTILS =====================


def extract_wandb_run_name(run_id: str) -> str:
    """Extract final W&B run name from a local run directory."""
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
    Fetch a single run's curve from W&B.
    Return a DataFrame([x_axis, metric]) or None on failure.
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
        print(f"[WARN] Missing required columns in run {path}")
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
    Aggregate multiple curves on the same x-axis and compute mean/std.
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


def plot_env_curves_for_group(
    base_group_name: str,
    metric: str,
    env_aggs: Dict[str, pd.DataFrame],
    x_axis: str,
    output_path: str,
):
    """
    Plot historical/random/inductive aggregated curves in the same figure.
    """
    has_any = any((not df.empty) for df in env_aggs.values())
    if not has_any:
        print(f"[WARN] No aggregated curves for group {base_group_name} metric={metric}.")
        return

    plt.figure(figsize=(8, 5))

    for neg in NEG_STRATEGIES:
        agg = env_aggs.get(neg)
        if agg is None or agg.empty:
            continue

        plt.plot(agg[x_axis], agg["mean"], linewidth=2.0, label=neg)
        plt.fill_between(
            agg[x_axis],
            agg["mean"] - agg["std"],
            agg["mean"] + agg["std"],
            alpha=0.15,
        )

    plt.xlabel(x_axis)
    plt.ylabel(metric)
    plt.title(f"{base_group_name} | {metric}")
    plt.legend(fontsize=8)
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

    if df.empty:
        print("[WARN] No runs remaining after filtering.")
        return

    base_keys = [c for c in BASE_GROUP_KEYS if c in df.columns]
    if not base_keys:
        base_keys = ["dataset_name"] if "dataset_name" in df.columns else []

    api = wandb.Api()

    for metric in METRICS:
        for key, sub in df.groupby(base_keys):
            if not isinstance(key, tuple):
                key = (key,)

            base_group_name = ", ".join(
                f"{col}={val}" for col, val in zip(base_keys, key)
            )

            print(
                f"[INFO] Processing metric={metric}, base_group={base_group_name}, "
                f"runs={len(sub)}"
            )

            if "negative_sample_strategy" not in sub.columns:
                print(f"[WARN] Missing 'negative_sample_strategy' column.")
                continue

            env_aggs: Dict[str, pd.DataFrame] = {}

            for neg in NEG_STRATEGIES:
                env_sub = sub[sub["negative_sample_strategy"] == neg]
                if env_sub.empty:
                    continue

                curves: Dict[str, pd.DataFrame] = {}

                for _, row in env_sub.iterrows():
                    run_id = str(row["run_id"])

                    parts = [neg]
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
                env_aggs[neg] = agg_df

            safe_group = (
                base_group_name.replace(" ", "_")
                .replace("/", "_")
                .replace("=", "-")
                .replace(",", "_")
            )
            metric_safe = metric.replace("/", "-")

            output_path = os.path.join(
                OUTPUT_DIR,
                f"{safe_group}_{metric_safe}.png",
            )

            plot_env_curves_for_group(
                base_group_name=base_group_name,
                metric=metric,
                env_aggs=env_aggs,
                x_axis=X_AXIS,
                output_path=output_path,
            )


if __name__ == "__main__":
    main()
