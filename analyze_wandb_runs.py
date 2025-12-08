import os
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt


WANDB_ROOT = "wandb"

ANALYSIS_ROOT = "wandb_analysis"
PLOTS_DIR = os.path.join(ANALYSIS_ROOT, "plots")

EPOCH_KEY = "epoch"
METRIC_KEY = "train/loss_epoch"  

COMPARE_LEFT = {"time_encoder": "nwi", "act_fn": "gelu"}
COMPARE_RIGHT = {"time_encoder": "original", "act_fn": "swiglu"}



def ensure_dirs():
    os.makedirs(ANALYSIS_ROOT, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def parse_args_list(args_list):
    """
    ["--dataset_name", "mooc", "--model_name", "DyGFormer", ...]
    to {"dataset_name": "mooc", "model_name": "DyGFormer", ...}
    """
    result = {}
    i = 0
    while i < len(args_list):
        key = args_list[i]
        if not isinstance(key, str) or not key.startswith("--"):
            i += 1
            continue
        key_clean = key.lstrip("-")
        if i + 1 < len(args_list) and not str(args_list[i + 1]).startswith("--"):
            val = args_list[i + 1]
            i += 2
        else:
            val = "True"
            i += 1
        result[key_clean] = val
    return result


def load_metadata(run_path):
    """
    {
        "dataset_name": ...,
        "model_name": ...,
        "time_encoder": ...,
        "act_fn": ...,
        "negative_sample_strategy": ...,
        "num_runs": int,
        ...
    }
    """
    meta_paths = [
        os.path.join(run_path, "wandb-metadata.json"),
        os.path.join(run_path, "files", "wandb-metadata.json"),
    ]

    meta = None
    for p in meta_paths:
        if os.path.isfile(p):
            with open(p, "r") as f:
                try:
                    meta = json.load(f)
                except Exception:
                    meta = None
            if meta is not None:
                break

    if meta is None:
        return None

    args_list = meta.get("args", [])
    args_dict = parse_args_list(args_list)

    dataset_name = args_dict.get("dataset_name")
    model_name = args_dict.get("model_name")
    act_fn = args_dict.get("act_fn")
    time_encoder = args_dict.get("time_encoder")
    neg_strategy = args_dict.get("negative_sample_strategy")
    num_runs = int(args_dict.get("num_runs", 1))

    if dataset_name is None or model_name is None or act_fn is None or time_encoder is None or neg_strategy is None:
        return None

    return {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "act_fn": act_fn,
        "time_encoder": time_encoder,
        "negative_sample_strategy": neg_strategy,
        "num_runs_script": num_runs,
        "metadata_raw": meta,
        "args_dict": args_dict,
    }


def parse_runtime_from_output(output_path):
    if not os.path.isfile(output_path):
        return False, None

    try:
        with open(output_path, "r") as f:
            lines = [ln.strip() for ln in f.readlines()]
    except Exception:
        return False, None

    last = ""
    for ln in reversed(lines):
        if ln:
            last = ln
            break

    if not last:
        return False, None

    if "Run " not in last or " cost " not in last or " seconds" not in last:
        return False, None

    try:
        after_cost = last.split("cost", 1)[1].strip()  # "67921.33 seconds."
        sec_str = after_cost.split()[0]
        runtime = float(sec_str)
    except Exception:
        return False, None

    return True, runtime


def load_loss_curve(history_path, epoch_key, metric_key):
    if not os.path.isfile(history_path):
        return None

    epochs = []
    vals = []

    try:
        with open(history_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if epoch_key in row and metric_key in row:
                    e = row[epoch_key]
                    v = row[metric_key]
                    if e is None or v is None:
                        continue
                    epochs.append(float(e))
                    vals.append(float(v))
    except Exception:
        return None

    if not epochs:
        return None

    epochs = np.array(epochs, dtype=float)
    vals = np.array(vals, dtype=float)
    order = np.argsort(epochs)
    return epochs[order], vals[order]


def aggregate_curves(curves):
    epoch_to_vals = defaultdict(list)
    for ep_arr, val_arr in curves:
        for e, v in zip(ep_arr, val_arr):
            epoch_to_vals[e].append(v)

    if not epoch_to_vals:
        return None

    epochs_sorted = np.array(sorted(epoch_to_vals.keys()), dtype=float)
    means = []
    stds = []
    counts = []

    for e in epochs_sorted:
        vals = np.array(epoch_to_vals[e], dtype=float)
        means.append(vals.mean())
        stds.append(vals.std(ddof=0))
        counts.append(len(vals))

    return (
        epochs_sorted,
        np.array(means, dtype=float),
        np.array(stds, dtype=float),
        np.array(counts, dtype=int),
    )




def main():
    ensure_dirs()

    if not os.path.isdir(WANDB_ROOT):
        print(f"wandb directory '{WANDB_ROOT}' not found.")
        return

    runs_meta = []
    curves_by_run = {}

    for d in os.listdir(WANDB_ROOT):
        run_path = os.path.join(WANDB_ROOT, d)
        if not os.path.isdir(run_path):
            continue
        if not d.startswith("run-"):
            continue

        run_id = d

        meta = load_metadata(run_path)
        if meta is None:
            continue

        dataset_name = meta["dataset_name"]
        model_name = meta["model_name"]
        act_fn = meta["act_fn"]
        time_encoder = meta["time_encoder"]
        neg_strategy = meta["negative_sample_strategy"]
        num_runs_script = meta["num_runs_script"]

        output_path = os.path.join(run_path, "files", "output.log")
        finished, runtime_sec = parse_runtime_from_output(output_path)

        history_path = os.path.join(run_path, "files", "wandb-history.jsonl")
        curve = load_loss_curve(history_path, EPOCH_KEY, METRIC_KEY)
        has_curve = curve is not None
        if has_curve:
            curves_by_run[run_id] = curve

        runs_meta.append(
            {
                "run_id": run_id,
                "run_path": run_path,
                "dataset_name": dataset_name,
                "model_name": model_name,
                "time_encoder": time_encoder,
                "act_fn": act_fn,
                "negative_sample_strategy": neg_strategy,
                "num_runs_script": num_runs_script,
                "finished": bool(finished),
                "runtime_seconds": runtime_sec if runtime_sec is not None else np.nan,
                "has_curve": has_curve,
            }
        )

    if not runs_meta:
        print("No valid W&B runs found.")
        return

    runs_meta_df = pd.DataFrame(runs_meta)
    runs_meta_csv = os.path.join(ANALYSIS_ROOT, "wandb_runs_meta.csv")
    runs_meta_df.to_csv(runs_meta_csv, index=False)
    print(f"Saved runs meta to: {runs_meta_csv}")

    finished_df = runs_meta_df[(runs_meta_df["finished"]) & (runs_meta_df["has_curve"])]

    # key: (dataset, model, time_encoder, act_fn, neg_strategy) -> list[run_id]
    config_to_runs = defaultdict(list)
    for _, row in finished_df.iterrows():
        key = (
            row["dataset_name"],
            row["model_name"],
            row["time_encoder"],
            row["act_fn"],
            row["negative_sample_strategy"],
        )
        config_to_runs[key].append(row["run_id"])

    curve_rows = []
    for key, run_ids in config_to_runs.items():
        dataset_name, model_name, time_encoder, act_fn, neg_strategy = key
        curves = []
        for rid in run_ids:
            if rid in curves_by_run:
                curves.append(curves_by_run[rid])
        if not curves:
            continue

        agg = aggregate_curves(curves)
        if agg is None:
            continue
        epochs_sorted, means, stds, counts = agg

        for e, m, s, c in zip(epochs_sorted, means, stds, counts):
            curve_rows.append(
                {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "time_encoder": time_encoder,
                    "act_fn": act_fn,
                    "negative_sample_strategy": neg_strategy,
                    "metric": METRIC_KEY,
                    "epoch": e,
                    "mean": m,
                    "std": s,
                    "num_runs_at_epoch": int(c),
                }
            )

    if not curve_rows:
        print("No aggregated curves produced.")
        return

    curves_df = pd.DataFrame(curve_rows)
    curves_csv = os.path.join(ANALYSIS_ROOT, "loss_curves_mean.csv")
    curves_df.to_csv(curves_csv, index=False)
    print(f"Saved aggregated loss curves to: {curves_csv}")

    #    (time_encoder=nwi, act_fn=gelu) vs (time_encoder=original, act_fn=swiglu)
    left_te = COMPARE_LEFT["time_encoder"]
    left_af = COMPARE_LEFT["act_fn"]
    right_te = COMPARE_RIGHT["time_encoder"]
    right_af = COMPARE_RIGHT["act_fn"]

    datasets = sorted(curves_df["dataset_name"].unique())
    envs = sorted(curves_df["negative_sample_strategy"].unique())

    for ds in datasets:
        for env in envs:
            df_left = curves_df[
                (curves_df["dataset_name"] == ds)
                & (curves_df["negative_sample_strategy"] == env)
                & (curves_df["time_encoder"] == left_te)
                & (curves_df["act_fn"] == left_af)
                & (curves_df["metric"] == METRIC_KEY)
            ]
            df_right = curves_df[
                (curves_df["dataset_name"] == ds)
                & (curves_df["negative_sample_strategy"] == env)
                & (curves_df["time_encoder"] == right_te)
                & (curves_df["act_fn"] == right_af)
                & (curves_df["metric"] == METRIC_KEY)
            ]

            if df_left.empty or df_right.empty:
                continue

            plt.figure(figsize=(7, 4))

            # left
            df_left_sorted = df_left.sort_values("epoch")
            x_left = df_left_sorted["epoch"].values
            y_left = df_left_sorted["mean"].values
            s_left = df_left_sorted["std"].values
            plt.plot(x_left, y_left, label=f"{left_te}+{left_af} (mean)")
            plt.fill_between(x_left, y_left - s_left, y_left + s_left, alpha=0.2)

            # right
            df_right_sorted = df_right.sort_values("epoch")
            x_right = df_right_sorted["epoch"].values
            y_right = df_right_sorted["mean"].values
            s_right = df_right_sorted["std"].values
            plt.plot(x_right, y_right, label=f"{right_te}+{right_af} (mean)")
            plt.fill_between(
                x_right, y_right - s_right, y_right + s_right, alpha=0.2
            )

            all_y = np.concatenate([y_left, y_right])
            y_min = float(all_y.min())
            y_max = float(all_y.max())
            if y_max > y_min:
                margin = 0.1 * (y_max - y_min)
                plt.ylim(y_min - margin, y_max + margin)

            plt.xlabel(EPOCH_KEY)
            plt.ylabel(METRIC_KEY)
            plt.title(f"{ds} ({env}) loss convergence")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()

            fname = f"{ds}_{env}_{left_te}_{left_af}_vs_{right_te}_{right_af}_{METRIC_KEY.replace('/', '_')}.png"
            out_path = os.path.join(PLOTS_DIR, fname)
            plt.savefig(out_path, dpi=200)
            plt.close()

            print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
