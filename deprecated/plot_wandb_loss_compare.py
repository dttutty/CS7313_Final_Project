import os
import json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

WANDB_DIR = "wandb"  
OUTPUT_DIR = "wandb_plots"

EPOCH_KEY = "epoch"
METRIC_KEY = "train/loss_epoch"  

ACT_FUNCS_TO_COMPARE = ("nwi_gelu", "original_swiglu")
# ====================


def parse_group(group_name: str):

    parts = group_name.split("_")
    if len(parts) < 5:
        return None

    dataset = parts[0]
    model = parts[1]
    time_encoder = parts[2]
    neg_strategy = parts[-1]
    act_fn = "_".join(parts[3:-1])

    return dataset, model, time_encoder, act_fn, neg_strategy


def load_run_history(history_path: str, epoch_key: str, metric_key: str):
    epochs = []
    metrics = []

    with open(history_path, "r") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            if epoch_key in row and metric_key in row:
                epochs.append(row[epoch_key])
                metrics.append(row[metric_key])

    if not epochs:
        return None

    epochs = np.array(epochs, dtype=float)
    metrics = np.array(metrics, dtype=float)

    order = np.argsort(epochs)
    return epochs[order], metrics[order]


def aggregate_runs(run_list):

    epoch_to_vals = defaultdict(list)
    for run in run_list:
        e = run["epoch"]
        v = run["metric"]
        for ep, val in zip(e, v):
            epoch_to_vals[ep].append(val)

    if not epoch_to_vals:
        return None

    epochs_sorted = np.array(sorted(epoch_to_vals.keys()), dtype=float)
    mean = np.array([np.mean(epoch_to_vals[ep]) for ep in epochs_sorted], dtype=float)
    std = np.array([np.std(epoch_to_vals[ep]) for ep in epochs_sorted], dtype=float)
    return epochs_sorted, mean, std


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # experiments[(dataset, model, time_encoder, neg_strategy, act_fn)] -> [runs...]
    experiments = defaultdict(list)

    if not os.path.isdir(WANDB_DIR):
        print(f"No wandb directory '{WANDB_DIR}' found.")
        return

    for run_dir in os.listdir(WANDB_DIR):
        run_path = os.path.join(WANDB_DIR, run_dir)
        if not os.path.isdir(run_path):
            continue

        metadata_path = os.path.join(run_path, "wandb-metadata.json")
        history_path = os.path.join(run_path, "files", "wandb-history.jsonl")

        if not (os.path.isfile(metadata_path) and os.path.isfile(history_path)):
            continue

        try:
            with open(metadata_path, "r") as f:
                meta = json.load(f)
        except Exception:
            continue

        group_name = meta.get("group")
        if not group_name:
            continue

        parsed = parse_group(group_name)
        if parsed is None:
            continue

        dataset, model, time_encoder, act_fn, neg_strategy = parsed

        if act_fn not in ACT_FUNCS_TO_COMPARE:
            continue

        history = load_run_history(history_path, EPOCH_KEY, METRIC_KEY)
        if history is None:
            continue

        epochs, metrics = history
        key = (dataset, model, time_encoder, neg_strategy, act_fn)
        experiments[key].append({"epoch": epochs, "metric": metrics})

    if not experiments:
        print("No matching runs found in wandb directory.")
        return

    base_to_act = defaultdict(dict)
    for (dataset, model, time_encoder, neg_strategy, act_fn), runs in experiments.items():
        base_key = (dataset, model, time_encoder, neg_strategy)
        base_to_act[base_key][act_fn] = runs

    for base_key, act_runs in base_to_act.items():
        if not all(a in act_runs for a in ACT_FUNCS_TO_COMPARE):
            continue

        dataset, model, time_encoder, neg_strategy = base_key
        print(f"Processing dataset={dataset}, model={model}, time_encoder={time_encoder}, neg={neg_strategy}")

        plt.figure(figsize=(7, 4))

        all_means = []

        for act_fn in ACT_FUNCS_TO_COMPARE:
            agg = aggregate_runs(act_runs[act_fn])
            if agg is None:
                continue
            epochs, mean, std = agg

            all_means.append(mean)

            plt.plot(epochs, mean, label=f"{act_fn} (mean)")
            plt.fill_between(epochs, mean - std, mean + std, alpha=0.2)

        if not all_means:
            plt.close()
            continue

        all_means_concat = np.concatenate(all_means)
        y_min = float(np.min(all_means_concat))
        y_max = float(np.max(all_means_concat))
        if y_max > y_min:
            margin = 0.1 * (y_max - y_min)
            plt.ylim(y_min - margin, y_max + margin)

        plt.xlabel(EPOCH_KEY)
        plt.ylabel(METRIC_KEY)
        plt.title(f"{dataset} {model} {time_encoder} ({neg_strategy})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        filename = f"{dataset}_{model}_{time_encoder}_{neg_strategy}_{METRIC_KEY.replace('/', '_')}.png"
        out_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
