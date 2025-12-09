#!/usr/bin/env python
import os
import json
import csv
import math

ROOT = "saved_results/DyGFormer"

CONFIGS = ["nwi_gelu", "original_gelu", "original_swiglu"]

# (section, key) pairs in each json
METRIC_KEYS = [
    ("validate metrics", "average_precision"),
    ("validate metrics", "roc_auc"),
    ("new node validate metrics", "average_precision"),
    ("new node validate metrics", "roc_auc"),
    ("test metrics", "average_precision"),
    ("test metrics", "roc_auc"),
    ("new node test metrics", "average_precision"),
    ("new node test metrics", "roc_auc"),
]

HEADER = [
    "config",        # 1: nwi_gelu / original_gelu / original_swiglu
    "dataset",       # 2: 
    "sampling",      # 3: historical / inductive / random

    # 4–5: validate metrics average_precision mean/var
    "val_ap_mean",
    "val_ap_var",

    # 6–7: validate metrics roc_auc mean/var
    "val_roc_mean",
    "val_roc_var",

    # 8–9: new node validate metrics average_precision mean/var
    "new_val_ap_mean",
    "new_val_ap_var",

    # 10–11: new node validate metrics roc_auc mean/var
    "new_val_roc_mean",
    "new_val_roc_var",

    # 12–13: test metrics average_precision mean/var
    "test_ap_mean",
    "test_ap_var",

    # 14–15: test metrics roc_auc mean/var
    "test_roc_mean",
    "test_roc_var",

    # 16–17: new node test metrics average_precision mean/var
    "new_test_ap_mean",
    "new_test_ap_var",

    # 18–19: new node test metrics roc_auc mean/var
    "new_test_roc_mean",
    "new_test_roc_var",
]


def mean_and_var(values):
    """Return (mean, sample_variance)."""
    n = len(values)
    if n == 0:
        return math.nan, math.nan
    m = sum(values) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    return m, var


rows = []

for config in CONFIGS:
    config_path = os.path.join(ROOT, config)
    if not os.path.isdir(config_path):
        continue

    for dataset in sorted(os.listdir(config_path)):
        dataset_path = os.path.join(config_path, dataset)
        if not os.path.isdir(dataset_path):
            continue

        for sampling in sorted(os.listdir(dataset_path)):
            sampling_path = os.path.join(dataset_path, sampling)
            if not os.path.isdir(sampling_path):
                continue

            # collect all json files under this leaf dir
            json_files = [
                f for f in os.listdir(sampling_path)
                if f.endswith(".json")
            ]
            if not json_files:
                continue

            # for each metric key, maintain a list of values across seeds
            collected = [[] for _ in range(len(METRIC_KEYS))]

            for fname in json_files:
                fpath = os.path.join(sampling_path, fname)
                with open(fpath, "r") as f:
                    data = json.load(f)

                for idx, (section, key) in enumerate(METRIC_KEYS):
                    try:
                        value_str = data[section][key]
                        value = float(value_str)
                        collected[idx].append(value)
                    except (KeyError, TypeError, ValueError):
                        # skip missing or malformed values
                        pass

            # compute mean and variance for each metric list
            stats = []
            for values in collected:
                m, v = mean_and_var(values)
                stats.extend([m, v])

            row = [config, dataset, sampling] + stats
            rows.append(row)

# write csv
output_csv = "summary_dygformer.csv"
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(HEADER)
    writer.writerows(rows)

print(f"Written {output_csv} with {len(rows)} rows.")
