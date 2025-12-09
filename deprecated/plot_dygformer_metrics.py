import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "summary_dygformer_filtered.csv"

DATASET_EDGES = {
    "CanParl": 74478,
    "Contacts": 2426279,
    "enron": 125235,
    "Flights": 1927145,
    "lastfm": 1293103,
    "mooc": 411749,
    "myket": 694121,
    "reddit": 672447,
    "SocialEvo": 2099519,
    "uci": 59835,
    "UNtrade": 507497,
    "UNvote": 1035742,
    "USLegis": 60396,
    "wikipedia": 157474,
}

METRIC_BASES = [
    "val_ap",
    "val_roc",
    "new_val_ap",
    "new_val_roc",
    "test_ap",
    "test_roc",
    "new_test_ap",
    "new_test_roc",
]


def get_edges(dataset: str) -> int:
    return DATASET_EDGES.get(dataset, 1)


def pretty_metric_name(base: str) -> str:
    mapping = {
        "val_ap": "Val AP",
        "val_roc": "Val ROC-AUC",
        "new_val_ap": "New-node Val AP",
        "new_val_roc": "New-node Val ROC-AUC",
        "test_ap": "Test AP",
        "test_roc": "Test ROC-AUC",
        "new_test_ap": "New-node Test AP",
        "new_test_roc": "New-node Test ROC-AUC",
    }
    return mapping.get(base, base)


def plot_pair_bars(df: pd.DataFrame, cfg_a: str, cfg_b: str, out_png: str):
    print(f"[+] Plotting {cfg_a} (left) vs {cfg_b} (right) -> {out_png}")

    sub = df[df["config"].isin([cfg_a, cfg_b])].copy()
    if sub.empty:
        raise ValueError(f"No rows for configs {cfg_a} and {cfg_b} in CSV.")

    datasets = sorted(
        sub["dataset"].unique(),
        key=lambda d: get_edges(d),
        reverse=True,
    )

    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True)
    axes = axes.flatten()

    x = np.arange(len(datasets))
    width = 0.35

    for idx, base in enumerate(METRIC_BASES):
        ax = axes[idx]
        mean_col = f"{base}_mean"
        var_col = f"{base}_var"

        if mean_col not in sub.columns or var_col not in sub.columns:
            ax.set_title(pretty_metric_name(base) + " (missing)")
            ax.axis("off")
            continue

        means_a, stds_a = [], []
        means_b, stds_b = [], []

        for d in datasets:
            row_a = sub[(sub["config"] == cfg_a) & (sub["dataset"] == d)]
            row_b = sub[(sub["config"] == cfg_b) & (sub["dataset"] == d)]

            m_a = float(row_a.iloc[0][mean_col]) if not row_a.empty else np.nan
            v_a = float(row_a.iloc[0][var_col]) if not row_a.empty else np.nan
            m_b = float(row_b.iloc[0][mean_col]) if not row_b.empty else np.nan
            v_b = float(row_b.iloc[0][var_col]) if not row_b.empty else np.nan

            means_a.append(m_a)
            stds_a.append(np.sqrt(v_a) if not np.isnan(v_a) else np.nan)
            means_b.append(m_b)
            stds_b.append(np.sqrt(v_b) if not np.isnan(v_b) else np.nan)

        means_a = np.array(means_a)
        means_b = np.array(means_b)
        stds_a = np.array(stds_a)
        stds_b = np.array(stds_b)

        ax.bar(
            x - width / 2,
            means_a,
            width,
            yerr=stds_a,
            capsize=4,
            label=cfg_a if idx == 0 else None,
        )
        ax.bar(
            x + width / 2,
            means_b,
            width,
            yerr=stds_b,
            capsize=4,
            label=cfg_b if idx == 0 else None,
        )

        valid_vals = np.concatenate([
            means_a[~np.isnan(means_a)],
            means_b[~np.isnan(means_b)],
        ])
        if len(valid_vals) > 0:
            ymin = valid_vals.min()
            ymax = valid_vals.max()
            margin = (ymax - ymin) * 0.20 if ymax > ymin else 0.01
            ax.set_ylim(ymin - margin, ymax + margin)

        ax.set_title(pretty_metric_name(base))
        ax.set_ylabel("score")

        if idx // 4 == 1:
            xticklabels = [f"{d}\n({get_edges(d)})" for d in datasets]
            ax.set_xticks(x)
            ax.set_xticklabels(xticklabels, rotation=30, ha="right")
        else:
            ax.set_xticks(x)
            ax.set_xticklabels([])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 1.02),
        )

    fig.suptitle(f"{cfg_a} (left) vs {cfg_b} (right)", y=1.05, fontsize=16)
    fig.tight_layout()

    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_png}")


def main():
    print("[+] Reading CSV...")
    df = pd.read_csv(CSV_PATH)
    print("[+] CSV loaded, shape:", df.shape)

    if "config" not in df.columns or "dataset" not in df.columns:
        raise ValueError("CSV must contain 'config' and 'dataset' columns.")

    plot_pair_bars(
        df,
        cfg_a="original_gelu",
        cfg_b="original_swiglu",
        out_png="bars_original_gelu_vs_original_swiglu.png",
    )

    plot_pair_bars(
        df,
        cfg_a="original_gelu",
        cfg_b="nwi_gelu",
        out_png="bars_original_gelu_vs_nwi_gelu.png",
    )

    print("[+] All done.")


if __name__ == "__main__":
    main()
