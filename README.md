# Dynamic Link Prediction with DyGLib

## Overview

This project uses Dynamic Graph Library (DyGLib) to study dynamic link prediction
on temporal graphs. We focus on the DyGFormer backbone and two variants:

- NWI (Non-Uniform, data-aware Weight Initialization) time encoder
- Gated SwiGLU feed-forward network

All training and evaluation code is adapted from DyGLib.

---

## 1. Environment Setup

We use Conda with an `environment.yml` file.

From the repository root:

```
conda env create -f environment.yml
conda activate dyglib-env
```

After activation, you should be able to run:

```
python -c "import torch; print(torch.__version__)"
```
to verify that PyTorch is correctly installed.

---

## 2. Datasets and Preprocessing

We use the datasets from:

  "Towards Better Evaluation for Dynamic Link Prediction"
  https://openreview.net/forum?id=1GVpwr2Tfdg

All datasets can be downloaded from the Zenodo record:

  https://zenodo.org/record/7213796

Place them under the DG_data folder in the repository root, e.g.

```
DG_data/
  wikipedia/
  reddit/
  mooc/
  uci/
  CanParl/
  USLegis/
  ...
```

### 2.1 Standard preprocessing

From the repository root:

```
cd preprocess_data/
```
# preprocess a single dataset
```
python preprocess_data.py --dataset_name wikipedia

```
# or preprocess all datasets

```
python preprocess_all_data.py
```

### 2.2 Dataset statistics for NWI

The NWI time encoder uses simple dataset-level statistics.

```
cd preprocess_data/
python data_statistics.py
```
Use the printed statistics to configure the NWI time encoder in `models/modules.py`. After this, you can use `--time_encoder nwi` during training.

---

## 3. Dynamic Link Prediction: Running the Code

All experiments are dynamic link prediction tasks using DyGLib’s pipeline.

### 3.1 Single run example

From the repository root:

```
python train_link_prediction.py \
  --dataset_name wikipedia \
  --model_name DyGFormer \
  --patch_size 2 \
  --max_input_sequence_length 64 \
  --num_runs 5 \
  --act_fn gelu \
  --time_encoder original \
  --gpu 0
```

Important flags:

- --dataset_name: wikipedia, reddit, mooc, uci, canparl, ...
- --model_name: DyGFormer, GraphMixer ...
- --act_fn: gelu or swiglu
- --time_encoder: original or nwi
- --gpu: GPU index

To run our variants:

- NWI time encoder: `--time_encoder nwi`
- SwiGLU FFN: `--act_fn swiglu`

---

## 4. Weights & Biases (wandb)

Training runs are logged to Weights & Biases.


`export WANDB_API_KEY=YOUR_API_KEY_HERE`

Wandb will record losses and metrics (e.g., validation/test AP) for each run.

---

## 5. Batch Experiments

We provide a simple script to compare standard configurations across datasets:

batch_run.sh

The default configuration list inside the script is:

```
configs=(
  "gelu original"   # baseline: GeLU + original time encoder
  "gelu nwi"        # NWI time encoder
  "swiglu original" # SwiGLU FFN
)
```

Edit batch_run.sh to change datasets or GPU IDs, then run:

`bash batch_run.sh`

---

## 6. Results and Analysis

All raw metrics are stored under:

```
saved_results/
  DyGFormer/
    ...
```

To aggregate metrics into CSV tables: `python aggregate_metrics.py`

To inspect which metrics are logged in wandb: `python inspect_wandb_run_columns.py`

To plot validation AP convergence curves: `python wandb_api_plot.py`, the generated plots will be saved to `wandb_analysis/`

