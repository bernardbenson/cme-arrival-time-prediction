# CME Arrival Time Prediction

Neural network model for predicting Coronal Mass Ejection (CME) arrival times at Earth using elongation angle measurements from STEREO satellites.

## Overview

This project trains a PyTorch neural network to predict CME travel time (hours from eruption to Earth arrival) based on:
- **Time_since_eruption**: Elapsed time since CME eruption
- **EA_diff_A**: Elongation angle difference from STEREO-A satellite
- **EA_diff_B**: Elongation angle difference from STEREO-B satellite

The model is trained on 13 CME events spanning 2010-2013.

## Model Architecture

```
Input(3) → Linear(128) + GELU + BatchNorm + Dropout(0.1)
         → Linear(64) + GELU + BatchNorm + Dropout(0.1)
         → Linear(32) + GELU + Dropout(0.05)
         → Linear(16) + GELU + Dropout(0.05)
         → Linear(1)
```

**Key features:**
- GELU activation for smoother gradients
- He/Kaiming weight initialization
- AdamW optimizer with weight decay (L2 regularization)
- Cosine annealing learning rate schedule with warm restarts
- Early stopping with patience of 15 epochs

## Project Structure

```
cme_arrival_time_prediction/
├── cme_training_pytorch.py     # Training script
├── data/                       # CME event data (13 events)
│   ├── 01_2010-04-03/
│   ├── 02_2010-05-23/
│   └── ...
├── results/                    # Experiment outputs
├── pyproject.toml              # Dependencies
└── README.md
```

## Setup

### Using uv (recommended)

```bash
uv sync
uv run python cme_training_pytorch.py
```

### Using pip

```bash
python -m venv venv
source venv/bin/activate
pip install pandas numpy scipy polars scikit-learn matplotlib torch
python cme_training_pytorch.py
```

## Usage

```bash
# Basic training (150 epochs)
python cme_training_pytorch.py

# Quick test run
python cme_training_pytorch.py --epochs 20

# Full options
python cme_training_pytorch.py \
    --name my_experiment \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.0005 \
    --weight-decay 0.01 \
    --device mps
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--name, -n` | auto | Experiment name (timestamped) |
| `--epochs, -e` | 150 | Training epochs |
| `--batch-size, -b` | 64 | Batch size |
| `--lr` | 0.0005 | Learning rate |
| `--weight-decay, -w` | 0.01 | L2 regularization |
| `--device` | auto | cpu, cuda, or mps |
| `--seed` | 42 | Random seed |
| `--output-dir, -o` | results | Output directory |

## Output

Each experiment creates a timestamped folder in `results/` containing:
- `results.csv` - Per-event metrics
- `summary.txt` - Experiment summary with statistics
- `model_config.json` - Architecture and hyperparameters
- `training_history_*.png` - Loss/MAE plots

## Requirements

- Python 3.12+
- PyTorch
- scikit-learn
- pandas, numpy, scipy, polars
- matplotlib
