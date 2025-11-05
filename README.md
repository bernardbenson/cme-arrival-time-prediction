# CME Arrival Time Prediction Project

This project trains neural network models to predict CME (Coronal Mass Ejection) arrival times using data from multiple CME events spanning 2010-2013.

## Project Structure

```
cme_arrival_time_prediction/
├── cme_training_script.py      # Main training script
├── data/                       # CME event data folders
│   ├── 01_2010-04-03/         # CME event data
│   ├── 02_2010-05-23/
│   ├── 03_2010-08-01/
│   └── ...                    # Additional CME events
├── archive/                    # Archived results and files
│   ├── results/               # Previous experiment results
│   ├── *.txt                  # Training results files
│   ├── *.csv                  # Training data files
│   └── *.png                  # Training history plots
├── pyproject.toml             # Project dependencies
├── uv.lock                    # Dependency lock file
└── README.md                  # This file
```

## How the Code Works

The `cme_training_script.py` contains a complete machine learning pipeline for CME arrival time prediction:

### Key Components

1. **Data Discovery**: Automatically discovers CME event folders in the `data/` directory with pattern `XX_YYYY-MM-DD`

2. **Data Loading**: For each CME event, loads:
   - Training data from `Train_*_NN.txt` files
   - Eruption time from `Erupt_time.txt`
   - Arrival time from `Arrival_time.txt`

3. **Neural Network Architecture**:
   - Input layer for 3 features: `Time_since_eruption`, `EA_diff_A`, `EA_diff_B`
   - Hidden layers: 64 → 32 → 16 neurons (ReLU activation)
   - Batch normalization and dropout for regularization
   - Linear output layer for regression

4. **Training Process**:
   - 80/20 train/validation split
   - StandardScaler for features, MinMaxScaler for targets
   - Adam optimizer with learning rate 0.0005
   - Early stopping and learning rate reduction
   - Configurable epochs (default: 150) with batch size 64

5. **Results Generation**:
   - Comprehensive experiment summaries
   - Model performance metrics (RMSE, MAE, prediction errors)
   - Training history plots
   - Model configuration documentation

### Features Used

The model uses three key features to predict CME travel time:
- **Time_since_eruption**: Time elapsed since CME eruption
- **EA_diff_A**: Earth-arrival time difference (ensemble member A)
- **EA_diff_B**: Earth-arrival time difference (ensemble member B)

## Setup Instructions

### Option 1: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

1. **Install uv** (if not already installed):
   ```bash
   # On macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Clone and setup the project**:
   ```bash
   cd cme_arrival_time_prediction
   uv sync
   ```

3. **Run the training script**:
   ```bash
   uv run python cme_training_script.py
   
   # Or with custom epochs
   uv run python cme_training_script.py --epochs 20
   ```

### Option 2: Using pip

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install pandas numpy scipy polars scikit-learn tensorflow keras matplotlib
   ```

3. **Run the training script**:
   ```bash
   python cme_training_script.py
   
   # Or with custom epochs
   python cme_training_script.py --epochs 20
   ```

## Usage

### Basic Usage

Run training on all CME events:
```bash
python cme_training_script.py
```

### Advanced Options

```bash
# Custom experiment name
python cme_training_script.py --name my_experiment

# Custom number of training epochs
python cme_training_script.py --epochs 20

# Custom output directory
python cme_training_script.py --output-dir my_results

# Combine options
python cme_training_script.py --name solar_study --epochs 50 --output-dir experiments
```

### Command Line Arguments

- `--name, -n`: Custom experiment name (auto-timestamped)
- `--epochs, -e`: Number of training epochs (default: 150)
- `--output-dir, -o`: Output directory for results (default: "results")

## Output Files

Each experiment generates a timestamped directory in `results/` containing:

- **`results.csv`**: Tabular results for all CME events with metrics
- **`summary.txt`**: Comprehensive experiment summary with statistics
- **`model_config.json`**: Neural network architecture details
- **`training_history_*.png`**: Training loss/MAE plots for the first CME event

### Recent Improvements

- **Configurable epochs**: Use `--epochs` to specify training duration (e.g., `--epochs 20` for quick experiments)
- **Fixed plotting bug**: Training history plots now correctly display both loss and MAE metrics
- **Organized output**: All experiment files (including plots) are saved in the same timestamped directory

## Data Format

Each CME event folder should contain:
- `Train_*_NN.txt`: Training data CSV with features and targets
- `Erupt_time.txt`: CME eruption timestamp
- `Arrival_time.txt`: CME Earth arrival timestamp

## Requirements

- Python 3.12+
- TensorFlow/Keras for neural networks
- scikit-learn for preprocessing and metrics
- pandas/numpy for data manipulation
- matplotlib for visualization