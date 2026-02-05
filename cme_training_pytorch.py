#!/usr/bin/env python3
"""
CME Arrival Time Prediction using PyTorch Neural Networks

This script trains neural network models to predict Coronal Mass Ejection (CME)
arrival times at Earth using elongation angle measurements from STEREO satellites.

The model predicts travel time (hours) from CME eruption to Earth arrival based on:
- Time since eruption
- Elongation angle differences from STEREO-A and STEREO-B satellites

Architecture improvements over TensorFlow baseline:
- GELU activation for smoother gradients
- Deeper network (128→64→32→16) for complex patterns
- He/Kaiming initialization for better training dynamics
- AdamW optimizer with weight decay for regularization
- Cosine annealing learning rate schedule

"""

# Standard library imports
import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset

# =============================================================================
# Constants
# =============================================================================

RANDOM_SEED = 42
DEFAULT_EPOCHS = 150
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.0005
DEFAULT_WEIGHT_DECAY = 0.01
FEATURE_COLUMNS = ["Time_since_eruption", "EA_diff_A", "EA_diff_B"]
TARGET_COLUMN = "Travel_time_y"

# Early stopping configuration
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 0.0001

# Learning rate scheduler configuration
LR_SCHEDULER_T0 = 10  # Initial restart period
LR_SCHEDULER_T_MULT = 2  # Period multiplier after each restart


def set_seed(seed: int = RANDOM_SEED) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# Dataset Class
# =============================================================================


class CMEDataset(Dataset):
    """
    PyTorch Dataset for CME training data.

    Handles loading and preprocessing of CME event data for neural network
    training. Supports both training and inference modes.

    Attributes:
        features: Tensor of input features (Time_since_eruption, EA_diff_A, EA_diff_B).
        targets: Tensor of target values (Travel_time_y).
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        """
        Initialize the CME dataset.

        Args:
            features: NumPy array of input features with shape (n_samples, n_features).
            targets: NumPy array of target values with shape (n_samples, 1).
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (features, target) tensors for the sample.
        """
        return self.features[idx], self.targets[idx]


# =============================================================================
# Neural Network Model
# =============================================================================


class CMEPredictor(nn.Module):
    """
    Neural network for CME arrival time prediction.

    Improved architecture with GELU activation, deeper layers, and better
    regularization compared to the TensorFlow baseline.

    Architecture:
        Input(3) → Linear(128) + GELU + BatchNorm + Dropout(0.1)
                 → Linear(64) + GELU + BatchNorm + Dropout(0.1)
                 → Linear(32) + GELU + Dropout(0.05)
                 → Linear(16) + GELU + Dropout(0.05)
                 → Linear(1)

    Attributes:
        layers: Sequential container of network layers.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dims: Tuple[int, ...] = (128, 64, 32, 16),
        dropout_rates: Tuple[float, ...] = (0.1, 0.1, 0.05, 0.05),
        use_batch_norm: Tuple[bool, ...] = (True, True, False, False),
    ) -> None:
        """
        Initialize the CME predictor model.

        Args:
            input_dim: Number of input features.
            hidden_dims: Tuple of hidden layer dimensions.
            dropout_rates: Tuple of dropout rates for each hidden layer.
            use_batch_norm: Tuple of booleans indicating batch norm usage per layer.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rates = dropout_rates
        self.use_batch_norm = use_batch_norm

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim, dropout_rate, batch_norm in zip(
            hidden_dims, dropout_rates, use_batch_norm
        ):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # GELU activation (smoother than ReLU for regression)
            layers.append(nn.GELU())

            # Batch normalization (if enabled for this layer)
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Dropout regularization
            layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        # Output layer (linear activation for regression)
        layers.append(nn.Linear(prev_dim, 1))

        self.layers = nn.Sequential(*layers)

        # Initialize weights using He/Kaiming initialization
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply He/Kaiming initialization to all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, 1) containing predicted travel times.
        """
        return self.layers(x)

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration as a dictionary.

        Returns:
            Dictionary containing model architecture details.
        """
        return {
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "dropout_rates": list(self.dropout_rates),
            "use_batch_norm": list(self.use_batch_norm),
            "activation": "GELU",
            "output_dim": 1,
        }


# =============================================================================
# Early Stopping
# =============================================================================


class EarlyStopping:
    """
    Early stopping to prevent overfitting during training.

    Monitors validation loss and stops training if no improvement is observed
    for a specified number of epochs.

    Attributes:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum change to qualify as an improvement.
        best_loss: Best validation loss observed.
        counter: Number of epochs without improvement.
        should_stop: Flag indicating whether to stop training.
        best_model_state: State dict of the best model.
    """

    def __init__(
        self,
        patience: int = EARLY_STOPPING_PATIENCE,
        min_delta: float = EARLY_STOPPING_MIN_DELTA,
        verbose: bool = True,
    ) -> None:
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum improvement required to reset patience counter.
            verbose: Whether to print early stopping messages.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss: Optional[float] = None
        self.counter = 0
        self.should_stop = False
        self.best_model_state: Optional[Dict[str, Any]] = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop based on validation loss.

        Args:
            val_loss: Current validation loss.
            model: Current model (used to save best weights).

        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict().copy()
        else:
            self.counter += 1
            if self.verbose:
                print(f"    EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def restore_best_weights(self, model: nn.Module) -> None:
        """
        Restore the best model weights.

        Args:
            model: Model to restore weights to.
        """
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


# =============================================================================
# Training Class
# =============================================================================


class CMETrainer:
    """
    Handles training loop, evaluation, and results generation for CME models.

    This class manages the complete training pipeline including data loading,
    model training, evaluation, and results persistence.

    Attributes:
        base_path: Path to the data directory.
        results: List of training results for each CME event.
        models: Dictionary storing trained models and scalers.
        experiment_name: Name of the current experiment.
        device: PyTorch device (CPU or CUDA).
    """

    def __init__(
        self,
        base_path: str = "data",
        experiment_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the CME trainer.

        Args:
            base_path: Path to the directory containing CME data folders.
            experiment_name: Custom name for the experiment.
            device: Device to use for training ('cpu', 'cuda', or 'mps').
        """
        self.base_path = Path(base_path)
        self.results: List[Dict[str, Any]] = []
        self.models: Dict[str, Dict[str, Any]] = {}
        self.experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = (
            experiment_name or f"cme_experiment_{self.experiment_timestamp}"
        )

        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

    def discover_cme_folders(self) -> List[Path]:
        """
        Discover all CME event folders in the data directory.

        Returns:
            Sorted list of paths to CME event folders.
        """
        folders = [
            f
            for f in self.base_path.iterdir()
            if f.is_dir() and f.name.count("_") == 1 and "-" in f.name
        ]
        folders.sort()
        print(f"Discovered {len(folders)} CME event folders:")
        for folder in folders:
            print(f"  {folder.name}")
        return folders

    def read_time_from_file(self, filename: Path) -> Optional[datetime]:
        """
        Read datetime from time reference files.

        Args:
            filename: Path to the time file (Erupt_time.txt or Arrival_time.txt).

        Returns:
            Parsed datetime or None if parsing fails.
        """
        try:
            with open(filename, "r") as f:
                last_line = f.readlines()[-1].strip()
            time_str = last_line.split()[0]
            return datetime.strptime(time_str, "%Y/%m/%dT%H:%M")
        except Exception as e:
            print(f"Error reading time from {filename}: {e}")
            return None

    def load_training_data(
        self, folder_path: Path
    ) -> Tuple[Optional[pd.DataFrame], Optional[float], Optional[str]]:
        """
        Load neural network training data from a CME folder.

        Args:
            folder_path: Path to the CME event folder.

        Returns:
            Tuple of (data DataFrame, actual travel time, data filename).
        """
        # Look for NN training file
        nn_files = list(folder_path.glob("Train_*_NN.txt"))

        if not nn_files:
            print(f"No NN training file found in {folder_path}")
            return None, None, None

        training_file = nn_files[0]
        print(f"Loading training data: {training_file.name}")

        # Load data
        data = pd.read_csv(training_file)

        # Load time reference files
        erupt_time = self.read_time_from_file(folder_path / "Erupt_time.txt")
        arrival_time = self.read_time_from_file(folder_path / "Arrival_time.txt")

        if erupt_time and arrival_time:
            actual_travel_time = (arrival_time - erupt_time).total_seconds() / 3600
        else:
            actual_travel_time = None

        return data, actual_travel_time, training_file.name

    def train_single_cme(
        self,
        folder_path: Path,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
    ) -> Optional[Dict[str, Any]]:
        """
        Train a model for a single CME event.

        Args:
            folder_path: Path to the CME event folder.
            epochs: Maximum number of training epochs.
            batch_size: Batch size for training.
            learning_rate: Initial learning rate.
            weight_decay: L2 regularization weight decay.

        Returns:
            Dictionary of results or None if training fails.
        """
        cme_name = folder_path.name
        print(f"\n{'=' * 60}")
        print(f"Training model for CME: {cme_name}")
        print(f"{'=' * 60}")

        # Load data
        data, actual_travel_time, data_file = self.load_training_data(folder_path)
        if data is None:
            return None

        print(f"Data shape: {data.shape}")
        if actual_travel_time:
            print(f"Actual travel time: {actual_travel_time:.2f} hours")
        else:
            print("Actual travel time: Unknown")

        # Prepare features and targets
        X = data[FEATURE_COLUMNS].values
        y = data[TARGET_COLUMN].values.reshape(-1, 1)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED
        )

        # Scale features
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)

        # Scale targets
        scaler_y = MinMaxScaler(feature_range=(0.0, 1.0))
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_val_scaled = scaler_y.transform(y_val)

        # Create datasets and dataloaders
        train_dataset = CMEDataset(X_train_scaled, y_train_scaled)
        val_dataset = CMEDataset(X_val_scaled, y_val_scaled)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Create model
        model = CMEPredictor(input_dim=X_train_scaled.shape[1])
        model = model.to(self.device)

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler (Cosine Annealing with Warm Restarts)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=LR_SCHEDULER_T0,
            T_mult=LR_SCHEDULER_T_MULT,
        )

        # Early stopping
        early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=True)

        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": [],
            "learning_rate": [],
        }

        print(f"Training neural network for {epochs} epochs...")

        # Training loop
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_mae = 0.0
            n_train_batches = 0

            for features, targets in train_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_mae += torch.mean(torch.abs(outputs - targets)).item()
                n_train_batches += 1

            train_loss /= n_train_batches
            train_mae /= n_train_batches

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_mae = 0.0
            n_val_batches = 0

            with torch.no_grad():
                for features, targets in val_loader:
                    features = features.to(self.device)
                    targets = targets.to(self.device)

                    outputs = model(features)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item()
                    val_mae += torch.mean(torch.abs(outputs - targets)).item()
                    n_val_batches += 1

            val_loss /= n_val_batches
            val_mae /= n_val_batches

            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            # Record history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_mae"].append(train_mae)
            history["val_mae"].append(val_mae)
            history["learning_rate"].append(current_lr)

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  Epoch {epoch + 1:3d}/{epochs}: "
                    f"loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                    f"mae={train_mae:.6f}, val_mae={val_mae:.6f}, lr={current_lr:.6f}"
                )

            # Early stopping check
            if early_stopping(val_loss, model):
                print(f"  Early stopping triggered at epoch {epoch + 1}")
                break

        # Restore best weights
        early_stopping.restore_best_weights(model)

        # Final evaluation on validation set
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(self.device)
                outputs = model(features)
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.numpy())

        y_pred_scaled = np.concatenate(all_preds, axis=0)
        y_val_scaled_np = np.concatenate(all_targets, axis=0)

        # Inverse transform predictions
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_val_original = scaler_y.inverse_transform(y_val_scaled_np)

        # Calculate metrics
        mse = float(np.mean((y_val_original - y_pred) ** 2))
        mae = float(np.mean(np.abs(y_val_original - y_pred)))
        rmse = float(np.sqrt(mse))

        # Prediction for actual CME (ensemble member 1, latest time)
        cme_data = data[data["Ensemble_member"] == 1.0].tail(1)
        if not cme_data.empty:
            X_cme = cme_data[FEATURE_COLUMNS].values
            X_cme_scaled = scaler_X.transform(X_cme)
            X_cme_tensor = torch.tensor(X_cme_scaled, dtype=torch.float32).to(
                self.device
            )

            with torch.no_grad():
                pred_scaled = model(X_cme_tensor).cpu().numpy()
            cme_prediction = float(scaler_y.inverse_transform(pred_scaled)[0, 0])
        else:
            cme_prediction = float(np.mean(y_pred))

        # Calculate prediction error
        prediction_error = (
            abs(actual_travel_time - cme_prediction) if actual_travel_time else None
        )

        # Store results
        result = {
            "cme_name": cme_name,
            "data_file": data_file,
            "actual_travel_time": actual_travel_time,
            "predicted_travel_time": cme_prediction,
            "prediction_error": prediction_error,
            "validation_mse": mse,
            "validation_mae": mae,
            "validation_rmse": rmse,
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "epochs_trained": len(history["train_loss"]),
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "best_val_loss": early_stopping.best_loss,
            "experiment_timestamp": self.experiment_timestamp,
            "experiment_name": self.experiment_name,
        }

        # Store model and scalers
        self.models[cme_name] = {
            "model": model,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "history": history,
            "config": model.get_config(),
        }

        print("Training completed:")
        print(f"  Validation RMSE: {rmse:.3f} hours")
        print(f"  Validation MAE: {mae:.3f} hours")
        if prediction_error is not None:
            print(f"  Prediction error: {prediction_error:.3f} hours")

        return result

    def train_all_cmes(
        self,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
    ) -> List[Dict[str, Any]]:
        """
        Train models for all discovered CME events.

        Args:
            epochs: Maximum number of training epochs per model.
            batch_size: Batch size for training.
            learning_rate: Initial learning rate.
            weight_decay: L2 regularization weight decay.

        Returns:
            List of result dictionaries for all trained models.
        """
        folders = self.discover_cme_folders()

        print(
            f"\nStarting training for {len(folders)} CME events with {epochs} epochs..."
        )

        for folder in folders:
            result = self.train_single_cme(
                folder,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
            if result:
                self.results.append(result)

        return self.results

    def save_results(self, output_dir: str = "results") -> Tuple[Path, Path, Path]:
        """
        Save training results including CSV, summary, and model configuration.

        Args:
            output_dir: Base directory for saving results.

        Returns:
            Tuple of paths to (CSV file, summary file, config file).
        """
        if not self.results:
            print("No results to save")
            raise ValueError("No results to save")

        # Create experiment-specific directory
        experiment_dir = Path(output_dir) / self.experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        csv_file = experiment_dir / "results.csv"
        summary_file = experiment_dir / "summary.txt"
        config_file = experiment_dir / "model_config.json"

        # Prepare CSV data
        csv_results = []
        for result in self.results:
            csv_row = {k: v for k, v in result.items()}
            csv_results.append(csv_row)

        # Save CSV
        results_df = pd.DataFrame(csv_results)
        results_df.to_csv(csv_file, index=False)
        print(f"\nResults saved to: {csv_file}")

        # Save model configuration
        model_config = self._generate_model_config()
        with open(config_file, "w") as f:
            json.dump(model_config, f, indent=2, default=str)
        print(f"Model configuration saved to: {config_file}")

        # Generate and save summary
        self._generate_experiment_summary(summary_file, results_df)

        # Plot training history for first CME
        if self.results:
            first_cme = self.results[0]["cme_name"]
            self._plot_training_history(first_cme, experiment_dir)

        return csv_file, summary_file, config_file

    def _generate_model_config(self) -> Dict[str, Any]:
        """
        Generate comprehensive model configuration dictionary.

        Returns:
            Dictionary containing model and training configuration.
        """
        # Get model config from first trained model
        first_model_key = next(iter(self.models.keys()))
        model_info = self.models[first_model_key]
        model_config = model_info["config"]

        # Build layer details
        layers = []
        layers.append(
            {
                "type": "Linear",
                "in_features": model_config["input_dim"],
                "out_features": model_config["hidden_dims"][0],
            }
        )

        for i, (hidden_dim, dropout, batch_norm) in enumerate(
            zip(
                model_config["hidden_dims"],
                model_config["dropout_rates"],
                model_config["use_batch_norm"],
            )
        ):
            layers.append({"type": "GELU"})
            if batch_norm:
                layers.append({"type": "BatchNorm1d", "num_features": hidden_dim})
            layers.append({"type": "Dropout", "p": dropout})

            # Add next linear layer
            if i < len(model_config["hidden_dims"]) - 1:
                layers.append(
                    {
                        "type": "Linear",
                        "in_features": hidden_dim,
                        "out_features": model_config["hidden_dims"][i + 1],
                    }
                )

        # Output layer
        layers.append(
            {
                "type": "Linear",
                "in_features": model_config["hidden_dims"][-1],
                "out_features": 1,
            }
        )

        return {
            "framework": "PyTorch",
            "pytorch_version": torch.__version__,
            "architecture_type": "feedforward_neural_network",
            "input_features": FEATURE_COLUMNS,
            "target": TARGET_COLUMN,
            "layers": layers,
            "hidden_dims": model_config["hidden_dims"],
            "activation": model_config["activation"],
            "dropout_rates": model_config["dropout_rates"],
            "use_batch_norm": model_config["use_batch_norm"],
            "weight_initialization": "kaiming_normal (He)",
            "optimizer": "AdamW",
            "learning_rate": DEFAULT_LEARNING_RATE,
            "weight_decay": DEFAULT_WEIGHT_DECAY,
            "loss_function": "MSELoss",
            "lr_scheduler": "CosineAnnealingWarmRestarts",
            "lr_scheduler_T0": LR_SCHEDULER_T0,
            "lr_scheduler_T_mult": LR_SCHEDULER_T_MULT,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
            "batch_size": DEFAULT_BATCH_SIZE,
            "feature_scaler": "StandardScaler",
            "target_scaler": "MinMaxScaler(0, 1)",
            "random_seed": RANDOM_SEED,
            "timestamp": self.experiment_timestamp,
        }

    def _generate_experiment_summary(
        self, summary_file: Path, results_df: pd.DataFrame
    ) -> None:
        """
        Generate comprehensive experiment summary with enhanced metrics.

        Args:
            summary_file: Path to save the summary file.
            results_df: DataFrame containing experiment results.
        """
        valid_results = results_df.dropna(subset=["prediction_error"])

        # Calculate enhanced statistics
        prediction_errors = valid_results["prediction_error"].values
        mean_error = prediction_errors.mean() if len(prediction_errors) > 0 else 0
        std_error = prediction_errors.std() if len(prediction_errors) > 0 else 0
        median_error = np.median(prediction_errors) if len(prediction_errors) > 0 else 0
        min_error = prediction_errors.min() if len(prediction_errors) > 0 else 0
        max_error = prediction_errors.max() if len(prediction_errors) > 0 else 0
        mae_of_predictions = mean_error  # MAE of prediction errors

        summary_lines = [
            "=" * 80,
            "CME ARRIVAL TIME PREDICTION EXPERIMENT SUMMARY",
            "=" * 80,
            f"Experiment Name: {self.experiment_name}",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Framework: PyTorch {torch.__version__}",
            "",
            "DATASET INFORMATION:",
            f"  Total CME events processed: {len(self.results)}",
            f"  Events with valid predictions: {len(valid_results)}",
            "  Date range: 2010-2013",
            f"  Features used: {', '.join(FEATURE_COLUMNS)}",
            f"  Target: {TARGET_COLUMN}",
            "",
            "MODEL ARCHITECTURE:",
        ]

        # Add model architecture details
        if self.models:
            first_model_key = next(iter(self.models.keys()))
            model_info = self.models[first_model_key]
            config = model_info["config"]

            summary_lines.extend(
                [
                    "  Architecture: Feedforward Neural Network",
                    f"  Hidden layers: {' → '.join(map(str, config['hidden_dims']))}",
                    f"  Activation: {config['activation']}",
                    f"  Dropout rates: {config['dropout_rates']}",
                    f"  Batch normalization: {config['use_batch_norm']}",
                    "  Weight initialization: Kaiming/He (normal)",
                    "",
                    "  Optimizer: AdamW",
                    f"  Learning rate: {DEFAULT_LEARNING_RATE}",
                    f"  Weight decay: {DEFAULT_WEIGHT_DECAY}",
                    f"  LR scheduler: CosineAnnealingWarmRestarts (T0={LR_SCHEDULER_T0}, T_mult={LR_SCHEDULER_T_MULT})",
                    "  Loss function: MSELoss",
                    "",
                    f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}",
                    f"  Batch size: {DEFAULT_BATCH_SIZE}",
                    f"  Random seed: {RANDOM_SEED}",
                ]
            )

        summary_lines.extend(
            [
                "",
                "PREDICTION ERROR STATISTICS:",
                f"  Mean Prediction Error: {mean_error:.3f} ± {std_error:.3f} hours",
                f"  Mean Absolute Error (MAE) of Predictions: {mae_of_predictions:.3f} hours",
                f"  Median Prediction Error: {median_error:.3f} hours",
                f"  Min Prediction Error: {min_error:.3f} hours",
                f"  Max Prediction Error: {max_error:.3f} hours",
                "",
                "VALIDATION METRICS (across all CME events):",
            ]
        )

        if not results_df.empty:
            summary_lines.extend(
                [
                    f"  Mean Validation RMSE: {results_df['validation_rmse'].mean():.3f} ± {results_df['validation_rmse'].std():.3f} hours",
                    f"  Mean Validation MAE: {results_df['validation_mae'].mean():.3f} ± {results_df['validation_mae'].std():.3f} hours",
                    f"  Mean Validation MSE: {results_df['validation_mse'].mean():.6f} ± {results_df['validation_mse'].std():.6f}",
                ]
            )

        summary_lines.extend(
            [
                "",
                "INDIVIDUAL CME RESULTS:",
                "-" * 40,
            ]
        )

        # Add individual results
        for _, row in results_df.iterrows():
            error_str = (
                f"{row['prediction_error']:.2f} hours"
                if pd.notna(row["prediction_error"])
                else "N/A"
            )
            summary_lines.extend(
                [
                    f"CME: {row['cme_name']}",
                    f"  Actual Travel Time: {row['actual_travel_time']:.2f} hours",
                    f"  Predicted Travel Time: {row['predicted_travel_time']:.2f} hours",
                    f"  Prediction Error: {error_str}",
                    f"  Validation RMSE: {row['validation_rmse']:.3f} hours",
                    f"  Training Samples: {row['training_samples']:,}",
                    f"  Epochs Trained: {row['epochs_trained']}",
                    "",
                ]
            )

        summary_lines.extend(
            [
                "=" * 80,
                f"Summary generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "=" * 80,
            ]
        )

        # Write summary to file
        with open(summary_file, "w") as f:
            f.write("\n".join(summary_lines))

        print(f"Experiment summary saved to: {summary_file}")

        # Print key statistics to console
        if not valid_results.empty:
            print("\nKEY RESULTS:")
            print(f"  Mean prediction error: {mean_error:.3f} ± {std_error:.3f} hours")
            print(f"  MAE of predictions: {mae_of_predictions:.3f} hours")
            best_idx = valid_results["prediction_error"].idxmin()
            worst_idx = valid_results["prediction_error"].idxmax()
            print(
                f"  Best prediction error: {min_error:.3f} hours ({valid_results.loc[best_idx, 'cme_name']})"
            )
            print(
                f"  Worst prediction error: {max_error:.3f} hours ({valid_results.loc[worst_idx, 'cme_name']})"
            )
            print(
                f"  Mean validation RMSE: {results_df['validation_rmse'].mean():.3f} hours"
            )

    def _plot_training_history(self, cme_name: str, output_dir: Path) -> None:
        """
        Plot and save training history for a specific CME.

        Args:
            cme_name: Name of the CME event to plot.
            output_dir: Directory to save the plot.
        """
        if cme_name not in self.models:
            print(f"No model found for {cme_name}")
            return

        history = self.models[cme_name]["history"]

        _, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Loss plot
        axes[0].plot(history["train_loss"], label="Training Loss", alpha=0.8)
        axes[0].plot(history["val_loss"], label="Validation Loss", alpha=0.8)
        axes[0].set_title(f"Model Loss - {cme_name}")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss (MSE)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # MAE plot
        axes[1].plot(history["train_mae"], label="Training MAE", alpha=0.8)
        axes[1].plot(history["val_mae"], label="Validation MAE", alpha=0.8)
        axes[1].set_title(f"Model MAE - {cme_name}")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Learning rate plot
        axes[2].plot(
            history["learning_rate"], label="Learning Rate", color="green", alpha=0.8
        )
        axes[2].set_title(f"Learning Rate Schedule - {cme_name}")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Learning Rate")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = output_dir / f"training_history_{cme_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training history plot saved to: {save_path}")
        plt.close()


# =============================================================================
# Main Function
# =============================================================================


def main(
    experiment_name: Optional[str] = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    device: Optional[str] = None,
) -> None:
    """
    Main entry point for CME arrival time prediction training.

    Args:
        experiment_name: Custom name for the experiment.
        epochs: Maximum number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization weight decay.
        device: Device to use for training.
    """
    print("CME Arrival Time Prediction Training Script (PyTorch)")
    print("=" * 60)

    # Set random seed for reproducibility
    set_seed(RANDOM_SEED)

    # Initialize trainer
    trainer = CMETrainer(experiment_name=experiment_name, device=device)

    # Train all models
    trainer.train_all_cmes(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    # Save results
    csv_file, summary_file, config_file = trainer.save_results()

    print("\nTraining completed successfully!")
    print(f"Results saved to: {csv_file}")
    print(f"Summary saved to: {summary_file}")
    print(f"Model config saved to: {config_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CME arrival time prediction models using PyTorch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cme_training_pytorch.py
  python cme_training_pytorch.py --epochs 20 --name quick_test
  python cme_training_pytorch.py --epochs 200 --lr 0.001 --weight-decay 0.005

Architecture:
  This script uses an improved neural network architecture compared to the
  TensorFlow baseline:
  - GELU activation (smoother gradients for regression)
  - Deeper network (128→64→32→16 hidden units)
  - He/Kaiming weight initialization
  - AdamW optimizer with weight decay
  - Cosine annealing learning rate schedule
        """,
    )

    parser.add_argument(
        "--name", "-n", type=str, help="Custom experiment name (auto-timestamped)"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="results",
        help="Output directory for results (default: results)",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for training (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Initial learning rate (default: {DEFAULT_LEARNING_RATE})",
    )
    parser.add_argument(
        "--weight-decay",
        "-w",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help=f"Weight decay for AdamW optimizer (default: {DEFAULT_WEIGHT_DECAY})",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        help="Device to use for training (default: auto-detect)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {RANDOM_SEED})",
    )

    args = parser.parse_args()

    # Set custom seed if provided
    if args.seed != RANDOM_SEED:
        set_seed(args.seed)

    # Generate experiment name with timestamp
    if args.name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{args.name}_{timestamp}"
    else:
        experiment_name = None

    print(f"Experiment name: {experiment_name or 'auto-generated'}")
    print(f"Output directory: {args.output_dir}")
    print(f"Training epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Random seed: {args.seed}")
    print()

    main(
        experiment_name=experiment_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
    )
