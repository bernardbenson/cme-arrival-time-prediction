#!/usr/bin/env python3
"""
CME Arrival Time Prediction Training Script

This script trains neural network models to predict CME (Coronal Mass Ejection) 
arrival times using data from multiple CME events spanning 2010-2013.

The script automatically discovers training data files in date-named folders
and trains individual models for each CME event.
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
import json
import argparse

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class CMETrainingScript:
    def __init__(self, base_path="data", experiment_name=None):
        self.base_path = Path(base_path)
        self.results = []
        self.models = {}
        self.experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"cme_experiment_{self.experiment_timestamp}"
        
    def discover_cme_folders(self):
        """Discover all CME event folders in the directory."""
        pattern = "[0-9][0-9]_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]"
        folders = [f for f in self.base_path.iterdir() 
                  if f.is_dir() and f.name.count('_') == 1 and '-' in f.name]
        folders.sort()
        print(f"Discovered {len(folders)} CME event folders:")
        for folder in folders:
            print(f"  {folder.name}")
        return folders
    
    def read_time_from_file(self, filename):
        """Read datetime from time files."""
        try:
            with open(filename, 'r') as f:
                last_line = f.readlines()[-1].strip()
            time_str = last_line.split()[0]
            return datetime.strptime(time_str, '%Y/%m/%dT%H:%M')
        except Exception as e:
            print(f"Error reading time from {filename}: {e}")
            return None
    
    def load_training_data(self, folder_path):
        """Load neural network training data from a CME folder."""
        # Look for NN training file
        pattern = f"Train_*_NN.txt"
        nn_files = list(folder_path.glob(pattern))
        
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
    
    def extract_model_config(self):
        """Extract model configuration by analyzing the create_neural_network method."""
        import inspect
        
        # Get the source code of the create_neural_network method
        source = inspect.getsource(self.create_neural_network)
        
        config = {
            "script_file": __file__,
            "method_name": "create_neural_network",
            "timestamp": self.experiment_timestamp,
            "tensorflow_version": tf.__version__,
            "architecture_type": "feedforward_neural_network"
        }
        
        # Extract layer information from source code
        layers = []
        lines = source.split('\n')
        
        for line in lines:
            line = line.strip()
            if 'Dense(' in line:
                # Extract Dense layer parameters
                if 'units=' in line or 'Dense(' in line:
                    layer_info = {"type": "Dense"}
                    # Extract units
                    if 'Dense(' in line:
                        start = line.find('Dense(') + 6
                        end = line.find(',', start) if ',' in line[start:] else line.find(')', start)
                        try:
                            layer_info["units"] = int(line[start:end])
                        except:
                            pass
                    # Extract activation
                    if 'activation=' in line:
                        start = line.find("activation='") + 12
                        end = line.find("'", start)
                        layer_info["activation"] = line[start:end]
                    layers.append(layer_info)
            elif 'Dropout(' in line:
                # Extract dropout rate
                start = line.find('Dropout(') + 8
                end = line.find(')', start)
                try:
                    dropout_rate = float(line[start:end])
                    if layers:  # Add to last layer
                        layers[-1]["dropout"] = dropout_rate
                except:
                    pass
            elif 'BatchNormalization()' in line:
                if layers:  # Add to last layer
                    layers[-1]["batch_normalization"] = True
            elif 'Adam(' in line:
                # Extract optimizer settings
                if 'learning_rate=' in line:
                    start = line.find('learning_rate=') + 14
                    end = line.find(')', start) if ')' in line[start:] else line.find(',', start)
                    try:
                        config["learning_rate"] = float(line[start:end])
                    except:
                        pass
                config["optimizer"] = "Adam"
            elif 'loss=' in line:
                start = line.find("loss='") + 6 if "loss='" in line else line.find('loss="') + 6
                end = line.find("'", start) if "loss='" in line else line.find('"', start)
                config["loss_function"] = line[start:end]
        
        config["layers"] = layers
        config["total_layers"] = len([l for l in layers if l.get("type") == "Dense"])
        
        return config
    
    def create_neural_network(self, input_dim):
        """Create a simplified neural network model for better generalization."""
        # Input layer
        inputs = Input(shape=(input_dim,))
        
        # Simplified architecture with fewer layers and less aggressive dropout
        # Layer 1: Feature extraction
        x = Dense(64, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        
        # Layer 2: Pattern recognition
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        
        # Layer 3: Final processing
        x = Dense(16, activation='relu')(x)
        x = Dropout(0.05)(x)
        
        # Output layer with linear activation for regression
        outputs = Dense(1, activation='linear')(x)
        
        # Create and compile model with lower learning rate for stability
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_single_cme(self, folder_path, epochs=150):
        """Train a model for a single CME event."""
        cme_name = folder_path.name
        print(f"\n{'='*60}")
        print(f"Training model for CME: {cme_name}")
        print(f"{'='*60}")
        
        # Load data
        data, actual_travel_time, data_file = self.load_training_data(folder_path)
        if data is None:
            return None
            
        print(f"Data shape: {data.shape}")
        print(f"Actual travel time: {actual_travel_time:.2f} hours" if actual_travel_time else "Unknown")
        
        # Prepare features and targets
        feature_cols = ['Time_since_eruption', 'EA_diff_A', 'EA_diff_B']
        target_col = 'Travel_time_y'
        
        X = data[feature_cols].values
        y = data[target_col].values.reshape(-1, 1)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        
        # Scale targets with wider range for better numerical stability
        scaler_y = MinMaxScaler(feature_range=(0.0, 1.0))
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_val_scaled = scaler_y.transform(y_val)
        
        # Create and train model
        model = self.create_neural_network(X_train_scaled.shape[1])
        
        # Aggressive early stopping to avoid wasting training time
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, min_delta=0.0001, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', patience=8, factor=0.5, min_lr=1e-6, verbose=1)
        ]
        
        # Train model with larger batch size for stability
        print(f"Training neural network for {epochs} epochs...")
        history = model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=epochs,
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )
        
        # Make predictions
        y_pred_scaled = model.predict(X_val_scaled, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_val_original = scaler_y.inverse_transform(y_val_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_val_original, y_pred)
        mae = mean_absolute_error(y_val_original, y_pred)
        rmse = np.sqrt(mse)
        
        # For prediction on actual CME data (ensemble member 1, latest time)
        cme_data = data[data['Ensemble_member'] == 1.0].tail(1)
        if not cme_data.empty:
            X_cme = cme_data[feature_cols].values
            X_cme_scaled = scaler_X.transform(X_cme)
            pred_scaled = model.predict(X_cme_scaled, verbose=0)
            cme_prediction = scaler_y.inverse_transform(pred_scaled)[0, 0]
        else:
            cme_prediction = np.mean(y_pred)
        
        # Extract model configuration
        model_config = self.extract_model_config()
        
        # Store results
        result = {
            'cme_name': cme_name,
            'data_file': data_file,
            'actual_travel_time': actual_travel_time,
            'predicted_travel_time': cme_prediction,
            'prediction_error': abs(actual_travel_time - cme_prediction) if actual_travel_time else None,
            'validation_mse': mse,
            'validation_mae': mae,
            'validation_rmse': rmse,
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'model_config': model_config,
            'experiment_timestamp': self.experiment_timestamp,
            'experiment_name': self.experiment_name
        }
        
        # Store model and scalers
        self.models[cme_name] = {
            'model': model,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'history': history
        }
        
        print(f"Training completed:")
        print(f"  Validation RMSE: {rmse:.3f} hours")
        print(f"  Validation MAE: {mae:.3f} hours")
        if actual_travel_time:
            print(f"  Prediction error: {result['prediction_error']:.3f} hours")
        
        return result
    
    def train_all_cmes(self, epochs=150):
        """Train models for all discovered CME events."""
        folders = self.discover_cme_folders()
        
        print(f"\nStarting training for {len(folders)} CME events with {epochs} epochs...")
        
        for folder in folders:
            result = self.train_single_cme(folder, epochs)
            if result:
                self.results.append(result)
        
        return self.results
    
    def save_results(self, output_dir="results"):
        """Save training results with timestamp and model specifications."""
        if not self.results:
            print("No results to save")
            return
            
        # Create experiment-specific directory
        experiment_dir = Path(output_dir) / f"{self.experiment_name}"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # File names without timestamp (since directory is timestamped)
        csv_file = experiment_dir / "results.csv"
        summary_file = experiment_dir / "summary.txt"
        config_file = experiment_dir / "model_config.json"
        
        # Prepare data for CSV (flatten model_config)
        csv_results = []
        for result in self.results:
            csv_row = {k: v for k, v in result.items() if k != 'model_config'}
            csv_results.append(csv_row)
        
        # Save CSV results
        results_df = pd.DataFrame(csv_results)
        results_df.to_csv(csv_file, index=False)
        print(f"\nResults saved to: {csv_file}")
        
        # Save model configuration
        if self.results:
            model_config = self.results[0]['model_config']
            with open(config_file, 'w') as f:
                json.dump(model_config, f, indent=2, default=str)
            print(f"Model configuration saved to: {config_file}")
        
        # Generate and save summary
        self.generate_experiment_summary(summary_file, results_df)
        
        return csv_file, summary_file, config_file
    
    def generate_experiment_summary(self, summary_file, results_df):
        """Generate comprehensive experiment summary."""
        valid_results = results_df.dropna(subset=['prediction_error'])
        
        summary_lines = [
            "=" * 80,
            f"CME ARRIVAL TIME PREDICTION EXPERIMENT SUMMARY",
            "=" * 80,
            f"Experiment Name: {self.experiment_name}",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Script Version: {self.experiment_timestamp}",
            "",
            "DATASET INFORMATION:",
            f"  Total CME events processed: {len(self.results)}",
            f"  Events with valid predictions: {len(valid_results)}",
            f"  Date range: 2010-2013",
            f"  Features used: Time_since_eruption, EA_diff_A, EA_diff_B",
            "",
            "MODEL ARCHITECTURE:",
        ]
        
        # Add model architecture details
        if self.results:
            config = self.results[0]['model_config']
            summary_lines.extend([
                f"  Architecture: {config.get('architecture_type', 'Unknown')}",
                f"  Total layers: {config.get('total_layers', 'Unknown')}",
                f"  Optimizer: {config.get('optimizer', 'Unknown')}",
                f"  Learning rate: {config.get('learning_rate', 'Unknown')}",
                f"  Loss function: {config.get('loss_function', 'Unknown')}",
                f"  TensorFlow version: {config.get('tensorflow_version', 'Unknown')}",
                "",
                "  Layer Details:"
            ])
            
            for i, layer in enumerate(config.get('layers', []), 1):
                layer_desc = f"    Layer {i}: {layer.get('type', 'Unknown')}"
                if 'units' in layer:
                    layer_desc += f" ({layer['units']} units)"
                if 'activation' in layer:
                    layer_desc += f", activation={layer['activation']}"
                if 'dropout' in layer:
                    layer_desc += f", dropout={layer['dropout']}"
                if layer.get('batch_normalization'):
                    layer_desc += ", batch_norm=True"
                summary_lines.append(layer_desc)
        
        summary_lines.extend([
            "",
            "PERFORMANCE METRICS:",
        ])
        
        if not valid_results.empty:
            summary_lines.extend([
                f"  Mean Prediction Error: {valid_results['prediction_error'].mean():.3f} ± {valid_results['prediction_error'].std():.3f} hours",
                f"  Median Prediction Error: {valid_results['prediction_error'].median():.3f} hours",
                f"  Min Prediction Error: {valid_results['prediction_error'].min():.3f} hours",
                f"  Max Prediction Error: {valid_results['prediction_error'].max():.3f} hours",
                "",
                f"  Mean Validation RMSE: {results_df['validation_rmse'].mean():.3f} ± {results_df['validation_rmse'].std():.3f} hours",
                f"  Mean Validation MAE: {results_df['validation_mae'].mean():.3f} ± {results_df['validation_mae'].std():.3f} hours",
                f"  Mean Validation MSE: {results_df['validation_mse'].mean():.6f} ± {results_df['validation_mse'].std():.6f}",
            ])
        
        summary_lines.extend([
            "",
            "INDIVIDUAL CME RESULTS:",
            "-" * 40,
        ])
        
        # Add individual results
        for _, row in results_df.iterrows():
            summary_lines.extend([
                f"CME: {row['cme_name']}",
                f"  Actual Travel Time: {row['actual_travel_time']:.2f} hours",
                f"  Predicted Travel Time: {row['predicted_travel_time']:.2f} hours",
                f"  Prediction Error: {row['prediction_error']:.2f} hours" if pd.notna(row['prediction_error']) else "  Prediction Error: N/A",
                f"  Validation RMSE: {row['validation_rmse']:.3f} hours",
                f"  Training Samples: {row['training_samples']:,}",
                ""
            ])
        
        summary_lines.extend([
            "=" * 80,
            f"Summary generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80
        ])
        
        # Write summary to file
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"Experiment summary saved to: {summary_file}")
        
        # Print key statistics to console
        if not valid_results.empty:
            print(f"\nKEY RESULTS:")
            print(f"  Mean prediction error: {valid_results['prediction_error'].mean():.3f} ± {valid_results['prediction_error'].std():.3f} hours")
            print(f"  Best prediction error: {valid_results['prediction_error'].min():.3f} hours ({valid_results.loc[valid_results['prediction_error'].idxmin(), 'cme_name']})")
            print(f"  Worst prediction error: {valid_results['prediction_error'].max():.3f} hours ({valid_results.loc[valid_results['prediction_error'].idxmax(), 'cme_name']})")
            print(f"  Mean validation RMSE: {results_df['validation_rmse'].mean():.3f} hours")
    
    def plot_training_history(self, cme_name):
        """Plot training history for a specific CME."""
        if cme_name not in self.models:
            print(f"No model found for {cme_name}")
            return
            
        history = self.models[cme_name]['history']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title(f'Model Loss - {cme_name}')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # MAE plot
        ax2.plot(history.history['mae'], label='Training MAE')
        ax2.plot(history.history['val_mae'], label='Validation MAE')
        ax2.set_title(f'Model MAE - {cme_name}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save to experiment directory if available, otherwise current directory
        if hasattr(self, 'experiment_name'):
            from pathlib import Path
            experiment_dir = Path("results") / self.experiment_name
            if experiment_dir.exists():
                save_path = experiment_dir / f'training_history_{cme_name}.png'
            else:
                save_path = f'training_history_{cme_name}.png'
        else:
            save_path = f'training_history_{cme_name}.png'
            
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
        plt.show()


def main(experiment_name=None, epochs=150):
    """Main function to run the training script."""
    print("CME Arrival Time Prediction Training Script")
    print("=" * 50)
    
    # Initialize trainer with optional custom experiment name
    trainer = CMETrainingScript(experiment_name=experiment_name)
    
    # Train all models
    results = trainer.train_all_cmes(epochs)
    
    # Save results
    csv_file, summary_file, config_file = trainer.save_results()
    
    print("\nTraining completed successfully!")
    print(f"Results saved to: {csv_file}")
    print(f"Summary saved to: {summary_file}")
    print(f"Model config saved to: {config_file}")
    
    # Optionally plot training history for first CME
    if results:
        first_cme = results[0]['cme_name']
        print(f"\nPlotting training history for {first_cme}...")
        trainer.plot_training_history(first_cme)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CME arrival time prediction models')
    parser.add_argument('--name', '-n', type=str, help='Custom experiment name (default: auto-generated with timestamp)')
    parser.add_argument('--output-dir', '-o', type=str, default='results', help='Output directory for results (default: results)')
    parser.add_argument('--epochs', '-e', type=int, default=150, help='Number of training epochs (default: 150)')
    
    args = parser.parse_args()
    
    # If name provided, combine with timestamp for full experiment name
    if args.name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{args.name}_{timestamp}"
    else:
        experiment_name = None
    
    print(f"Experiment name: {experiment_name or 'auto-generated'}")
    print(f"Output directory: {args.output_dir}")
    print(f"Training epochs: {args.epochs}")
    print()
    
    main(experiment_name, args.epochs)