import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as utils
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import random
from typing import Dict, List, Tuple, Any, Optional
import optuna
from optuna.pruners import MedianPruner, PatientPruner
from automatic_assessment.sklearn.datasets import Dataset

# Set seeds for reproducibility
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- Visualization Module ---
class Visualizer:
    """Handles all plotting logic for training, optimization, and evaluation."""
    
    @staticmethod
    def plot_learning_curves(history: Dict[str, List[float]], title: str = "Training Progress"):
        """Plots Training vs Validation Loss over epochs."""
        epochs = range(1, len(history['train_loss']) + 1)
        
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history['train_loss'], label='Train Loss')
        if history['val_loss']:
            plt.plot(epochs, history['val_loss'], label='Val Loss', linestyle='--')
        
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss (Scaled)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("learning_curve.png")
        print("Saved learning_curve.png")
        # plt.show()

    @staticmethod
    def plot_optimization_history(study: optuna.Study):
        """Plots the optimization history (Objective Value vs Trial)."""
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not trials:
            print("No complete trials to plot.")
            return
            
        values = [t.value for t in trials]
        trial_nums = [t.number for t in trials]

        plt.figure(figsize=(10, 5))
        plt.scatter(trial_nums, values, color='blue', alpha=0.6, label='Trial')
        
        # Plot best value so far
        best_values = [np.min(values[:i+1]) for i in range(len(values))]
        plt.plot(trial_nums, best_values, color='red', linewidth=2, label='Best So Far')
        
        plt.title('Hyperparameter Optimization History')
        plt.xlabel('Trial')
        plt.ylabel('Mean CV Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("optuna_history.png")
        print("Saved optuna_history.png")

    @staticmethod
    def plot_evaluation(y_true: np.ndarray, y_pred: np.ndarray, target_names: Optional[List[str]] = None, dataset_name: str = "Test"):
        """
        Plots Parity Plot (Predicted vs Actual) and R2 Score per Target.
        """
        # 1. Parity Plot
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        # Flatten for global scatter
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Downsample for scatter plot if data is huge
        if len(y_true_flat) > 2000:
            indices = np.random.choice(len(y_true_flat), size=2000, replace=False)
            plt.scatter(y_true_flat[indices], y_pred_flat[indices], alpha=0.3, s=10)
        else:
            plt.scatter(y_true_flat, y_pred_flat, alpha=0.3, s=10)
        
        min_val = min(y_true_flat.min(), y_pred_flat.min())
        max_val = max(y_true_flat.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.title(f'{dataset_name}: Global Parity Plot')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')

        # 2. Per-Target R2
        plt.subplot(1, 2, 2)
        
        # Calculate R2 per target
        if y_true.ndim == 1 or y_true.shape[1] == 1:
            r2_scores = np.array([r2_score(y_true, y_pred)])
        else:
            r2_scores = r2_score(y_true, y_pred, multioutput='raw_values')
            
        n_targets = len(r2_scores)
        if target_names is None:
            target_names = [f"Target {i}" for i in range(n_targets)]
        
        # Colors based on score
        colors = ['green' if x > 0 else 'red' for x in r2_scores]
        
        x_pos = np.arange(n_targets)
        bars = plt.bar(x_pos, r2_scores, color=colors, alpha=0.7, edgecolor='black')
        plt.axhline(0, color='black', linewidth=0.8)
        
        plt.title(f'{dataset_name}: Performance per Target')
        plt.xlabel('Target')
        plt.ylabel('R2 Score')
        plt.xticks(x_pos, target_names, rotation=45, ha='right')
        
        # Limit y-axis if R2 is terribly negative to keep plot readable
        bottom_lim = max(-1.0, min(r2_scores)) - 0.1
        plt.ylim(bottom=bottom_lim, top=1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.3)

        # Add value labels on bars
        for bar, v in zip(bars, r2_scores):
            height = bar.get_height()
            # Position text above bar for positive, below for negative
            y_pos = height + 0.02 if height > 0 else height - 0.05
            # Ensure text doesn't go below bottom limit visually
            if y_pos < bottom_lim: y_pos = bottom_lim + 0.05
            
            plt.text(bar.get_x() + bar.get_width()/2, y_pos, f"{v:.2f}", ha='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"evaluation_metrics_{dataset_name.lower()}.png")
        print(f"Saved evaluation_metrics_{dataset_name.lower()}.png")

# --- Preprocessing Module ---
class DataProcessor:
    """Encapsulates the fitting and transforming logic to prevent data leaks."""
    def __init__(self, pca_variance: float = 0.95):
        self.imputer = SimpleImputer(strategy='mean')
        self.x_scaler = StandardScaler()
        self.pca = PCA(n_components=pca_variance)
        self.y_scaler = StandardScaler() # Crucial for multi-output regression
        
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fits scalers/PCA on X and y, then transforms them."""
        # X processing
        X = self.imputer.fit_transform(X)
        X = self.x_scaler.fit_transform(X)
        X = self.pca.fit_transform(X)
        
        # y processing (reshape needed for scaler if 1D)
        if len(y.shape) == 1: 
            y = y.reshape(-1, 1)
        y = self.y_scaler.fit_transform(y)
        return X, y

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Applies existing transformations to X (and y if provided)."""
        # X processing
        X = self.imputer.transform(X)
        X = self.x_scaler.transform(X)
        X = self.pca.transform(X)
        
        # y processing
        if y is not None:
            if len(y.shape) == 1: 
                y = y.reshape(-1, 1)
            y = self.y_scaler.transform(y)
        return X, y
    
    def inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """Reverts scaling on predictions."""
        return self.y_scaler.inverse_transform(y_scaled)

# --- Model Definition ---
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int], output_dim: int = 1, dropout_rate: float = 0.0):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, output_dim)) # Regression output
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# --- Training Logic ---
def run_training(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_val: Optional[np.ndarray], 
    y_val: Optional[np.ndarray],
    params: Dict[str, Any],
    device: torch.device,
) -> Tuple[nn.Module, float, int, Dict[str, List[float]]]:
    """
    Executes the training loop for a single model configuration.
    Handles early stopping.
    """
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

    # Validation Data
    X_val_t, y_val_t = None, None
    if X_val is not None and y_val is not None:
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val).to(device)

    # Initialize Model
    model = MLP(
        input_dim=X_train.shape[1], 
        hidden_layers=params['hidden_layers'], 
        output_dim=y_train.shape[1],
        dropout_rate=params['dropout']
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    use_early_stopping = (X_val_t is not None)
    
    for epoch in range(params['epochs']):
        model.train()
        epoch_train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # CHECK FOR NAN/INF
            if torch.isnan(loss) or torch.isinf(loss):
                return model, float('inf'), epoch, history # Return inf so Optuna handles it

            loss.backward()
            
            # FIX: Gradient Clipping prevents exploding gradients
            utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        if use_early_stopping:
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
            
            history['val_loss'].append(val_loss)

            # Removed per-epoch pruning to avoid "Fold Reset" trap

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= params['patience']:
                break
        else:
            # No validation, just keep the last state
            best_model_state = model.state_dict()
            best_val_loss = 0.0 
            best_epoch = epoch
            
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    return model, best_val_loss, best_epoch, history

# --- Optimization Objective ---
def objective(trial, X, y, groups, device):
    """Optuna objective function for Bayesian Optimization."""
    
    # Define Hyperparameter Search Space
    hidden_layer_options = [[64, 32], [128, 64], [128, 64, 32], [256, 128]]
    hidden_layer_idx = trial.suggest_categorical('hidden_layer_idx', range(len(hidden_layer_options)))
    
    params = {
        'hidden_layers': hidden_layer_options[hidden_layer_idx],
        'dropout': trial.suggest_float('dropout', 0.1, 0.5), # Increased min dropout
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]), # Removed 16
        'epochs': 50, 
        'patience': 10
    }
    
    # Use GroupKFold for speed (5 splits instead of 28) inside optimization
    folds = LeaveOneGroupOut()
    # folds = GroupKFold(n_splits=5)
    val_losses = []
    best_epochs = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds.split(X, y, groups)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Use DataProcessor for clean pipeline
        processor = DataProcessor()
        X_train_fold, y_train_fold = processor.fit_transform(X_train_fold, y_train_fold)
        X_val_fold, y_val_fold = processor.transform(X_val_fold, y_val_fold)
        
        # Train
        _, val_loss, best_epoch, _ = run_training(
            X_train_fold, y_train_fold, 
            X_val_fold, y_val_fold, 
            params, device
        )
        
        # Handle Inf return from training
        if val_loss == float('inf'):
             return float('inf')

        val_losses.append(val_loss)
        best_epochs.append(best_epoch)
        
        # PRUNING STRATEGY:
        # Prune after the first fold is done if it's terrible.
        intermediate_value = np.mean(val_losses)
        trial.report(intermediate_value, step=fold_idx)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    # Store average best epoch for final training
    trial.set_user_attr("avg_best_epoch", int(np.mean(best_epochs)))
            
    return np.mean(val_losses)

if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- 1. Initialize Dataset ---
    dataset = Dataset(dataset_type="1s", recreate=False)
    target_names = dataset.TARGET_COLS
    print(f"Dataset aggregated per 1s")

    # --- 2. Prepare Data ---
    # This splits users into Train/Val set and a completely held-out Test set
    X_train_val_df, X_test_df, y_train_val, y_test, users_train_val, users_test = dataset.get_train_test_split()
    
    # Convert DataFrames to numpy for PyTorch
    X_train_val = X_train_val_df.values
    X_test = X_test_df.values
    
    print(f"Train/Val Samples: {len(X_train_val)} (Users: {len(np.unique(users_train_val))})")
    print(f"Test Samples: {len(X_test)} (Users: {len(np.unique(users_test))})")
    print(f"Number of Features: {X_train_val.shape[1]}")
    
    # --- 3. Bayesian Optimization (Optuna) ---
    print("\n--- Starting Bayesian Optimization ---")
    # FIX: Use a Patient Pruner
    study = optuna.create_study(
        direction="minimize",
        pruner=PatientPruner(MedianPruner(), patience=1, min_delta=0.0)
    )
    study.optimize(
        lambda trial: objective(trial, X_train_val, y_train_val, users_train_val, device), 
        n_trials=50  # Adjust number of trials as needed
    )
    
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    Visualizer.plot_optimization_history(study)

    # Reconstruct best params
    hidden_layer_options = [[64, 32], [128, 64], [128, 64, 32], [256, 128]]
    best_params = trial.params.copy()
    best_params['hidden_layers'] = hidden_layer_options[best_params.pop('hidden_layer_idx')]
    
    # Use the average optimal epochs found during CV + buffer
    avg_best_epoch = int(trial.user_attrs["avg_best_epoch"])
    print(f"  Average Best Epoch from CV: {avg_best_epoch}")
    best_params['epochs'] = avg_best_epoch + 5
    best_params['patience'] = 100 # Disable early stopping for final run

    # --- 4. Final Training & Evaluation ---
    print("\n--- Training Final Model with Validation Split ---")
    
    # Create a validation split from the Train/Val set to report validation metrics
    # We use GroupKFold to ensure we validate on unseen users, similar to the CV process
    # folds = GroupKFold(n_splits=5)
    folds = LeaveOneGroupOut()
    train_idx, val_idx = next(folds.split(X_train_val, y_train_val, users_train_val))
    
    X_train_final = X_train_val[train_idx]
    y_train_final = y_train_val[train_idx]
    X_val_final = X_train_val[val_idx]
    y_val_final = y_train_val[val_idx]

    # Final Processing (Fit on training split only)
    final_processor = DataProcessor()
    X_train_final_proc, y_train_final_proc = final_processor.fit_transform(X_train_final, y_train_final)
    
    # Transform Validation and Test sets
    X_val_final_proc, y_val_final_proc = final_processor.transform(X_val_final, y_val_final)
    X_test_proc, y_test_proc = final_processor.transform(X_test, y_test)

    # Train
    final_model, _, _, history = run_training(
        X_train_final_proc, y_train_final_proc, 
        X_val_final_proc, y_val_final_proc, 
        best_params, device
    )
    
    Visualizer.plot_learning_curves(history, title="Final Model Training Loss")
    
    # --- 5. Evaluation Helper ---
    def evaluate_dataset(name: str, X_proc: np.ndarray, y_real: np.ndarray):
        final_model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_proc).to(device)
            y_pred_scaled = final_model(X_t).cpu().numpy()
        
        # Inverse transform predictions to original scale
        y_pred_real = final_processor.inverse_transform_y(y_pred_scaled)
            
        # Calculate Metrics - Real Scale
        mse_real = mean_squared_error(y_real, y_pred_real)
        rmse_real = np.sqrt(mse_real)
        r2_real = r2_score(y_real, y_pred_real)

        print(f"\n{name} Set Results (Real Scale):")
        print(f"MSE: {mse_real:.4f}")
        print(f"RMSE: {rmse_real:.4f}")
        print(f"R2: {r2_real:.4f}")
        
        Visualizer.plot_evaluation(y_real, y_pred_real, target_names=target_names, dataset_name=name)

    # Evaluate Validation Set
    evaluate_dataset("Validation", X_val_final_proc, y_val_final)

    # Evaluate Test Set
    # evaluate_dataset("Test", X_test_proc, y_test)