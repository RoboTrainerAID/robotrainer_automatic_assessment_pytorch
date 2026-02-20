import torch
import torch.nn as nn
from typing import Dict, Any, List

from ..base import BaseModel


# =========================================================
# helpers
# =========================================================

def masked_mean(x: torch.Tensor, lengths: torch.Tensor):
    device = x.device
    mask = torch.arange(x.size(1), device=device)[None, :] < lengths[:, None]
    mask = mask.float().unsqueeze(-1)
    x = x * mask
    return x.sum(1) / lengths.clamp(min=1).unsqueeze(-1)

def compute_lengths(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the effective length of time series by finding the last non-zero element.
    Args:
        x: (N, T) tensor of time series data.
    Returns:
        lengths: (N,) tensor of lengths (clamped to min 1).
    """
    # valid: (N, T) boolean mask
    valid = (x.abs() > 1e-8)

    flipped_valid = valid.flip(1)
    
    last_valid_idx_from_end = flipped_valid.float().argmax(dim=1)
    
    lengths = x.size(1) - last_valid_idx_from_end
    
    return lengths.clamp(min=1).cpu()


# =========================================================
# MODEL
# =========================================================

class LSTMBaseline(BaseModel):
    model_name = "LSTM_Baseline"

    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)

        ts_shape = input_dims[0]
        # ts_shape: (Batch, N_Paths, N_TS_Features, Time)
        self.n_paths = ts_shape[1]
        self.n_ts = ts_shape[2]
        self.max_timesteps = ts_shape[3]

        path_shape = input_dims[1]
        self.f_path = path_shape[2]

        user_shape = input_dims[2]
        self.f_user = user_shape[1]

        hp = hyperparams

        self.lstm_hidden = hp["lstm_hidden"]

        # LSTM input_size is 1 because we process each time-series feature independently 
        # (flattened B * P * TS) in the current logic.
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.lstm_hidden,
            num_layers=hp["lstm_layers"],
            dropout=hp["dropout_lstm"] if hp["lstm_layers"] > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )

        manual_dim = self.n_paths * self.f_path + self.f_user
        fused_dim = self.n_paths * self.n_ts * self.lstm_hidden + manual_dim

        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, hp["regressor_dim"]),
            nn.ReLU(),
            nn.Dropout(hp["dropout_reg"]),
            nn.Linear(hp["regressor_dim"], output_dim),
        )

    # -----------------------------------------------------

    def encode_timeseries(self, x_ts: torch.Tensor) -> torch.Tensor:
        # x_ts: (B, P, TS, T)
        B, P, TS, T = x_ts.shape

        # Flatten hierarchy to (N, T) where N = B*P*TS
        # We process every single time series independently
        flat_ts = x_ts.view(B * P * TS, T)
        
        # Compute lengths on the flattened data
        lengths = compute_lengths(flat_ts)

        # Prepare for LSTM
        # input: (N, T, 1)
        x = flat_ts.unsqueeze(-1)

        # Pack sequence to ignore padding
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        # Run LSTM
        # out: PackedSequence
        # (h_n, c_n): (num_layers, N, hidden)
        _, (h_n, _) = self.lstm(packed)

        # Take last hidden state
        # h_n[-1] is (N, hidden)
        pooled = h_n[-1]

        # Reshape back to (B, P * TS * hidden)
        pooled = pooled.view(B, P * TS * self.lstm_hidden)
        return pooled

    # -----------------------------------------------------

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x_ts, x_path, x_user = x

        ts_feat = self.encode_timeseries(x_ts)

        # Flatten path features: (B, P, F_p) -> (B, P*F_p)
        x_path_flat = x_path.reshape(x_path.size(0), -1)

        manual = torch.cat([
            x_path_flat,
            x_user
        ], dim=1)

        fused = torch.cat([ts_feat, manual], dim=1)
        return self.regressor(fused)

    # =====================================================
    # OPTUNA
    # =====================================================

    @staticmethod
    def get_hyperparameter_space(trial) -> Dict[str, Any]:
        return {
            "lstm_hidden": trial.suggest_categorical("lstm_hidden", [16, 32, 48, 64]),
            "lstm_layers": trial.suggest_int("lstm_layers", 1, 2),
            "dropout_lstm": trial.suggest_float("dropout_lstm", 0.0, 0.3),
            "regressor_dim": trial.suggest_categorical("regressor_dim", [64, 128, 256]),
            "dropout_reg": trial.suggest_float("dropout_reg", 0.0, 0.4),
            "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [6]),
            "correlation_threshold": trial.suggest_float("correlation_threshold", 0.4, 0.8, step=0.01)
        }

    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        return {
            "lstm_hidden": 64,
            "lstm_layers": 1,
            "dropout_lstm": 0.1,
            "regressor_dim": 256,
            "dropout_reg": 0.2,
            "lr": 2e-4,
            "weight_decay": 1e-3,
            "batch_size": 6,
            "correlation_threshold": 0.5
        }
