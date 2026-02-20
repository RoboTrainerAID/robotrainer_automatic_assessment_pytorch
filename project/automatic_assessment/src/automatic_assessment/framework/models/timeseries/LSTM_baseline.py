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


def compute_lengths_flat(x_flat: torch.Tensor):
    # x_flat: (N,T)
    valid = (x_flat.abs() > 1e-8)

    # last valid index from end
    last_valid = valid.flip(1).float().argmax(dim=1)
    lengths = x_flat.size(1) - last_valid
    return lengths.clamp(min=1)


# =========================================================
# MODEL
# =========================================================

class LSTMBaseline(BaseModel):
    model_name = "LSTM_Baseline"

    def __init__(self, input_dims, output_dim, hyperparams):
        super().__init__(input_dims, output_dim, hyperparams)

        ts_shape = input_dims[0]
        self.n_paths = ts_shape[1]
        self.n_ts = ts_shape[2]
        self.max_timesteps = ts_shape[3]

        path_shape = input_dims[1]
        self.f_path = path_shape[2]

        user_shape = input_dims[2]
        self.f_user = user_shape[1]

        hp = hyperparams

        self.lstm_hidden = hp["lstm_hidden"]

        self.lstm = nn.LSTM(
            input_size=self.n_ts,
            hidden_size=self.lstm_hidden,
            num_layers=hp["lstm_layers"],
            dropout=hp["dropout_lstm"] if hp["lstm_layers"] > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )

        manual_dim = self.n_paths * self.f_path + self.f_user
        fused_dim = self.n_paths * self.lstm_hidden + manual_dim

        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, hp["regressor_dim"]),
            nn.ReLU(),
            nn.Dropout(hp["dropout_reg"]),
            nn.Linear(hp["regressor_dim"], output_dim),
        )

    # -----------------------------------------------------

    def encode_timeseries(self, x_ts):
        # (B,P,TS,T)
        B, P, TS, T = x_ts.shape

        # compute path lengths
        x_path = x_ts.abs().sum(dim=2)  # collapse TS
        x_path = x_path.view(B * P, T)
        l = compute_lengths_flat(x_path).cpu()

        # reorder → (B,P,T,TS)
        x = x_ts.permute(0, 1, 3, 2)

        # flatten → (B*P,T,TS)
        x = x.reshape(B * P, T, TS)

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            l,
            batch_first=True,
            enforce_sorted=False,
        )

        out, _ = self.lstm(packed)

        padded, _ = nn.utils.rnn.pad_packed_sequence(
            out,
            batch_first=True,
            total_length=T,
        )

        pooled = masked_mean(padded, l.to(padded.device))

        pooled = pooled.view(B, P * pooled.shape[-1])
        return pooled

    # -----------------------------------------------------

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x_ts, x_path, x_user = x

        ts_feat = self.encode_timeseries(x_ts)

        manual = torch.cat([
            x_path.reshape(x_path.size(0), -1),
            x_user
        ], dim=1)

        fused = torch.cat([ts_feat, manual], dim=1)
        return self.regressor(fused)

    # OPTUNA unchanged

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
            "lstm_hidden": 8,
            "lstm_layers": 1,
            "dropout_lstm": 0.1,
            "regressor_dim": 128,
            "dropout_reg": 0.2,
            "lr": 2e-4,
            "weight_decay": 1e-3,
            "batch_size": 6,
            "correlation_threshold": 0.3
        }
