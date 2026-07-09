"""
HybridCNNGRUFusion — combination model: merges the empirically best
pieces of the existing architectures (grouped-representation sweep §3b in
open_improvements.md) and fixes their known weaknesses.

What is taken from where, and what was changed:

1. From CNN_BaselineNOEMBED (best, val RMSE 0.852): the per-group Conv1d
   encoder (k5 + k3) with masked pooling and NO bottleneck on the
   mean-aggregation route — the tight path bottleneck was the documented
   failure of CNN_Baseline (O1.2).

2. From LSTM_Baseline (2nd, 0.858): sequence order. A small GRU runs on
   top of the conv features — but on a masked-downsampled grid (stride 8
   for the 50 Hz mechanical group), so it trains in a fraction of the
   LSTM's 14 min while the CNN front-end preserves the fast dynamics.
   Packed with mask-derived lengths (padding is skipped, as before).
   Switchable off via temporal_mode='cnn' (then: pure conv encoder).

3. From BASE_BaselineFLAT (3rd, 0.863): flatten aggregation preserves
   path identity — but it only worked there because per-path vectors
   were compressed FIRST (path_dim=4 -> 80 flat dims, vs. NOEMBEDFLAT's
   2240). Adopted as a compressed-flatten TRACK: per-path fused vectors
   are compressed to flat_dim before flattening.

4. DUAL aggregation instead of either/or: the regressor sees the
   compressed flatten track (path identity) AND the uncompressed masked
   mean over paths (the 0.852 winner's route) side by side — the
   bottleneck can no longer strangle the signal because the wide mean
   route bypasses it.

5. Upgrades that cost (almost) nothing: masked mean+STD statistics
   pooling over time (variability information the plain mean discards)
   and per-group coverage fractions as explicit missingness features (O2a).

Revision v2 (2026-07-08, after the first benchmark run + ablations):
v1 scored 0.893 val RMSE (BASE_FLAT 0.863) with best epochs collapsing to
1-4 — it overfitted almost immediately, bleeding on the weak-signal
targets while already BEATING BASE_FLAT on the disturbance targets
(Robotrainer Left/Right). No-early-stopping ablations located the causes:

a. The LayerNorm on the fused per-path vector was REMOVED — it whitened
   the representation per sample, accelerating optimization straight into
   overfitting (ablation: best mean-val 0.466@ep2 with norm vs
   0.449@ep4 without). The GRU was exonerated (cnn-only was worse).
b. LayerScale on the ts embeddings (gamma init 0.3): the model starts
   close to the proven path-features signal; the extra ts capacity —
   the overfitting fuel for the weak targets — fades in gradually.
c. Feature-level dropout on the fused per-path vector BEFORE both
   aggregation tracks (v1 had dropout only inside the flatten track, so
   the regressor saw the mean track noise-free).
d. Tighter regressor defaults (regressor_dim 384, dropout_reg 0.5);
   lr stays 1e-3 — on the 9-fold probe it beat 5e-4 under simulated
   patience-2 stopping (0.411 vs 0.424; best mean epoch moved 2 -> 5).
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List

from ..base import BaseModel
from .masking import (
    timestep_mask,
    mask_lengths,
    masked_mean_std_over_time,
    masked_avg_pool1d,
)
from ...data.schema import split_inputs, group_shapes


class HybridCNNGRUFusion(BaseModel):
    model_name = "Hybrid_CNN_GRU_Fusion"

    # Groups longer than this get downsampled before the GRU
    # (in practice: only the 50 Hz mechanical group).
    DOWNSAMPLE_MIN_LEN = 500

    def __init__(self, input_dims: list, output_dim: int, hyperparams: Dict[str, Any]) -> None:
        super().__init__(input_dims, output_dim, hyperparams)

        path_shape, user_shape, groups = group_shapes(input_dims)
        self.n_paths = path_shape[1]
        self.f_path = path_shape[2]
        self.f_user = user_shape[1]
        self.n_groups = len(groups)

        hp = hyperparams
        channels = hp.get("cnn_channels", 12)
        self.temporal_mode = hp.get("temporal_mode", "cnn_gru")
        gru_hidden = hp.get("gru_hidden", 12)
        gru_stride = hp.get("gru_stride", 8)
        flat_dim = hp.get("flat_dim", 8)

        # --- Per-group conv encoder (identical shape to the 0.852 winner) ---
        self.cnns = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(g_x_shape[2], channels, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            for (g_x_shape, _) in groups
        ])

        # --- Optional recurrent stage on the downsampled conv features ---
        if self.temporal_mode == "cnn_gru":
            self.grus = nn.ModuleList([
                nn.GRU(input_size=channels, hidden_size=gru_hidden, batch_first=True)
                for _ in groups
            ])
            self.gru_strides = [
                gru_stride if g_x_shape[3] > self.DOWNSAMPLE_MIN_LEN else 1
                for (g_x_shape, _) in groups
            ]
            emb_dim = 2 * gru_hidden      # mean+std pooling of GRU outputs
        elif self.temporal_mode == "cnn":
            emb_dim = 2 * channels        # mean+std pooling of conv features
        else:
            raise ValueError(f"Invalid temporal_mode: '{self.temporal_mode}'. Choose 'cnn' or 'cnn_gru'.")

        # LayerScale (v2b): ts embeddings + coverages fade in from a small
        # factor, so early training is dominated by the proven path features.
        ts_feat_dim = emb_dim * self.n_groups + self.n_groups
        self.ts_gamma = nn.Parameter(torch.full((ts_feat_dim,), 0.3))

        # Fused per-path vector: [scaled group embeddings | coverages | path features]
        # (v2a: no per-sample normalization — the LayerNorm here accelerated
        #  optimization straight into epoch-2 overfitting.)
        self.path_combined_dim = ts_feat_dim + self.f_path

        # Feature dropout before BOTH aggregation tracks (v2c)
        self.path_drop = nn.Dropout(hp.get("dropout_path", 0.15))

        # Compressed flatten track (BASE_FLAT's trick): compress first, then flatten
        self.flat_compress = nn.Sequential(
            nn.Linear(self.path_combined_dim, flat_dim),
            nn.ReLU(),
        )

        # Regressor sees BOTH tracks: flatten (path identity) + mean (winner route)
        fused_dim = flat_dim * self.n_paths + self.path_combined_dim + self.f_user
        regressor_dim = hp.get("regressor_dim", 384)
        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, regressor_dim),
            nn.ReLU(),
            nn.Dropout(hp.get("dropout_reg", 0.5)),
            nn.Linear(regressor_dim, output_dim),
        )

    # -----------------------------------------------------

    def encode_timeseries(self, groups) -> torch.Tensor:
        """
        Per group: conv encoder -> (optional masked downsample + GRU) ->
        masked mean+std pooling; plus the coverage fraction per path.

        Args:
            groups: list of (x_g, mask_g) with x_g: (B, P, C_g, T_g)

        Returns:
            (B, P, emb_dim*n_groups + n_groups)
        """
        embs, covs = [], []
        for i, (cnn, (x_g, mask_g)) in enumerate(zip(self.cnns, groups)):
            B, P, C, T = x_g.shape

            # per-timestep validity from the explicit mask (never from values!)
            t_mask = timestep_mask(mask_g).view(B * P, T)
            covs.append((t_mask.sum(dim=1) / T).view(B, P, 1))

            feat = cnn(x_g.reshape(B * P, C, T))             # (B*P, ch, T)

            if self.temporal_mode == "cnn_gru":
                feat, t_mask = masked_avg_pool1d(feat, t_mask, self.gru_strides[i])
                seq = feat.permute(0, 2, 1)                  # (B*P, T', ch)
                packed = nn.utils.rnn.pack_padded_sequence(
                    seq, mask_lengths(t_mask), batch_first=True, enforce_sorted=False,
                )
                out, _ = self.grus[i](packed)
                feat, _ = nn.utils.rnn.pad_packed_sequence(
                    out, batch_first=True, total_length=seq.size(1),
                )                                            # (B*P, T', hidden)
            else:
                feat = feat.permute(0, 2, 1)                 # (B*P, T, ch)

            pooled = masked_mean_std_over_time(feat, t_mask) # (B*P, 2*dim)
            embs.append(pooled.view(B, P, -1))

        return torch.cat(embs + covs, dim=-1)

    # -----------------------------------------------------

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: Input list (x_path, x_user, g0_x, g0_mask, ...)

        Returns:
            prediction: (B, output_dim)
        """
        x_path, x_user, groups = split_inputs(x)
        B = x_path.size(0)

        ts_paths = self.encode_timeseries(groups) * self.ts_gamma    # (B, P, emb*G + G)

        path_combined = self.path_drop(
            torch.cat([ts_paths, x_path], dim=-1))                   # (B, P, D_path)

        flat_track = self.flat_compress(path_combined).view(B, -1)   # (B, P*flat_dim)
        mean_track = path_combined.mean(dim=1)                       # (B, D_path)

        fused = torch.cat([flat_track, mean_track, x_user], dim=1)
        return self.regressor(fused)

    # =====================================================
    # OPTUNA
    # =====================================================

    @staticmethod
    def get_hyperparameter_space(trial: Any) -> Dict[str, Any]:
        params = {
            "cnn_channels": trial.suggest_categorical("cnn_channels", [8, 12, 16]),
            "temporal_mode": trial.suggest_categorical("temporal_mode", ["cnn", "cnn_gru"]),
            "flat_dim": trial.suggest_categorical("flat_dim", [4, 6, 8, 12]),
            "dropout_path": trial.suggest_float("dropout_path", 0.05, 0.4),
            "regressor_dim": trial.suggest_categorical("regressor_dim", [256, 384, 512, 768]),
            "dropout_reg": trial.suggest_float("dropout_reg", 0.2, 0.6),
            "lr": trial.suggest_float("lr", 2e-4, 2e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 5e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [6]),
            "n_path_features": trial.suggest_int("n_path_features", 20, 70, step=5),
        }
        if params["temporal_mode"] == "cnn_gru":
            params["gru_hidden"] = trial.suggest_categorical("gru_hidden", [8, 12, 16])
            params["gru_stride"] = trial.suggest_categorical("gru_stride", [4, 8, 16])
        return params

    @staticmethod
    def get_default_parameters() -> Dict[str, Any]:
        return {
            "cnn_channels": 12,
            "temporal_mode": "cnn_gru",
            "gru_hidden": 12,
            "gru_stride": 8,
            "flat_dim": 8,
            "dropout_path": 0.15,
            "regressor_dim": 384,
            "dropout_reg": 0.5,
            "lr": 1e-3,
            "weight_decay": 3e-3,
            "batch_size": 6,
            "n_path_features": 35,
        }
