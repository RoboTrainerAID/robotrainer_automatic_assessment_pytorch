"""
Single source of truth for the model input layout ("X schema").

Every component that touches the input tuple — dataset loading, fold
scaling, feature selection, models, pipeline bookkeeping — imports its
layout knowledge from here. Do not hard-code input indices anywhere else.

Layout
------
    X = (x_path, x_user, g0_x, g0_mask, g1_x, g1_mask, ...)

- x_path: (N, P, F_path) float32 — path-level features
- x_user: (N, F_user)    float32 — user-level features (demographics)
- per timeseries channel group (config.TS_MODEL_GROUPS order,
  e.g. mechanical @ 50 Hz, physiological @ 2 Hz, gait @ 2 Hz):
    g_x:    (N, P, C_g, T_g) float32 — 0.0 where no data
    g_mask: (N, P, C_g, T_g) bool    — True where the bin holds data

Groups have their own sampling rates and lengths and are NOT aligned to
each other. Validity must always be derived from the masks, never from
the values (padding is re-zeroed after standardization).
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Tuple

import torch

import automatic_assessment.dataset.config as ds_config

# Index layout of the X tuple
X_PATH = 0
X_USER = 1
GROUPS_START = 2


@dataclass
class TSGroup:
    """One timeseries channel group (values + validity mask)."""
    name: str
    rate_hz: float
    channels: List[str]
    x: torch.Tensor      # (N, P, C, T) float32, 0.0 at invalid positions
    mask: torch.Tensor   # (N, P, C, T) bool

    @property
    def spec(self) -> Dict[str, Any]:
        return {"name": self.name, "rate_hz": self.rate_hz, "channels": self.channels}


def group_specs() -> List[Dict[str, Any]]:
    """Channel-group definitions in deterministic config order (validated)."""
    ds_config.validate_ts_groups()
    return [
        {"name": name, "rate_hz": float(g["rate_hz"]), "channels": list(g["channels"])}
        for name, g in ds_config.TS_MODEL_GROUPS.items()
    ]


def build_X(x_path: torch.Tensor, x_user: torch.Tensor, groups: List[TSGroup], idx=None) -> tuple:
    """Assembles the canonical X tuple (optionally sliced by idx)."""
    if idx is None:
        parts = [x_path, x_user]
        for g in groups:
            parts.extend([g.x, g.mask])
    else:
        parts = [x_path[idx], x_user[idx]]
        for g in groups:
            parts.extend([g.x[idx], g.mask[idx]])
    return tuple(parts)


def split_inputs(x) -> Tuple[Any, Any, List[Tuple[Any, Any]]]:
    """
    Unpacks a model input list/tuple.

    Returns:
        (x_path, x_user, groups) with groups = [(g_x, g_mask), ...]
    """
    x_path, x_user = x[X_PATH], x[X_USER]
    groups = [(x[i], x[i + 1]) for i in range(GROUPS_START, len(x), 2)]
    return x_path, x_user, groups


def group_shapes(input_dims) -> Tuple[Any, Any, List[Tuple[Any, Any]]]:
    """
    Same split for the shape list produced by BaseModel.get_input_dims.

    Returns:
        (path_shape, user_shape, [(g_x_shape, g_mask_shape), ...])
    """
    path_shape, user_shape = input_dims[X_PATH], input_dims[X_USER]
    groups = [(input_dims[i], input_dims[i + 1]) for i in range(GROUPS_START, len(input_dims), 2)]
    return path_shape, user_shape, groups


def iter_group_tensors(X) -> Iterator[Tuple[int, Any, Any]]:
    """Yields (tuple_index_of_values, g_x, g_mask) for every group in X."""
    for i in range(GROUPS_START, len(X), 2):
        yield i, X[i], X[i + 1]


def input_shapes_info(X, y) -> Dict[str, Any]:
    """Shape summary for experiment bookkeeping (config.yaml)."""
    return {
        "shape_X_path": list(X[X_PATH].shape),
        "shape_X_user": list(X[X_USER].shape),
        "shape_ts_groups": [list(x.shape) for _, x, _ in iter_group_tensors(X)],
        "shape_y": list(y.shape),
    }
