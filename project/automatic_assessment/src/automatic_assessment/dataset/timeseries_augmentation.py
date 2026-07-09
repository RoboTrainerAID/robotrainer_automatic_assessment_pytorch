"""
Data augmentation: synthetic training users via per-channel jitter.

A clone is a noisy RE-MEASUREMENT of the same person:
- ONLY the raw sensor channels (config.TIMESERIES_TO_LOAD) receive noise,
  independently per (path, channel), with sigma = noise_factor * robust
  std of that very trace (IQR-based — outlier spikes do not inflate the
  noise). Constant traces (e.g. the imputed zero-disturbance series) have
  robust std 0 and therefore stay exactly constant.
- Strictly positive channels (physiological + gait events) get
  MULTIPLICATIVE noise and are truncated away from zero — additive
  Gaussian could produce impossible values (negative heart rate,
  negative stride time).
- Derived series (magnitudes, power, work) are REMOVED from the clone
  and re-computed from the noisy raw components by the caller
  (TimeseriesDerivedTS), keeping the clone physically consistent.
- Timestamps, scalar features, meta, targets, demographics, and task
  difficulty stay identical (they belong to the person/protocol, not to
  the sensor noise).

Clone user id = original_id * 100 + clone_index (1..max_ratio); every
clone gets its own deterministic RNG stream derived from the configured
base seed, so regeneration is reproducible regardless of iteration order.

Extensibility: `method` in config.AUGMENTATION selects the noise
generator; add new methods (e.g. time_warp, magnitude_warp) to
_AUGMENTATION_METHODS below with the same signature.
"""

import copy
from typing import Any, Dict

import numpy as np

import automatic_assessment.dataset.config as config
from automatic_assessment.dataset.timeseries_loader import PathData
from automatic_assessment.dataset.timeseries_derived_ts import TimeseriesDerivedTS


def robust_std(values: np.ndarray) -> float:
    """IQR-based std estimate (normal-consistent); 0 for constant traces."""
    if values.size < 4:
        return float(np.std(values))
    q75, q25 = np.percentile(values, [75, 25])
    return float((q75 - q25) / 1.349)


def jitter_series(values: np.ndarray, noise_factor: float, positive: bool,
                  rng: np.random.Generator) -> np.ndarray:
    """
    Bounded, scale-aware jitter for one trace.

    Signed channels:   x + N(0, noise_factor * robust_std(x))
    Positive channels: x * (1 + N(0, noise_factor)), truncated > 0
    """
    if values.size == 0:
        return values.copy()

    if positive:
        noise = rng.normal(0.0, noise_factor, size=values.shape)
        out = values * (1.0 + noise)
        # Truncate away from zero: never smaller than 10% of the original
        # sample (a 5%-sigma jitter virtually never reaches this bound).
        return np.maximum(out, 0.1 * values)

    sigma = noise_factor * robust_std(values)
    if sigma <= 0.0:
        return values.copy()  # constant trace stays constant
    return values + rng.normal(0.0, sigma, size=values.shape)


_AUGMENTATION_METHODS = {
    "jitter": jitter_series,
}


class TimeseriesAugmenter:
    """Creates synthetic clone users from processed PathData structures."""

    def __init__(self, cfg: Any = config):
        self.cfg = cfg
        aug = cfg.AUGMENTATION
        self.max_ratio = int(aug["max_ratio"])
        self.noise_factor = float(aug["noise_factor"])
        self.base_seed = int(aug["seed"])
        method = aug["method"]
        if method not in _AUGMENTATION_METHODS:
            raise ValueError(
                f"Unknown augmentation method '{method}'. "
                f"Available: {sorted(_AUGMENTATION_METHODS)}"
            )
        self.method_name = method
        self._noise_fn = _AUGMENTATION_METHODS[method]
        self._raw_channels = set(cfg.TIMESERIES_TO_LOAD)
        self._positive_channels = cfg.augmentation_positive_channels()

    # ------------------------------------------------------------------

    def clone_user(self, user_paths: Dict[int, PathData], user_id: int,
                   clone_idx: int) -> Dict[int, PathData]:
        """
        One synthetic user: deep copy with jittered raw channels.
        Derived channels are dropped (recomputed later); everything else
        (timestamps, scalars, meta) stays identical.
        """
        new_id = int(user_id) * 100 + clone_idx
        rng = np.random.default_rng(self.base_seed + new_id)

        clone_paths: Dict[int, PathData] = {}
        for path_id, pd_obj in user_paths.items():
            clone = PathData(
                user_id=new_id,
                path_id=path_id,
                meta=copy.deepcopy(pd_obj.meta),
                scalar_features=dict(pd_obj.scalar_features),
                motion_start_timestamp=pd_obj.motion_start_timestamp,
            )
            for ts_name, arr in pd_obj.timeseries.items():
                if ts_name not in self._raw_channels:
                    continue  # derived series are re-computed from noisy raws
                new_arr = arr.copy()
                if new_arr.ndim == 2 and new_arr.shape[0] > 0 and new_arr.shape[1] > self.cfg.COL_VALUE:
                    new_arr[:, self.cfg.COL_VALUE] = self._noise_fn(
                        new_arr[:, self.cfg.COL_VALUE],
                        self.noise_factor,
                        ts_name in self._positive_channels,
                        rng,
                    )
                clone.timeseries[ts_name] = new_arr
            clone_paths[path_id] = clone
        return clone_paths

    def augment_users(self, ts_map: Dict[int, Dict[int, PathData]], max_ratio: int = None
                      ) -> Dict[int, Dict[int, PathData]]:
        """
        Generates clones for every user in ts_map (originals stay untouched)
        and re-derives magnitudes/power/work on the clones.

        Returns a dict {clone_user_id: {path_id: PathData}} containing
        ONLY the clones.
        """
        n_clones = self.max_ratio if max_ratio is None else int(max_ratio)
        if n_clones <= 0:
            return {}

        clones: Dict[int, Dict[int, PathData]] = {}
        for user_id, user_paths in ts_map.items():
            if int(user_id) >= 100:
                raise ValueError(
                    f"Refusing to augment user {user_id}: id >= 100 collides "
                    "with the clone id scheme (original_id * 100 + i)."
                )
            for i in range(1, n_clones + 1):
                clone_paths = self.clone_user(user_paths, user_id, i)
                clones[int(user_id) * 100 + i] = clone_paths

        # Re-derive magnitudes / power / work from the noisy raw channels
        TimeseriesDerivedTS(clones).process()

        print(f"Augmentation: generated {len(clones)} synthetic users "
              f"({n_clones} clones/user, method='{self.method_name}', "
              f"noise_factor={self.noise_factor}, seed={self.base_seed}).")
        return clones
