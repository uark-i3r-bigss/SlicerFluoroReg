"""Display images for the detector: windowed X-ray with DRR overlays blended in colour (numpy only)."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def window_to_unit(image: np.ndarray, lo_percentile: float = 0.5, hi_percentile: float = 99.5) -> np.ndarray:
    """Linear window between two percentiles of the finite pixels, clipped to [0, 1]."""
    img = np.asarray(image, dtype=np.float64)
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros(img.shape)
    lo, hi = np.percentile(finite, [lo_percentile, hi_percentile])
    if hi <= lo:
        return np.zeros(img.shape)
    return np.clip((np.nan_to_num(img, nan=lo) - lo) / (hi - lo), 0.0, 1.0)


def composite_rgb(xray: np.ndarray, overlays: Sequence[Tuple[np.ndarray, Sequence[float]]],
                  opacity: float = 0.5) -> np.ndarray:
    """rows x cols x 3 uint8: the windowed X-ray in grey with each DRR (same grid) blended in its colour.

    A DRR pixel's weight is its value over the 99th percentile of the DRR's non-zero pixels, clipped
    to [0, 1] and scaled by opacity; zero DRR pixels leave the X-ray untouched.
    """
    base = window_to_unit(xray)
    rgb = np.repeat(base[..., None], 3, axis=2)
    for drr, color in overlays:
        d = np.nan_to_num(np.asarray(drr, dtype=np.float64))
        if d.shape != base.shape:
            raise ValueError(f"DRR shape {d.shape} differs from the X-ray's {base.shape}")
        positive = d[d > 0]
        if positive.size == 0:
            continue
        a = (float(opacity) * np.clip(d / np.percentile(positive, 99.0), 0.0, 1.0))[..., None]
        rgb = rgb * (1.0 - a) + np.asarray(color, dtype=np.float64)[None, None, :] * a
    return np.round(rgb * 255.0).astype(np.uint8)
