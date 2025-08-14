from .common import _dbg
import numpy as np
from typing import Optional

def _interp_temp_to_height_3km(height_profile, temp_profile, height_surface):
    """
    Linear interpolation of temperature to (surface + 3000 m).
    Profiles are (levels, y, x). Returns T_3km in °C, NaN if out of bounds.
    """
    h = np.asarray(height_profile, dtype=float)      # (L, Y, X)
    T = np.asarray(temp_profile, dtype=float)        # (L, Y, X)
    hsfc = np.asarray(height_surface, dtype=float)   # (Y, X)

    # Kelvin -> °C if needed
    if np.nanmax(T) > 200.0:
        T = T - 273.15

    target = hsfc + 3000.0                           # (Y, X)

    L = h.shape[0]
    Y, X = hsfc.shape
    N = Y * X
    h2 = h.reshape(L, N)                             # (L, N)
    T2 = T.reshape(L, N)                             # (L, N)
    tgt = target.reshape(1, N)                       # (1, N)

    # Find index k such that h2[k, j] <= tgt <= h2[k+1, j]
    mask = h2 <= tgt                                 # (L, N)
    k = mask.sum(axis=0) - 1                         # (N,)
    k = np.clip(k, 0, L - 2)

    idx = np.arange(N)
    h1 = h2[k, idx];       h2u = h2[k + 1, idx]
    t1 = T2[k, idx];       t2 = T2[k + 1, idx]

    with np.errstate(divide='ignore', invalid='ignore'):
        w = ((tgt.ravel() - h1) / (h2u - h1))
        T3 = t1 + w * (t2 - t1)

        col_min = np.nanmin(h2, axis=0)
        col_max = np.nanmax(h2, axis=0)
        oob = (tgt.ravel() < col_min) | (tgt.ravel() > col_max) | ~np.isfinite(w)

        # Optional: guard against non-monotonic profiles
        nonmono = np.any(np.diff(h2, axis=0) <= 0, axis=0)
        if np.any(nonmono):
            oob = oob | nonmono

        T3[oob] = np.nan

    return T3.reshape(Y, X)

def _compute_2level_lapse_rate(temp_surface: np.ndarray, temp_700: np.ndarray,
                               height_surface: np.ndarray, height_700: np.ndarray) -> np.ndarray:
    # Convert temperatures to °C if K
    if np.nanmax(temp_surface) > 200.0:
        temp_surface_c = temp_surface - 273.15
        temp_700_c = temp_700 - 273.15
    else:
        temp_surface_c = temp_surface
        temp_700_c = temp_700

    height_diff_total = height_700 - height_surface
    valid_thickness = height_diff_total > 1500.0     # require >= 1.5 km

    target_height_3km = height_surface + 3000.0
    interp_factor = np.where(valid_thickness,
                             (target_height_3km - height_surface) / height_diff_total,
                             np.nan)

    temp_3km_c = temp_surface_c + interp_factor * (temp_700_c - temp_surface_c)

    lapse_rate = np.where(valid_thickness,
                          (temp_surface_c - temp_3km_c) / 3.0,
                          np.nan)
    _dbg(f"📊 Using 2-level interpolation fallback")
    _dbg(f"   Valid thickness points: {np.nansum(valid_thickness)} of {valid_thickness.size}")
    return lapse_rate

def lapse_rate_03km(temp_surface: np.ndarray, temp_700: np.ndarray,
                    height_surface: np.ndarray, height_700: np.ndarray,
                    height_profile: Optional[np.ndarray] = None,
                    temp_profile: Optional[np.ndarray] = None) -> np.ndarray:
    if (height_profile is not None) and (temp_profile is not None):
        try:
            t_sfc_c = (temp_surface - 273.15) if (np.nanmax(temp_surface) > 200.0) else temp_surface
            t_3km_c = _interp_temp_to_height_3km(height_profile, temp_profile, height_surface)
            lapse_rate = (t_sfc_c - t_3km_c) / 3.0
            _dbg("✅ Using manual profile interpolation for 0–3 km lapse rate")
        except Exception as e:
            _dbg(f"⚠️ Profile interpolation failed: {e}")
            lapse_rate = _compute_2level_lapse_rate(temp_surface, temp_700, height_surface, height_700)
    else:
        lapse_rate = _compute_2level_lapse_rate(temp_surface, temp_700, height_surface, height_700)

    # Physical clipping (°C/km)
    lapse_rate = np.where(np.isfinite(lapse_rate), np.clip(lapse_rate, 2.0, 10.0), np.nan)
    return lapse_rate
