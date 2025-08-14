# derived_params/_psychrometrics.py
import numpy as np

# Physical constants
EPSILON = 0.62197            # Rd/Rv
CPD = 1004.0                 # J kg-1 K-1
G = 9.80665                  # m s-2

def _to_pa(pressure):
    """
    Convert pressure to Pa if needed.
    Heuristic: if max < 5000, assume hPa and multiply by 100.
    """
    p = np.asarray(pressure, dtype=float)
    if np.nanmax(p) < 5000.0:
        return p * 100.0
    return p

# Alduchov–Eskridge saturation vapor pressure (water/ice)
def es_w_pa(temp_c):
    Tc = np.asarray(temp_c, dtype=float)
    return 610.94 * np.exp(17.625 * Tc / (Tc + 243.04))

def es_i_pa(temp_c):
    Tc = np.asarray(temp_c, dtype=float)
    return 611.21 * np.exp(22.587 * Tc / (Tc + 273.86))

def es_mixed_pa(temp_c):
    Tc = np.asarray(temp_c, dtype=float)
    return np.where(Tc < 0.0, es_i_pa(Tc), es_w_pa(Tc))

def e_from_dewpoint_pa(td_c):
    """Actual vapor pressure from dewpoint using mixed-phase formula (Pa)."""
    return es_mixed_pa(td_c)

def lv_j_per_kg(temp_k):
    """Latent heat of vaporization (J/kg), weakly temperature dependent."""
    Tk = np.asarray(temp_k, dtype=float)
    return 2.501e6 - 2361.0 * (Tk - 273.15)

def mixing_ratio_from_e(p_pa, e_pa):
    """Mixing ratio (kg/kg) from total pressure and vapor pressure."""
    p = np.asarray(p_pa, dtype=float)
    e = np.asarray(e_pa, dtype=float)
    return EPSILON * e / np.maximum(p - e, 1.0)  # guard tiny denominators

def saturation_mixing_ratio(p_pa, temp_c):
    """Saturation mixing ratio (kg/kg) at (P,T) using mixed-phase e_s."""
    p = np.asarray(p_pa, dtype=float)
    e_s = es_mixed_pa(temp_c)
    e_s = np.minimum(e_s, 0.99 * p)              # avoid blow-ups at near-sat
    return mixing_ratio_from_e(p, e_s)

def wet_bulb_bisection(temp_c, td_c, pressure, max_iter=24, tol=5e-6):
    """
    Vectorized wet-bulb temperature via the psychrometric equation and bisection.
    Inputs: temp_c (°C), td_c (°C), pressure (Pa or hPa).
    Returns: Tw (°C).
    """
    T_c = np.asarray(temp_c, dtype=float)
    Td_c = np.asarray(td_c, dtype=float)
    P_pa = _to_pa(pressure)

    # Physical bracket: Tw ∈ [min(T, Td), max(T, Td)]
    lo = np.minimum(T_c, np.maximum(Td_c, -80.0))
    hi = np.maximum(T_c, Td_c)

    # Actual vapor pressure & mixing ratio
    e = e_from_dewpoint_pa(Td_c)                 # Pa
    w = mixing_ratio_from_e(P_pa, e)             # kg/kg

    def f(Tw_c):
        ws = saturation_mixing_ratio(P_pa, Tw_c)                 # kg/kg
        L = lv_j_per_kg(Tw_c + 273.15)                           # J/kg
        rhs = w + (CPD / L) * (T_c - Tw_c)                       # kg/kg
        return ws - rhs

    # Ensure the bracket encloses a root (if not, widen toward cold side)
    f_lo = f(lo)
    f_hi = f(hi)
    bad = (f_lo * f_hi) > 0
    if np.any(bad):
        lo = np.where(bad, lo - 5.0, lo)

    Tw = 0.5 * (lo + hi)
    for _ in range(max_iter):
        fm = f(Tw)
        done = np.abs(fm) < tol
        if np.all(done | ~np.isfinite(fm)):
            break
        left = fm > 0.0
        hi = np.where(left, Tw, hi)
        lo = np.where(left, lo, Tw)
        Tw = 0.5 * (lo + hi)

    # Keep within bounds; propagate NaNs
    mn = np.minimum(T_c, Td_c)
    mx = np.maximum(T_c, Td_c)
    Tw = np.clip(Tw, mn, mx)
    Tw = np.where(np.isfinite(T_c) & np.isfinite(Td_c) & np.isfinite(P_pa), Tw, np.nan)
    return Tw