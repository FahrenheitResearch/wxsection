#!/usr/bin/env python3
import os, time, re, pathlib, numpy as np

# Make logs quiet by default; set HRRR_DEBUG=1/true/yes/on to see debug
os.environ.setdefault("HRRR_DEBUG", "0")

from derived_params.wet_bulb_temperature import wet_bulb_temperature
from derived_params.wet_bulb_temperature_metpy import wet_bulb_temperature_metpy
from derived_params.mixing_ratio_2m import mixing_ratio_2m
from derived_params.lapse_rate_03km import lapse_rate_03km
from derived_params._wet_bulb_approximation import _wet_bulb_approximation
from derived_params._mixing_ratio_approximation import _mixing_ratio_approximation
from derived_params._psychrometrics import (
    saturation_mixing_ratio, lv_j_per_kg, CPD, _to_pa,
    e_from_dewpoint_pa, mixing_ratio_from_e
)

def rand_grid(seed=0, Y=120, X=150):
    rng = np.random.default_rng(seed)
    T = rng.uniform(-20, 40, size=(Y, X))
    Td = T - np.abs(rng.normal(2, 3, size=T.shape))
    P = rng.uniform(900, 1020, size=T.shape)  # hPa
    return T, Td, P

def test_wet_bulb():
    print("\n[WB] wet-bulb residual/bounds/alias parity...")
    T, Td, P = rand_grid(0, 80, 120)
    Tw = wet_bulb_temperature(T, Td, P)

    # residual small
    P_pa = _to_pa(P)
    ws = saturation_mixing_ratio(P_pa, Tw)
    w = mixing_ratio_from_e(P_pa, e_from_dewpoint_pa(Td))
    res = ws - (w + (CPD / lv_j_per_kg(Tw + 273.15)) * (T - Tw))
    p99 = np.nanpercentile(np.abs(res), 99)
    print(f"  residual p99 = {p99:.2e}")
    assert p99 < 5e-5

    # bounds
    assert np.all(Tw <= np.maximum(T, Td) + 1e-9)
    assert np.all(Tw >= np.minimum(T, Td) - 1e-9)

    # alias equality
    Tw_legacy = wet_bulb_temperature_metpy(T, Td, P)
    assert np.allclose(Tw, Tw_legacy, equal_nan=True)
    print("  ✓ wet-bulb OK")

def test_mixing_ratio():
    print("\n[MR] mixing ratio vs approximation...")
    T, Td, P = rand_grid(1, 120, 180)
    mr = mixing_ratio_2m(Td, P)
    mr_appx = _mixing_ratio_approximation(Td, P)
    diff = np.abs(mr - mr_appx)
    med = np.nanmedian(diff)
    p95 = np.nanpercentile(diff, 95)
    print(f"  median diff={med:.3f} g/kg, p95={p95:.3f} g/kg")
    assert med <= 0.2 and p95 <= 0.6
    assert np.all(mr[np.isfinite(mr)] >= 0)
    print("  ✓ mixing ratio OK")

def test_lapse_rate():
    print("\n[LR] 0–3 km lapse rate: profile interpolation & fallback...")
    L, Y, X = 30, 200, 260
    hsfc = np.linspace(0, 1500, Y*X).reshape(Y, X)
    h = hsfc[None, ...] + np.linspace(0, 12000, L)[:, None, None]
    Tprof = 290 - 0.0065 * h  # K
    Tsfc = Tprof[0]
    T700 = Tsfc - 5.0
    h700 = hsfc + 3000.0

    t0 = time.perf_counter()
    lr = lapse_rate_03km(Tsfc, T700, hsfc, h700, h, Tprof)
    dt = time.perf_counter() - t0
    mean_error = np.nanmean(np.abs(lr - 6.5))
    print(f"  perf: {dt:.3f}s for {Y*X:,} pts; mean error vs 6.5°C/km = {mean_error:.3f}")
    assert mean_error < 0.05

    # No column loops (spot check the source)
    src = pathlib.Path('derived_params/lapse_rate_03km.py').read_text()
    assert not re.search(r'for\s+\w+\s+in\s+range\(h2\.shape\[1\]\)', src)
    print("  ✓ lapse rate OK")

def test_no_metpy_imports():
    print("\n[IMP] grep for metpy imports...")
    import subprocess
    res = subprocess.run(['bash', '-lc', "grep -R \"from metpy\\|import metpy\" -n -- 'derived_params' || true"],
                         capture_output=True, text=True)
    out = (res.stdout or '').strip()
    if out:
        print(out)
    assert out == ""
    print("  ✓ no metpy imports")

def main():
    print("="*60)
    print("MetPy-free HRRR refactor: single final verification")
    print("="*60)
    test_wet_bulb()
    test_mixing_ratio()
    test_lapse_rate()
    test_no_metpy_imports()
    print("\nAll checks passed.")

if __name__ == "__main__":
    main()