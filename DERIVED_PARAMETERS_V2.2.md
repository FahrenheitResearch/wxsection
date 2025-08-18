# 📊 HRRR Derived Parameters - Enhanced Reference (v2.2)

> **⚠️ NOTICE: This document has been superseded by the main [DERIVED_PARAMETERS.md](./DERIVED_PARAMETERS.md) which contains the current v2.2 documentation with 108 parameters.**  
> **This file is maintained for detailed implementation reference and backward compatibility.**

> **Enhanced reference documentation for meteorological derived parameters**

## 📑 Table of Contents

- [Overview](#overview)
- [Quick Reference](#quick-reference)
- [Core Severe Weather Parameters](#core-severe-weather-parameters)
  - [Supercell Composite Parameter (SCP)](#supercell-composite-parameter-scp)
  - [Significant Tornado Parameter (STP)](#significant-tornado-parameter-stp)
  - [Energy-Helicity Index (EHI)](#energy-helicity-index-ehi)
  - [Significant Hail Parameter (SHIP)](#significant-hail-parameter-ship)
- [Wind Shear Parameters](#wind-shear-parameters)
  - [Bulk Wind Shear](#bulk-wind-shear)
  - [Effective Layer Parameters](#effective-layer-parameters)
  - [Wind Vector Calculations](#wind-vector-calculations)
- [Thermodynamic Parameters](#thermodynamic-parameters)
  - [Wet Bulb Temperature](#wet-bulb-temperature)
  - [CAPE/CIN Calculations](#capecin-calculations)
  - [Low-Level CAPE](#low-level-cape-0-3km)
- [Stability Indices](#stability-indices)
  - [Lifted Index](#lifted-index-li)
  - [Showalter Index](#showalter-index-si)
  - [SWEAT Index](#sweat-index)
  - [Bulk Richardson Number](#bulk-richardson-number-brn)
- [Fire Weather Parameters](#fire-weather-parameters)
  - [Haines Index](#haines-index)
  - [Ventilation Rate](#ventilation-rate)
  - [Enhanced Smoke Dispersion Index](#enhanced-smoke-dispersion-index-esdi)
- [Heat Stress Indices](#heat-stress-indices)
  - [WBGT Variants](#wbgt-wet-bulb-globe-temperature)
  - [Mixing Ratio](#mixing-ratio-2m)
- [Specialized Research Parameters](#specialized-research-parameters)
  - [Violent Tornado Parameter (VTP)](#violent-tornado-parameter-vtp)
  - [Mesocyclone Strength Parameter (MSP)](#mesocyclone-strength-parameter-msp)
  - [Vorticity Generation Parameter (VGP)](#vorticity-generation-parameter-vgp)
- [Composite Indices](#composite-indices)
- [Surface & Boundary Layer](#surface--boundary-layer-parameters)
- [Usage Examples](#usage-examples)
- [Implementation Architecture](#implementation-architecture)
- [Contributing](#contributing)

---

## 🌟 Overview (Legacy Reference)

**For current v2.2 documentation, see [DERIVED_PARAMETERS.md](./DERIVED_PARAMETERS.md)**

This legacy reference provides detailed implementation documentation for meteorological parameters. The main documentation now covers **108 total parameters** with full SPC alignment.

### v2.2 Status
- ✅ **108 meteorological parameters** across all severe weather categories (updated count)
- ✅ **SPC-compliant** canonical implementations with proper EBWD/20 normalization
- ✅ **Centralized constants** module prevents parameter drift
- ✅ **Transport wind methodology** for improved fire weather calculations
- ✅ **Status badge system** for operational confidence

### Categories

| Category | Parameters | Primary Use |
|----------|------------|-------------|
| **Severe Weather** | SCP, STP, EHI, SHIP, VTP | Supercell & tornado forecasting |
| **Wind Shear** | 0-1km, 0-6km, effective layers | Storm organization assessment |
| **Thermodynamics** | CAPE/CIN, wet bulb, lapse rates | Instability analysis |
| **Stability** | LI, SI, SWEAT, BRN | Convective potential |
| **Fire Weather** | Haines, ventilation, smoke dispersion | Fire behavior & smoke management |
| **Heat Stress** | WBGT variants | Occupational safety |
| **Research** | VTP, MSP, VGP | Advanced tornado diagnostics |

---

## 🚀 Quick Reference

### Most Critical Parameters

| Parameter | Formula | Threshold | Meaning |
|-----------|---------|-----------|---------|
| **SCP** | `(muCAPE/1000)×(ESRH/50)×shear×CIN` | > 1 | Supercell potential |
| **STP** | `(MLCAPE/1500)×(SRH/150)×(shear/12)×LCL×CIN` | > 1 | EF2+ tornado risk |
| **EHI** | `(CAPE/1600)×(SRH/50)×damping` | > 2 | Significant tornado |
| **SHIP** | `(muCAPE/1500)×(mr/13.6)×(lapse/7)×(shear/20)×temp` | > 1 | 2"+ hail |
| **BRN** | `CAPE/(0.5×shear²)` | 10-45 | Supercell window |
| **VTP** | `7-term product including low-level CAPE` | > 1 | Violent tornado |

---

## ⛈️ Core Severe Weather Parameters

### Supercell Composite Parameter (SCP)

#### 📐 Formula
```
SCP = (muCAPE/1000) × (ESRH/50) × shear_term × CIN_weight

where:
  shear_term = clip((EBWD-10)/10, 0, 1)  # SPC piecewise scaling
  CIN_weight = 1.0 if muCIN > -40 else -40/muCIN
```

#### 📥 Inputs
| Parameter | Description | Units | Source |
|-----------|-------------|-------|--------|
| `mucape` | Most-Unstable CAPE | J/kg | HRRR MUCAPE |
| `effective_srh` | Effective Storm Relative Helicity | m²/s² | HRRR ESRHL |
| `effective_shear` | Effective Bulk Wind Difference | m/s | Derived |
| `mucin` | Most-Unstable CIN (optional) | J/kg | HRRR MUCIN |

#### ⚙️ Implementation Details
- **Shear Term Scaling:**
  - `EBWD < 10 m/s`: 0 (insufficient shear)
  - `10-20 m/s`: Linear scaling (EBWD-10)/10
  - `EBWD ≥ 20 m/s`: 1.0 (optimal shear)
  
- **CIN Penalty:**
  - `muCIN > -40 J/kg`: No penalty (CIN_weight = 1.0)
  - `muCIN ≤ -40 J/kg`: Proportional reduction (-40/muCIN)

- **Quality Control:**
  - Forces negative SRH to 0 before calculation
  - Masks areas with CAPE < 50 J/kg
  - Clamps output to [-2, 10] for display

#### 📊 Interpretation
| SCP Value | Meaning | Action |
|-----------|---------|--------|
| < 0 | Left-moving supercell potential | Monitor anticyclonic storms |
| 0-1 | Marginal supercell potential | General thunderstorms likely |
| 1-4 | Moderate supercell potential | **Issue watches** |
| 4-10 | High supercell potential | **Tornado watches likely** |
| > 10 | Extreme overlap environment | **PDS watches/emergencies** |

#### 📁 Source
`derived_params/supercell_composite_parameter.py:3-117`

---

### Significant Tornado Parameter (STP)

#### 📐 Formula
```
STP = (MLCAPE/1500) × (SRH_01km/150) × (shear_06km/12) × 
      ((2000-MLLCL)/1000) × ((MLCIN+200)/150)
```

#### 📥 Inputs
| Parameter | Description | Units | Range |
|-----------|-------------|-------|-------|
| `mlcape` | Mixed Layer CAPE | J/kg | 0-5000+ |
| `mlcin` | Mixed Layer CIN | J/kg | -200 to 0 |
| `srh_01km` | 0-1km Storm Relative Helicity | m²/s² | 0-600+ |
| `shear_06km` | 0-6km bulk wind shear | m/s | 0-60+ |
| `lcl_height` | Mixed Layer LCL height | m AGL | 0-4000 |

#### ⚙️ Implementation Details

**Term-by-term breakdown:**

1. **CAPE Term:** `MLCAPE/1500`
   - No artificial cap (physics provides natural limits)
   
2. **LCL Term:** `(2000-MLLCL)/1000`
   - LCL < 1000m → 1.0 (very favorable)
   - 1000-2000m → Linear decrease
   - LCL > 2000m → 0.0 (too high)

3. **SRH Term:** `SRH_01km/150`
   - Preserves positive values only
   
4. **Shear Term:** Complex scaling
   - shear < 12.5 m/s → 0 (insufficient)
   - 12.5-30 m/s → shear/12
   - shear > 30 m/s → cap at 1.5

5. **CIN Term:** `(MLCIN+200)/150`
   - CIN > -50 J/kg → 1.0 (weak cap)
   - -50 to -200 → Linear decrease
   - CIN < -200 J/kg → 0.0 (too strong)

#### 📊 Interpretation
| STP Value | Tornado Risk | Historical Context |
|-----------|--------------|-------------------|
| < 0.5 | Very low | Background conditions |
| 0.5-1 | Low | Marginal tornado potential |
| **1-3** | **Moderate** | **EF2+ tornadoes possible** |
| **3-6** | **High** | **Significant outbreak likely** |
| **6-10** | **Very High** | **Major outbreak conditions** |
| > 10 | Extreme | Historic outbreak potential |

#### 📁 Source
`derived_params/significant_tornado_parameter.py:3-68`

---

### Energy-Helicity Index (EHI)

#### 📐 Formula
```
EHI_raw = (CAPE/1600) × (SRH_03km/50)
EHI_final = sign(EHI_raw) × damping_factor

where damping_factor:
  if |EHI| ≤ 5: no damping
  if |EHI| > 5: 5 + log(|EHI|/5)  # Logarithmic compression
```

#### 📥 Inputs
- `cape`: Surface-based or mixed-layer CAPE (J/kg)
- `srh_03km`: 0-3km Storm Relative Helicity (m²/s²) - **preserves sign**

#### ⚙️ Implementation Details
- **Anti-saturation damping** prevents "red sea" oversaturation in extreme environments
- **Sign preservation:** 
  - Positive EHI → Right-moving (cyclonic) supercells
  - Negative EHI → Left-moving (anticyclonic) supercells
- **Damping threshold:** 5.0 (operational experience)

#### 📊 Interpretation
| EHI Value | Meaning | Tornado Potential |
|-----------|---------|-------------------|
| < -2 | Strong left-mover potential | Anticyclonic tornadoes possible |
| -2 to 0 | Weak left-mover signal | Limited anticyclonic risk |
| 0-1 | Marginal | General rotation |
| **1-2** | **Moderate** | **Tornadoes possible** |
| **2-5** | **Significant** | **Strong tornadoes likely** |
| > 5 | Extreme | Violent tornadoes possible |

#### 📁 Source
`derived_params/energy_helicity_index.py:3-56`

---

### Significant Hail Parameter (SHIP)

#### 📐 Formula
```
SHIP = (muCAPE/1500) × (MU_mr/13.6) × (lapse_700_500/7) × 
       (shear_06km/20) × ((frz_lvl - T500)/8)

All five terms capped at 1.0 per SPC v1.1 specification
```

#### 📥 Inputs
| Parameter | Description | Units | Typical Range |
|-----------|-------------|-------|---------------|
| `mucape` | Most-Unstable CAPE | J/kg | 0-5000 |
| `lapse_rate_700_500` | Mid-level lapse rate | °C/km | 5-9 |
| `wind_shear_06km` | Deep-layer shear | m/s | 0-40 |
| `freezing_level` | 0°C isotherm height | m AGL | 2000-5000 |
| `temp_500` | 500mb temperature | °C | -20 to -5 |
| `mixing_ratio_2m` | Surface moisture (proxy for MU) | g/kg | 5-20 |

#### ⚙️ Implementation Details
**Five required terms (SPC SHIP v1.1):**

1. **CAPE term:** `min(muCAPE/1500, 1.0)`
2. **Moisture term:** `min(mr/13.6, 1.0)` - Uses 2m as MU proxy
3. **Lapse rate term:** `min(lapse/7.0, 1.0)`
4. **Shear term:** `min(shear/20, 1.0)`
5. **Temperature term:** `min((frz_lvl_km - T500)/8, 1.0)`

#### 📊 Interpretation
| SHIP Value | Hail Size | Probability |
|------------|-----------|-------------|
| < 0.5 | < 1" | Low |
| 0.5-1 | 1-2" | Marginal severe |
| **1-2** | **2-2.75"** | **Significant hail likely** |
| **2-4** | **2.75-4"** | **Giant hail possible** |
| > 4 | > 4" | Extreme hail threat |

#### 📁 Source
`derived_params/significant_hail_parameter.py:4-108`

---

## 💨 Wind Shear Parameters

### Bulk Wind Shear

#### 📐 Formula
```
shear_magnitude = sqrt(u_shear² + v_shear²)
```

#### Variants
| Layer | Purpose | Critical Values |
|-------|---------|-----------------|
| **0-1km** | Low-level shear, tornado potential | > 10 m/s favorable |
| **0-3km** | Mesocyclone development | > 15 m/s favorable |
| **0-6km** | Deep-layer shear, supercell longevity | 15-25 m/s optimal |

#### 📁 Sources
- `derived_params/wind_shear_magnitude.py:3`
- `derived_params/wind_shear_vector_01km.py`
- `derived_params/wind_shear_vector_06km.py`

---

### Effective Layer Parameters

#### Effective Storm Relative Helicity
**Adjusts helicity calculation based on actual storm depth**

```python
if MLCAPE >= 100 and MLCIN >= -250:
    effective_base = max(0, LCL - 500)  # Start below LCL
    effective_top = EL  # Equilibrium level
    ESRH = calculate_srh(effective_base, effective_top)
```

#### Effective Bulk Wind Difference
**Shear through the effective storm depth**

```python
if storm_is_surface_based:
    EBWD = standard_06km_shear
else:
    EBWD = shear_through_effective_layer
```

#### 📁 Sources
- `derived_params/effective_srh.py`
- `derived_params/effective_shear.py`

---

### Wind Vector Calculations

#### Wind Speed & Direction (10m)
```python
# Wind speed
speed = sqrt(u10² + v10²)

# Wind direction (meteorological convention)
dir_math = atan2(v10, u10) * 180/π
dir_met = (270 - dir_math) % 360  # North=0°, clockwise
```

#### Crosswind Component
**For aviation and fire spread applications**
```python
crosswind = wind_speed × sin(wind_dir - runway_heading)
```

#### 📁 Sources
- `derived_params/wind_speed_10m.py`
- `derived_params/wind_direction_10m.py`
- `derived_params/crosswind_component.py`

---

## 🌡️ Thermodynamic Parameters

### Wet Bulb Temperature

#### Implementation Strategy
```python
def wet_bulb_temperature(T, Td, P):
    try:
        # Primary: Iterative bisection (high accuracy)
        Tw = wet_bulb_bisection(T, Td, P)
        if fraction_nan > 0.2:
            raise RuntimeError("Excessive NaN values")
        return Tw
    except:
        # Fallback: Fast approximation
        return wet_bulb_approximation(T, Td, P)
```

#### Methods
1. **Bisection Method** - Solves psychrometric equations iteratively
2. **Stull Approximation** - Fast empirical formula for operational use

#### 📁 Source
`derived_params/wet_bulb_temperature.py:7-24`

---

### CAPE/CIN Calculations

#### Backup Calculations (when HRRR fields unavailable)

| Type | Parcel Definition | Use Case |
|------|-------------------|----------|
| **Surface-Based** | Uses surface T/Td | Afternoon convection |
| **Mixed-Layer** | 100mb mean parcel | General severe weather |
| **Most-Unstable** | Max θe in lowest 300mb | Elevated convection |

#### Implementation
```python
# Surface-based CAPE example
parcel = lift_parcel(T_sfc, Td_sfc, P_sfc)
CAPE = integrate_positive_area(parcel, environment)
CIN = integrate_negative_area(parcel, environment)
```

#### 📁 Sources
- `derived_params/calculate_surface_based_cape.py`
- `derived_params/calculate_mixed_layer_cape.py`
- `derived_params/calculate_most_unstable_cape.py`

---

### Low-Level CAPE (0-3km)

#### 📐 Formula
```python
# Realistic approximation based on CAPE magnitude
if MLCAPE < 1000:
    fraction = 0.25  # Higher fraction for weak CAPE
elif MLCAPE < 3000:
    fraction = 0.20  # Moderate CAPE
else:
    fraction = 0.15  # Lower fraction for high CAPE

CAPE_03km = MLCAPE × fraction
CAPE_03km = min(CAPE_03km, 600)  # Rarely exceeds 600 J/kg
```

#### Physical Basis
- Low-level CAPE typically 10-30% of total CAPE
- Critical for tornado potential assessment
- Values > 200 J/kg are significant
- Values > 400 J/kg are rare but indicate extreme low-level buoyancy

#### 📁 Source
`derived_params/cape_03km.py:3-45`

---

### 0-3km Lapse Rate

#### 📐 Formula
```
LR_03km = (T_surface - T_3km) / 3.0  [°C/km]
```

#### Implementation
1. **Profile interpolation** - Best method using full atmospheric profile
2. **Two-level fallback** - Linear interpolation between surface and 700mb

```python
# Find temperature at exactly 3km AGL
T_3km = interpolate_to_height(height_profile, temp_profile, 
                              surface_height + 3000)
lapse_rate = (T_surface_C - T_3km_C) / 3.0
```

#### Typical Values
- **< 5°C/km**: Stable layer
- **5-7°C/km**: Conditionally unstable
- **7-9°C/km**: Unstable (favorable for convection)
- **> 9°C/km**: Very unstable (strong convection likely)

#### 📁 Source
`derived_params/lapse_rate_03km.py:80-98`

---

## 📈 Stability Indices

### Lifted Index (LI)

#### 📐 Formula
```
LI = T_500mb_environment - T_500mb_parcel
```

#### Interpretation Scale
| LI Value | Stability | Convective Potential |
|----------|-----------|---------------------|
| > 2 | Very stable | No thunderstorms |
| 0 to 2 | Stable | Unlikely thunderstorms |
| **0 to -3** | **Marginally unstable** | **Thunderstorms possible** |
| **-3 to -6** | **Moderately unstable** | **Thunderstorms likely** |
| **< -6** | **Very unstable** | **Severe thunderstorms likely** |
| < -9 | Extremely unstable | Extreme severe weather |

---

### Showalter Index (SI)

#### 📐 Formula
```
SI = T_500mb_environment - T_500mb_parcel_from_850mb
```

#### Key Differences from LI
- Uses 850mb parcel instead of surface
- Better for elevated convection assessment
- Less sensitive to shallow surface layers

---

### SWEAT Index

#### 📐 Formula
```
SWEAT = 12×TT + 20×max(TT-49,0) + 2×WS850 + WS500 + WDIR
where:
  TT = Total Totals = (T850 + Td850) - 2×T500
  WS850 = 850mb wind speed term (if > 15 m/s)
  WS500 = 500mb wind speed term (if > 15 m/s)
  WDIR = Directional shear term (complex conditions)
```

#### Interpretation
| SWEAT | Severe Weather Potential |
|-------|-------------------------|
| < 150 | Low |
| 150-300 | Moderate |
| **300-400** | **High** |
| > 400 | Very high |

---

### Bulk Richardson Number (BRN)

#### 📐 Formula
```
BRN = CAPE / (0.5 × shear²)
```

#### Storm Type Discrimination
| BRN Range | Expected Storm Type | Characteristics |
|-----------|-------------------|-----------------|
| < 10 | Shear-dominated | Storms struggle, quick dissipation |
| **10-45** | **Supercells** | **Optimal balance** |
| 45-100 | Multicells | Organized but less rotation |
| > 100 | Pulse storms | Brief, disorganized |

#### 📁 Source
`derived_params/bulk_richardson_number.py:3-46`

---

## 🔥 Fire Weather Parameters

### Haines Index

#### 📐 Formula
```
Haines Index = A + B

A (Stability):
  1: T850-T700 < 4°C
  2: 4-8°C
  3: > 8°C

B (Moisture):
  1: T850-Td850 < 6°C
  2: 6-10°C
  3: > 10°C
```

#### Fire Behavior Interpretation
| Haines | Fire Potential | Expected Behavior |
|--------|---------------|-------------------|
| 2-3 | Very low | Normal fire behavior |
| 4 | Low | Limited erratic behavior |
| **5** | **Moderate** | **Increased fire activity** |
| **6** | **High** | **Erratic fire behavior likely** |

---

### Ventilation Rate

#### 📐 Formula
```
VR = Wind_Speed × Boundary_Layer_Height
```

#### Smoke Management Applications
| VR (m²/s) | Dispersion | Prescribed Burn Suitability |
|-----------|------------|----------------------------|
| < 1,000 | Very poor | Do not burn |
| 1,000-6,000 | Poor | Marginal conditions |
| **6,000-20,000** | **Fair** | **Acceptable for most burns** |
| 20,000-50,000 | Good | Excellent dispersion |
| > 50,000 | Excellent | Optimal conditions |

---

### Enhanced Smoke Dispersion Index (ESDI)

#### 📐 Formula
```
ESDI = shear_factor × stability_factor × BL_factor × wind_factor

where:
  shear_factor = min(shear × 100, 2.0)
  stability_factor = 2.0 (unstable), 1.0 (neutral), 0.5 (stable)
  BL_factor = clip(BL_height/1500, 0.1, 2.0)
  wind_factor = clip(wind_speed/10, 0.1, 2.0)
```

#### Interpretation
| ESDI | Dispersion Quality | Action |
|------|-------------------|--------|
| < 0.5 | Very poor | Avoid burning |
| 0.5-1.0 | Poor | High smoke impacts |
| **1.0-3.0** | **Moderate** | **Acceptable conditions** |
| 3.0-5.0 | Good | Low smoke impacts |
| > 5.0 | Excellent | Minimal impacts |

---

## 🌡️ Heat Stress Indices

### WBGT (Wet Bulb Globe Temperature)

#### Variants

| Type | Formula | Application |
|------|---------|-------------|
| **Shade** | `0.7×Tw + 0.3×T` | Indoor/shaded work |
| **Outdoor** | Complex with solar load | Direct sun exposure |
| **Simplified** | Empirical approximation | Quick assessment |

#### Heat Stress Categories (WBGT °C)
| WBGT | Risk Level | Work/Rest Ratio | Actions |
|------|------------|-----------------|---------|
| < 25.6 | Low | Continuous | Normal operations |
| 25.6-27.8 | Moderate | 45 min/15 min | Increase water breaks |
| **27.8-29.4** | **High** | **30 min/30 min** | **Mandatory rest periods** |
| **29.4-31.1** | **Very High** | **15 min/45 min** | **Light duty only** |
| > 31.1 | Extreme | Stop work | Emergency protocols |

---

### Mixing Ratio (2m)

#### 📐 Formula
```python
# Saturation vapor pressure
e_s = 6.112 × exp(17.67 × Td / (Td + 243.5))

# Mixing ratio
mr = 622 × e_s / (P - e_s)  [g/kg]
```

#### Applications
- Moisture advection assessment
- Convective inhibition analysis
- Fire weather fuel moisture
- Agricultural monitoring

---

## 🌪️ Specialized Research Parameters

### Violent Tornado Parameter (VTP)

#### 📐 Formula
```
VTP = (MLCAPE/1500) × (EBWD/20) × (ESRH/150) × 
      ((2000-MLLCL)/1000) × ((200+MLCIN)/150) × 
      (CAPE_03km/50) × (LR_03km/6.5)
```

#### Advanced Features
- **7-term multiplicative formula** (Hampshire et al. 2018)
- **Low-level focus** with 0-3km CAPE and lapse rate
- **Effective-layer gating** for realistic environments
- **Hard ceiling at 8.0** to prevent unrealistic values

#### Implementation Gates
```python
# Only calculate where storms can realistically develop
if MLCAPE >= 100 and MLCIN >= -150 and LCL <= 2000:
    calculate_VTP()
else:
    VTP = 0
```

#### Interpretation
| VTP | Violent Tornado Risk | Historical Analogs |
|-----|---------------------|-------------------|
| < 0.5 | Very low | Background |
| 0.5-1 | Low | Marginal EF2+ risk |
| **1-2** | **Moderate** | **EF3+ possible** |
| **2-4** | **High** | **EF4+ possible** |
| > 4 | Extreme | EF4-5 likely |

#### 📁 Source
`derived_params/violent_tornado_parameter.py:3-124`

---

### Mesocyclone Strength Parameter (MSP)

#### 📐 Formula
```
MSP = (UH/100) × vertical_enhancement × shear_enhancement

where:
  vertical_enhancement = 1 + 0.5 × min(w/20, 1.5)
  shear_enhancement = min(shear/25, 1.2)
```

#### Physical Basis
- **Updraft helicity** as primary mesocyclone indicator
- **Vertical velocity** enhances through stretching
- **Environmental shear** maintains mesocyclone

#### Applications
- Mesocyclone tracking and intensity nowcasting
- Tornado warning decision support
- Supercell lifecycle monitoring

---

### Vorticity Generation Parameter (VGP)

#### 📐 Formula
```
VGP = (CAPE/1000) × (shear_01km/1000) × 0.1  [m/s²]
```

#### Physical Process
1. **Horizontal vorticity** from low-level shear
2. **Tilting** by updraft into vertical
3. **Stretching** intensifies rotation
4. **Result:** Tornado-scale vorticity

#### Critical Thresholds
| VGP (m/s²) | Tornado Potential | Process Efficiency |
|------------|------------------|-------------------|
| < 0.1 | Very low | Minimal vorticity generation |
| 0.1-0.2 | Low | Some tilting/stretching |
| **0.2-0.5** | **Moderate** | **Efficient vorticity generation** |
| > 0.5 | High | Rapid intensification likely |

---

## 🔄 Composite Indices

### Parameter Combinations

| Index | Components | Purpose |
|-------|------------|---------|
| **Craven SigSvr** | MLCAPE × Shear | Simple severe weather |
| **Craven-Brooks** | 0.4×CAPE + 0.4×Shear + 0.2×SRH | Weighted composite |
| **Composite Severe** | SCP + STP + UH | Multi-hazard assessment |
| **Modified STP** | Enhanced effective layer version | Research applications |
| **Right-Mover Composite** | Positive SRH emphasis | Cyclonic supercell focus |

---

## 🌍 Surface & Boundary Layer Parameters

### Surface Richardson Number

#### 📐 Formula
```
Ri = (g/T) × (∂T/∂z) / (∂u/∂z)²
```

#### Stability Regimes
| Ri | Stability | Turbulence |
|----|-----------|------------|
| < -1 | Very unstable | Strong convective turbulence |
| -1 to 0 | Unstable | Moderate turbulence |
| 0 to 0.25 | Neutral to critical | Transition zone |
| > 0.25 | Stable | Suppressed turbulence |

---

### Additional BL Parameters

| Parameter | Formula | Application |
|-----------|---------|-------------|
| **Monin-Obukhov Length** | `L = -u*³T/(kg(w'T'))` | BL scaling parameter |
| **Convective Velocity** | `w* = (g×zi×H/T)^(1/3)` | Convective intensity |
| **TKE Estimate** | Based on shear and buoyancy | Turbulence intensity |

---

## 💻 Usage Examples

### Basic Parameter Calculation
```python
from derived_params import compute_derived_parameter

# Calculate SCP
config = {
    'function': 'supercell_composite_parameter',
    'inputs': ['mucape', 'effective_srh', 'effective_shear', 'mucin']
}

input_data = {
    'mucape': cape_array,
    'effective_srh': srh_array,
    'effective_shear': shear_array,
    'mucin': cin_array
}

scp = compute_derived_parameter('scp', input_data, config)
```

### Batch Processing
```python
# Calculate multiple parameters efficiently
parameters = ['scp', 'stp', 'ehi', 'ship']
results = {}

for param in parameters:
    results[param] = compute_derived_parameter(
        param, 
        input_data, 
        param_configs[param]
    )
```

### Quality Control Example
```python
# Check for extreme values
if np.nanmax(scp) > 10:
    logger.warning(f"Extreme SCP detected: {np.nanmax(scp):.1f}")
    
# Mask invalid regions
scp_masked = np.where(mucape > 100, scp, np.nan)
```

---

## 🏗️ Implementation Architecture

### System Design

```
┌─────────────────────────────────────┐
│         User Application            │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│    compute_derived_parameter()      │
│         (Dispatch Layer)            │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│     Individual Calculation          │
│         Functions (70+)             │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│      Common Utilities               │
│   (QC, validation, helpers)         │
└─────────────────────────────────────┘
```

### Key Features

#### 1. Centralized Dispatch
```python
_DERIVED_FUNCTIONS = {
    'scp': supercell_composite_parameter,
    'stp': significant_tornado_parameter,
    # ... 70+ functions
}
```

#### 2. Configuration-Driven
```json
{
  "scp": {
    "function": "supercell_composite_parameter",
    "inputs": ["mucape", "effective_srh", "effective_shear"],
    "units": "dimensionless",
    "cmap": "SCP",
    "levels": [-2, 0, 1, 2, 4, 8, 10]
  }
}
```

#### 3. Quality Control Pipeline
- Input validation
- Physical bounds checking
- NaN/missing data handling
- Extreme value logging
- Output masking

#### 4. Performance Optimization
- Vectorized NumPy operations
- Lazy evaluation where possible
- Caching for repeated calculations
- Parallel processing support

---

## 🤝 Contributing

### Adding New Parameters

1. **Create calculation function** in `derived_params/`
```python
def new_parameter(input1: np.ndarray, input2: np.ndarray) -> np.ndarray:
    """
    Calculate new meteorological parameter
    
    Args:
        input1: Description (units)
        input2: Description (units)
    
    Returns:
        Parameter values (units)
    """
    # Implementation
    result = formula(input1, input2)
    
    # Quality control
    result = np.where(valid_conditions, result, np.nan)
    
    return result
```

2. **Register in dispatch table** (`__init__.py`)
```python
_DERIVED_FUNCTIONS['new_parameter'] = new_parameter
```

3. **Add configuration** (`parameters/derived.json`)
```json
{
  "new_param": {
    "function": "new_parameter",
    "inputs": ["input1", "input2"],
    "units": "units",
    "description": "What it measures"
  }
}
```

4. **Document thoroughly** with:
   - Mathematical formula
   - Physical interpretation
   - Typical value ranges
   - Use cases and applications

### Code Standards
- Type hints for all functions
- Comprehensive docstrings
- Unit tests with edge cases
- Performance benchmarks for real-time use

---

## 📚 References

### Primary Sources
- **Thompson et al. (2003):** SCP formulation
- **Thompson et al. (2012):** STP updates
- **Hampshire et al. (2018):** VTP development
- **SPC Mesoanalysis:** Operational implementations
- **NOAA/NWS directives:** Standard calculations

### Additional Reading
- [SPC Mesoanalysis Help](https://www.spc.noaa.gov/exper/mesoanalysis/help/)
- [HRRR Model Documentation](https://rapidrefresh.noaa.gov/hrrr/)
- [AMS Glossary](https://glossary.ametsoc.org/)

---

## 📄 License & Citation

This documentation and code are part of the HRRR Map Project. When using these parameters in research or operations, please cite:

```
HRRR Derived Parameters Library
https://github.com/yourusername/hrrr-map-project
Version 2.0, 2024
```

---

**Last Updated:** 2024
**Version:** 2.0
**Total Parameters:** 70+
**Contact:** [your-email@domain.com]

---

*This documentation represents the collective knowledge of the severe weather forecasting community. We acknowledge the contributions of NOAA/NWS/SPC and all researchers whose work enables these calculations.*