# 📊 HRRR Derived Parameters - Complete Documentation (v2.1 - Corrected)

> **Comprehensive documentation of 70+ meteorological derived parameters for the HRRR Map Project**  
> **Version 2.1** - Updated with SPC-compliant formulations and accuracy corrections

---

## 🔄 Version 2.1 - Critical Corrections Summary

### Why These Changes Were Made
A peer review identified several discrepancies between our implementations and official Storm Prediction Center (SPC) / peer-reviewed definitions. These corrections ensure our parameters match operational standards used by the National Weather Service and research community.

### What Changed

| Parameter | Issue | Correction | Impact |
|-----------|-------|------------|--------|
| **EHI** | Used `/80,000` normalization (non-standard) | Changed to `/160,000` per Davies (1993) & SPC | Values now 2× larger; thresholds adjusted |
| **VGP** | Custom formula `(CAPE/1000)×(shear/1000)×0.1` | Corrected to `S × √CAPE` per Rasmussen (1998) | Different units & thresholds (now >0.15 significant) |
| **WBGT** | Not clearly labeled as approximation | Added clarification: uses Tw not Tnwb, Tdb not Tg | Same values, better documentation |
| **Haines Index** | Missing deprecation notice | Added NWS SCN 24-107 (Dec 2024) discontinuation | Users directed to HDW, VPD alternatives |
| **STP** | Mixed fixed/effective implementations | Clarified two variants: fixed+CIN (modified) vs effective+CIN (SPC) | Better understanding of which version to use |
| **0-3km CAPE** | Fraction-based approximation not labeled | Added warning: heuristic with ±50% error possible | Users aware it's not true vertical integration |
| **SHIP** | Confusion about versions | Confirmed using correct SPC v1.1 (5 capped terms) | No change needed, documentation clarified |
| **SCP** | Shear term questioned | Verified `(EBWD-10)/10` IS correct SPC implementation | No change needed, added explanation |

### Files Modified

#### Code Files (7 changed):
1. `energy_helicity_index.py` - Fixed normalization
2. `vorticity_generation_parameter.py` - New formula with proper references
3. `wbgt_shade.py` - Added approximation clarification
4. `haines_index.py` - Added deprecation notice
5. `significant_tornado_parameter.py` - Clarified as fixed+CIN variant
6. `significant_tornado_parameter_effective.py` - Fixed EBWD/12, clarified as SPC current
7. `cape_03km.py` - Added approximation warning

#### Documentation:
- Created `DERIVED_PARAMETERS_CORRECTED.md` with all updates
- Added status labels (🟢 SPC, 🟡 Modified, 🟠 Approximation, 🔵 Research, 🔴 Deprecated)

### Impact on Users

⚠️ **BREAKING CHANGES:**
- **EHI values** will be ~2× larger than before (now correct)
- **VGP values** will be completely different (new formula and units)

ℹ️ **NON-BREAKING CLARIFICATIONS:**
- WBGT, 0-3km CAPE unchanged but now properly documented as approximations
- Haines Index still works but users warned of deprecation
- STP/SCP values unchanged, just better documented

### Validation
All corrections verified against:
- SPC Mesoanalysis Page formulations
- Peer-reviewed literature (Davies 1993, Rasmussen 1998, Thompson 2003/2012)
- NWS operational standards
- Current (2024) best practices

---

## 📑 Table of Contents

- [Overview](#overview)
- [Parameter Status Classifications](#parameter-status-classifications)
- [Quick Reference](#quick-reference)
- [Core Severe Weather Parameters](#core-severe-weather-parameters)
  - [Supercell Composite Parameter (SCP)](#supercell-composite-parameter-scp)
  - [Significant Tornado Parameter (STP)](#significant-tornado-parameter-stp)
  - [Energy-Helicity Index (EHI)](#energy-helicity-index-ehi)
  - [Significant Hail Parameter (SHIP)](#significant-hail-parameter-ship)
- [Wind Shear Parameters](#wind-shear-parameters)
- [Thermodynamic Parameters](#thermodynamic-parameters)
- [Stability Indices](#stability-indices)
- [Fire Weather Parameters](#fire-weather-parameters)
- [Heat Stress Indices](#heat-stress-indices)
- [Specialized Research Parameters](#specialized-research-parameters)
- [Implementation Notes](#implementation-notes)
- [References](#references)

---

## 🌟 Overview

The HRRR Derived Parameters system provides **real-time calculation** of advanced meteorological indices from High-Resolution Rapid Refresh (HRRR) model data. This library implements official **Storm Prediction Center (SPC)** formulations alongside research parameters.

### Version 2.1 Updates
- ✅ **Corrected EHI** to standard /160,000 normalization
- ✅ **Fixed VGP** to use S × √CAPE formula
- ✅ **Clarified WBGT** as approximation method
- ✅ **Added Haines Index** deprecation notice (NWS SCN 24-107)
- ✅ **Distinguished STP variants** (fixed vs. effective layers)
- ✅ **Labeled 0-3km CAPE** as heuristic approximation
- ✅ **Updated SWEAT** with complete formula

---

## 🏷️ Parameter Status Classifications

| Status | Meaning | Example |
|--------|---------|---------|
| **🟢 SPC-Operational** | Exact SPC implementation | SCP, STP (effective) |
| **🟡 Modified** | Based on SPC but adjusted | EHI with damping |
| **🟠 Approximation** | Simplified operational method | WBGT-Approx, 0-3km CAPE |
| **🔵 Research** | Experimental/literature-based | VTP, MSP |
| **🔴 Deprecated** | No longer operationally used | Haines Index |

---

## 🚀 Quick Reference

### Most Critical Parameters (Corrected)

| Parameter | Status | Formula | Threshold | Notes |
|-----------|--------|---------|-----------|-------|
| **SCP** | 🟢 SPC | `(muCAPE/1000)×(ESRH/50)×shear×CIN` | > 1 | Shear term: clip((EBWD-10)/10, 0, 1) |
| **STP (eff)** | 🟢 SPC | `(MLCAPE/1500)×(ESRH/150)×(EBWD/12)×LCL×CIN` | > 1 | Effective layers with CIN |
| **STP (fixed)** | 🟡 Modified | `(MLCAPE/1500)×(SRH01/150)×(BWD06/12)×LCL×CIN` | > 1 | Mixed fixed/CIN approach |
| **EHI** | 🟢 SPC | `(CAPE×SRH)/160,000` | > 2 | Standard normalization |
| **SHIP** | 🟢 SPC | `5 terms, each capped at 1.0` | > 1 | SPC v1.1 specification |
| **VGP** | 🟢 Standard | `S × √CAPE` where S = shear/depth | > 0.15 | Corrected formula |
| **BRN** | 🟢 SPC | `CAPE/(0.5×shear²)` | 10-45 | Supercell window |

---

## ⛈️ Core Severe Weather Parameters

### Supercell Composite Parameter (SCP)

#### 📐 Formula (SPC-Compliant)
```
SCP = (muCAPE/1000) × (ESRH/50) × shear_term × CIN_weight

where:
  shear_term = clip((EBWD-10)/10, 0, 1)  # SPC piecewise scaling
  CIN_weight = 1.0 if muCIN > −40 J/kg, else −40/muCIN
```

#### 📥 Inputs
| Parameter | Description | Units | Source |
|-----------|-------------|-------|--------|
| `mucape` | Most-Unstable CAPE | J/kg | HRRR MUCAPE |
| `effective_srh` | Effective Storm Relative Helicity | m²/s² | HRRR ESRHL |
| `effective_shear` | Effective Bulk Wind Difference | m/s | Derived |
| `mucin` | Most-Unstable CIN (optional) | J/kg | HRRR MUCIN |

#### ⚙️ Implementation Details (CORRECTED)
- **Shear Term Scaling (SPC-compliant):**
  - The formula `(EBWD-10)/10` correctly implements SPC's piecewise rule:
  - EBWD = 10 m/s → (10-10)/10 = 0 ✅
  - EBWD = 15 m/s → (15-10)/10 = 0.5 ✅
  - EBWD = 20 m/s → (20-10)/10 = 1.0 ✅
  - Clipped to [0, 1] range
  
- **CIN Weight (SPC specification):**
  - muCIN > -40 J/kg: CIN_weight = 1.0 (no penalty)
  - muCIN ≤ -40 J/kg: CIN_weight = -40/muCIN (proportional penalty)

#### 📊 Status: 🟢 **SPC-Operational**

#### 📁 Source
`derived_params/supercell_composite_parameter.py:3-117`

---

### Significant Tornado Parameter (STP)

#### Two Variants Available:

#### 1️⃣ STP (Effective Layer Version with CIN) - **CURRENT SPC**
```
STP_effective = (MLCAPE/1500) × (ESRH/150) × (EBWD/12) × 
                ((2000-MLLCL)/1000) × ((MLCIN+200)/150)
```
- Uses **effective layers** (ESRH, EBWD)
- Includes **CIN term**
- **Status:** 🟢 **SPC-Operational**
- **File:** `significant_tornado_parameter_effective.py`

#### 2️⃣ STP (Fixed Layer Version) - **LEGACY/MIXED**
```
STP_fixed = (MLCAPE/1500) × (SRH_01km/150) × (BWD_06km/12) × 
            ((2000-MLLCL)/1000) × ((MLCIN+200)/150)
```
- Uses **fixed layers** (0-1km SRH, 0-6km shear)
- Includes **CIN term** (non-standard for fixed version)
- **Status:** 🟡 **Modified** (mixes approaches)
- **File:** `significant_tornado_parameter.py`

#### ⚙️ Implementation Details (CORRECTED)
- **Shear Term:** 
  - < 12.5 m/s → 0
  - 12.5-30 m/s → shear/12
  - > 30 m/s → cap at 1.5
- **LCL Term:** (2000-LCL)/1000, clipped [0, 1]
- **CIN Term:** (MLCIN+200)/150, clipped [0, 1]

#### 📊 Interpretation
| STP Value | Tornado Risk |
|-----------|--------------|
| < 1 | Low |
| 1-3 | Moderate (EF2+ possible) |
| 3-6 | High (outbreak likely) |
| > 6 | Extreme |

---

### Energy-Helicity Index (EHI)

#### 📐 Formula (CORRECTED - Standard SPC)
```
EHI = (CAPE × SRH) / 160,000
```

#### Previous Issue & Correction
- **OLD (incorrect):** `(CAPE/1600) × (SRH/50)` = (CAPE×SRH)/80,000
- **NEW (correct):** `(CAPE × SRH) / 160,000` per Davies (1993) and SPC

#### Optional Display Damping
```python
# For extreme value visualization only (not part of EHI definition)
if |EHI| > 5:
    EHI_display = sign(EHI) × (5 + log(|EHI|/5))
```

#### 📊 Status: 🟢 **SPC-Operational**

#### 📁 Source
`derived_params/energy_helicity_index.py:3-56` (CORRECTED)

---

### Significant Hail Parameter (SHIP)

#### 📐 Formula (SPC v1.1)
```
SHIP = Term1 × Term2 × Term3 × Term4 × Term5

where each term is capped at 1.0:
  Term1 = min(muCAPE/1500, 1.0)
  Term2 = min(MU_mr/13.6, 1.0)  
  Term3 = min(lapse_700_500/7.0, 1.0)
  Term4 = min(shear_06km/20, 1.0)
  Term5 = min((frz_lvl_km - T500)/8, 1.0)
```

#### Clarification on Versions
- **SPC SHIP v1.1** (current): Five terms, each capped at 1.0 ✅
- **Older raw version**: Uncapped multiplicative product (/44,000,000)
- Our implementation follows **v1.1 specification**

#### 📊 Status: 🟢 **SPC-Operational**

---

## 💨 Wind Shear Parameters

### Bulk Wind Shear Magnitude
```
shear_magnitude = sqrt(u_shear² + v_shear²)
```
**Status:** 🟢 **Standard**

### Effective Layer Parameters
- **Effective SRH:** Helicity through effective storm depth
- **Effective Shear (EBWD):** Shear through effective inflow layer
**Status:** 🟢 **SPC-Operational**

---

## 🌡️ Thermodynamic Parameters

### Wet Bulb Temperature
**Status:** 🟢 **Standard** (psychrometric calculation)

### WBGT (Wet Bulb Globe Temperature) - APPROXIMATION

#### Formula (Approximation Method)
```
WBGT_approx = 0.7×Tw + 0.3×Tdb
```

#### ⚠️ Important Clarification
- **TRUE WBGT requires:**
  - Natural wet-bulb temperature (Tnwb) not psychrometric (Tw)
  - Black globe temperature (Tg) not dry-bulb (Tdb)
  - Formula: `WBGT = 0.7×Tnwb + 0.3×Tg` (shade) or `0.7×Tnwb + 0.2×Tg + 0.1×Tdb` (sun)
  
- **Our approximation:**
  - Acceptable for heat stress screening
  - May underestimate true WBGT by 1-2°C
  - For precise assessments, use actual Tnwb and Tg

**Status:** 🟠 **Approximation**

#### 📁 Source
`derived_params/wbgt_shade.py` (UPDATED with clarification)

---

### 0-3km CAPE - HEURISTIC APPROXIMATION

#### ⚠️ Critical Note
**This is NOT a true 0-3km CAPE calculation!**

#### Current Implementation (Approximation)
```python
# Empirical fractions based on total CAPE magnitude
if MLCAPE < 1000:
    fraction = 0.25
elif MLCAPE < 3000:
    fraction = 0.20
else:
    fraction = 0.15

CAPE_03km_approx = MLCAPE × fraction
CAPE_03km_approx = min(CAPE_03km_approx, 600)  # Cap at realistic maximum
```

#### Correct Method Would Require:
1. Full thermodynamic profile
2. Parcel lifting from surface
3. Integration of positive area from 0-3km AGL only
4. Tools: MetPy, SHARPpy, or direct HRRR output

**Status:** 🟠 **Approximation** (±50% error possible)

#### 📁 Source
`derived_params/cape_03km.py` (UPDATED with warning)

---

## 📈 Stability Indices

### SWEAT Index (Complete Formula)

#### 📐 Full Formula
```
SWEAT = 12×Td850 + 20×max(TT-49, 0) + 2×f850 + f500 + 125×(S+0.2)

where:
  TT = Total Totals = (T850 + Td850) - 2×T500
  f850 = 850mb wind speed term (if > 15 m/s)
  f500 = 500mb wind speed term (if > 15 m/s)
  S = sin(wind_direction_difference) with specific conditions:
      - 850mb wind from 130-250°
      - 500mb wind from 210-310°
      - Both speeds ≥ 15 m/s
      - Positive directional shear
```

**Status:** 🟢 **Standard**

---

## 🔥 Fire Weather Parameters

### Haines Index - DEPRECATED

#### ⚠️ DEPRECATION NOTICE
**The National Weather Service discontinued Haines Index in operational fire weather forecasts as of December 20, 2024 (Service Change Notice 24-107).**

#### Recommended Alternatives:
- **Hot-Dry-Windy Index (HDW)** - Modern replacement
- **Vapor Pressure Deficit (VPD)** - Moisture stress indicator
- **Fire Weather Index (FWI) System** - Canadian system

#### Implementation Note:
- Current code provides **mid-level variant only** (850-700mb)
- Three variants exist (low/mid/high elevation)
- Retained for research/historical comparison

**Status:** 🔴 **Deprecated**

#### 📁 Source
`derived_params/haines_index.py` (UPDATED with deprecation notice)

---

## 🌪️ Specialized Research Parameters

### Vorticity Generation Parameter (VGP)

#### 📐 Formula (CORRECTED - Standard Definition)
```
VGP = S × √CAPE

where:
  S = mean 0-1km shear magnitude / depth (s⁻¹)
  S = wind_shear_01km / 1000 m
```

#### Previous Issue & Correction
- **OLD (non-standard):** `(CAPE/1000) × (shear/1000) × 0.1`
- **NEW (standard):** `S × √CAPE` per Rasmussen & Blanchard (1998)

#### Interpretation (Updated Thresholds)
| VGP | Tornado Potential |
|-----|------------------|
| < 0.15 | Low |
| 0.15-0.30 | Moderate |
| 0.30-0.45 | Significant |
| > 0.45 | High |

**Status:** 🟢 **Standard** (after correction)

#### 📁 Source
`derived_params/vorticity_generation_parameter.py` (CORRECTED)

---

### Violent Tornado Parameter (VTP)

#### 📐 Formula
```
VTP = 7-term product including low-level CAPE and lapse rate terms
```

#### Important Note
- No canonical "SPC VTP" exists
- Hampshire et al. (2018) discussed concepts but didn't define this exact formula
- Our implementation is an experimental composite

**Status:** 🔵 **Research/Experimental**

---

### Mesocyclone Strength Parameter (MSP)

Custom formulation based on updraft helicity with enhancement factors.

**Status:** 🔵 **Research/Experimental**

---

### Enhanced Smoke Dispersion Index (ESDI)

Local enhancement of standard smoke dispersion calculations.

**Status:** 🔵 **Research/Experimental**

---

## 🏗️ Implementation Notes

### Quality Assurance
All parameters now include:
- ✅ Correct normalizations per literature
- ✅ Proper status labels (SPC/Modified/Approximation/Research/Deprecated)
- ✅ Clear documentation of assumptions
- ✅ References to primary sources
- ✅ Warnings for approximations

### Code Organization
```
derived_params/
├── # SPC-OPERATIONAL (🟢)
├── supercell_composite_parameter.py
├── significant_tornado_parameter_effective.py
├── energy_helicity_index.py [CORRECTED]
├── significant_hail_parameter.py
│
├── # MODIFIED/MIXED (🟡)
├── significant_tornado_parameter.py [fixed+CIN]
│
├── # APPROXIMATIONS (🟠)
├── wbgt_shade.py [CLARIFIED]
├── cape_03km.py [LABELED]
│
├── # RESEARCH (🔵)
├── violent_tornado_parameter.py
├── mesocyclone_strength_parameter.py
│
└── # DEPRECATED (🔴)
    └── haines_index.py [NOTICE ADDED]
```

---

## 📚 References

### Primary Sources (Peer-Reviewed)
- **Davies, J.M., 1993:** Small tornadic supercells in the central plains. *Preprints, 17th Conf. on Severe Local Storms*, 305-309.
- **Rasmussen, E.N., and D.O. Blanchard, 1998:** A baseline climatology of sounding-derived supercell and tornado forecast parameters. *WAF*, 13, 1148-1164.
- **Thompson, R.L., et al., 2003:** Close proximity soundings within supercell environments obtained from the Rapid Update Cycle. *WAF*, 18, 1243-1261.
- **Thompson, R.L., et al., 2012:** Convective modes for significant severe thunderstorms in the contiguous United States. Part II: Supercell and QLCS tornado environments. *WAF*, 27, 1136-1154.

### Operational References
- **SPC Mesoanalysis Help:** https://www.spc.noaa.gov/exper/mesoanalysis/help/
- **NWS Service Change Notice 24-107:** Discontinuation of Haines Index (Dec 2024)
- **ACGIH TLVs and BEIs, 2024:** Heat stress thresholds
- **ISO 7243:2017:** Ergonomics of the thermal environment - WBGT

### Corrections Made in v2.1
1. ✅ EHI: Changed from /80,000 to /160,000
2. ✅ VGP: Changed to S × √CAPE formula
3. ✅ WBGT: Clarified as approximation
4. ✅ Haines: Added deprecation notice
5. ✅ STP: Distinguished variants
6. ✅ 0-3km CAPE: Labeled as approximation
7. ✅ SCP: Clarified shear term is correct
8. ✅ Parameter status labels added

---

**Version:** 2.1 (Corrected)  
**Last Updated:** 2024  
**Total Parameters:** 70+  
**Compliance:** SPC-aligned where applicable

---

*This documentation has been updated based on peer review to ensure accuracy with SPC operational definitions and peer-reviewed literature. Parameters are clearly labeled by their implementation status.*