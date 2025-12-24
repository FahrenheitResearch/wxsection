"""SPC-style colormaps for weather visualization"""

from matplotlib.colors import LinearSegmentedColormap


def create_all_colormaps():
    """Create SPC-style colormaps"""
    colormaps = {}
    
    # NWS Reflectivity
    ref_colors = ['#646464', '#04e9e7', '#019ff4', '#0300f4', '#02fd02',
                  '#01c501', '#008e00', '#fdf802', '#e5bc00', '#fd9500',
                  '#fd0000', '#d40000', '#bc0000', '#f800fd', '#9854c6']
    colormaps['NWSReflectivity'] = LinearSegmentedColormap.from_list('NWSRef', ref_colors)
    
    # CAPE - White to Yellow to Red (white background for zero CAPE)
    cape_colors = ['#ffffff', '#fffaf0', '#fff5e6', '#ffeecc', '#ffe6b3', '#ffdd99', '#ffcc66', '#ff9933', '#ff6600', '#ff3300', '#cc0000']
    colormaps['CAPE'] = LinearSegmentedColormap.from_list('CAPE', cape_colors)
    
    # CIN - Deep blue for strong CIN (most negative) to white for no CIN (zero)
    cin_colors = ['#0000ff', '#3333ff', '#6666ff', '#9999ff', '#b3b3ff', '#ccccff', '#e6e6ff', '#f0f0ff', '#fafafa', '#ffffff']
    colormaps['CIN'] = LinearSegmentedColormap.from_list('CIN', cin_colors)
    
    # Lifted Index - Blue to Red
    li_colors = ['#0000ff', '#4169e1', '#87ceeb', '#f0f8ff', '#ffffff', '#ffe4e1', '#ffa07a', '#ff4500', '#ff0000']
    colormaps['LiftedIndex'] = LinearSegmentedColormap.from_list('LiftedIndex', li_colors)
    
    # Hail - Green to Purple
    hail_colors = ['#00ff00', '#32cd32', '#9acd32', '#ffff00', '#ffa500', '#ff4500', '#ff0000', '#800080']
    colormaps['Hail'] = LinearSegmentedColormap.from_list('Hail', hail_colors)
    
    # NOAA Smoke - Light Blue to Blue to Green to Yellow to Orange to Red to Purple (less transparent)
    smoke_colors = ['#e6f3ff', '#87ceeb', '#90ee90', '#ffff00', '#ffa500', '#ff4500', '#ff0000', '#800080']
    colormaps['NOAASmoke'] = LinearSegmentedColormap.from_list('NOAASmoke', smoke_colors)
    
    # Tornado Diagnostics Colormaps
    
    # STP colormap - light below STP=1, then increasingly warm colors for higher values
    stp_colors = ['#f7f7f7', '#e0e0e0', '#cccccc', '#ffeda0', '#fed976', '#feb24c', 
                 '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026', '#800026']
    colormaps['STP'] = LinearSegmentedColormap.from_list('STP', stp_colors)
    
    # SCP colormap - white background transitioning to warm colors for positive values
    scp_colors = ['#ffffff', '#f7f7f7', '#f0f0f0', '#ffe6e6', '#ffcccc', 
                 '#ffaaaa', '#ff8888', '#ff6666', '#ff4444', '#ff2222', '#ff0000']
    colormaps['SCP'] = LinearSegmentedColormap.from_list('SCP', scp_colors)
    
    # EHI colormap - white background transitioning to warm colors for positive values
    ehi_colors = ['#ffffff', '#fff5f5', '#ffeeee', '#ffd4d4', '#ffb8b8', 
                 '#ff9999', '#ff7777', '#ff5555', '#ff3333', '#ff0000', '#cc0000']
    colormaps['EHI'] = LinearSegmentedColormap.from_list('EHI', ehi_colors)
    
    # Storm-Relative Helicity - red scales for rotation
    srh_colors = ['#ffeeee', '#ffcccc', '#ffaaaa', '#ff8888', '#ff6666', 
                 '#ff4444', '#ff2222', '#ff0000', '#dd0000', '#bb0000', '#990000']
    colormaps['SRH'] = LinearSegmentedColormap.from_list('SRH', srh_colors)
    
    # Wind Shear - viridis-like for wind magnitude
    shear_colors = ['#440154', '#482777', '#3f4a8a', '#31678e', '#26838f', 
                   '#1f9d8a', '#6cce5a', '#b6de2b', '#fee825']
    colormaps['WindShear'] = LinearSegmentedColormap.from_list('WindShear', shear_colors)
    
    # BRN colormap - emphasizes the 10-45 "supercell sweet spot"
    # Light blue for extreme shear (<10), green-yellow for supercells (10-45), orange-red for weak shear (>50)
    brn_colors = ['#08306b', '#4292c6', '#9ecae1', '#41ab5d', '#78c679', '#addd8e', 
                 '#fee391', '#fec44f', '#fe9929', '#ec7014', '#cc4c02']
    colormaps['BRN'] = LinearSegmentedColormap.from_list('BRN', brn_colors)
    
    # LCL colormap - inverted (low LCL = good = green/blue, high LCL = bad = yellow/red)
    lcl_colors = ['#004529', '#238b45', '#41ab5d', '#74c476', '#a1d99b', 
                 '#c7e9c0', '#edf8e9', '#fff7ec', '#fee8c8', '#fdd49e', 
                 '#fdbb84', '#fc8d59', '#ef6548', '#d7301f', '#990000']
    colormaps['LCL'] = LinearSegmentedColormap.from_list('LCL', lcl_colors)
    
    # Personality Composite Colormaps
    
    # Seqouigrove Weird-West Composite - Desert vibes with moisture pop
    # Blue (boring dry) -> White (neutral) -> Yellow/Orange (getting interesting) -> Red/Purple (peak weirdness)
    sw2c_colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7', 
                  '#fdbf6f', '#ff7f00', '#e31a1c', '#b10026', '#67001f']
    colormaps['SeqouigroveWeirdWest'] = LinearSegmentedColormap.from_list('SW2C', sw2c_colors)
    
    # K-MAX Composite - Pure hype: everything-to-eleven
    # Green (meh) -> Yellow (getting spicy) -> Orange (serious business) -> Red (HYPE) -> Purple (legendary)
    kmax_colors = ['#1a9850', '#66bd63', '#a6d96a', '#d9ef8b', '#fee08b', 
                  '#fdae61', '#f46d43', '#d73027', '#a50026', '#762a83']
    colormaps['KazooMAXX'] = LinearSegmentedColormap.from_list('KMAX', kmax_colors)
    
    # Seqouiagrove Thermal Range - Diurnal temperature spread vibes
    # Cool blue (small range) -> Warm amber (moderate) -> Hot orange/red (big swings) -> Deep crimson (epic ranges)
    thermal_colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#fdbf6f', 
                     '#fd8d3c', '#e31a1c', '#bd0026', '#800026', '#67001f']
    colormaps['SeqouiagroveThermal'] = LinearSegmentedColormap.from_list('SThermal', thermal_colors)
    
    # Destroyer Reality-Check - Anti-hype truth colormap
    # Dark gray (hype/bust) -> Red (weak) -> Orange (marginal) -> Yellow (decent) -> Green (legit) -> Bright green (chase-worthy)
    destroyer_colors = ['#2b2b2b', '#636363', '#969696', '#cc4c02', '#fd8d3c', 
                       '#fecc5c', '#ffffb2', '#c7e9b4', '#7fcdbb', '#2c7fb8']
    colormaps['DestroyerReality'] = LinearSegmentedColormap.from_list('DReality', destroyer_colors)
    
    # Samuel Outflow Propensity - Cold pool science colormap  
    # Light blue (weak outflow) -> White (marginal) -> Yellow (moderate) -> Orange (strong) -> Red (violent outflow) -> Dark red (gust front city)
    samuel_colors = ['#c6dbef', '#9ecae1', '#6baed6', '#f7f7f7', '#fee391', 
                    '#fec44f', '#fe9929', '#ec7014', '#cc4c02', '#8c2d04']
    colormaps['SamuelOutflow'] = LinearSegmentedColormap.from_list('SOutflow', samuel_colors)
    
    # Mason-Flappity Bayou Buzz - Gulf Coast high-impact weather composite
    # Navy (synoptically benign) -> Blue (seabreeze) -> Teal (ordinary storms) -> Yellow (watch-worthy) -> Orange (enhanced severe) -> Red (high-impact events) -> Deep red (hurricane/historic)
    mf_buzz_colors = ['#08306b', '#2171b5', '#4292c6', '#6baed6', '#9ecae1',
                     '#fee391', '#fec44f', '#fe9929', '#d94701', '#a63603']
    colormaps['MasonFlappityBuzz'] = LinearSegmentedColormap.from_list('MFBuzz', mf_buzz_colors)
    
    # Wind Direction - Circular colormap that wraps properly
    # North=Red, East=Yellow, South=Green, West=Blue, back to North=Red
    wind_dir_colors = ['#ff0000', '#ff4000', '#ff8000', '#ffbf00', '#ffff00',  # N to E (red to yellow)
                      '#bfff00', '#80ff00', '#40ff00', '#00ff00',  # E to S (yellow to green)
                      '#00ff40', '#00ff80', '#00ffbf', '#00ffff',  # S to SW (green to cyan)
                      '#00bfff', '#0080ff', '#0040ff', '#0000ff',  # SW to W (cyan to blue)
                      '#4000ff', '#8000ff', '#bf00ff', '#ff00ff',  # W to NW (blue to magenta)
                      '#ff00bf', '#ff0080', '#ff0040', '#ff0000']  # NW to N (magenta to red)
    colormaps['WindDirection'] = LinearSegmentedColormap.from_list('WindDir', wind_dir_colors)
    
    # Wind Gust - White/transparent for calm, visible colors only for significant gusts
    wind_gust_colors = ['#ffffff', '#f8f8f8', '#f0f0f0',  # 0-15 m/s: nearly invisible
                       '#c6e9c6', '#90d490', '#5fbf5f',  # 15-25 m/s: light green (breezy)
                       '#ffeb3b', '#ffc107', '#ff9800',  # 25-35 m/s: yellow to orange (strong)
                       '#ff5722', '#f44336', '#d32f2f',  # 35-45 m/s: orange to red (severe)
                       '#b71c1c', '#880e4f', '#4a148c']  # 45+ m/s: dark red to purple (extreme)
    colormaps['WindGust'] = LinearSegmentedColormap.from_list('WindGust', wind_gust_colors)

    # ============================================================
    # DIURNAL TEMPERATURE COLORMAPS
    # ============================================================

    # Diurnal Range - Cool blue (small range) -> Warm amber/orange (moderate) -> Hot red (large swings)
    # Small DTR = humid/cloudy, Large DTR = arid/clear
    diurnal_range_colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6',  # 0-10°C: blues (small range)
                           '#fef0d9', '#fdd49e', '#fdbb84', '#fc8d59',  # 10-20°C: oranges (moderate)
                           '#e34a33', '#b30000', '#7f0000']  # 20-30+°C: reds (extreme range)
    colormaps['DiurnalRange'] = LinearSegmentedColormap.from_list('DiurnalRange', diurnal_range_colors)

    # Heating Rate - White/light yellow (slow) -> Orange (moderate) -> Red/Magenta (rapid heating)
    # Rapid heating = strong insolation, fire weather concern, convective destabilization
    heating_rate_colors = ['#ffffcc', '#ffeda0', '#fed976', '#feb24c',  # 0-2°C/hr: yellows (gentle)
                          '#fd8d3c', '#fc4e2a', '#e31a1c',  # 2-4°C/hr: oranges (moderate)
                          '#bd0026', '#800026', '#67001f']  # 4+°C/hr: reds (rapid)
    colormaps['HeatingRate'] = LinearSegmentedColormap.from_list('HeatingRate', heating_rate_colors)

    # Cooling Rate - Light cyan (slow) -> Blue (moderate) -> Deep blue/purple (rapid cooling)
    # Rapid cooling = clear skies, radiational cooling, frost/fog potential
    cooling_rate_colors = ['#f7fcf0', '#e0f3db', '#ccebc5', '#a8ddb5',  # 0-2°C/hr: light greens (gentle)
                          '#7bccc4', '#4eb3d3', '#2b8cbe',  # 2-4°C/hr: teals/blues (moderate)
                          '#0868ac', '#084081', '#081d58']  # 4+°C/hr: deep blues (rapid)
    colormaps['CoolingRate'] = LinearSegmentedColormap.from_list('CoolingRate', cooling_rate_colors)

    return colormaps