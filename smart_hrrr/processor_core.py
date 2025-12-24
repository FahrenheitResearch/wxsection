# Slim HRRRProcessor; delegates heavy pieces into smart_hrrr.derived

import time
from pathlib import Path
from typing import Dict, Any, Optional
import xarray as xr

from field_registry import FieldRegistry
from model_config import get_model_registry
from config.colormaps import create_all_colormaps
from core import downloader, grib_loader, plotting
from . import derived as derived_mod


class HRRRProcessor:
    """Weather data processor with extensible field configurations
    
    Supports multiple weather models including HRRR and RRFS
    """
    
    def __init__(self, config_dir: Path = None, model: str = 'hrrr'):
        """Initialize weather processor
        
        Args:
            config_dir: Directory containing parameter configuration files
            model: Weather model to use ('hrrr' or 'rrfs')
        """
        self.registry = FieldRegistry(config_dir)
        self.colormaps = create_all_colormaps()
        # Hardcoded CONUS region since regional processing is removed
        self.regions = {
            'conus': {
                'name': 'CONUS',
                'extent': [-130, -65, 20, 50],
                'barb_thinning': 60
            }
        }
        self.current_region = 'conus'  # Fixed region
        # NEW: simple in-memory cache for the current cycle/FHR
        self._data_cache: Dict[tuple[str, str, int], xr.DataArray] = {}
        
        # Model configuration
        self.model_registry = get_model_registry()
        self.model_name = model.lower()
        self.model_config = self.model_registry.get_model(self.model_name)
        
        if not self.model_config:
            raise ValueError(f"Unknown model: {model}. Available models: {self.model_registry.list_models()}")

    def set_region(self, region_name: str) -> bool:
        """Set the current region for plotting (only conus supported now)"""
        self.current_region = 'conus'
        return True

    def download_model_file(self, cycle, forecast_hour, output_dir, file_type='pressure'):
        """Download weather model file with appropriate source fallbacks"""
        return downloader.download_model_file(cycle, forecast_hour, output_dir, file_type, self.model_config)
    
    def download_hrrr_file(self, cycle, forecast_hour, output_dir, file_type='wrfprs'):
        """Legacy method for backward compatibility - redirects to download_model_file"""
        return downloader.download_hrrr_file(cycle, forecast_hour, output_dir, file_type, self.model_config)

    def load_field_data(self, grib_file, field_name, field_config):
        """Load specific field data - now uses robust multi-dataset approach when needed"""
        return grib_loader.load_field(grib_file, field_name, field_config, self.model_name)
    
    def load_uh_layer(self, path, top, bottom):
        """Return max-1h UH for a given AG layer (m AGL) from a HRRR wrfsfc file."""
        return grib_loader.load_uh_layer(path, top, bottom)

    # Delegate heavy methods to smart_hrrr.derived
    def load_derived_parameter(self, field_name, field_config, grib_file, wrfsfc_file=None):
        """Load and compute derived parameter from input fields"""
        return derived_mod.load_derived_parameter(self, field_name, field_config, grib_file, wrfsfc_file)

    def _load_composite_data(self, field_name, field_config, grib_file, wrfsfc_file=None):
        """Load data for composite plots that need multiple input fields"""
        return derived_mod.load_composite_data(self, field_name, field_config, grib_file, wrfsfc_file)

    def create_spc_plot(self, data, field_name, field_config, cycle, forecast_hour, output_dir: Path):
        """Create enhanced SPC-style plot with comprehensive metadata"""
        return plotting.create_plot(data, field_name, field_config, cycle, forecast_hour, output_dir,
                                   self.regions, self.current_region, self.colormaps, self.registry)

    def process_fields(self, fields_to_process, cycle, forecast_hour, output_dir):
        """Process a list of fields for a given cycle and forecast hour

        Args:
            fields_to_process: List of field names to process
            cycle: Model cycle datetime object
            forecast_hour: Forecast hour integer
            output_dir: Output directory path
        """
        from pathlib import Path
        import logging

        logger = logging.getLogger(__name__)
        output_dir = Path(output_dir)

        # Determine GRIB directory - check parent if we're in a category subdirectory
        # GRIB files should be in the F## directory, not category subdirectories
        grib_dir = output_dir
        if output_dir.parent.name.startswith('F'):
            # We're in a category subdirectory like F00/severe/, look for GRIB in F00/
            grib_dir = output_dir.parent

        # Find or download GRIB files
        grib_files = {}
        for file_type in ["wrfprs", "wrfsfc"]:
            try:
                # First check if GRIB file already exists in grib_dir
                grib_pattern = f"*{file_type}*.grib2"
                existing = list(grib_dir.glob(grib_pattern))
                if existing:
                    grib_files[file_type] = existing[0]
                    continue

                # Skip download if use_local_grib is set
                if getattr(self, 'use_local_grib', False):
                    logger.debug(f"use_local_grib set, skipping download for {file_type}")
                    continue

                # Download to grib_dir (not category subdirectory)
                grib_file = self.download_model_file(cycle, forecast_hour, grib_dir, file_type)
                if grib_file and grib_file.exists():
                    grib_files[file_type] = grib_file
            except Exception as e:
                logger.warning(f"Failed to get {file_type}: {e}")
        
        if not grib_files:
            raise RuntimeError("No GRIB files available for processing")
        
        # Process each field
        processed_count = 0
        failed_fields = []
        
        for field_name in fields_to_process:
            try:
                # Get field configuration
                field_config = self.registry.get_field_config(field_name)
                if not field_config:
                    logger.error(f"No configuration found for field: {field_name}")
                    failed_fields.append(field_name)
                    continue
                
                # Load field data
                if field_config.get('derived'):
                    # Derived parameter
                    data = self.load_derived_parameter(field_name, field_config, 
                                                     grib_files.get('wrfprs'), 
                                                     grib_files.get('wrfsfc'))
                else:
                    # Direct GRIB field
                    primary_file = grib_files.get('wrfprs') or grib_files.get('wrfsfc')
                    if not primary_file:
                        raise ValueError(f"No suitable GRIB file for {field_name}")
                    data = self.load_field_data(primary_file, field_name, field_config)
                
                if data is None:
                    logger.error(f"Failed to load data for {field_name}")
                    failed_fields.append(field_name)
                    continue
                
                # Create plot
                self.create_spc_plot(data, field_name, field_config, cycle, forecast_hour, output_dir)
                processed_count += 1
                logger.info(f"✓ Processed {field_name}")
                
            except Exception as e:
                logger.error(f"✗ Failed to process {field_name}: {e}")
                failed_fields.append(field_name)
        
        logger.info(f"Processed {processed_count}/{len(fields_to_process)} fields")
        if failed_fields:
            logger.warning(f"Failed fields: {', '.join(failed_fields)}")
        
        return {
            'processed': processed_count,
            'failed': failed_fields,
            'total': len(fields_to_process)
        }