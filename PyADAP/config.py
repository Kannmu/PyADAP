"""Configuration Management for PyADAP 3.0

This module provides centralized configuration management with intelligent defaults
and validation for statistical analysis parameters.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from pathlib import Path


@dataclass
class StatisticalConfig:
    """Statistical analysis configuration."""
    alpha: float = 0.05
    power: float = 0.8
    min_effect_size: float = 0.2
    effect_size_thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'cohens_d': {'small': 0.2, 'medium': 0.5, 'large': 0.8},
        'eta_squared': {'small': 0.01, 'medium': 0.06, 'large': 0.14},
        'omega_squared': {'small': 0.01, 'medium': 0.06, 'large': 0.14}
    })
    auto_test_selection: bool = True
    prefer_nonparametric: bool = False
    use_robust_tests: bool = False
    multiple_comparisons_method: str = 'bonferroni'  # 'bonferroni', 'holm', 'fdr_bh', 'fdr_by'
    check_normality: bool = True
    check_homogeneity: bool = True
    check_independence: bool = True
    use_bootstrap: bool = False
    bootstrap_n_samples: int = 1000
    normality_tests: List[str] = field(default_factory=lambda: ['shapiro', 'anderson', 'kstest'])
    sphericity_correction: str = 'greenhouse_geisser'  # 'greenhouse_geisser', 'huynh_feldt', 'lower_bound'
    multiple_comparisons: str = 'bonferroni'  # 'bonferroni', 'holm', 'fdr_bh', 'fdr_by'
    outlier_method: str = 'iqr'  # 'iqr', 'zscore', 'modified_zscore', 'isolation_forest'
    outlier_threshold: float = 3.0
    
    # Backward compatibility property
    @property
    def alpha_level(self) -> float:
        """Backward compatibility for alpha_level."""
        return self.alpha
    
    @alpha_level.setter
    def alpha_level(self, value: float) -> None:
        """Backward compatibility setter."""
        self.alpha = value
    

@dataclass
class DataConfig:
    """Data processing configuration."""
    auto_detect_types: bool = True
    # Missing data handling
    handle_missing: str = 'auto'  # 'auto', 'drop', 'impute', 'flag'
    missing_threshold: float = 0.1  # Drop variables with >10% missing
    # Outlier detection and handling
    detect_outliers: bool = True
    outlier_action: str = 'flag'  # 'flag', 'remove', 'transform'
    # Data transformation
    auto_transform: bool = False
    transform_method: str = 'auto'  # 'auto', 'log', 'sqrt', 'boxcox', 'yeo_johnson', 'none'
    # Data scaling/normalization
    auto_scaling: bool = False
    scaling_method: str = 'zscore'  # 'zscore', 'minmax', 'robust', 'none'
    # Categorical encoding
    encoding_categorical: str = 'auto'  # 'auto', 'dummy', 'effect', 'ordinal'
    
    # Properties for backward compatibility
    @property
    def missing_data_strategy(self) -> str:
        """Backward compatibility for missing_data_strategy."""
        return self.handle_missing
    
    @missing_data_strategy.setter
    def missing_data_strategy(self, value: str) -> None:
        """Backward compatibility setter."""
        self.handle_missing = value
    
    @property
    def missing_data_threshold(self) -> float:
        """Backward compatibility for missing_data_threshold."""
        return self.missing_threshold
    
    @missing_data_threshold.setter
    def missing_data_threshold(self, value: float) -> None:
        """Backward compatibility setter."""
        self.missing_threshold = value
    
    @property
    def outlier_method(self) -> str:
        """Get outlier method from statistical config."""
        # This will be accessed from statistical config
        return 'iqr'  # Default fallback
    
    @outlier_method.setter
    def outlier_method(self, value: str) -> None:
        """Set outlier method (for backward compatibility)."""
        # Note: This is a compatibility setter, actual outlier_method is in StatisticalConfig
        pass
    
    @property
    def outlier_threshold(self) -> float:
        """Get outlier threshold from statistical config."""
        # This will be accessed from statistical config
        return 3.0  # Default fallback
    
    @outlier_threshold.setter
    def outlier_threshold(self, value: float) -> None:
        """Set outlier threshold (for backward compatibility)."""
        # Note: This is a compatibility setter, actual outlier_threshold is in StatisticalConfig
        pass
    
    @property
    def normalization_method(self) -> str:
        """Backward compatibility for normalization_method."""
        return self.scaling_method
    
    @normalization_method.setter
    def normalization_method(self, value: str) -> None:
        """Backward compatibility setter."""
        self.scaling_method = value
    
    @property
    def transformation_method(self) -> str:
        """Backward compatibility for transformation_method."""
        return self.transform_method
    
    @transformation_method.setter
    def transformation_method(self, value: str) -> None:
        """Backward compatibility setter."""
        self.transform_method = value
    

@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    # Plot generation control
    generate_plots: bool = True
    # Plot style and appearance
    style: str = 'seaborn-v0_8-whitegrid'
    palette: str = 'husl'
    figure_size: tuple = (10, 6)
    dpi: int = 300
    font_family: str = 'Arial'
    font_size: int = 12
    # Plot display options
    show_plots: bool = True
    show_grid: bool = True
    show_legend: bool = True
    tight_layout: bool = True
    # Plot saving options
    save_plots: bool = True
    save_format: str = 'png'  # 'png', 'pdf', 'svg'
    
    # Properties for backward compatibility
    @property
    def color_palette(self) -> str:
        """Backward compatibility for color_palette."""
        return self.palette
    
    @color_palette.setter
    def color_palette(self, value: str) -> None:
        """Backward compatibility setter."""
        self.palette = value
    
    @property
    def plot_format(self) -> str:
        """Backward compatibility for plot_format."""
        return self.save_format
    
    @plot_format.setter
    def plot_format(self, value: str) -> None:
        """Backward compatibility setter."""
        self.save_format = value
    

@dataclass
class OutputConfig:
    """Output configuration."""
    create_report: bool = True
    report_format: str = 'html'  # 'html', 'pdf', 'docx'
    save_data: bool = True
    save_plots: bool = True
    output_precision: int = 4
    include_raw_output: bool = False
    output_directory: Path = field(default_factory=lambda: Path.cwd() / 'results')
    create_subdirectory: bool = True
    file_prefix: str = 'analysis'
    include_timestamp: bool = True
    export_excel: bool = True
    export_csv: bool = True
    export_html: bool = True
    export_pdf: bool = False
    enable_logging: bool = True
    log_level: str = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    log_to_file: bool = True
    

class Config:
    """Main configuration class for PyADAP 3.0."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_file: Path to configuration file (JSON format)
        """
        self.statistical = StatisticalConfig()
        self.data = DataConfig()
        self.visualization = VisualizationConfig()
        self.output = OutputConfig()
        
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: str) -> None:
        """Load configuration from JSON file.
        
        Args:
            config_file: Path to configuration file
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Update configurations
            if 'statistical' in config_data:
                self._update_config(self.statistical, config_data['statistical'])
            if 'data' in config_data:
                self._update_config(self.data, config_data['data'])
            if 'visualization' in config_data:
                self._update_config(self.visualization, config_data['visualization'])
            if 'output' in config_data:
                self._update_config(self.output, config_data['output'])
                
        except Exception as e:
            print(f"Warning: Could not load configuration file {config_file}: {e}")
    
    def save_to_file(self, config_file: str) -> None:
        """Save configuration to JSON file.
        
        Args:
            config_file: Path to save configuration file
        """
        config_data = {
            'statistical': self._config_to_dict(self.statistical),
            'data': self._config_to_dict(self.data),
            'visualization': self._config_to_dict(self.visualization),
            'output': self._config_to_dict(self.output)
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    def _update_config(self, config_obj, config_dict: dict) -> None:
        """Update configuration object with dictionary values."""
        for key, value in config_dict.items():
            if hasattr(config_obj, key):
                # Handle Path objects
                if key == 'output_directory' and isinstance(value, str):
                    setattr(config_obj, key, Path(value))
                # Handle tuples (like figure_size)
                elif key == 'figure_size' and isinstance(value, list):
                    setattr(config_obj, key, tuple(value))
                else:
                    setattr(config_obj, key, value)
    
    def _config_to_dict(self, config_obj) -> dict:
        """Convert configuration object to dictionary."""
        if hasattr(config_obj, '__dict__'):
            result = {}
            for key, value in config_obj.__dict__.items():
                # Handle Path objects
                if isinstance(value, Path):
                    result[key] = str(value)
                # Handle tuples
                elif isinstance(value, tuple):
                    result[key] = list(value)
                else:
                    result[key] = value
            return result
        return {}
    
    def copy(self) -> 'Config':
        """Create a copy of the configuration.
        
        Returns:
            New Config instance with copied values
        """
        import copy as copy_module
        new_config = Config()
        new_config.statistical = copy_module.deepcopy(self.statistical)
        new_config.data = copy_module.deepcopy(self.data)
        new_config.visualization = copy_module.deepcopy(self.visualization)
        new_config.output = copy_module.deepcopy(self.output)
        return new_config
    
    def validate(self) -> List[str]:
        """Validate configuration settings.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate statistical config
        if not 0 < self.statistical.alpha < 1:
            errors.append("Alpha must be between 0 and 1")
        
        if not 0 < self.statistical.power < 1:
            errors.append("Power must be between 0 and 1")
        
        if self.statistical.outlier_threshold <= 0:
            errors.append("Outlier threshold must be positive")
        
        # Validate data config
        if not 0 <= self.data.missing_threshold <= 1:
            errors.append("Missing threshold must be between 0 and 1")
        
        # Validate visualization config
        if self.visualization.dpi <= 0:
            errors.append("DPI must be positive")
        
        if self.visualization.font_size <= 0:
            errors.append("Font size must be positive")
        
        # Validate output config
        if self.output.output_precision < 0:
            errors.append("Output precision must be non-negative")
        
        return errors
    
    def get_default_config_path(self) -> str:
        """Get default configuration file path."""
        home_dir = Path.home()
        config_dir = home_dir / '.pyadap'
        config_dir.mkdir(exist_ok=True)
        return str(config_dir / 'config.json')
    
    def reset_to_defaults(self) -> None:
        """Reset all configurations to default values."""
        self.statistical = StatisticalConfig()
        self.data = DataConfig()
        self.visualization = VisualizationConfig()
        self.output = OutputConfig()


# Global default configuration instance
default_config = Config()