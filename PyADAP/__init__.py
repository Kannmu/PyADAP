"""PyADAP 3.0
=============

PyADAP: Python Automated Data Analysis Pipeline - Version 3.0

A comprehensive, automated statistical analysis pipeline with enhanced scientific rigor
and intelligent automation capabilities.

Key Features in 3.0:
- Enhanced statistical analysis with automatic assumption checking
- Intelligent test selection based on data characteristics
- Advanced data preprocessing and outlier detection
- Comprehensive effect size calculations
- Automated report generation with statistical interpretations
- Modern GUI with real-time feedback
- Robust error handling and validation

Author: Kannmu
Date: 2024/12/19
License: MIT License
Repository: https://github.com/Kannmu/PyADAP
"""

__version__ = "3.0.0"
__author__ = "Kannmu"
__email__ = "kannmu@163.com"
__license__ = "MIT"

from .core import DataManager, StatisticalAnalyzer, Pipeline
from .gui import MainWindow
from .visualization import Plotter
from .utils import Logger, Validator
from .config import Config

# Aliases for backward compatibility
ModernInterface = MainWindow
AdvancedPlotter = Plotter

# Main pipeline function for backward compatibility
def run_analysis(data_path: str, **kwargs):
    """Run complete statistical analysis pipeline.
    
    Args:
        data_path: Path to data file
        **kwargs: Additional configuration options
        
    Returns:
        Analysis results and generated reports
    """
    pipeline = Pipeline()
    return pipeline.run(data_path, **kwargs)

__all__ = [
    'DataManager',
    'StatisticalAnalyzer', 
    'Pipeline',
    'MainWindow',
    'Plotter',
    'ModernInterface',  # Alias for MainWindow
    'AdvancedPlotter',  # Alias for Plotter
    'Logger',
    'Validator',
    'Config',
    'run_analysis'
]