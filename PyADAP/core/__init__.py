"""Core modules for PyADAP 3.0

This package contains the core functionality for data management,
statistical analysis, and pipeline orchestration.
"""

from .data_manager import DataManager
from .statistical_analyzer import StatisticalAnalyzer
from .pipeline import AnalysisPipeline

# Alias for backward compatibility
Pipeline = AnalysisPipeline

__all__ = ['DataManager', 'StatisticalAnalyzer', 'AnalysisPipeline', 'Pipeline']