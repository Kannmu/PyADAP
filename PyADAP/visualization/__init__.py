"""Visualization module for PyADAP 3.0

This module provides comprehensive visualization capabilities for statistical analysis,
including plots for data exploration, assumption checking, and result presentation.
"""

from .plotter import Plotter
from .statistical_plots import StatisticalPlots
from .diagnostic_plots import DiagnosticPlots

# Alias for backward compatibility
PlotManager = Plotter

__all__ = [
    'Plotter',
    'PlotManager',
    'StatisticalPlots', 
    'DiagnosticPlots'
]