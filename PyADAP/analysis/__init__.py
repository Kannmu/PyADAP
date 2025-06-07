"""Statistical analysis modules for PyADAP 3.0

This package provides comprehensive statistical analysis capabilities including:
- Descriptive statistics
- Inferential statistics (t-tests, ANOVA, non-parametric tests)
- Correlation and regression analysis
- Chi-square tests
- Effect size calculations
- Power analysis
- Assumption checking
"""

from .descriptive import DescriptiveAnalysis
from .inferential import InferentialAnalysis
from .correlation import CorrelationAnalysis
from .regression import RegressionAnalysis
from .nonparametric import NonParametricAnalysis
from .chi_square import ChiSquareAnalysis
from .effect_size import EffectSizeAnalysis
from .power_analysis import PowerAnalysis
from .assumptions import AssumptionChecker
from .analyzer import StatisticalAnalyzer

__all__ = [
    'DescriptiveAnalysis',
    'InferentialAnalysis', 
    'CorrelationAnalysis',
    'RegressionAnalysis',
    'NonParametricAnalysis',
    'ChiSquareAnalysis',
    'EffectSizeAnalysis',
    'PowerAnalysis',
    'AssumptionChecker',
    'StatisticalAnalyzer'
]