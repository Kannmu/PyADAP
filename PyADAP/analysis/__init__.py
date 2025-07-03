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

__all__ = [
    'DescriptiveAnalysis'
]