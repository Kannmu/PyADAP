"""Statistical and analysis constants for PyADAP.

This module centralizes all magic numbers and thresholds used throughout
the PyADAP codebase to improve maintainability and consistency.
"""


# Statistical significance levels
DEFAULT_ALPHA = 0.05
STRICT_ALPHA = 0.01
VERY_STRICT_ALPHA = 0.001

# Effect size thresholds (Cohen's conventions)
EFFECT_SIZE_THRESHOLDS = {
    'cohens_d': {
        'small': 0.2,
        'medium': 0.5,
        'large': 0.8
    },
    'eta_squared': {
        'small': 0.01,
        'medium': 0.06,
        'large': 0.14
    },
    'omega_squared': {
        'small': 0.01,
        'medium': 0.06,
        'large': 0.14
    },
    'correlation': {
        'small': 0.1,
        'medium': 0.3,
        'large': 0.5
    }
}

# Sample size thresholds
MIN_SAMPLE_SIZE = 3
SMALL_SAMPLE_SIZE = 30
MEDIUM_SAMPLE_SIZE = 100
LARGE_SAMPLE_SIZE = 1000
SHAPIRO_WILK_MAX_SIZE = 5000

# Bootstrap and resampling
DEFAULT_BOOTSTRAP_SAMPLES = 1000
MIN_BOOTSTRAP_SAMPLES = 100
MAX_BOOTSTRAP_SAMPLES = 10000

# Data quality thresholds
HIGH_MISSING_THRESHOLD = 0.2  # 20%
MODERATE_MISSING_THRESHOLD = 0.1  # 10%
LOW_MISSING_THRESHOLD = 0.05  # 5%

# Outlier detection
IQR_OUTLIER_MULTIPLIER = 1.5
Z_SCORE_THRESHOLD = 3.0
MODIFIED_Z_SCORE_THRESHOLD = 3.5

# Categorical variable thresholds
CATEGORICAL_UNIQUE_RATIO_THRESHOLD = 0.05
MAX_CATEGORICAL_LEVELS = 20
MIN_CATEGORICAL_LEVELS = 2

# Visualization settings
DEFAULT_DPI = 100
DEFAULT_FIGURE_WIDTH = 8
DEFAULT_FIGURE_HEIGHT = 6
MAX_DISPLAY_ROWS = 1000
DEFAULT_ROWS_PER_PAGE = 100

# File and data limits
MAX_FILENAME_LENGTH = 255
DEFAULT_DECIMAL_PLACES = 3
PERCENTAGE_MULTIPLIER = 100

# Confidence intervals
DEFAULT_CONFIDENCE_LEVEL = 0.95
CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]

# Quantiles for robust statistics
LOWER_QUANTILE = 0.05
UPPER_QUANTILE = 0.95
FIRST_QUARTILE = 0.25
THIRD_QUARTILE = 0.75
MEDIAN = 0.5

# Power analysis
DEFAULT_POWER = 0.8
MIN_POWER = 0.5
MAX_POWER = 0.99

# Quality score bounds
MIN_QUALITY_SCORE = 0
MAX_QUALITY_SCORE = 100
PERFECT_QUALITY_SCORE = 100

# GUI and display settings
PROGRESS_BAR_MAX = 100
DEFAULT_COLUMN_WIDTH = 100
TREE_MIN_WIDTH = 50

# Statistical test specific constants
MIN_CHI_SQUARE_EXPECTED = 5
MIN_FISHER_EXACT_SAMPLE = 20
MIN_MCNEMAR_SAMPLE = 25

# Normality test thresholds
MIN_DAGOSTINO_SAMPLE = 8
MIN_JARQUE_BERA_SAMPLE = 2
MIN_ANDERSON_SAMPLE = 8

# Variance analysis
LEVENE_MIN_GROUPS = 2
BARTLETT_MIN_GROUPS = 2

# Correlation analysis
MIN_CORRELATION_SAMPLE = 3
SPEARMAN_TIE_CORRECTION = True

# HTML report settings
HTML_Z_INDEX_MODAL = 1000
HTML_FULL_WIDTH = 100

# File format extensions
SUPPORTED_DATA_FORMATS = ['.csv', '.xlsx', '.xls', '.tsv', '.txt']
SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg', '.svg', '.pdf']

# Error messages and warnings
ERROR_MESSAGES = {
    'insufficient_data': 'Insufficient data for analysis',
    'invalid_alpha': 'Alpha level must be between 0 and 1',
    'invalid_power': 'Power must be between 0 and 1',
    'missing_variables': 'Required variables not found in data',
    'invalid_file_format': 'Unsupported file format',
    'normality_violation': 'Data does not meet normality assumptions',
    'homogeneity_violation': 'Data does not meet homogeneity assumptions'
}

# Success thresholds for various metrics
SUCCESS_THRESHOLDS = {
    'data_completeness': 0.95,  # 95% complete data
    'test_power': 0.8,          # 80% statistical power
    'effect_size_medium': 0.5,  # Medium effect size
    'correlation_moderate': 0.3  # Moderate correlation
}