"""Helper utilities for PyADAP 3.0

This module provides various utility functions for data processing,
formatting, and common operations.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, Dict, Any, List
from pathlib import Path
import re
import os
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings


def normalize_data(data: Union[pd.Series, np.ndarray], 
                  method: str = "z_score") -> Union[pd.Series, np.ndarray]:
    """Normalize data using various methods.
    
    Args:
        data: Data to normalize
        method: Normalization method ('z_score', 'min_max', 'robust')
        
    Returns:
        Normalized data
        
    Raises:
        ValueError: If method is not supported
    """
    if isinstance(data, pd.Series):
        values = data.values.reshape(-1, 1)
        is_series = True
        index = data.index
    else:
        values = np.array(data).reshape(-1, 1)
        is_series = False
        index = None
    
    # Remove NaN values for scaling
    mask = ~np.isnan(values.flatten())
    if not mask.any():
        return data  # All NaN, return as is
    
    if method.lower() == "z_score":
        scaler = StandardScaler()
    elif method.lower() == "min_max":
        scaler = MinMaxScaler()
    elif method.lower() == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    
    # Fit scaler on non-NaN values
    scaler.fit(values[mask].reshape(-1, 1))
    
    # Transform all values (NaN will remain NaN)
    normalized = np.full_like(values.flatten(), np.nan)
    normalized[mask] = scaler.transform(values[mask].reshape(-1, 1)).flatten()
    
    if is_series:
        return pd.Series(normalized, index=index)
    else:
        return normalized


def standardize_data(data: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """Standardize data (z-score normalization).
    
    Args:
        data: Data to standardize
        
    Returns:
        Standardized data
    """
    return normalize_data(data, method="z_score")


def calculate_effect_size(group1: Union[pd.Series, np.ndarray],
                         group2: Union[pd.Series, np.ndarray],
                         effect_type: str = "cohen_d") -> float:
    """Calculate effect size between two groups.
    
    Args:
        group1: First group data
        group2: Second group data
        effect_type: Type of effect size ('cohen_d', 'glass_delta', 'hedges_g')
        
    Returns:
        Effect size value
        
    Raises:
        ValueError: If effect_type is not supported
    """
    g1 = pd.Series(group1).dropna()
    g2 = pd.Series(group2).dropna()
    
    if len(g1) == 0 or len(g2) == 0:
        return np.nan
    
    mean1, mean2 = g1.mean(), g2.mean()
    std1, std2 = g1.std(ddof=1), g2.std(ddof=1)
    n1, n2 = len(g1), len(g2)
    
    if effect_type.lower() == "cohen_d":
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0.0
        return (mean1 - mean2) / pooled_std
    
    elif effect_type.lower() == "glass_delta":
        # Use control group (group2) standard deviation
        if std2 == 0:
            return 0.0
        return (mean1 - mean2) / std2
    
    elif effect_type.lower() == "hedges_g":
        # Bias-corrected Cohen's d
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0.0
        cohen_d = (mean1 - mean2) / pooled_std
        # Bias correction factor
        correction = 1 - (3 / (4 * (n1 + n2) - 9))
        return cohen_d * correction
    
    else:
        raise ValueError(f"Unsupported effect size type: {effect_type}")


def calculate_correlation_effect_size(r: float, n: int) -> Dict[str, float]:
    """Calculate effect size measures for correlation.
    
    Args:
        r: Correlation coefficient
        n: Sample size
        
    Returns:
        Dictionary with effect size measures
    """
    r_squared = r**2
    
    # Cohen's conventions for correlation
    if abs(r) < 0.1:
        magnitude = "negligible"
    elif abs(r) < 0.3:
        magnitude = "small"
    elif abs(r) < 0.5:
        magnitude = "medium"
    else:
        magnitude = "large"
    
    return {
        "r": r,
        "r_squared": r_squared,
        "magnitude": magnitude,
        "variance_explained": r_squared * 100
    }


def format_p_value(p_value: float, threshold: float = 0.001, 
                  decimal_places: int = 3) -> str:
    """Format p-value for reporting.
    
    Args:
        p_value: P-value to format
        threshold: Threshold below which to report as "< threshold"
        decimal_places: Number of decimal places
        
    Returns:
        Formatted p-value string
    """
    if pd.isna(p_value):
        return "NaN"
    
    if p_value < threshold:
        return f"< {threshold}"
    else:
        return f"{p_value:.{decimal_places}f}"


def format_number(number: float, decimal_places: int = 3, 
                 scientific_threshold: float = 0.001) -> str:
    """Format number for reporting.
    
    Args:
        number: Number to format
        decimal_places: Number of decimal places
        scientific_threshold: Threshold below which to use scientific notation
        
    Returns:
        Formatted number string
    """
    if pd.isna(number):
        return "NaN"
    
    if abs(number) < scientific_threshold and number != 0:
        return f"{number:.{decimal_places}e}"
    else:
        return f"{number:.{decimal_places}f}"


def format_confidence_interval(lower: float, upper: float, 
                              confidence_level: float = 0.95,
                              decimal_places: int = 3) -> str:
    """Format confidence interval for reporting.
    
    Args:
        lower: Lower bound
        upper: Upper bound
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        decimal_places: Number of decimal places
        
    Returns:
        Formatted confidence interval string
    """
    if pd.isna(lower) or pd.isna(upper):
        return "NaN"
    
    percentage = int(confidence_level * 100)
    lower_str = format_number(lower, decimal_places)
    upper_str = format_number(upper, decimal_places)
    
    return f"{percentage}% CI [{lower_str}, {upper_str}]"


def create_safe_filename(filename: str, max_length: int = 255) -> str:
    """Create a safe filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
        
    Returns:
        Safe filename
    """
    # Remove or replace invalid characters
    safe_chars = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    safe_chars = safe_chars.strip(' .')
    
    # Ensure it's not empty
    if not safe_chars:
        safe_chars = "unnamed_file"
    
    # Truncate if too long
    if len(safe_chars) > max_length:
        name, ext = os.path.splitext(safe_chars)
        max_name_length = max_length - len(ext)
        safe_chars = name[:max_name_length] + ext
    
    return safe_chars


def ensure_directory(directory: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object of the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_memory_usage(obj: Any) -> float:
    """Get memory usage of an object in MB.
    
    Args:
        obj: Object to measure
        
    Returns:
        Memory usage in MB
    """
    import sys
    
    if isinstance(obj, pd.DataFrame):
        return obj.memory_usage(deep=True).sum() / 1024**2
    elif isinstance(obj, pd.Series):
        return obj.memory_usage(deep=True) / 1024**2
    elif isinstance(obj, np.ndarray):
        return obj.nbytes / 1024**2
    else:
        return sys.getsizeof(obj) / 1024**2


def detect_data_types(df: pd.DataFrame, 
                     sample_size: Optional[int] = 1000) -> Dict[str, str]:
    """Automatically detect appropriate data types for DataFrame columns.
    
    Args:
        df: DataFrame to analyze
        sample_size: Number of rows to sample for analysis (None for all)
        
    Returns:
        Dictionary mapping column names to suggested data types
    """
    if sample_size and len(df) > sample_size:
        sample_df = df.sample(n=sample_size, random_state=42)
    else:
        sample_df = df
    
    suggestions = {}
    
    for col in sample_df.columns:
        series = sample_df[col].dropna()
        
        if len(series) == 0:
            suggestions[col] = "object"
            continue
        
        # Check if numeric
        try:
            pd.to_numeric(series, errors='raise')
            # Check if integer
            if series.dtype in ['int64', 'int32'] or all(series == series.astype(int)):
                suggestions[col] = "integer"
            else:
                suggestions[col] = "numeric"
            continue
        except (ValueError, TypeError):
            pass
        
        # Check if datetime
        try:
            pd.to_datetime(series, errors='raise')
            suggestions[col] = "datetime"
            continue
        except (ValueError, TypeError):
            pass
        
        # Check if boolean
        unique_values = set(series.astype(str).str.lower().unique())
        bool_values = {'true', 'false', '1', '0', 'yes', 'no', 't', 'f', 'y', 'n'}
        if unique_values.issubset(bool_values) and len(unique_values) <= 2:
            suggestions[col] = "boolean"
            continue
        
        # Check if categorical
        unique_count = series.nunique()
        total_count = len(series)
        
        if unique_count <= 20 or unique_count / total_count < 0.05:
            suggestions[col] = "categorical"
        else:
            suggestions[col] = "text"
    
    return suggestions


def convert_data_types(df: pd.DataFrame, 
                      type_mapping: Dict[str, str]) -> pd.DataFrame:
    """Convert DataFrame columns to specified data types.
    
    Args:
        df: DataFrame to convert
        type_mapping: Dictionary mapping column names to target types
        
    Returns:
        DataFrame with converted types
    """
    df_converted = df.copy()
    
    for col, target_type in type_mapping.items():
        if col not in df_converted.columns:
            continue
        
        try:
            if target_type == "numeric":
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
            elif target_type == "integer":
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce').astype('Int64')
            elif target_type == "categorical":
                df_converted[col] = df_converted[col].astype('category')
            elif target_type == "datetime":
                df_converted[col] = pd.to_datetime(df_converted[col], errors='coerce')
            elif target_type == "boolean":
                # Convert common boolean representations
                bool_map = {
                    'true': True, 'false': False,
                    '1': True, '0': False,
                    'yes': True, 'no': False,
                    'y': True, 'n': False,
                    't': True, 'f': False
                }
                df_converted[col] = df_converted[col].astype(str).str.lower().map(bool_map)
            elif target_type == "text":
                df_converted[col] = df_converted[col].astype(str)
        
        except Exception as e:
            warnings.warn(f"Failed to convert column '{col}' to {target_type}: {e}")
    
    return df_converted


def calculate_missing_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive missing data summary.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        DataFrame with missing data statistics
    """
    missing_summary = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100,
        'Data_Type': df.dtypes,
        'Non_Missing_Count': df.count(),
        'Unique_Values': [df[col].nunique() for col in df.columns]
    })
    
    missing_summary = missing_summary.sort_values('Missing_Percentage', ascending=False)
    missing_summary = missing_summary.reset_index(drop=True)
    
    return missing_summary


def generate_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive data summary.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Dictionary with data summary statistics
    """
    summary = {
        'shape': df.shape,
        'memory_usage_mb': get_memory_usage(df),
        'missing_data': calculate_missing_data_summary(df),
        'data_types': detect_data_types(df),
        'numeric_summary': df.describe() if len(df.select_dtypes(include=[np.number]).columns) > 0 else None,
        'categorical_summary': {}
    }
    
    # Categorical summary
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        summary['categorical_summary'][col] = {
            'unique_count': df[col].nunique(),
            'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
            'frequency': df[col].value_counts().head(5).to_dict()
        }
    
    return summary


def create_correlation_matrix(df: pd.DataFrame, 
                            method: str = "pearson",
                            min_periods: int = 30) -> pd.DataFrame:
    """Create correlation matrix with specified method.
    
    Args:
        df: DataFrame with numeric columns
        method: Correlation method ('pearson', 'spearman', 'kendall')
        min_periods: Minimum number of observations for correlation
        
    Returns:
        Correlation matrix
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        raise ValueError("No numeric columns found for correlation analysis")
    
    return numeric_df.corr(method=method, min_periods=min_periods)


def bootstrap_statistic(data: Union[pd.Series, np.ndarray],
                       statistic_func: callable,
                       n_bootstrap: int = 1000,
                       confidence_level: float = 0.95,
                       random_state: Optional[int] = None) -> Dict[str, float]:
    """Calculate bootstrap confidence intervals for a statistic.
    
    Args:
        data: Data to bootstrap
        statistic_func: Function to calculate statistic (e.g., np.mean)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with statistic and confidence intervals
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    data_clean = pd.Series(data).dropna().values
    n = len(data_clean)
    
    if n < 2:
        return {'statistic': np.nan, 'lower_ci': np.nan, 'upper_ci': np.nan}
    
    # Original statistic
    original_stat = statistic_func(data_clean)
    
    # Bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data_clean, size=n, replace=True)
        bootstrap_stats.append(statistic_func(bootstrap_sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_ci = np.percentile(bootstrap_stats, lower_percentile)
    upper_ci = np.percentile(bootstrap_stats, upper_percentile)
    
    return {
        'statistic': original_stat,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci,
        'bootstrap_std': np.std(bootstrap_stats),
        'bootstrap_samples': bootstrap_stats
    }


def calculate_sample_size_recommendation(effect_size: float,
                                       power: float = 0.8,
                                       alpha: float = 0.05,
                                       test_type: str = "t_test") -> Dict[str, Any]:
    """Calculate recommended sample size for given parameters.
    
    Args:
        effect_size: Expected effect size
        power: Desired statistical power
        alpha: Significance level
        test_type: Type of statistical test
        
    Returns:
        Dictionary with sample size recommendations
    """
    try:
        from statsmodels.stats.power import ttest_power
        
        if test_type.lower() in ["t_test", "ttest"]:
            # Binary search for sample size
            n_low, n_high = 2, 10000
            
            while n_high - n_low > 1:
                n_mid = (n_low + n_high) // 2
                current_power = ttest_power(effect_size, n_mid, alpha)
                
                if current_power < power:
                    n_low = n_mid
                else:
                    n_high = n_mid
            
            recommended_n = n_high
            actual_power = ttest_power(effect_size, recommended_n, alpha)
            
            return {
                'recommended_sample_size': recommended_n,
                'actual_power': actual_power,
                'target_power': power,
                'effect_size': effect_size,
                'alpha': alpha
            }
        
        else:
            return {
                'error': f"Sample size calculation not implemented for {test_type}"
            }
    
    except ImportError:
        return {
            'error': "statsmodels not available for sample size calculation"
        }
    except Exception as e:
        return {
            'error': f"Sample size calculation failed: {e}"
        }


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame column names for better usability.
    
    Args:
        df: DataFrame to clean
        
    Returns:
        DataFrame with cleaned column names
    """
    df_clean = df.copy()
    
    # Clean column names
    new_columns = []
    for col in df_clean.columns:
        # Convert to string
        clean_col = str(col)
        
        # Remove leading/trailing whitespace
        clean_col = clean_col.strip()
        
        # Replace spaces and special characters with underscores
        clean_col = re.sub(r'[^a-zA-Z0-9_]', '_', clean_col)
        
        # Remove multiple consecutive underscores
        clean_col = re.sub(r'_+', '_', clean_col)
        
        # Remove leading/trailing underscores
        clean_col = clean_col.strip('_')
        
        # Ensure it doesn't start with a number
        if clean_col and clean_col[0].isdigit():
            clean_col = 'col_' + clean_col
        
        # Ensure it's not empty
        if not clean_col:
            clean_col = f'unnamed_column_{len(new_columns)}'
        
        new_columns.append(clean_col)
    
    # Handle duplicate column names
    seen = {}
    final_columns = []
    for col in new_columns:
        if col in seen:
            seen[col] += 1
            final_columns.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            final_columns.append(col)
    
    df_clean.columns = final_columns
    return df_clean