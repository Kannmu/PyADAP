"""Validation utilities for PyADAP 3.0

This module provides comprehensive validation functions for data, parameters,
and statistical assumptions.
"""

import numpy as np
import pandas as pd
from typing import Any, List, Dict, Union, Optional, Tuple, Callable
from pathlib import Path
import warnings
from scipy import stats
from sklearn.preprocessing import LabelEncoder


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class Validator:
    """Comprehensive validation utilities for PyADAP."""
    
    @staticmethod
    def validate_dataframe(df: Any, name: str = "DataFrame") -> pd.DataFrame:
        """Validate that input is a pandas DataFrame.
        
        Args:
            df: Input to validate
            name: Name for error messages
            
        Returns:
            Validated DataFrame
            
        Raises:
            ValidationError: If input is not a valid DataFrame
        """
        if not isinstance(df, pd.DataFrame):
            raise ValidationError(f"{name} must be a pandas DataFrame, got {type(df)}")
        
        if df.empty:
            raise ValidationError(f"{name} cannot be empty")
        
        return df
    
    @staticmethod
    def validate_columns(df: pd.DataFrame, columns: Union[str, List[str]], 
                        name: str = "columns") -> List[str]:
        """Validate that columns exist in DataFrame.
        
        Args:
            df: DataFrame to check
            columns: Column name(s) to validate
            name: Name for error messages
            
        Returns:
            List of validated column names
            
        Raises:
            ValidationError: If columns don't exist
        """
        if isinstance(columns, str):
            columns = [columns]
        
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            available_cols = list(df.columns)
            raise ValidationError(
                f"{name} {missing_cols} not found in DataFrame. "
                f"Available columns: {available_cols}"
            )
        
        return columns
    
    @staticmethod
    def validate_numeric_columns(df: pd.DataFrame, columns: Union[str, List[str]],
                               allow_missing: bool = True) -> List[str]:
        """Validate that columns contain numeric data.
        
        Args:
            df: DataFrame to check
            columns: Column name(s) to validate
            allow_missing: Whether to allow missing values
            
        Returns:
            List of validated numeric column names
            
        Raises:
            ValidationError: If columns are not numeric
        """
        columns = Validator.validate_columns(df, columns, "Numeric columns")
        
        non_numeric = []
        for col in columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                # Try to convert to numeric
                try:
                    pd.to_numeric(df[col], errors='raise')
                except (ValueError, TypeError):
                    non_numeric.append(col)
        
        if non_numeric:
            raise ValidationError(
                f"Columns {non_numeric} are not numeric. "
                f"Please ensure all values are numbers."
            )
        
        if not allow_missing:
            missing_data = [col for col in columns if df[col].isnull().any()]
            if missing_data:
                raise ValidationError(
                    f"Columns {missing_data} contain missing values. "
                    f"Missing values are not allowed for this operation."
                )
        
        return columns
    
    @staticmethod
    def validate_categorical_columns(df: pd.DataFrame, columns: Union[str, List[str]]) -> List[str]:
        """Validate that columns contain categorical data.
        
        Args:
            df: DataFrame to check
            columns: Column name(s) to validate
            
        Returns:
            List of validated categorical column names
            
        Raises:
            ValidationError: If columns are not suitable for categorical analysis
        """
        columns = Validator.validate_columns(df, columns, "Categorical columns")
        
        for col in columns:
            unique_values = df[col].nunique()
            total_values = len(df[col].dropna())
            
            # Check if column has too many unique values to be categorical
            if unique_values > total_values * 0.5 and unique_values > 20:
                warnings.warn(
                    f"Column '{col}' has {unique_values} unique values out of {total_values} "
                    f"total values. This might not be suitable for categorical analysis."
                )
        
        return columns
    
    @staticmethod
    def validate_sample_size(df: pd.DataFrame, min_size: int = 3, 
                           group_column: Optional[str] = None) -> bool:
        """Validate sample size requirements.
        
        Args:
            df: DataFrame to check
            min_size: Minimum required sample size
            group_column: Column to group by for group-wise validation
            
        Returns:
            True if sample size is adequate
            
        Raises:
            ValidationError: If sample size is insufficient
        """
        if group_column is None:
            total_size = len(df.dropna())
            if total_size < min_size:
                raise ValidationError(
                    f"Insufficient sample size: {total_size} < {min_size}"
                )
        else:
            Validator.validate_columns(df, group_column, "Group column")
            group_sizes = df.groupby(group_column).size()
            small_groups = group_sizes[group_sizes < min_size]
            
            if not small_groups.empty:
                raise ValidationError(
                    f"Insufficient sample size in groups: {dict(small_groups)}. "
                    f"Minimum required: {min_size}"
                )
        
        return True
    
    @staticmethod
    def validate_file_path(file_path: Union[str, Path], 
                          must_exist: bool = True,
                          allowed_extensions: Optional[List[str]] = None) -> Path:
        """Validate file path.
        
        Args:
            file_path: Path to validate
            must_exist: Whether file must exist
            allowed_extensions: List of allowed file extensions
            
        Returns:
            Validated Path object
            
        Raises:
            ValidationError: If path is invalid
        """
        path = Path(file_path)
        
        if must_exist and not path.exists():
            raise ValidationError(f"File does not exist: {path}")
        
        if allowed_extensions:
            if path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
                raise ValidationError(
                    f"File extension '{path.suffix}' not allowed. "
                    f"Allowed extensions: {allowed_extensions}"
                )
        
        return path
    
    @staticmethod
    def validate_alpha_level(alpha: float) -> float:
        """Validate significance level (alpha).
        
        Args:
            alpha: Significance level to validate
            
        Returns:
            Validated alpha value
            
        Raises:
            ValidationError: If alpha is not valid
        """
        if not isinstance(alpha, (int, float)):
            raise ValidationError(f"Alpha must be a number, got {type(alpha)}")
        
        if not 0 < alpha < 1:
            raise ValidationError(f"Alpha must be between 0 and 1, got {alpha}")
        
        return float(alpha)
    
    @staticmethod
    def validate_effect_size_interpretation(effect_size: float, 
                                          test_type: str = "cohen_d") -> str:
        """Validate and interpret effect size.
        
        Args:
            effect_size: Effect size value
            test_type: Type of effect size (cohen_d, eta_squared, etc.)
            
        Returns:
            Effect size interpretation
        """
        if not isinstance(effect_size, (int, float)):
            raise ValidationError(f"Effect size must be a number, got {type(effect_size)}")
        
        abs_effect = abs(effect_size)
        
        if test_type.lower() == "cohen_d":
            if abs_effect < 0.2:
                return "negligible"
            elif abs_effect < 0.5:
                return "small"
            elif abs_effect < 0.8:
                return "medium"
            else:
                return "large"
        
        elif test_type.lower() in ["eta_squared", "partial_eta_squared"]:
            if abs_effect < 0.01:
                return "negligible"
            elif abs_effect < 0.06:
                return "small"
            elif abs_effect < 0.14:
                return "medium"
            else:
                return "large"
        
        elif test_type.lower() == "r":
            if abs_effect < 0.1:
                return "negligible"
            elif abs_effect < 0.3:
                return "small"
            elif abs_effect < 0.5:
                return "medium"
            else:
                return "large"
        
        else:
            return "unknown"
    
    @staticmethod
    def validate_normality_assumption(data: Union[pd.Series, np.ndarray],
                                    alpha: float = 0.05,
                                    test: str = "shapiro") -> Dict[str, Any]:
        """Validate normality assumption.
        
        Args:
            data: Data to test
            alpha: Significance level
            test: Test to use (shapiro, ks, anderson)
            
        Returns:
            Dictionary with test results
        """
        data_clean = pd.Series(data).dropna()
        
        if len(data_clean) < 3:
            raise ValidationError("Insufficient data for normality test (n < 3)")
        
        result = {"test": test, "assumption_met": False, "warning": None}
        
        try:
            if test.lower() == "shapiro":
                if len(data_clean) > 5000:
                    result["warning"] = "Shapiro-Wilk test not recommended for n > 5000"
                    test = "ks"  # Fall back to KS test
                else:
                    statistic, p_value = stats.shapiro(data_clean)
                    result.update({"statistic": statistic, "p_value": p_value})
            
            if test.lower() == "ks":
                # Kolmogorov-Smirnov test against normal distribution
                statistic, p_value = stats.kstest(data_clean, 'norm')
                result.update({"statistic": statistic, "p_value": p_value})
            
            elif test.lower() == "anderson":
                result_anderson = stats.anderson(data_clean, dist='norm')
                # Use 5% significance level (index 2)
                critical_value = result_anderson.critical_values[2]
                result.update({
                    "statistic": result_anderson.statistic,
                    "critical_value": critical_value,
                    "p_value": None  # Anderson-Darling doesn't provide p-value directly
                })
                result["assumption_met"] = result_anderson.statistic < critical_value
                return result
            
            result["assumption_met"] = result["p_value"] > alpha
            
        except Exception as e:
            result["error"] = str(e)
            result["warning"] = f"Normality test failed: {e}"
        
        return result
    
    @staticmethod
    def validate_homogeneity_assumption(groups: List[Union[pd.Series, np.ndarray]],
                                      alpha: float = 0.05,
                                      test: str = "levene") -> Dict[str, Any]:
        """Validate homogeneity of variance assumption.
        
        Args:
            groups: List of groups to test
            alpha: Significance level
            test: Test to use (levene, bartlett)
            
        Returns:
            Dictionary with test results
        """
        if len(groups) < 2:
            raise ValidationError("At least 2 groups required for homogeneity test")
        
        # Clean data
        clean_groups = [pd.Series(group).dropna() for group in groups]
        
        # Check sample sizes
        for i, group in enumerate(clean_groups):
            if len(group) < 2:
                raise ValidationError(f"Group {i} has insufficient data (n < 2)")
        
        result = {"test": test, "assumption_met": False}
        
        try:
            if test.lower() == "levene":
                statistic, p_value = stats.levene(*clean_groups)
            elif test.lower() == "bartlett":
                statistic, p_value = stats.bartlett(*clean_groups)
            else:
                raise ValidationError(f"Unknown homogeneity test: {test}")
            
            result.update({
                "statistic": statistic,
                "p_value": p_value,
                "assumption_met": p_value > alpha
            })
            
        except Exception as e:
            result["error"] = str(e)
            result["warning"] = f"Homogeneity test failed: {e}"
        
        return result
    
    @staticmethod
    def validate_independence_assumption(data: pd.DataFrame,
                                       subject_column: Optional[str] = None,
                                       time_column: Optional[str] = None) -> Dict[str, Any]:
        """Validate independence assumption.
        
        Args:
            data: DataFrame to check
            subject_column: Column identifying subjects
            time_column: Column identifying time points
            
        Returns:
            Dictionary with validation results
        """
        result = {"assumption_met": True, "warnings": []}
        
        # Check for repeated measures
        if subject_column and subject_column in data.columns:
            subject_counts = data[subject_column].value_counts()
            repeated_subjects = subject_counts[subject_counts > 1]
            
            if not repeated_subjects.empty:
                result["assumption_met"] = False
                result["warnings"].append(
                    f"Repeated measures detected: {len(repeated_subjects)} subjects "
                    f"have multiple observations"
                )
        
        # Check for temporal dependencies
        if time_column and time_column in data.columns:
            if data[time_column].nunique() > 1:
                result["warnings"].append(
                    "Multiple time points detected. Consider temporal dependencies."
                )
        
        # Check for potential clustering
        if len(data) > 1000:
            result["warnings"].append(
                "Large sample size detected. Consider potential clustering effects."
            )
        
        return result
    
    @staticmethod
    def validate_linearity_assumption(x: Union[pd.Series, np.ndarray],
                                    y: Union[pd.Series, np.ndarray],
                                    alpha: float = 0.05) -> Dict[str, Any]:
        """Validate linearity assumption using residual analysis.
        
        Args:
            x: Independent variable
            y: Dependent variable
            alpha: Significance level
            
        Returns:
            Dictionary with validation results
        """
        from scipy.stats import pearsonr
        from sklearn.linear_model import LinearRegression
        
        # Clean data
        df_temp = pd.DataFrame({'x': x, 'y': y}).dropna()
        if len(df_temp) < 3:
            raise ValidationError("Insufficient data for linearity test (n < 3)")
        
        x_clean = df_temp['x'].values.reshape(-1, 1)
        y_clean = df_temp['y'].values
        
        result = {"assumption_met": True, "warnings": []}
        
        try:
            # Fit linear regression
            reg = LinearRegression().fit(x_clean, y_clean)
            y_pred = reg.predict(x_clean)
            residuals = y_clean - y_pred
            
            # Test for non-linear patterns in residuals
            # Simple test: correlation between residuals and predicted values
            corr, p_value = pearsonr(y_pred, residuals)
            
            result.update({
                "residual_correlation": corr,
                "p_value": p_value,
                "assumption_met": abs(corr) < 0.3 and p_value > alpha
            })
            
            if abs(corr) >= 0.3:
                result["warnings"].append(
                    f"Strong correlation between residuals and fitted values ({corr:.3f}). "
                    "Consider non-linear relationship."
                )
            
        except Exception as e:
            result["error"] = str(e)
            result["assumption_met"] = False
        
        return result
    
    @staticmethod
    def validate_multicollinearity(df: pd.DataFrame, 
                                 columns: List[str],
                                 threshold: float = 0.8) -> Dict[str, Any]:
        """Validate multicollinearity assumption.
        
        Args:
            df: DataFrame to check
            columns: Columns to check for multicollinearity
            threshold: Correlation threshold for concern
            
        Returns:
            Dictionary with validation results
        """
        columns = Validator.validate_numeric_columns(df, columns)
        
        if len(columns) < 2:
            return {"assumption_met": True, "warning": "Less than 2 variables provided"}
        
        # Calculate correlation matrix
        corr_matrix = df[columns].corr()
        
        # Find high correlations (excluding diagonal)
        high_corr_pairs = []
        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > threshold:
                    high_corr_pairs.append({
                        'var1': columns[i],
                        'var2': columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        result = {
            "assumption_met": len(high_corr_pairs) == 0,
            "correlation_matrix": corr_matrix,
            "high_correlations": high_corr_pairs,
            "threshold": threshold
        }
        
        if high_corr_pairs:
            result["warning"] = f"High correlations detected: {len(high_corr_pairs)} pairs"
        
        return result
    
    @staticmethod
    def validate_outliers(data: Union[pd.Series, np.ndarray],
                         method: str = "iqr",
                         threshold: float = 1.5) -> Dict[str, Any]:
        """Validate and detect outliers.
        
        Args:
            data: Data to check
            method: Method to use (iqr, zscore, modified_zscore)
            threshold: Threshold for outlier detection
            
        Returns:
            Dictionary with outlier information
        """
        data_clean = pd.Series(data).dropna()
        
        if len(data_clean) < 3:
            raise ValidationError("Insufficient data for outlier detection (n < 3)")
        
        result = {"method": method, "threshold": threshold}
        
        if method.lower() == "iqr":
            Q1 = data_clean.quantile(0.25)
            Q3 = data_clean.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = data_clean[(data_clean < lower_bound) | (data_clean > upper_bound)]
            
        elif method.lower() == "zscore":
            z_scores = np.abs(stats.zscore(data_clean))
            outliers = data_clean[z_scores > threshold]
            
        elif method.lower() == "modified_zscore":
            median = np.median(data_clean)
            mad = np.median(np.abs(data_clean - median))
            modified_z_scores = 0.6745 * (data_clean - median) / mad
            outliers = data_clean[np.abs(modified_z_scores) > threshold]
            
        else:
            raise ValidationError(f"Unknown outlier detection method: {method}")
        
        result.update({
            "outlier_count": len(outliers),
            "outlier_percentage": len(outliers) / len(data_clean) * 100,
            "outlier_indices": outliers.index.tolist() if hasattr(outliers, 'index') else [],
            "outlier_values": outliers.tolist()
        })
        
        return result
    
    @staticmethod
    def validate_power_analysis(effect_size: float, 
                              sample_size: int,
                              alpha: float = 0.05,
                              test_type: str = "t_test") -> Dict[str, Any]:
        """Validate statistical power.
        
        Args:
            effect_size: Expected effect size
            sample_size: Sample size
            alpha: Significance level
            test_type: Type of statistical test
            
        Returns:
            Dictionary with power analysis results
        """
        try:
            from statsmodels.stats.power import ttest_power, FTestAnovaPower
            
            result = {
                "effect_size": effect_size,
                "sample_size": sample_size,
                "alpha": alpha,
                "test_type": test_type
            }
            
            if test_type.lower() in ["t_test", "ttest"]:
                power = ttest_power(effect_size, sample_size, alpha)
            elif test_type.lower() == "anova":
                # Simplified ANOVA power calculation using FTestAnovaPower
                anova_power_calc = FTestAnovaPower()
                power = anova_power_calc.power(effect_size, sample_size, alpha, k_groups=2)
            else:
                result["warning"] = f"Power analysis not implemented for {test_type}"
                return result
            
            result["power"] = power
            result["adequate_power"] = power >= 0.8
            
            if power < 0.8:
                result["warning"] = f"Low statistical power ({power:.3f}). Consider increasing sample size."
            
        except ImportError:
            result["error"] = "statsmodels not available for power analysis"
        except Exception as e:
            result["error"] = f"Power analysis failed: {e}"
        
        return result