"""Descriptive statistics analysis for PyADAP 3.0

This module provides comprehensive descriptive statistics including:
- Central tendency measures
- Variability measures
- Distribution shape measures
- Frequency analysis
- Cross-tabulation
- Data quality assessment
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import scipy.stats as stats
from collections import Counter
import warnings

from ..utils import Logger, get_logger, Validator, format_number, format_p_value
from ..config import Config

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


class DescriptiveAnalysis:
    """Class for performing descriptive statistical analysis."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the descriptive analysis.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.logger = get_logger("PyADAP.DescriptiveAnalysis")
        self.validator = Validator()
    
    def analyze(self, data: pd.DataFrame, 
                variables: Dict[str, List[str]],
                **kwargs) -> Dict[str, Any]:
        """Perform comprehensive descriptive analysis.
        
        Args:
            data: Input DataFrame
            variables: Dictionary with variable assignments
            **kwargs: Additional analysis options
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            self.logger.info("Starting descriptive analysis")
            
            # Validate inputs
            self.validator.validate_dataframe(data)
            
            results = {
                'analysis_type': 'descriptive',
                'sample_size': len(data),
                'variables_analyzed': {},
                'summary_statistics': {},
                'frequency_analysis': {},
                'distribution_analysis': {},
                'cross_tabulation': {},
                'data_quality': {},
                'recommendations': []
            }
            
            # Get all variables to analyze
            all_vars = []
            for var_type, var_list in variables.items():
                if var_list:
                    all_vars.extend(var_list)
                    results['variables_analyzed'][var_type] = var_list
            
            # Remove duplicates while preserving order
            all_vars = list(dict.fromkeys(all_vars))
            
            # Filter variables that exist in data
            existing_vars = [var for var in all_vars if var in data.columns]
            
            if not existing_vars:
                self.logger.warning("No valid variables found for analysis")
                return results
            
            # Perform different types of descriptive analysis
            results['summary_statistics'] = self._calculate_summary_statistics(data, existing_vars)
            results['frequency_analysis'] = self._perform_frequency_analysis(data, existing_vars)
            results['distribution_analysis'] = self._analyze_distributions(data, existing_vars)
            results['cross_tabulation'] = self._perform_cross_tabulation(data, variables)
            results['data_quality'] = self._assess_data_quality(data, existing_vars)
            
            # Generate recommendations
            results['recommendations'] = self._generate_recommendations(data, existing_vars, results)
            
            self.logger.info(f"Descriptive analysis completed for {len(existing_vars)} variables")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in descriptive analysis: {str(e)}")
            raise
    
    def _calculate_summary_statistics(self, data: pd.DataFrame, 
                                    variables: List[str]) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics.
        
        Args:
            data: Input DataFrame
            variables: List of variables to analyze
            
        Returns:
            Dictionary of summary statistics
        """
        summary_stats = {
            'numeric': {},
            'categorical': {},
            'overall': {}
        }
        
        # Separate numeric and categorical variables
        numeric_vars = [var for var in variables if data[var].dtype in ['int64', 'float64']]
        categorical_vars = [var for var in variables if var not in numeric_vars]
        
        # Numeric variables statistics
        for var in numeric_vars:
            try:
                clean_data = data[var].dropna()
                
                if len(clean_data) == 0:
                    summary_stats['numeric'][var] = {'error': 'No valid data'}
                    continue
                
                var_stats = {
                    # Central tendency
                    'count': len(clean_data),
                    'mean': float(clean_data.mean()),
                    'median': float(clean_data.median()),
                    'mode': self._calculate_mode(clean_data),
                    
                    # Variability
                    'std': float(clean_data.std()),
                    'variance': float(clean_data.var()),
                    'range': float(clean_data.max() - clean_data.min()),
                    'iqr': float(clean_data.quantile(0.75) - clean_data.quantile(0.25)),
                    'mad': float(stats.median_abs_deviation(clean_data)),
                    'cv': float(clean_data.std() / clean_data.mean()) if clean_data.mean() != 0 else np.inf,
                    
                    # Distribution shape
                    'skewness': float(stats.skew(clean_data)),
                    'kurtosis': float(stats.kurtosis(clean_data)),
                    
                    # Extremes
                    'min': float(clean_data.min()),
                    'max': float(clean_data.max()),
                    
                    # Percentiles
                    'q1': float(clean_data.quantile(0.25)),
                    'q3': float(clean_data.quantile(0.75)),
                    'p5': float(clean_data.quantile(0.05)),
                    'p95': float(clean_data.quantile(0.95)),
                    
                    # Missing data
                    'missing_count': int(data[var].isna().sum()),
                    'missing_percent': float(data[var].isna().sum() / len(data) * 100),
                    
                    # Outliers
                    'outliers_iqr': self._count_outliers_iqr(clean_data),
                    'outliers_zscore': self._count_outliers_zscore(clean_data),
                    
                    # Confidence intervals
                    'ci_mean_95': self._calculate_ci_mean(clean_data, 0.95),
                    'ci_median_95': self._calculate_ci_median(clean_data, 0.95)
                }
                
                # Add interpretation
                var_stats['interpretation'] = self._interpret_numeric_stats(var_stats)
                
                summary_stats['numeric'][var] = var_stats
                
            except Exception as e:
                self.logger.warning(f"Error calculating statistics for {var}: {str(e)}")
                summary_stats['numeric'][var] = {'error': str(e)}
        
        # Categorical variables statistics
        for var in categorical_vars:
            try:
                clean_data = data[var].dropna()
                
                if len(clean_data) == 0:
                    summary_stats['categorical'][var] = {'error': 'No valid data'}
                    continue
                
                value_counts = clean_data.value_counts()
                proportions = clean_data.value_counts(normalize=True)
                
                var_stats = {
                    'count': len(clean_data),
                    'unique_count': int(clean_data.nunique()),
                    'mode': clean_data.mode().iloc[0] if len(clean_data.mode()) > 0 else None,
                    'mode_frequency': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'mode_proportion': float(proportions.iloc[0]) if len(proportions) > 0 else 0,
                    
                    # Missing data
                    'missing_count': int(data[var].isna().sum()),
                    'missing_percent': float(data[var].isna().sum() / len(data) * 100),
                    
                    # Frequency distribution
                    'frequency_table': value_counts.to_dict(),
                    'proportion_table': proportions.to_dict(),
                    
                    # Diversity measures
                    'entropy': self._calculate_entropy(clean_data),
                    'gini_simpson': self._calculate_gini_simpson(clean_data),
                    
                    # Concentration
                    'concentration_ratio': self._calculate_concentration_ratio(clean_data),
                    'herfindahl_index': self._calculate_herfindahl_index(clean_data)
                }
                
                # Add interpretation
                var_stats['interpretation'] = self._interpret_categorical_stats(var_stats)
                
                summary_stats['categorical'][var] = var_stats
                
            except Exception as e:
                self.logger.warning(f"Error calculating statistics for {var}: {str(e)}")
                summary_stats['categorical'][var] = {'error': str(e)}
        
        # Overall dataset statistics
        summary_stats['overall'] = {
            'total_observations': len(data),
            'total_variables': len(variables),
            'numeric_variables': len(numeric_vars),
            'categorical_variables': len(categorical_vars),
            'complete_cases': int(data[variables].dropna().shape[0]),
            'complete_cases_percent': float(data[variables].dropna().shape[0] / len(data) * 100),
            'total_missing': int(data[variables].isna().sum().sum()),
            'missing_percent': float(data[variables].isna().sum().sum() / (len(data) * len(variables)) * 100)
        }
        
        return summary_stats
    
    def _perform_frequency_analysis(self, data: pd.DataFrame, 
                                  variables: List[str]) -> Dict[str, Any]:
        """Perform frequency analysis for categorical variables.
        
        Args:
            data: Input DataFrame
            variables: List of variables to analyze
            
        Returns:
            Dictionary of frequency analysis results
        """
        frequency_results = {}
        
        # Focus on categorical variables and low-cardinality numeric variables
        for var in variables:
            try:
                clean_data = data[var].dropna()
                
                if len(clean_data) == 0:
                    continue
                
                # Determine if variable should be treated as categorical
                is_categorical = (data[var].dtype in ['object', 'category'] or 
                                data[var].nunique() <= 20)
                
                if is_categorical:
                    value_counts = clean_data.value_counts()
                    proportions = clean_data.value_counts(normalize=True)
                    cumulative_props = proportions.cumsum()
                    
                    # Create frequency table
                    freq_table = pd.DataFrame({
                        'Frequency': value_counts,
                        'Proportion': proportions,
                        'Percentage': proportions * 100,
                        'Cumulative_Proportion': cumulative_props,
                        'Cumulative_Percentage': cumulative_props * 100
                    })
                    
                    frequency_results[var] = {
                        'frequency_table': freq_table.to_dict('index'),
                        'most_frequent': value_counts.index[0],
                        'least_frequent': value_counts.index[-1],
                        'frequency_range': int(value_counts.max() - value_counts.min()),
                        'evenness': self._calculate_evenness(clean_data),
                        'modal_percentage': float(proportions.iloc[0] * 100)
                    }
                    
                    # Chi-square goodness of fit test (uniform distribution)
                    if len(value_counts) > 1:
                        try:
                            expected = len(clean_data) / len(value_counts)
                            chi2_stat, chi2_p = stats.chisquare(value_counts)
                            
                            frequency_results[var]['uniformity_test'] = {
                                'chi2_statistic': float(chi2_stat),
                                'p_value': float(chi2_p),
                                'interpretation': 'Non-uniform distribution' if chi2_p < 0.05 else 'Approximately uniform distribution'
                            }
                        except:
                            pass
                
            except Exception as e:
                self.logger.warning(f"Error in frequency analysis for {var}: {str(e)}")
        
        return frequency_results
    
    def _analyze_distributions(self, data: pd.DataFrame, 
                             variables: List[str]) -> Dict[str, Any]:
        """Analyze distribution characteristics of variables.
        
        Args:
            data: Input DataFrame
            variables: List of variables to analyze
            
        Returns:
            Dictionary of distribution analysis results
        """
        distribution_results = {}
        
        # Focus on numeric variables
        numeric_vars = [var for var in variables if data[var].dtype in ['int64', 'float64']]
        
        for var in numeric_vars:
            try:
                clean_data = data[var].dropna()
                
                if len(clean_data) < 3:
                    continue
                
                dist_analysis = {
                    'normality_tests': self._test_normality(clean_data),
                    'distribution_fit': self._fit_distributions(clean_data),
                    'shape_characteristics': self._analyze_shape(clean_data),
                    'tail_behavior': self._analyze_tails(clean_data)
                }
                
                distribution_results[var] = dist_analysis
                
            except Exception as e:
                self.logger.warning(f"Error in distribution analysis for {var}: {str(e)}")
        
        return distribution_results
    
    def _perform_cross_tabulation(self, data: pd.DataFrame, 
                                variables: Dict[str, List[str]]) -> Dict[str, Any]:
        """Perform cross-tabulation analysis.
        
        Args:
            data: Input DataFrame
            variables: Dictionary with variable assignments
            
        Returns:
            Dictionary of cross-tabulation results
        """
        crosstab_results = {}
        
        # Get categorical variables
        all_vars = []
        for var_list in variables.values():
            all_vars.extend(var_list)
        
        categorical_vars = [var for var in all_vars if var in data.columns and 
                          (data[var].dtype in ['object', 'category'] or data[var].nunique() <= 10)]
        
        # Perform pairwise cross-tabulations
        for i, var1 in enumerate(categorical_vars):
            for var2 in categorical_vars[i+1:]:
                try:
                    # Create cross-tabulation
                    crosstab = pd.crosstab(data[var1], data[var2], margins=True)
                    
                    # Calculate proportions
                    prop_table = pd.crosstab(data[var1], data[var2], normalize='all')
                    row_prop = pd.crosstab(data[var1], data[var2], normalize='index')
                    col_prop = pd.crosstab(data[var1], data[var2], normalize='columns')
                    
                    # Chi-square test of independence
                    chi2_stat, chi2_p, dof, expected = stats.chi2_contingency(crosstab.iloc[:-1, :-1])
                    
                    # Cramér's V
                    n = crosstab.iloc[-1, -1]  # Total count
                    cramers_v = np.sqrt(chi2_stat / (n * (min(crosstab.shape) - 2)))
                    
                    crosstab_key = f"{var1}_vs_{var2}"
                    crosstab_results[crosstab_key] = {
                        'contingency_table': crosstab.to_dict(),
                        'proportion_table': prop_table.to_dict(),
                        'row_proportions': row_prop.to_dict(),
                        'column_proportions': col_prop.to_dict(),
                        'chi_square_test': {
                            'statistic': float(chi2_stat),
                            'p_value': float(chi2_p),
                            'degrees_of_freedom': int(dof),
                            'interpretation': 'Variables are associated' if chi2_p < 0.05 else 'Variables are independent'
                        },
                        'effect_size': {
                            'cramers_v': float(cramers_v),
                            'interpretation': self._interpret_cramers_v(cramers_v)
                        }
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Error in cross-tabulation for {var1} vs {var2}: {str(e)}")
        
        return crosstab_results
    
    def _assess_data_quality(self, data: pd.DataFrame, 
                           variables: List[str]) -> Dict[str, Any]:
        """Assess data quality for the variables.
        
        Args:
            data: Input DataFrame
            variables: List of variables to analyze
            
        Returns:
            Dictionary of data quality assessment
        """
        quality_assessment = {
            'missing_data': {},
            'outliers': {},
            'duplicates': {},
            'data_types': {},
            'value_ranges': {},
            'consistency': {},
            'overall_quality_score': 0
        }
        
        # Missing data analysis
        for var in variables:
            missing_count = data[var].isna().sum()
            missing_percent = missing_count / len(data) * 100
            
            quality_assessment['missing_data'][var] = {
                'count': int(missing_count),
                'percentage': float(missing_percent),
                'severity': self._assess_missing_severity(missing_percent)
            }
        
        # Outlier analysis for numeric variables
        numeric_vars = [var for var in variables if data[var].dtype in ['int64', 'float64']]
        for var in numeric_vars:
            clean_data = data[var].dropna()
            if len(clean_data) > 0:
                outliers_iqr = self._count_outliers_iqr(clean_data)
                outliers_zscore = self._count_outliers_zscore(clean_data)
                
                quality_assessment['outliers'][var] = {
                    'iqr_outliers': outliers_iqr,
                    'zscore_outliers': outliers_zscore,
                    'outlier_percentage': float(max(outliers_iqr, outliers_zscore) / len(clean_data) * 100)
                }
        
        # Duplicate analysis
        duplicate_rows = data[variables].duplicated().sum()
        quality_assessment['duplicates'] = {
            'duplicate_rows': int(duplicate_rows),
            'duplicate_percentage': float(duplicate_rows / len(data) * 100)
        }
        
        # Data type consistency
        for var in variables:
            quality_assessment['data_types'][var] = {
                'current_type': str(data[var].dtype),
                'suggested_type': self._suggest_data_type(data[var]),
                'type_consistency': self._check_type_consistency(data[var])
            }
        
        # Value range analysis
        for var in numeric_vars:
            clean_data = data[var].dropna()
            if len(clean_data) > 0:
                quality_assessment['value_ranges'][var] = {
                    'min': float(clean_data.min()),
                    'max': float(clean_data.max()),
                    'range': float(clean_data.max() - clean_data.min()),
                    'has_negative': bool((clean_data < 0).any()),
                    'has_zero': bool((clean_data == 0).any()),
                    'reasonable_range': self._assess_value_range(clean_data, var)
                }
        
        # Calculate overall quality score
        quality_assessment['overall_quality_score'] = self._calculate_quality_score(quality_assessment)
        
        return quality_assessment
    
    def _generate_recommendations(self, data: pd.DataFrame, 
                                variables: List[str],
                                results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on the analysis.
        
        Args:
            data: Input DataFrame
            variables: List of variables analyzed
            results: Analysis results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Sample size recommendations
        if len(data) < 30:
            recommendations.append("Consider collecting more data. Sample size < 30 may limit statistical power.")
        elif len(data) < 100:
            recommendations.append("Sample size is adequate but consider collecting more data for robust analysis.")
        
        # Missing data recommendations
        missing_data = results.get('data_quality', {}).get('missing_data', {})
        high_missing_vars = [var for var, info in missing_data.items() 
                           if info.get('percentage', 0) > 20]
        if high_missing_vars:
            recommendations.append(f"Variables with high missing data (>20%): {', '.join(high_missing_vars)}. Consider imputation or removal.")
        
        # Outlier recommendations
        outliers = results.get('data_quality', {}).get('outliers', {})
        high_outlier_vars = [var for var, info in outliers.items() 
                           if info.get('outlier_percentage', 0) > 10]
        if high_outlier_vars:
            recommendations.append(f"Variables with many outliers (>10%): {', '.join(high_outlier_vars)}. Consider outlier treatment.")
        
        # Distribution recommendations
        summary_stats = results.get('summary_statistics', {}).get('numeric', {})
        for var, stats_info in summary_stats.items():
            if isinstance(stats_info, dict) and 'skewness' in stats_info:
                skewness = abs(stats_info['skewness'])
                if skewness > 2:
                    recommendations.append(f"Variable '{var}' is highly skewed. Consider transformation.")
                elif skewness > 1:
                    recommendations.append(f"Variable '{var}' is moderately skewed. Transformation may be beneficial.")
        
        # Variability recommendations
        for var, stats_info in summary_stats.items():
            if isinstance(stats_info, dict) and 'cv' in stats_info:
                cv = stats_info['cv']
                if cv > 1:
                    recommendations.append(f"Variable '{var}' has high variability (CV > 1). Consider standardization.")
        
        # Cross-tabulation recommendations
        crosstab = results.get('cross_tabulation', {})
        significant_associations = [key for key, info in crosstab.items() 
                                  if info.get('chi_square_test', {}).get('p_value', 1) < 0.05]
        if significant_associations:
            recommendations.append(f"Significant associations found: {', '.join(significant_associations)}. Consider in further analysis.")
        
        # Data quality recommendations
        quality_score = results.get('data_quality', {}).get('overall_quality_score', 100)
        if quality_score < 70:
            recommendations.append("Overall data quality is low. Consider data cleaning and preprocessing.")
        elif quality_score < 85:
            recommendations.append("Data quality is moderate. Some preprocessing may improve analysis quality.")
        
        return recommendations
    
    # Helper methods
    def _calculate_mode(self, data: pd.Series) -> Union[float, str, None]:
        """Calculate mode of a series."""
        try:
            mode_result = data.mode()
            return mode_result.iloc[0] if len(mode_result) > 0 else None
        except:
            return None
    
    def _count_outliers_iqr(self, data: pd.Series) -> int:
        """Count outliers using IQR method."""
        try:
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return int(((data < lower_bound) | (data > upper_bound)).sum())
        except:
            return 0
    
    def _count_outliers_zscore(self, data: pd.Series) -> int:
        """Count outliers using Z-score method."""
        try:
            z_scores = np.abs(stats.zscore(data))
            return int((z_scores > 3).sum())
        except:
            return 0
    
    def _calculate_ci_mean(self, data: pd.Series, confidence: float) -> Tuple[float, float]:
        """Calculate confidence interval for mean."""
        try:
            mean = data.mean()
            sem = stats.sem(data)
            h = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
            return (float(mean - h), float(mean + h))
        except:
            return (np.nan, np.nan)
    
    def _calculate_ci_median(self, data: pd.Series, confidence: float) -> Tuple[float, float]:
        """Calculate confidence interval for median (bootstrap)."""
        try:
            # Simple bootstrap for median CI
            n_bootstrap = 1000
            bootstrap_medians = []
            
            for _ in range(n_bootstrap):
                sample = np.random.choice(data, size=len(data), replace=True)
                bootstrap_medians.append(np.median(sample))
            
            alpha = 1 - confidence
            lower = np.percentile(bootstrap_medians, 100 * alpha / 2)
            upper = np.percentile(bootstrap_medians, 100 * (1 - alpha / 2))
            
            return (float(lower), float(upper))
        except:
            return (np.nan, np.nan)
    
    def _calculate_entropy(self, data: pd.Series) -> float:
        """Calculate Shannon entropy."""
        try:
            proportions = data.value_counts(normalize=True)
            return float(-np.sum(proportions * np.log2(proportions)))
        except:
            return np.nan
    
    def _calculate_gini_simpson(self, data: pd.Series) -> float:
        """Calculate Gini-Simpson diversity index."""
        try:
            proportions = data.value_counts(normalize=True)
            return float(1 - np.sum(proportions ** 2))
        except:
            return np.nan
    
    def _calculate_concentration_ratio(self, data: pd.Series) -> float:
        """Calculate concentration ratio (top category proportion)."""
        try:
            proportions = data.value_counts(normalize=True)
            return float(proportions.iloc[0])
        except:
            return np.nan
    
    def _calculate_herfindahl_index(self, data: pd.Series) -> float:
        """Calculate Herfindahl-Hirschman Index."""
        try:
            proportions = data.value_counts(normalize=True)
            return float(np.sum(proportions ** 2))
        except:
            return np.nan
    
    def _calculate_evenness(self, data: pd.Series) -> float:
        """Calculate evenness index."""
        try:
            proportions = data.value_counts(normalize=True)
            entropy = -np.sum(proportions * np.log(proportions))
            max_entropy = np.log(len(proportions))
            return float(entropy / max_entropy) if max_entropy > 0 else 0
        except:
            return np.nan
    
    def _test_normality(self, data: pd.Series) -> Dict[str, Any]:
        """Test normality using multiple tests."""
        tests = {}
        
        try:
            # Shapiro-Wilk test (for n < 5000)
            if len(data) < 5000:
                shapiro_stat, shapiro_p = stats.shapiro(data)
                tests['shapiro_wilk'] = {
                    'statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'interpretation': 'Normal' if shapiro_p > 0.05 else 'Non-normal'
                }
        except:
            pass
        
        try:
            # D'Agostino's normality test
            if len(data) >= 8:
                dagostino_stat, dagostino_p = stats.normaltest(data)
                tests['dagostino'] = {
                    'statistic': float(dagostino_stat),
                    'p_value': float(dagostino_p),
                    'interpretation': 'Normal' if dagostino_p > 0.05 else 'Non-normal'
                }
        except:
            pass
        
        try:
            # Jarque-Bera test
            jb_stat, jb_p = stats.jarque_bera(data)
            tests['jarque_bera'] = {
                'statistic': float(jb_stat),
                'p_value': float(jb_p),
                'interpretation': 'Normal' if jb_p > 0.05 else 'Non-normal'
            }
        except:
            pass
        
        return tests
    
    def _fit_distributions(self, data: pd.Series) -> Dict[str, Any]:
        """Fit common distributions to data."""
        distributions = {}
        
        # List of distributions to try
        dist_names = ['norm', 'lognorm', 'expon', 'gamma', 'beta']
        
        for dist_name in dist_names:
            try:
                dist = getattr(stats, dist_name)
                params = dist.fit(data)
                
                # Kolmogorov-Smirnov test
                ks_stat, ks_p = stats.kstest(data, lambda x: dist.cdf(x, *params))
                
                distributions[dist_name] = {
                    'parameters': [float(p) for p in params],
                    'ks_statistic': float(ks_stat),
                    'ks_p_value': float(ks_p),
                    'goodness_of_fit': 'Good' if ks_p > 0.05 else 'Poor'
                }
            except:
                continue
        
        return distributions
    
    def _analyze_shape(self, data: pd.Series) -> Dict[str, Any]:
        """Analyze distribution shape characteristics."""
        try:
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)
            
            return {
                'skewness': float(skewness),
                'skewness_interpretation': self._interpret_skewness(skewness),
                'kurtosis': float(kurtosis),
                'kurtosis_interpretation': self._interpret_kurtosis(kurtosis)
            }
        except:
            return {}
    
    def _analyze_tails(self, data: pd.Series) -> Dict[str, Any]:
        """Analyze tail behavior."""
        try:
            # Calculate tail ratios
            p1 = data.quantile(0.01)
            p5 = data.quantile(0.05)
            p95 = data.quantile(0.95)
            p99 = data.quantile(0.99)
            median = data.median()
            
            left_tail_ratio = (median - p1) / (p95 - p5) if (p95 - p5) != 0 else np.inf
            right_tail_ratio = (p99 - median) / (p95 - p5) if (p95 - p5) != 0 else np.inf
            
            return {
                'left_tail_ratio': float(left_tail_ratio),
                'right_tail_ratio': float(right_tail_ratio),
                'tail_symmetry': 'Symmetric' if abs(left_tail_ratio - right_tail_ratio) < 0.5 else 'Asymmetric'
            }
        except:
            return {}
    
    def _interpret_numeric_stats(self, stats_dict: Dict[str, Any]) -> Dict[str, str]:
        """Interpret numeric statistics."""
        interpretations = {}
        
        # Coefficient of variation
        cv = stats_dict.get('cv', 0)
        if cv < 0.1:
            interpretations['variability'] = 'Low variability'
        elif cv < 0.3:
            interpretations['variability'] = 'Moderate variability'
        else:
            interpretations['variability'] = 'High variability'
        
        # Skewness
        skewness = stats_dict.get('skewness', 0)
        interpretations['skewness'] = self._interpret_skewness(skewness)
        
        # Kurtosis
        kurtosis = stats_dict.get('kurtosis', 0)
        interpretations['kurtosis'] = self._interpret_kurtosis(kurtosis)
        
        return interpretations
    
    def _interpret_categorical_stats(self, stats_dict: Dict[str, Any]) -> Dict[str, str]:
        """Interpret categorical statistics."""
        interpretations = {}
        
        # Concentration
        mode_prop = stats_dict.get('mode_proportion', 0)
        if mode_prop > 0.8:
            interpretations['concentration'] = 'Highly concentrated'
        elif mode_prop > 0.5:
            interpretations['concentration'] = 'Moderately concentrated'
        else:
            interpretations['concentration'] = 'Well distributed'
        
        # Diversity
        entropy = stats_dict.get('entropy', 0)
        unique_count = stats_dict.get('unique_count', 1)
        max_entropy = np.log2(unique_count) if unique_count > 1 else 1
        relative_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        if relative_entropy > 0.8:
            interpretations['diversity'] = 'High diversity'
        elif relative_entropy > 0.5:
            interpretations['diversity'] = 'Moderate diversity'
        else:
            interpretations['diversity'] = 'Low diversity'
        
        return interpretations
    
    def _interpret_skewness(self, skewness: float) -> str:
        """Interpret skewness value."""
        if abs(skewness) < 0.5:
            return 'Approximately symmetric'
        elif abs(skewness) < 1:
            return 'Moderately skewed'
        else:
            return 'Highly skewed'
    
    def _interpret_kurtosis(self, kurtosis: float) -> str:
        """Interpret kurtosis value."""
        if abs(kurtosis) < 0.5:
            return 'Approximately normal (mesokurtic)'
        elif kurtosis > 0.5:
            return 'Heavy-tailed (leptokurtic)'
        else:
            return 'Light-tailed (platykurtic)'
    
    def _interpret_cramers_v(self, cramers_v: float) -> str:
        """Interpret Cramér's V effect size."""
        if cramers_v < 0.1:
            return 'Negligible association'
        elif cramers_v < 0.3:
            return 'Weak association'
        elif cramers_v < 0.5:
            return 'Moderate association'
        else:
            return 'Strong association'
    
    def _assess_missing_severity(self, missing_percent: float) -> str:
        """Assess severity of missing data."""
        if missing_percent < 5:
            return 'Low'
        elif missing_percent < 20:
            return 'Moderate'
        else:
            return 'High'
    
    def _suggest_data_type(self, series: pd.Series) -> str:
        """Suggest appropriate data type for a series."""
        if series.dtype in ['int64', 'float64']:
            return 'numeric'
        elif series.dtype in ['object', 'category']:
            if series.nunique() <= 10:
                return 'categorical'
            else:
                return 'text'
        else:
            return str(series.dtype)
    
    def _check_type_consistency(self, series: pd.Series) -> bool:
        """Check if data type is consistent with data."""
        try:
            if series.dtype in ['int64', 'float64']:
                # Check if all non-null values are actually numeric
                return pd.to_numeric(series.dropna(), errors='coerce').notna().all()
            return True
        except:
            return False
    
    def _assess_value_range(self, data: pd.Series, var_name: str) -> bool:
        """Assess if value range is reasonable."""
        try:
            # Simple heuristics for common variable types
            min_val, max_val = data.min(), data.max()
            
            # Age variables
            if 'age' in var_name.lower():
                return 0 <= min_val <= 120 and 0 <= max_val <= 120
            
            # Percentage variables
            if any(term in var_name.lower() for term in ['percent', 'pct', 'rate']):
                return 0 <= min_val <= 100 and 0 <= max_val <= 100
            
            # Default: check for extreme values
            return not (abs(min_val) > 1e6 or abs(max_val) > 1e6)
        except:
            return True
    
    def _calculate_quality_score(self, quality_assessment: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-100)."""
        try:
            score = 100
            
            # Penalize for missing data
            missing_data = quality_assessment.get('missing_data', {})
            for var_info in missing_data.values():
                missing_pct = var_info.get('percentage', 0)
                if missing_pct > 20:
                    score -= 20
                elif missing_pct > 10:
                    score -= 10
                elif missing_pct > 5:
                    score -= 5
            
            # Penalize for outliers
            outliers = quality_assessment.get('outliers', {})
            for var_info in outliers.values():
                outlier_pct = var_info.get('outlier_percentage', 0)
                if outlier_pct > 15:
                    score -= 15
                elif outlier_pct > 10:
                    score -= 10
                elif outlier_pct > 5:
                    score -= 5
            
            # Penalize for duplicates
            duplicate_pct = quality_assessment.get('duplicates', {}).get('duplicate_percentage', 0)
            if duplicate_pct > 10:
                score -= 15
            elif duplicate_pct > 5:
                score -= 10
            
            # Penalize for type inconsistencies
            data_types = quality_assessment.get('data_types', {})
            inconsistent_types = sum(1 for var_info in data_types.values() 
                                   if not var_info.get('type_consistency', True))
            score -= inconsistent_types * 5
            
            return max(0, min(100, score))
        except:
            return 50  # Default moderate score if calculation fails