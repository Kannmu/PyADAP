"""Advanced Statistical Analysis for PyADAP 3.0

This module provides comprehensive statistical analysis with automatic test selection,
effect size calculations, power analysis, and robust reporting.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    shapiro, normaltest, jarque_bera, anderson, kstest,
    levene, bartlett, fligner,
    ttest_ind, ttest_rel,
    mannwhitneyu, wilcoxon, kruskal,
    f_oneway, friedmanchisquare,
    chi2_contingency, pearsonr, spearmanr
)
from statsmodels.stats.power import (
    ttest_power
)
# Import effect size functions from utils.helpers
# jarque_bera is already imported from scipy.stats
from statsmodels.stats.stattools import durbin_watson
import pingouin as pg

from ..config import Config
from ..utils import Logger, Validator


class TestType(Enum):
    """Statistical test types."""
    NORMALITY = "normality"
    HOMOGENEITY = "homogeneity"
    INDEPENDENCE = "independence"
    COMPARISON = "comparison"
    CORRELATION = "correlation"
    REGRESSION = "regression"
    ANOVA = "anova"
    NONPARAMETRIC = "nonparametric"
    NONPARAMETRIC_RM_ANOVA = "nonparametric_rm_anova"


@dataclass
class StatisticalResult:
    """Container for statistical test results."""
    test_name: str
    test_type: TestType
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    effect_size_interpretation: Optional[str] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    power: Optional[float] = None
    sample_size: Optional[int] = None
    assumptions_met: Optional[Dict[str, bool]] = None
    interpretation: Optional[str] = None
    recommendations: Optional[List[str]] = None
    additional_info: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


class StatisticalAnalyzer:
    """Advanced statistical analysis with intelligent test selection."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize StatisticalAnalyzer.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.logger = Logger()
        self.validator = Validator()
        
        # Results storage
        self.results: Dict[str, StatisticalResult] = {}
        self.assumptions_results: Dict[str, Dict[str, Any]] = {}
        
        # Analysis metadata
        self.data: Optional[pd.DataFrame] = None
        self.dependent_vars: List[str] = []
        self.independent_vars: List[str] = []
        self.subject_column: Optional[str] = None
        
    def set_data(self, data: pd.DataFrame, 
                 dependent_vars: List[str],
                 independent_vars: List[str],
                 subject_column: Optional[str] = None) -> None:
        """Set data and variables for analysis.
        
        Args:
            data: DataFrame containing the data
            dependent_vars: List of dependent variable names
            independent_vars: List of independent variable names
            subject_column: Name of subject identifier column
        """
        self.data = data.copy()
        self.dependent_vars = dependent_vars
        self.independent_vars = independent_vars
        self.subject_column = subject_column
        
        # Validate variables exist in data
        all_vars = dependent_vars + independent_vars
        if subject_column:
            all_vars.append(subject_column)
        
        missing_vars = [var for var in all_vars if var not in data.columns]
        if missing_vars:
            raise ValueError(f"Variables not found in data: {missing_vars}")
        
        self.logger.info(f"Data set for analysis: {len(data)} observations, {len(dependent_vars)} DVs, {len(independent_vars)} IVs")
    
    def check_assumptions(self) -> Dict[str, Dict[str, Any]]:
        """Comprehensive assumption checking for statistical tests.
        
        Returns:
            Dictionary containing assumption test results
        """
        if self.data is None:
            raise ValueError("No data set. Call set_data() first.")
        
        self.assumptions_results = {}
        
        # Check normality for each dependent variable
        self._check_normality()
        
        # Check homogeneity of variance
        self._check_homogeneity()
        
        # Check independence (if repeated measures)
        self._check_independence()
        
        # Check linearity (for regression/correlation)
        self._check_linearity()
        
        # Check multicollinearity
        self._check_multicollinearity()
        
        return self.assumptions_results
    
    def _check_normality(self) -> None:
        """Check normality assumption for dependent variables."""
        normality_results = {}
        
        for dv in self.dependent_vars:
            if dv not in self.data.columns:
                continue
            
            data_col = self.data[dv].dropna()
            
            if len(data_col) < 3:
                normality_results[dv] = {
                    'sufficient_data': False,
                    'message': 'Insufficient data for normality testing'
                }
                continue
            
            tests = {}
            
            # Shapiro-Wilk test (best for small samples)
            if len(data_col) <= 5000:
                try:
                    stat, p = shapiro(data_col)
                    tests['shapiro_wilk'] = {
                        'statistic': stat,
                        'p_value': p,
                        'normal': p > self.config.statistical.alpha_level
                    }
                except (ValueError, RuntimeError) as e:
                    self.logger.debug(f"Shapiro-Wilk test failed: {str(e)}")
            
            # D'Agostino-Pearson test
            if len(data_col) >= 8:
                try:
                    stat, p = normaltest(data_col)
                    tests['dagostino_pearson'] = {
                        'statistic': stat,
                        'p_value': p,
                        'normal': p > self.config.statistical.alpha_level
                    }
                except (ValueError, RuntimeError) as e:
                    self.logger.debug(f"D'Agostino-Pearson test failed: {str(e)}")
            
            # Jarque-Bera test
            if len(data_col) >= 2:
                try:
                    stat, p = jarque_bera(data_col)
                    tests['jarque_bera'] = {
                        'statistic': stat,
                        'p_value': p,
                        'normal': p > self.config.statistical.alpha_level
                    }
                except (ValueError, RuntimeError, IndexError) as e:
                    self.logger.debug(f"Jarque-Bera test failed for column {dv}: {str(e)}")
            
            # Anderson-Darling test
            try:
                result = anderson(data_col, dist='norm')
                # Use 5% significance level
                critical_value = result.critical_values[2]  # 5% level
                tests['anderson_darling'] = {
                    'statistic': result.statistic,
                    'critical_value': critical_value,
                    'normal': result.statistic < critical_value
                }
            except (ValueError, RuntimeError, IndexError) as e:
                self.logger.debug(f"Anderson-Darling test failed for column {dv}: {str(e)}")
            
            # Kolmogorov-Smirnov test
            try:
                # Compare with normal distribution with same mean and std
                mean, std = data_col.mean(), data_col.std()
                stat, p = kstest(data_col, lambda x: stats.norm.cdf(x, mean, std))
                tests['kolmogorov_smirnov'] = {
                    'statistic': stat,
                    'p_value': p,
                    'normal': p > self.config.statistical.alpha_level
                }
            except (ValueError, RuntimeError, IndexError) as e:
                self.logger.debug(f"Kolmogorov-Smirnov test failed for column {dv}: {str(e)}")
            
            # Overall assessment
            normal_count = sum(1 for test in tests.values() if test.get('normal', False))
            total_tests = len(tests)
            
            normality_results[dv] = {
                'tests': tests,
                'overall_normal': normal_count >= total_tests * 0.6,  # Majority rule
                'normal_test_ratio': normal_count / total_tests if total_tests > 0 else 0,
                'skewness': data_col.skew(),
                'kurtosis': data_col.kurtosis(),
                'sufficient_data': True
            }
        
        self.assumptions_results['normality'] = normality_results
    
    def _check_homogeneity(self) -> None:
        """Check homogeneity of variance assumption."""
        if not self.independent_vars:
            return
        
        homogeneity_results = {}
        
        for dv in self.dependent_vars:
            for iv in self.independent_vars:
                if dv not in self.data.columns or iv not in self.data.columns:
                    continue
                
                # Group data by independent variable
                groups = [group[dv].dropna() for name, group in self.data.groupby(iv)]
                groups = [g for g in groups if len(g) >= 2]  # Need at least 2 observations per group
                
                if len(groups) < 2:
                    continue
                
                tests = {}
                
                # Levene's test (most robust)
                try:
                    stat, p = levene(*groups)
                    tests['levene'] = {
                        'statistic': stat,
                        'p_value': p,
                        'homogeneous': p > self.config.statistical.alpha_level
                    }
                except:
                    pass
                
                # Bartlett's test (assumes normality)
                try:
                    stat, p = bartlett(*groups)
                    tests['bartlett'] = {
                        'statistic': stat,
                        'p_value': p,
                        'homogeneous': p > self.config.statistical.alpha_level
                    }
                except:
                    pass
                
                # Fligner-Killeen test (non-parametric)
                try:
                    stat, p = fligner(*groups)
                    tests['fligner_killeen'] = {
                        'statistic': stat,
                        'p_value': p,
                        'homogeneous': p > self.config.statistical.alpha_level
                    }
                except:
                    pass
                
                # Overall assessment
                homogeneous_count = sum(1 for test in tests.values() if test.get('homogeneous', False))
                total_tests = len(tests)
                
                homogeneity_results[f"{dv}_by_{iv}"] = {
                    'tests': tests,
                    'overall_homogeneous': homogeneous_count >= total_tests * 0.6,
                    'homogeneous_test_ratio': homogeneous_count / total_tests if total_tests > 0 else 0,
                    'group_variances': [g.var() for g in groups],
                    'variance_ratio': max([g.var() for g in groups]) / min([g.var() for g in groups]) if groups else None
                }
        
        self.assumptions_results['homogeneity'] = homogeneity_results
    
    def _check_independence(self) -> None:
        """Check independence assumption."""
        if not self.subject_column:
            return
        
        independence_results = {}
        
        # Check for repeated measures
        subject_counts = self.data[self.subject_column].value_counts()
        repeated_measures = (subject_counts > 1).any()
        
        independence_results['repeated_measures_detected'] = repeated_measures
        
        if repeated_measures:
            independence_results['max_observations_per_subject'] = subject_counts.max()
            independence_results['subjects_with_multiple_observations'] = (subject_counts > 1).sum()
            
            # For time series data, check autocorrelation
            for dv in self.dependent_vars:
                if dv in self.data.columns:
                    # Sort by subject and time (if available)
                    sorted_data = self.data.sort_values([self.subject_column])
                    
                    # Durbin-Watson test for autocorrelation
                    try:
                        dw_stat = durbin_watson(sorted_data[dv].dropna())
                        independence_results[f'{dv}_durbin_watson'] = {
                            'statistic': dw_stat,
                            'interpretation': self._interpret_durbin_watson(dw_stat)
                        }
                    except:
                        pass
        
        self.assumptions_results['independence'] = independence_results
    
    def _interpret_durbin_watson(self, dw_stat: float) -> str:
        """Interpret Durbin-Watson statistic."""
        if dw_stat < 1.5:
            return "Positive autocorrelation detected"
        elif dw_stat > 2.5:
            return "Negative autocorrelation detected"
        else:
            return "No significant autocorrelation"
    
    def _check_linearity(self) -> None:
        """Check linearity assumption for regression/correlation."""
        linearity_results = {}
        
        # Check linearity between continuous variables
        continuous_vars = [var for var in self.dependent_vars + self.independent_vars 
                          if var in self.data.select_dtypes(include=[np.number]).columns]
        
        for i, var1 in enumerate(continuous_vars):
            for var2 in continuous_vars[i+1:]:
                if var1 in self.data.columns and var2 in self.data.columns:
                    # Remove missing values
                    clean_data = self.data[[var1, var2]].dropna()
                    
                    if len(clean_data) < 10:
                        continue
                    
                    # Pearson correlation (assumes linearity)
                    r_pearson, p_pearson = pearsonr(clean_data[var1], clean_data[var2])
                    
                    # Spearman correlation (monotonic relationship)
                    r_spearman, p_spearman = spearmanr(clean_data[var1], clean_data[var2])
                    
                    # If Pearson and Spearman correlations are similar, relationship is likely linear
                    linearity_score = 1 - abs(abs(r_pearson) - abs(r_spearman))
                    
                    linearity_results[f"{var1}_vs_{var2}"] = {
                        'pearson_r': r_pearson,
                        'spearman_r': r_spearman,
                        'linearity_score': linearity_score,
                        'likely_linear': linearity_score > 0.8,
                        'sample_size': len(clean_data)
                    }
        
        self.assumptions_results['linearity'] = linearity_results
    
    def _check_multicollinearity(self) -> None:
        """Check multicollinearity among independent variables."""
        if len(self.independent_vars) < 2:
            return
        
        # Only check numeric variables
        numeric_ivs = [var for var in self.independent_vars 
                      if var in self.data.select_dtypes(include=[np.number]).columns]
        
        if len(numeric_ivs) < 2:
            return
        
        multicollinearity_results = {}
        
        # Correlation matrix
        corr_matrix = self.data[numeric_ivs].corr()
        
        # Find high correlations
        high_correlations = []
        for i, var1 in enumerate(numeric_ivs):
            for var2 in numeric_ivs[i+1:]:
                corr_value = corr_matrix.loc[var1, var2]
                if abs(corr_value) > 0.8:
                    high_correlations.append({
                        'var1': var1,
                        'var2': var2,
                        'correlation': corr_value
                    })
        
        # Calculate VIF (Variance Inflation Factor)
        vif_results = {}
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            
            clean_data = self.data[numeric_ivs].dropna()
            if len(clean_data) > len(numeric_ivs):
                for i, var in enumerate(numeric_ivs):
                    vif = variance_inflation_factor(clean_data.values, i)
                    vif_results[var] = {
                        'vif': vif,
                        'problematic': vif > 5.0  # Common threshold
                    }
        except:
            pass
        
        multicollinearity_results = {
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlations': high_correlations,
            'vif_results': vif_results,
            'multicollinearity_detected': len(high_correlations) > 0 or any(v.get('problematic', False) for v in vif_results.values())
        }
        
        self.assumptions_results['multicollinearity'] = multicollinearity_results
    
    def auto_select_test(self, dependent_var: str, independent_var: str) -> str:
        """Automatically select appropriate statistical test with intelligent decision-making.
        
        Args:
            dependent_var: Name of dependent variable
            independent_var: Name of independent variable
            
        Returns:
            Name of recommended test
        """
        if self.data is None:
            raise ValueError("No data set. Call set_data() first.")
        
        # Get variable types
        dv_type = self._get_variable_type(dependent_var)
        iv_type = self._get_variable_type(independent_var)
        
        # Get number of groups
        n_groups = self.data[independent_var].nunique()
        
        # Check if data is paired/repeated measures
        is_paired = self._is_paired_data(dependent_var, independent_var)
        
        # Check assumptions with intelligent decision-making
        assumptions = self.assumptions_results.get('normality', {}).get(dependent_var, {})
        is_normal = assumptions.get('overall_normal', False)
        
        homogeneity = self.assumptions_results.get('homogeneity', {}).get(f"{dependent_var}_by_{independent_var}", {})
        is_homogeneous = homogeneity.get('overall_homogeneous', True)
        
        # Check sphericity for repeated measures (Author: PyADAP Development Team - Revised)
        sphericity_met = True # Default, will be updated
        sphericity_test_performed = False
        if is_paired and n_groups > 2:
            sphericity_met = self._check_sphericity_assumption(dependent_var, independent_var)
            sphericity_result = self.assumptions_results.get('sphericity', {}).get(f"{dependent_var}_by_{independent_var}", {})
            sphericity_test_performed = sphericity_result.get('test_performed', False)
            
            if sphericity_test_performed:
                sphericity_met = sphericity_result.get('sphericity_met', False) # Use actual test result
            else:
                # If test was not performed for >2 levels (e.g. insufficient data, error during test)
                # Conservatively assume sphericity is violated to prompt correction or non-parametric test.
                self.logger.warning(f"Sphericity test for {dependent_var} by {independent_var} not performed or failed. Assuming sphericity violated for test selection.")
                sphericity_met = False
        
        # Intelligent decision tree for test selection (Author: PyADAP Development Team)
        if dv_type == 'continuous':
            if iv_type == 'categorical':
                if n_groups == 2:
                    if is_paired:
                        # For paired data: normality check with fallback
                        if is_normal:
                            return 'paired_t_test'
                        else:
                            # Try data transformation first
                            if self._should_attempt_transformation(dependent_var):
                                return 'paired_t_test_with_transformation'
                            else:
                                return 'wilcoxon_signed_rank'
                    else:
                        # For independent groups: normality and homogeneity checks
                        if is_normal and is_homogeneous:
                            return 'independent_t_test'
                        elif is_normal and not is_homogeneous:
                            return 'welch_t_test'  # Welch's t-test for unequal variances
                        else:
                            # Try transformation if normality fails
                            if self._should_attempt_transformation(dependent_var):
                                return 'independent_t_test_with_transformation'
                            else:
                                return 'mann_whitney_u'
                elif n_groups > 2:
                    if is_paired:
                        # Repeated measures ANOVA with sphericity consideration
                        if is_normal and sphericity_met:
                            return 'repeated_measures_anova'
                        elif is_normal and not sphericity_met:
                            return 'repeated_measures_anova_corrected'  # With sphericity correction
                        else:
                            # Try transformation or use non-parametric
                            if self._should_attempt_transformation(dependent_var):
                                return 'repeated_measures_anova_with_transformation'
                            else:
                                return 'friedman_test'
                    else:
                        # One-way ANOVA with assumption checks
                        if is_normal and is_homogeneous:
                            return 'one_way_anova'
                        elif is_normal and not is_homogeneous:
                            return 'welch_anova'  # Welch's ANOVA for unequal variances
                        else:
                            # Try transformation or use non-parametric
                            if self._should_attempt_transformation(dependent_var):
                                return 'one_way_anova_with_transformation'
                            else:
                                return 'kruskal_wallis'
            elif iv_type == 'continuous':
                # Correlation analysis with normality consideration
                if is_normal:
                    return 'pearson_correlation'
                else:
                    # Try transformation for better linearity
                    if self._should_attempt_transformation(dependent_var):
                        return 'pearson_correlation_with_transformation'
                    else:
                        return 'spearman_correlation'
        
        elif dv_type == 'categorical':
            if iv_type == 'categorical':
                return 'chi_square_test' if n_groups > 1 else 'binomial_test'
        
        return 'descriptive_statistics'  # Fallback
    
    def _get_variable_type(self, var_name: str) -> str:
        """Determine variable type (continuous, categorical, ordinal)."""
        if var_name not in self.data.columns:
            return 'unknown'
        
        dtype = self.data[var_name].dtype
        
        if pd.api.types.is_numeric_dtype(dtype):
            # Check if it's actually categorical (few unique values)
            unique_ratio = self.data[var_name].nunique() / len(self.data[var_name])
            if unique_ratio < 0.05 and self.data[var_name].nunique() < 10:
                return 'categorical'
            return 'continuous'
        else:
            return 'categorical'
    
    def _is_paired_data(self, dependent_var: str, independent_var: str) -> bool:
        """Check if data represents paired/repeated measures."""
        if not self.subject_column:
            return False
        
        # Check if each subject has multiple observations
        subject_counts = self.data.groupby(self.subject_column)[dependent_var].count()
        return (subject_counts > 1).any()
    
    def _check_sphericity_assumption(self, dependent_var: str, independent_var: str) -> bool:
        """Check sphericity assumption for repeated measures ANOVA using Mauchly's test.
        
        Author: PyADAP Development Team (Revised)
        """
        result_key = f"{dependent_var}_by_{independent_var}"
        self.assumptions_results.setdefault('sphericity', {})

        if self.subject_column is None:
            self.logger.warning(f"Subject column not set. Cannot perform sphericity test for {result_key}.")
            self.assumptions_results['sphericity'][result_key] = {
                'sphericity_met': True,  # Defaulting to True to avoid blocking, but this is not ideal.
                'test_performed': False,
                'message': "Subject column not set."
            }
            return True

        try:
            within_levels = self.data[independent_var].nunique()
            if within_levels <= 2:
                self.logger.info(f"Sphericity assumption is met by definition for {result_key} (<=2 levels).")
                self.assumptions_results['sphericity'][result_key] = {
                    'sphericity_met': True,
                    'test_performed': False,
                    'message': "Sphericity met by definition (<=2 levels)."
                }
                return True

            data_for_sphericity = self.data[[self.subject_column, independent_var, dependent_var]].dropna()
            
            # Check if there's enough data after dropping NaNs
            # Pingouin's sphericity needs at least k subjects for k levels, and k > 2 levels.
            if data_for_sphericity.empty or data_for_sphericity[self.subject_column].nunique() < within_levels or data_for_sphericity[self.subject_column].nunique() < 3: # Need at least 3 subjects for a meaningful test with >2 levels
                self.logger.warning(f"Not enough data or subjects to perform sphericity test for {result_key} after handling NaNs.")
                self.assumptions_results['sphericity'][result_key] = {
                    'sphericity_met': False, # Assume violated to be conservative if test cannot run
                    'test_performed': False,
                    'message': "Not enough data/subjects after NaN handling or for k > 2 levels."
                }
                return False

            sphericity_test_output = pg.sphericity(data=data_for_sphericity, dv=dependent_var, within=independent_var, subject=self.subject_column)
            
            # pingouin.sphericity returns a tuple or a DataFrame depending on version/context.
            # Assuming DataFrame output based on typical pingouin style for consistency.
            if isinstance(sphericity_test_output, pd.DataFrame):
                sphericity_met = sphericity_test_output['spher'].iloc[0]
                W_statistic = sphericity_test_output['W'].iloc[0]
                chi_sq_statistic = sphericity_test_output['chi2'].iloc[0]
                df = sphericity_test_output['dof'].iloc[0]
                p_value = sphericity_test_output['pval'].iloc[0]
            elif isinstance(sphericity_test_output, tuple): # Fallback for tuple output
                sphericity_met, W_statistic, chi_sq_statistic, df, p_value = sphericity_test_output[0], sphericity_test_output[1], sphericity_test_output[2], sphericity_test_output[3], sphericity_test_output[4]
            else:
                raise TypeError(f"Unexpected output type from pg.sphericity: {type(sphericity_test_output)}")

            self.logger.info(f"Sphericity test for {result_key}: W={W_statistic:.4f}, p={p_value:.4f}. Sphericity {'met' if sphericity_met else 'violated'}.")
            
            self.assumptions_results['sphericity'][result_key] = {
                'sphericity_met': sphericity_met,
                'W_statistic': W_statistic,
                'chi_square_statistic': chi_sq_statistic,
                'df': df,
                'p_value': p_value,
                'test_performed': True
            }
            return sphericity_met
            
        except Exception as e:
            self.logger.error(f"Could not check sphericity assumption for {result_key}: {str(e)}")
            self.assumptions_results['sphericity'][result_key] = {
                'sphericity_met': False, # Assume violated on error to be safe
                'test_performed': False,
                'message': f"Error during test: {str(e)}"
            }
            return False
    
    def _should_attempt_transformation(self, variable: str) -> bool:
        """Determine if data transformation should be attempted.
        
        Author: PyADAP Development Team
        """
        try:
            data = self.data[variable].dropna()
            
            # Check if data is suitable for transformation
            if len(data) < 10:
                return False  # Too few observations
            
            # Check if data has positive values (required for log, Box-Cox)
            if data.min() <= 0:
                return False
            
            # Check if data shows strong skewness that might benefit from transformation
            from scipy.stats import skew
            skewness = abs(skew(data))
            
            # Attempt transformation if moderate to high skewness
            return skewness > 1.0
            
        except Exception as e:
            self.logger.warning(f"Could not assess transformation suitability for {variable}: {str(e)}")
            return False
    
    def _apply_best_transformation(self, variable: str):
        """Apply the best transformation for normality.
        
        Author: PyADAP Development Team
        """
        try:
            from scipy import stats
            import numpy as np
            
            data = self.data[variable].dropna()
            if len(data) < 10 or data.min() <= 0:
                return None
            
            transformations = {
                'log': np.log,
                'sqrt': np.sqrt,
                'reciprocal': lambda x: 1/x
            }
            
            best_p_value = 0
            best_transformation = None
            best_data = None
            
            # Test each transformation
            for name, func in transformations.items():
                try:
                    transformed = func(data)
                    if np.isfinite(transformed).all():
                        _, p_value = stats.shapiro(transformed)
                        if p_value > best_p_value:
                            best_p_value = p_value
                            best_transformation = name
                            best_data = transformed
                except:
                    continue
            
            # Try Box-Cox if available
            try:
                transformed, _ = stats.boxcox(data)
                _, p_value = stats.shapiro(transformed)
                if p_value > best_p_value:
                    best_p_value = p_value
                    best_transformation = 'boxcox'
                    best_data = transformed
            except:
                pass
            
            if best_transformation:
                self.logger.info(f"Applied {best_transformation} transformation to {variable} (Shapiro p-value: {best_p_value:.4f})")
                return best_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Transformation failed for {variable}: {str(e)}")
            return None
    
    def _run_test_with_transformation(self, dependent_var: str, independent_var: str, test_name: str) -> StatisticalResult:
        """Run statistical test with data transformation.
        
        Author: PyADAP Development Team
        """
        try:
            # Store original data
            original_data = self.data[dependent_var].copy()
            
            # Apply transformation
            transformed_data = self._apply_best_transformation(dependent_var)
            if transformed_data is None:
                # Fallback to non-parametric test
                return self._run_fallback_test(dependent_var, independent_var)
            
            # Temporarily replace data
            self.data[dependent_var] = transformed_data
            
            # Re-check assumptions with transformed data
            self._check_normality_single_var(dependent_var)
            
            # Run the base test dynamically
            base_test_name = test_name.replace('_with_transformation', '')
            base_test_method_name = f"_run_{base_test_name}"
            
            if hasattr(self, base_test_method_name):
                base_test_method = getattr(self, base_test_method_name)
                # Determine if the method needs two variables (dv, iv) or just one (dv for correlation-like)
                import inspect
                sig = inspect.signature(base_test_method)
                if len(sig.parameters) == 3: # Expects self, dv, iv
                    result = base_test_method(dependent_var, independent_var)
                elif len(sig.parameters) == 2: # Expects self, dv (or var1 for correlation)
                    # This case needs careful handling if independent_var is also needed for context
                    # For now, assuming it's a simple call like correlation where iv is the second var
                    result = base_test_method(dependent_var, independent_var) # Pearson/Spearman take var1, var2
                else:
                    self.logger.error(f"Base test method {base_test_method_name} has unexpected signature.")
                    result = self._run_fallback_test(dependent_var, independent_var)
            else:
                self.logger.error(f"Base test method {base_test_method_name} not found after transformation attempt.")
                result = self._run_fallback_test(dependent_var, independent_var)
            
            # Add transformation note to result
            result.notes = f"Data transformation applied. {result.notes or ''}"
            
            # Restore original data
            self.data[dependent_var] = original_data
            
            return result
            
        except Exception as e:
            # Restore original data and fallback
            if 'original_data' in locals():
                self.data[dependent_var] = original_data
            self.logger.error(f"Transformation failed for {dependent_var}: {str(e)}")
            return self._run_fallback_test(dependent_var, independent_var)
    
    def _run_test_with_correction(self, dependent_var: str, independent_var: str, test_name: str) -> StatisticalResult:
        """Run statistical test with appropriate corrections.
        
        Author: PyADAP Development Team
        """
        try:
            if test_name == 'repeated_measures_anova_corrected':
                return self._run_repeated_measures_anova_corrected(dependent_var, independent_var)
            else:
                # Fallback to base test
                base_test = test_name.replace('_corrected', '')
                if base_test == 'repeated_measures_anova':
                    return self._run_repeated_measures_anova(dependent_var, independent_var)
                else:
                    return self._run_descriptive_statistics(dependent_var, independent_var)
                    
        except Exception as e:
            self.logger.error(f"Corrected test failed for {dependent_var}: {str(e)}")
            return self._run_fallback_test(dependent_var, independent_var)
    
    def _run_fallback_test(self, dependent_var: str, independent_var: str) -> Optional[StatisticalResult]:
        """Run fallback non-parametric test when parametric tests fail.
        
        Author: PyADAP Development Team
        """
        try:
            # Determine appropriate non-parametric test
            n_groups = self.data[independent_var].nunique()
            is_paired = self._is_paired_data(dependent_var, independent_var)
            
            if n_groups == 2:
                if is_paired:
                    return self._run_wilcoxon_signed_rank(dependent_var, independent_var)
                else:
                    return self._run_mann_whitney_u(dependent_var, independent_var)
            elif n_groups > 2:
                if is_paired:
                    return self._run_friedman_test(dependent_var, independent_var)
                else:
                    return self._run_kruskal_wallis(dependent_var, independent_var)
            else:
                return self._run_descriptive_statistics(dependent_var, independent_var)
                
        except Exception as e:
            self.logger.error(f"Fallback test failed for {dependent_var}: {str(e)}")
            return None
    
    def run_comprehensive_analysis(self) -> Dict[str, StatisticalResult]:
        """Run comprehensive statistical analysis with intelligent decision-making.
        
        Returns:
            Dictionary of statistical results
        """
        if self.data is None:
            raise ValueError("No data set. Call set_data() first.")
        
        # Check assumptions first
        self.check_assumptions()
        
        # Store original data for potential transformations (Author: PyADAP Development Team)
        self._original_data = self.data.copy()
        
        # Run analyses for each DV-IV combination with intelligent decision flow
        for dv in self.dependent_vars:
            for iv in self.independent_vars:
                # Pass assumptions from check_assumptions to auto_select_test
                # This requires auto_select_test to accept an assumptions dict
                # For now, auto_select_test re-checks or relies on stored self.assumptions_results
                # Consider refactoring auto_select_test to accept assumptions for clarity
                test_name = self.auto_select_test(dv, iv) 
                result_key = f"{dv}_by_{iv}"
                result = None
                self.logger.info(f"For {result_key}, auto_select_test chose: {test_name}")

                try:
                    # Transformation logic needs to be integrated with the new test selection
                    # This is a simplified integration. A more robust way would be to pass the base test name
                    # to _run_test_with_transformation and let it call the correct underlying method.
                    
                    # Determine if transformation is suggested by auto_select_test or general config
                    # Current auto_select_test doesn't explicitly return '_with_transformation' suffix.
                    # We rely on checking normality assumption from self.assumptions_results here.
                    normality_passed = self.assumptions_results.get('normality', {}).get(dv, {}).get('overall_normal', True)
                    apply_transform = self.config.analysis.apply_transformations and not normality_passed

                    if test_name == 'repeated_measures_anova_corrected':
                        if apply_transform:
                            # _run_test_with_transformation needs to know which actual method to call after transform
                            # We pass the method name string, and _run_test_with_transformation will call it.
                            result = self._run_test_with_transformation(dv, iv, '_run_repeated_measures_anova_corrected')
                        else:
                            result = self._run_repeated_measures_anova_corrected(dv, iv)
                    elif test_name == 'repeated_measures_anova':
                        if apply_transform:
                            result = self._run_test_with_transformation(dv, iv, '_run_repeated_measures_anova')
                        else:
                            result = self._run_repeated_measures_anova(dv, iv)
                    # Add other specific test calls here if they need special handling with transformation
                    # For example, if _run_test_with_transformation needs to call them by method name string:
                    # elif test_name == 'independent_t_test':
                    #     if apply_transform:
                    #        result = self._run_test_with_transformation(dv, iv, '_run_independent_t_test')
                    #     else:
                    #        result = self._run_independent_t_test(dv, iv)
                    # ... and so on for other tests that might be transformed.
                    
                    # General case for other tests if not handled above for transformation
                    else:
                        if apply_transform and hasattr(self, f"_run_{test_name}"):
                             # Assuming _run_test_with_transformation can take the base test name
                             # and call the corresponding _run_{base_test_name} method.
                             # This requires _run_test_with_transformation to be adapted or a new helper.
                             # For now, let's assume _run_test_with_transformation is smart enough or we call _execute_test.
                             # The original _run_test_with_transformation took a test_name that was then cleaned.
                             # We might need to adjust _run_test_with_transformation to accept a callable or method name.\
                             # Let's simplify: if transform is needed, _run_test_with_transformation handles it, then calls _execute_test with original test_name.\
                             # This means _run_test_with_transformation needs to be aware of the original test_name selected by auto_select_test.\
                             # The current _run_test_with_transformation has a hardcoded list of tests it can run after transform.
                             # This needs to be more dynamic.
                             # For now, let's assume _run_test_with_transformation is called with the *selected* test_name.
                             # And it internally figures out the base test to run.
                            self.logger.info(f"Attempting transformation for {test_name} on {result_key} as normality failed.")
                            result = self._run_test_with_transformation(dv, iv, test_name) # Pass the original test_name
                        elif hasattr(self, f"_run_{test_name}"):
                            result = getattr(self, f"_run_{test_name}")(dv, iv)
                        elif test_name: # If test_name is not a direct _run_ method, use _execute_test as a fallback dispatcher
                            result = self._execute_test(test_name, dv, iv)
                        else:
                            self.logger.warning(f"No specific run method or _execute_test case for {test_name} on {result_key}. Running descriptives.")
                            result = self._run_descriptive_statistics(dv, iv)

                    if result:
                        self.results[result_key] = result
                        self.logger.info(f"Analysis for {result_key} completed. Test: {result.test_name}, P-value: {result.p_value}")
                    else:
                        self.logger.warning(f"Analysis for {result_key} (Test: {test_name}) did not yield a result.")
                
                except Exception as e:
                    self.logger.error(f"Failed to run analysis for {result_key} (intended test: {test_name}): {str(e)}", exc_info=True)
                    # Fallback to non-parametric test or descriptives
                    try:
                        self.logger.info(f"Attempting fallback test for {result_key} due to error.")
                        fallback_result = self._run_fallback_test(dv, iv)
                        if fallback_result:
                            self.results[result_key] = fallback_result
                            self.logger.info(f"Fallback test for {result_key} completed: {fallback_result.test_name}")
                        else:
                            self.logger.warning(f"Fallback test for {result_key} also failed or returned no result. Storing error.")
                            self.results[result_key] = StatisticalResult(
                                test_name=f"Error in analysis for {result_key}",
                                test_type=TestType.COMPARISON,  # Or a more generic error type
                                statistic=np.nan,
                                p_value=np.nan,
                                interpretation=f"Analysis and fallback failed: {str(e)}",
                                notes=f"Original error: {str(e)}."
                            )
                    except Exception as fe:
                        self.logger.error(f"Fallback test itself failed for {result_key}: {str(fe)}", exc_info=True)
                        self.results[result_key] = StatisticalResult(
                            test_name=f"Critical Error in analysis for {result_key}",
                            test_type=TestType.COMPARISON, # Or a more generic error type
                            statistic=np.nan,
                            p_value=np.nan,
                            interpretation=f"Analysis and fallback failed critically. Original error: {str(e)}. Fallback error: {str(fe)}",
                            notes=f"Original error: {str(e)}. Fallback error: {str(fe)}."
                        )
        
        return self.results
    
    def _run_independent_t_test(self, dv: str, iv: str) -> StatisticalResult:
        """Run independent samples t-test."""
        groups = [group[dv].dropna() for name, group in self.data.groupby(iv)]
        
        if len(groups) != 2:
            raise ValueError("Independent t-test requires exactly 2 groups")
        
        group1, group2 = groups
        
        # Perform t-test
        statistic, p_value = ttest_ind(group1, group2)
        
        # Calculate effect size (Cohen's d)
        effect_size = self._calculate_cohens_d(group1, group2)
        effect_interpretation = self._interpret_cohens_d(effect_size)
        
        # Calculate confidence interval
        ci = self._calculate_t_test_ci(group1, group2)
        
        # Calculate power
        power = self._calculate_t_test_power(group1, group2, effect_size)
        
        # Check assumptions
        assumptions = {
            'normality': self.assumptions_results.get('normality', {}).get(dv, {}).get('overall_normal', False),
            'homogeneity': self.assumptions_results.get('homogeneity', {}).get(f"{dv}_by_{iv}", {}).get('overall_homogeneous', True)
        }
        
        return StatisticalResult(
            test_name="Independent Samples t-test",
            test_type=TestType.COMPARISON,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            effect_size_interpretation=effect_interpretation,
            confidence_interval=ci,
            power=power,
            sample_size=len(group1) + len(group2),
            assumptions_met=assumptions,
            interpretation=self._interpret_p_value(p_value),
            recommendations=self._generate_recommendations("independent_t_test", assumptions, power)
        )
    
    def _calculate_cohens_d(self, group1: pd.Series, group2: pd.Series) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        return (group1.mean() - group2.mean()) / pooled_std
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _run_welch_t_test(self, dv: str, iv: str) -> StatisticalResult:
        """Run Welch's t-test (unequal variances t-test).
        
        Author: PyADAP Development Team
        """
        groups = [group[dv].dropna() for name, group in self.data.groupby(iv)]
        
        if len(groups) != 2:
            raise ValueError("Welch's t-test requires exactly 2 groups")
        
        group1, group2 = groups
        
        # Perform Welch's t-test (equal_var=False)
        statistic, p_value = ttest_ind(group1, group2, equal_var=False)
        
        # Calculate effect size (Cohen's d)
        effect_size = self._calculate_cohens_d(group1, group2)
        effect_interpretation = self._interpret_cohens_d(effect_size)
        
        # Calculate confidence interval for Welch's t-test
        ci = self._calculate_welch_t_test_ci(group1, group2)
        
        # Calculate power
        power = self._calculate_t_test_power(group1, group2, effect_size)
        
        # Check assumptions
        assumptions = {
            'normality': self.assumptions_results.get('normality', {}).get(dv, {}).get('overall_normal', False),
            'homogeneity': False  # Welch's t-test assumes unequal variances
        }
        
        return StatisticalResult(
            test_name="Welch's t-test",
            test_type=TestType.COMPARISON,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            effect_size_interpretation=effect_interpretation,
            confidence_interval=ci,
            power=power,
            sample_size=len(group1) + len(group2),
            assumptions_met=assumptions,
            interpretation=self._interpret_p_value(p_value),
            recommendations=self._generate_recommendations("welch_t_test", assumptions, power),
            notes="Welch's t-test used due to unequal variances"
        )
    
    def _run_welch_anova(self, dv: str, iv: str) -> StatisticalResult:
        """Run Welch's ANOVA (unequal variances one-way ANOVA) using pingouin.
        
        Author: PyADAP Development Team (Revised)
        """
        try:
            groups_data = self.data.groupby(iv)[dv]
            if groups_data.ngroups < 2:
                raise ValueError("Welch's ANOVA requires at least 2 groups")

            # Pingouin's welch_anova expects data in long format
            anova_results = pg.welch_anova(data=self.data, dv=dv, between=iv)
            
            statistic = anova_results['F'].iloc[0]
            p_value = anova_results['p-unc'].iloc[0]
            # Pingouin's welch_anova does not directly return eta-squared.
            # We can calculate Omega-squared or Epsilon-squared as effect size for Welch's ANOVA.
            # For simplicity, let's use a common approximation for eta-squared or report what pingouin provides if any.
            # Pingouin's anova function (not welch_anova) returns partial_eta_sq ('np2') or eta_sq ('n2').
            # We might need to calculate it manually or use another library if pingouin.welch_anova doesn't offer it.
            # Let's calculate eta-squared using the formula: SS_between / SS_total
            # This is an approximation for Welch's ANOVA context.
            # Alternatively, pg.anova might be used with welch=True if available, or calculate manually.
            # For now, let's use the SS values if available from pingouin's output or calculate manually.
            # The anova_results from pg.welch_anova is a simple table with F, p, dof.
            # We will use the _calculate_eta_squared_welch helper for an approximation.
            raw_groups = [group_data.dropna() for _, group_data in groups_data]
            effect_size = self._calculate_eta_squared_welch(raw_groups) # Using existing helper
            effect_interpretation = self._interpret_eta_squared(effect_size)
            
            assumptions = {
                'normality': self.assumptions_results.get('normality', {}).get(dv, {}).get('overall_normal', False),
                'homogeneity': False  # Welch's ANOVA is used when homogeneity is violated
            }
            
            return StatisticalResult(
                test_name="Welch's ANOVA",
                test_type=TestType.ANOVA, # Changed from COMPARISON to ANOVA for consistency
                statistic=statistic,
                p_value=p_value,
                effect_size=effect_size,
                effect_size_interpretation=effect_interpretation,
                sample_size=len(self.data.dropna(subset=[dv, iv])),
                assumptions_met=assumptions,
                interpretation=self._interpret_p_value(p_value),
                recommendations=self._generate_recommendations("welch_anova", assumptions, None),
                notes="Welch's ANOVA used due to unequal variances (via pingouin).",
                additional_info={'df_num': anova_results['ddof1'].iloc[0], 'df_den': anova_results['ddof2'].iloc[0]}
            )
            
        except Exception as e:
            self.logger.error(f"Welch's ANOVA failed for {dv} by {iv}: {str(e)}")
            # Fallback to Kruskal-Wallis
            return self._run_kruskal_wallis(dv, iv)
    
    def _run_repeated_measures_anova_corrected(self, dv: str, iv: str) -> StatisticalResult:
        """Run repeated measures ANOVA with sphericity correction using pingouin.
        
        Author: PyADAP Development Team (Revised)
        """
        result_key = f"{dv}_by_{iv}"
        sphericity_details = self.assumptions_results.get('sphericity', {}).get(result_key, {})
        
        correction_method = 'gg' # Default to Greenhouse-Geisser
        epsilon_val = None
        correction_note = "Sphericity correction applied."

        # Pingouin's rm_anova can auto-select correction if sphericity is violated.
        # We can also be more explicit based on Mauchly's p-value and epsilon if available from sphericity test.
        # For simplicity here, we'll rely on pingouin's auto or a default if sphericity_met is False.

        sphericity_met = sphericity_details.get('sphericity_met', True) # Assume met if no info
        if sphericity_details.get('test_performed', False) and not sphericity_met:
            # Pingouin's rm_anova has an 'auto' correction which applies GG if Mauchly p < .05
            # and HF if Mauchly p < .05 AND GG epsilon > .75. We can just set correction to True or 'auto'.
            # For more control, one might inspect sphericity_test_output['eps'] from pg.sphericity
            # but pg.rm_anova handles this well with correction='auto' or correction=True.
            correction_method = 'auto' # Let pingouin decide GG or HF based on its internal epsilon calculation
            correction_note = "Sphericity correction (auto GG/HF) applied due to Mauchly's test."
            # If we had epsilon from pg.sphericity, we could log it or choose explicitly:
            # epsilon_val = sphericity_details.get('epsilon_value_from_pg_sphericity') # Fictional key for now
            # if epsilon_val:
            #    correction_note += f" (ε_mauchly_approx = {epsilon_val:.3f})"

        elif not sphericity_details.get('test_performed', False) and self.data[iv].nunique() > 2:
            # If sphericity test couldn't be performed (e.g. data issues) for >2 levels, apply correction conservatively.
            correction_method = 'gg' # Conservative Greenhouse-Geisser
            correction_note = "Sphericity correction (GG) conservatively applied as test could not be performed."

        try:
            # Use pingouin for repeated measures ANOVA with specified correction
            # The 'correction' parameter in pg.rm_anova handles this.
            # If sphericity_met is True, correction='auto' or correction=True will not apply correction.
            # If sphericity_met is False, correction='auto' or correction=True will apply GG or HF.
            # We use 'auto' when sphericity is violated, otherwise no correction is applied by pingouin by default.
            
            rm_anova_results = pg.rm_anova(
                data=self.data,
                dv=dv,
                within=iv,
                subject=self.subject_column,
                correction=correction_method if not sphericity_met else False, # Apply correction only if sphericity is violated
                detailed=True
            )

            # Extract relevant values from the anova table
            # The row for the 'within' factor is usually the first one if no between-subject factors.
            # If there are multiple rows (e.g. interactions), this needs careful handling.
            # Assuming 'iv' is the main effect of interest.
            factor_row = rm_anova_results[rm_anova_results['Source'] == iv]
            if factor_row.empty:
                 # Fallback if the exact factor name isn't found, take the first relevant row.
                 # This might happen if pingouin names sources differently in some contexts.
                 # A more robust way would be to identify the correct row based on 'within' factor.
                 if not rm_anova_results.empty:
                     factor_row = rm_anova_results.iloc[[0]] # Take the first row as a guess
                 else:
                     raise ValueError("ANOVA results table is empty or factor not found.")
            
            p_value_col = 'p-GG-corr' if 'p-GG-corr' in factor_row.columns and not sphericity_met and correction_method in ['gg', 'auto'] else \
                          'p-HF-corr' if 'p-HF-corr' in factor_row.columns and not sphericity_met and correction_method in ['hf', 'auto'] else \
                          'p-unc' # Default to uncorrected p-value
            
            statistic = factor_row['F'].iloc[0]
            p_value = factor_row[p_value_col].iloc[0]
            effect_size = factor_row['np2'].iloc[0]  # partial eta-squared
            df_num = factor_row['ddof1'].iloc[0]
            df_den = factor_row['ddof2'].iloc[0]

            final_test_name = "Repeated Measures ANOVA"
            if not sphericity_met and correction_method != False:
                if 'GG' in p_value_col:
                    final_test_name += " (Greenhouse-Geisser corrected)"
                    correction_note = f"Greenhouse-Geisser correction applied. Epsilon (GG): {factor_row['eps'].iloc[0]:.3f}"
                elif 'HF' in p_value_col:
                    final_test_name += " (Huynh-Feldt corrected)"
                    correction_note = f"Huynh-Feldt correction applied. Epsilon (HF): {factor_row['eps'].iloc[0]:.3f}"
                else: # If 'auto' was used and it decided no correction or used uncorrected p-value
                    correction_note = "Sphericity violated, but uncorrected p-value used or correction type unclear from output."
            elif sphericity_met:
                correction_note = "Sphericity assumption met."

            return StatisticalResult(
                test_name=final_test_name,
                test_type=TestType.ANOVA,
                statistic=statistic,
                p_value=p_value,
                effect_size=effect_size,
                effect_size_interpretation=self._interpret_eta_squared(effect_size),
                sample_size=self.data[self.subject_column].nunique(), # Number of subjects
                interpretation=self._interpret_p_value(p_value),
                assumptions_met={'sphericity': sphericity_met},
                notes=correction_note,
                additional_info={'df_num': df_num, 'df_den': df_den, 'epsilon': factor_row['eps'].iloc[0] if 'eps' in factor_row.columns else None}
            )
            
        except Exception as e:
            self.logger.error(f"Corrected repeated measures ANOVA failed for {result_key}: {str(e)}. Falling back to Friedman test.")
            # Fallback to Friedman test if rm_anova with correction fails
            return self._run_friedman_test(dv, iv)
    
    def _check_normality_single_var(self, variable: str):
        """Check normality for a single variable.
        
        Author: PyADAP Development Team
        """
        try:
            data = self.data[variable].dropna()
            if len(data) < 3:
                return
            
            # Shapiro-Wilk test
            _, p_shapiro = shapiro(data)
            
            # Store result
            if 'normality' not in self.assumptions_results:
                self.assumptions_results['normality'] = {}
            
            self.assumptions_results['normality'][variable] = {
                'shapiro_wilk': {'statistic': _, 'p_value': p_shapiro},
                'overall_normal': p_shapiro > self.config.statistical.alpha
            }
            
        except Exception as e:
            self.logger.warning(f"Normality check failed for {variable}: {str(e)}")
    
    def _calculate_welch_t_test_ci(self, group1: pd.Series, group2: pd.Series, alpha: float = 0.05) -> Tuple[float, float]:
        """Calculate confidence interval for Welch's t-test.
        
        Author: PyADAP Development Team
        """
        try:
            from scipy.stats import t
            
            n1, n2 = len(group1), len(group2)
            m1, m2 = group1.mean(), group2.mean()
            s1, s2 = group1.std(ddof=1), group2.std(ddof=1)
            
            # Welch's degrees of freedom
            df = (s1**2/n1 + s2**2/n2)**2 / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
            
            # Standard error
            se = np.sqrt(s1**2/n1 + s2**2/n2)
            
            # Critical value
            t_crit = t.ppf(1 - alpha/2, df)
            
            # Confidence interval
            diff = m1 - m2
            margin = t_crit * se
            
            return (diff - margin, diff + margin)
            
        except Exception as e:
            self.logger.warning(f"Could not calculate Welch's t-test CI: {str(e)}")
            return (np.nan, np.nan)
    
    def _calculate_eta_squared_welch(self, groups) -> float:
        """Calculate eta-squared for Welch's ANOVA (approximation).
        
        Author: PyADAP Development Team
        """
        try:
            # Simple approximation of eta-squared for unequal variances
            all_data = np.concatenate(groups)
            grand_mean = np.mean(all_data)
            
            # Between-group sum of squares (weighted)
            ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)
            
            # Total sum of squares
            ss_total = np.sum((all_data - grand_mean)**2)
            
            return ss_between / ss_total if ss_total > 0 else 0
            
        except Exception as e:
            self.logger.warning(f"Could not calculate eta-squared for Welch's ANOVA: {str(e)}")
            return 0
    
    def _calculate_t_test_ci(self, group1: pd.Series, group2: pd.Series, alpha: float = 0.05) -> Tuple[float, float]:
        """Calculate confidence interval for t-test."""
        n1, n2 = len(group1), len(group2)
        mean_diff = group1.mean() - group2.mean()
        
        # Standard error of the difference
        se_diff = np.sqrt(group1.var(ddof=1)/n1 + group2.var(ddof=1)/n2)
        
        # Degrees of freedom
        df = n1 + n2 - 2
        
        # Critical t-value
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        # Confidence interval
        margin_error = t_critical * se_diff
        
        return (mean_diff - margin_error, mean_diff + margin_error)
    
    def _calculate_t_test_power(self, group1: pd.Series, group2: pd.Series, effect_size: float) -> float:
        """Calculate statistical power for t-test."""
        try:
            n1, n2 = len(group1), len(group2)
            # Use harmonic mean for unequal sample sizes
            n_harmonic = 2 * n1 * n2 / (n1 + n2)
            
            power = ttest_power(effect_size, n_harmonic, self.config.statistical.alpha_level)
            return power
        except:
            return None
    
    def _run_paired_t_test(self, dv: str, iv: str) -> StatisticalResult:
        """Run paired samples t-test using pingouin for enhanced results.

        Author: PyADAP Development Team (Revised)
        """
        if not self.subject_column:
            raise ValueError("Subject column must be specified for paired t-test.")

        # Ensure the independent variable has exactly two levels for pairing
        iv_levels = self.data[iv].unique()
        if len(iv_levels) != 2:
            raise ValueError(f"Paired t-test requires exactly 2 levels for the independent variable '{iv}'. Found {len(iv_levels)}.")

        # Pivot data to wide format for paired t-test
        try:
            # Drop duplicates to avoid pivot errors, then pivot
            wide_data = self.data.drop_duplicates(subset=[self.subject_column, iv]) \
                                .pivot(index=self.subject_column, columns=iv, values=dv)
        except ValueError as e:
             # If duplicates persist (e.g. multiple measurements for same subject under same condition not averaged before this step)
             # A more robust solution would be to average them here or ensure data is preprocessed.
             # For now, we log and raise, or could attempt an average.
            self.logger.error(f"Error pivoting data for paired t-test ({dv} by {iv} for subject {self.subject_column}): {e}. Ensure one observation per subject per condition.")
            # Attempt to average duplicates if that's the desired strategy
            try:
                self.logger.info("Attempting to average duplicates for pivot...")
                wide_data = self.data.groupby([self.subject_column, iv])[dv].mean().unstack()
            except Exception as e_avg:
                raise ValueError(f"Could not pivot data for paired t-test due to: {e}, and averaging failed: {e_avg}")
        
        wide_data = wide_data.dropna()

        if wide_data.shape[0] < 2: # Need at least 2 subjects with complete pairs
            raise ValueError("Not enough paired observations for t-test after dropping NaNs.")

        group1_data = wide_data[iv_levels[0]]
        group2_data = wide_data[iv_levels[1]]

        # Perform paired t-test using pingouin
        ttest_results = pg.ttest(group1_data, group2_data, paired=True)
        
        statistic = ttest_results['T'].iloc[0]
        p_value = ttest_results['p-val'].iloc[0]
        effect_size = ttest_results['cohen-d'].iloc[0]
        effect_interpretation = self._interpret_cohens_d(abs(effect_size)) # abs for interpretation magnitude
        ci = tuple(ttest_results['CI95%'].iloc[0])
        power = ttest_results['power'].iloc[0]
        dof = ttest_results['dof'].iloc[0]

        normality_check_result = self._check_normality_of_differences(group1_data, group2_data)
        assumptions = {
            'normality_of_differences': normality_check_result
        }

        return StatisticalResult(
            test_name="Paired Samples t-test",
            test_type=TestType.COMPARISON,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            effect_size_interpretation=effect_interpretation,
            confidence_interval=ci,
            power=power,
            sample_size=len(wide_data), # Number of pairs
            assumptions_met=assumptions,
            interpretation=self._interpret_p_value(p_value),
            recommendations=self._generate_recommendations("paired_t_test", assumptions, power),
            additional_info={'degrees_of_freedom': dof, 'mean_difference': ttest_results['diff'].iloc[0]}
        )

    def _check_normality_of_differences(self, group1_data: pd.Series, group2_data: pd.Series) -> Dict[str, Any]:
        """Check normality of the differences between two paired groups.

        Args:
            group1_data: Data for the first group/condition.
            group2_data: Data for the second group/condition.

        Returns:
            A dictionary containing the Shapiro-Wilk test statistic, p-value, and a boolean indicating if normally distributed.
        """
        differences = (group1_data - group2_data).dropna()
        if len(differences) < 3:
            self.logger.warning("Not enough data points to check normality of differences.")
            return {'statistic': np.nan, 'p_value': np.nan, 'is_normal': None, 'note': 'Insufficient data'}
        
        try:
            statistic, p_value = shapiro(differences)
            is_normal = p_value > self.config.statistical.alpha_level
            return {'statistic': statistic, 'p_value': p_value, 'is_normal': is_normal}
        except Exception as e:
            self.logger.error(f"Error during normality check of differences: {e}")
            return {'statistic': np.nan, 'p_value': np.nan, 'is_normal': None, 'note': f'Error: {e}'}
    
    def _run_mann_whitney_u(self, dv: str, iv: str) -> StatisticalResult:
        """Run Mann-Whitney U test using pingouin for enhanced results.

        Author: PyADAP Development Team (Revised)
        """
        groups_data = self.data.groupby(iv)[dv]
        group_names = list(groups_data.groups.keys())

        if len(group_names) != 2:
            raise ValueError(f"Mann-Whitney U test requires exactly 2 groups for independent variable '{iv}'. Found {len(group_names)}.")

        group1 = groups_data.get_group(group_names[0]).dropna()
        group2 = groups_data.get_group(group_names[1]).dropna()

        if len(group1) < 1 or len(group2) < 1:
            raise ValueError("One or both groups have no data after dropping NaNs for Mann-Whitney U test.")

        # Perform Mann-Whitney U test using pingouin
        mwu_results = pg.mwu(group1, group2, alternative='two-sided')
        
        statistic = mwu_results['U-val'].iloc[0]
        p_value = mwu_results['p-val'].iloc[0]
        # Pingouin's mwu returns RBC (Rank-Biserial Correlation) as 'RBC' or common language effect size 'CLES'
        # We'll use RBC as it's a common effect size for MWU
        effect_size = mwu_results['RBC'].iloc[0] 
        effect_interpretation = self._interpret_rank_biserial_correlation(abs(effect_size))
        # Power for non-parametric tests is complex and not directly provided by pingouin's mwu output.
        # It can be estimated via simulation or specific packages if critical.
        power = np.nan # Placeholder, as pingouin.mwu doesn't directly return power.

        # Normality assumption is not required for Mann-Whitney U, but it's often used when normality is violated for t-tests.
        # We can note the normality status of the original groups if checked previously.
        normality_status_group1 = self.assumptions_results.get('normality', {}).get(f"{dv}_{group_names[0]}", {}).get('overall_normal', 'Not checked')
        normality_status_group2 = self.assumptions_results.get('normality', {}).get(f"{dv}_{group_names[1]}", {}).get('overall_normal', 'Not checked')
        
        assumption_notes = f"Normality for group {group_names[0]} ('{dv}'): {normality_status_group1}. " \
                           f"Normality for group {group_names[1]} ('{dv}'): {normality_status_group2}."

        return StatisticalResult(
            test_name="Mann-Whitney U test",
            test_type=TestType.NONPARAMETRIC_COMPARISON,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            effect_size_interpretation=effect_interpretation,
            sample_size={'group1': len(group1), 'group2': len(group2), 'total': len(group1) + len(group2)},
            interpretation=self._interpret_p_value(p_value),
            power=power,
            notes=f"{assumption_notes} Power calculation for Mann-Whitney U is not directly available."
        )
    
    def _run_wilcoxon_signed_rank(self, dv: str, iv: str) -> StatisticalResult:
        """Run Wilcoxon signed-rank test using pingouin for enhanced results.

        Author: PyADAP Development Team (Revised)
        """
        if not self.subject_column:
            raise ValueError("Subject column must be specified for Wilcoxon signed-rank test.")

        iv_levels = self.data[iv].unique()
        if len(iv_levels) != 2:
            raise ValueError(f"Wilcoxon signed-rank test requires exactly 2 levels for the independent variable '{iv}'. Found {len(iv_levels)}.")

        try:
            wide_data = self.data.drop_duplicates(subset=[self.subject_column, iv]) \
                                .pivot(index=self.subject_column, columns=iv, values=dv)
        except ValueError as e:
            self.logger.error(f"Error pivoting data for Wilcoxon test ({dv} by {iv} for subject {self.subject_column}): {e}. Ensure one observation per subject per condition.")
            try:
                self.logger.info("Attempting to average duplicates for pivot...")
                wide_data = self.data.groupby([self.subject_column, iv])[dv].mean().unstack()
            except Exception as e_avg:
                raise ValueError(f"Could not pivot data for Wilcoxon test due to: {e}, and averaging failed: {e_avg}")

        wide_data = wide_data.dropna()

        if wide_data.shape[0] < 1: # Wilcoxon can run with fewer pairs than t-test, but still needs some.
            raise ValueError("Not enough paired observations for Wilcoxon test after dropping NaNs.")

        group1_data = wide_data[iv_levels[0]]
        group2_data = wide_data[iv_levels[1]]

        # Perform Wilcoxon signed-rank test using pingouin
        wilcoxon_results = pg.wilcoxon(group1_data, group2_data, alternative='two-sided')
        
        statistic = wilcoxon_results['W-val'].iloc[0]
        p_value = wilcoxon_results['p-val'].iloc[0]
        effect_size = wilcoxon_results['RBC'].iloc[0] # Rank-Biserial Correlation
        effect_interpretation = self._interpret_rank_biserial_correlation(abs(effect_size))
        power = np.nan # Placeholder, as pingouin.wilcoxon doesn't directly return power.

        # Wilcoxon assumes the differences are symmetrically distributed. This is harder to check programmatically simply.
        # Normality of differences is a stronger assumption (for paired t-test).
        # We can note the normality of differences if checked for a t-test fallback scenario.
        normality_diff_check = self._check_normality_of_differences(group1_data, group2_data)
        assumption_notes = f"Normality of differences: statistic={normality_diff_check.get('statistic', 'N/A'):.3f}, p={normality_diff_check.get('p_value', 'N/A'):.3f}, normal={normality_diff_check.get('is_normal', 'N/A')}. " \
                           "Wilcoxon assumes symmetric distribution of differences."

        return StatisticalResult(
            test_name="Wilcoxon Signed-Rank test",
            test_type=TestType.NONPARAMETRIC_COMPARISON,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            effect_size_interpretation=effect_interpretation,
            sample_size=len(wide_data), # Number of pairs
            interpretation=self._interpret_p_value(p_value),
            power=power,
            notes=f"{assumption_notes} Power calculation for Wilcoxon test is not directly available."
        )
    
    def _run_one_way_anova(self, dv: str, iv: str) -> StatisticalResult:
        """Run one-way ANOVA using pingouin for enhanced results.

        Author: PyADAP Development Team (Revised)
        """
        # Filter out groups with no data or too few data points for variance calculation
        valid_groups_data = []
        group_names_list = []
        for name, group_df in self.data.groupby(iv):
            group_dv = group_df[dv].dropna()
            if len(group_dv) > 1: # Need at least 2 data points to calculate variance
                valid_groups_data.append(group_dv)
                group_names_list.append(name)
            else:
                self.logger.warning(f"Group '{name}' for IV '{iv}' has insufficient data for ANOVA (DV: '{dv}'). Skipping this group.")

        if len(valid_groups_data) < 2:
            raise ValueError(f"ANOVA requires at least 2 groups with sufficient data. Found {len(valid_groups_data)} valid groups for IV '{iv}'.")

        # Perform one-way ANOVA using pingouin
        # Ensure data is in long format for pingouin's anova
        current_data_subset = self.data[[dv, iv]].dropna()
        aov_results = pg.anova(data=current_data_subset, dv=dv, between=iv, detailed=True)
        
        statistic = aov_results['F'].iloc[0]
        p_value = aov_results['p-unc'].iloc[0]
        effect_size = aov_results['np2'].iloc[0]  # Partial eta-squared from pingouin
        effect_interpretation = self._interpret_eta_squared(effect_size)
        df_between = aov_results['DF'].iloc[0] # Degrees of freedom for the factor
        df_within = aov_results['DF'].iloc[1]  # Degrees of freedom for residuals/within
        sample_size = sum(len(g) for g in valid_groups_data)

        # Power calculation for ANOVA
        try:
            power = pg.power_anova(eta_sq=effect_size, k=len(valid_groups_data), n=sample_size/len(valid_groups_data), alpha=self.config.statistical.alpha_level)
        except Exception as e:
            self.logger.warning(f"Could not calculate power for One-Way ANOVA: {e}")
            power = np.nan

        # Check assumptions: Normality and Homogeneity of variances
        normality_results = {} # Store per group
        for i, group_data in enumerate(valid_groups_data):
            if len(group_data) >=3:
                stat_shapiro, p_shapiro = shapiro(group_data)
                normality_results[group_names_list[i]] = {'shapiro_statistic': stat_shapiro, 'shapiro_p_value': p_shapiro, 'is_normal': p_shapiro > self.config.statistical.alpha_level}
            else:
                normality_results[group_names_list[i]] = {'shapiro_statistic': np.nan, 'shapiro_p_value': np.nan, 'is_normal': None, 'note': 'Insufficient data for normality test'}
        
        homogeneity_stat, homogeneity_p, homogeneity_met = np.nan, np.nan, None
        if len(valid_groups_data) >= 2 and all(len(g) >=3 for g in valid_groups_data): # Levene needs at least 2 groups with some data
            try:
                # pg.homoscedasticity expects a list of array-like
                levene_result = pg.homoscedasticity([g.to_numpy() for g in valid_groups_data], method='levene', alpha=self.config.statistical.alpha_level)
                homogeneity_stat = levene_result['W'].iloc[0]
                homogeneity_p = levene_result['pval'].iloc[0]
                homogeneity_met = bool(levene_result['equal_var'].iloc[0]) # Ensure it's a Python bool
            except Exception as e:
                self.logger.warning(f"Levene's test failed: {e}")
                homogeneity_met = None # Indicate test failed
        else:
            homogeneity_met = None # Not enough data/groups for test

        assumptions = {
            'normality_by_group': normality_results,
            'homogeneity_of_variances': {'levene_statistic': homogeneity_stat, 'levene_p_value': homogeneity_p, 'is_homogeneous': homogeneity_met}
        }

        return StatisticalResult(
            test_name="One-Way ANOVA",
            test_type=TestType.ANOVA,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            effect_size_interpretation=effect_interpretation,
            sample_size=sample_size,
            interpretation=self._interpret_p_value(p_value),
            power=power,
            assumptions_met=assumptions,
            recommendations=self._generate_recommendations("one_way_anova", assumptions, power),
            additional_info={'df_between': df_between, 'df_within': df_within}
        )
    
    def _calculate_eta_squared(self, groups: List[pd.Series]) -> float:
        """Calculate eta-squared effect size for ANOVA."""
        # Between-group sum of squares
        grand_mean = np.concatenate(groups).mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
        
        # Total sum of squares
        ss_total = sum(((g - grand_mean)**2).sum() for g in groups)
        
        return ss_between / ss_total if ss_total > 0 else 0
    
    def _interpret_eta_squared(self, eta_sq: float) -> str:
        """Interpret eta-squared effect size."""
        if eta_sq < 0.01:
            return "negligible"
        elif eta_sq < 0.06:
            return "small"
        elif eta_sq < 0.14:
            return "medium"
        else:
            return "large"
    
    def _run_repeated_measures_anova(self, dv: str, iv: str) -> StatisticalResult:
        """Run repeated measures ANOVA, deciding on correction based on sphericity.
        
        Author: PyADAP Development Team (Revised)
        """
        result_key = f"{dv}_by_{iv}"
        sphericity_details = self.assumptions_results.get('sphericity', {}).get(result_key, {})
        sphericity_met = sphericity_details.get('sphericity_met', True) # Default to True if not found
        sphericity_test_performed = sphericity_details.get('test_performed', False)

        if not self.subject_column or self.subject_column not in self.data.columns:
            self.logger.warning(f"Subject column ('{self.subject_column}') not specified or not found for repeated measures ANOVA on {result_key}. Falling back to one-way ANOVA.")
            return self._run_one_way_anova(dv, iv)
        
        # Check if there are enough levels for sphericity test and ANOVA
        if self.data[iv].nunique() < 2:
            self.logger.warning(f"Repeated measures ANOVA for {result_key} requires at least 2 levels for IV '{iv}'. Found {self.data[iv].nunique()}. Skipping.")
            return StatisticalResult.error_result(f"Insufficient levels for IV '{iv}' in Repeated Measures ANOVA.")

        # This method is called when auto_select_test determined an uncorrected RM ANOVA is appropriate.
        # Sphericity is assumed to be met or not applicable (e.g. <=2 levels).
        try:
            aov = pg.rm_anova(data=self.data, dv=dv, within=iv, subject=self.subject_column, detailed=True, correction=False)
            
            factor_row = aov[aov['Source'] == iv]
            if factor_row.empty:
                if not aov.empty:
                    # Fallback: if the exact factor name isn't found, take the first relevant row.
                    # This might happen if pingouin names sources differently or for simple designs.
                    factor_row = aov.iloc[[0]] 
                else:
                    raise ValueError("ANOVA results table is empty or factor not found.")

            statistic = factor_row['F'].iloc[0]
            p_value = factor_row['p-unc'].iloc[0]
            effect_size = factor_row['np2'].iloc[0] # partial eta-squared
            df_num = factor_row['ddof1'].iloc[0]
            df_den = factor_row['ddof2'].iloc[0]

            sphericity_note = "Sphericity assumption met or test not applicable (e.g. <=2 levels); uncorrected ANOVA performed."
            # Optionally, include more details from sphericity_details if needed for the report
            # sphericity_details = self.assumptions_results.get('sphericity', {}).get(f"{dv}_by_{iv}", {})
            # if sphericity_details.get('test_performed', False) and sphericity_details.get('sphericity_met', True):
            #     sphericity_note = f"Mauchly's test: W={sphericity_details.get('W_statistic', 'N/A')}, p={sphericity_details.get('p_value', 'N/A')}. Sphericity met."
            # elif not sphericity_details.get('test_performed', False) and self.data[iv].nunique() <= 2:
            #     sphericity_note = "Sphericity assumption holds (<=2 levels)."

            return StatisticalResult(
                test_name="Repeated Measures ANOVA (uncorrected)",
                test_type=TestType.ANOVA,
                statistic=statistic,
                p_value=p_value,
                effect_size=effect_size,
                effect_size_interpretation=self._interpret_eta_squared(effect_size),
                sample_size=self.data[self.subject_column].nunique(), # Number of subjects
                interpretation=self._interpret_p_value(p_value),
                assumptions_met={'sphericity': True}, # Assumed met for this path
                notes=sphericity_note,
                additional_info={'df_num': df_num, 'df_den': df_den}
            )
        except Exception as e:
            self.logger.error(f"Uncorrected repeated measures ANOVA for {dv}_by_{iv} failed: {str(e)}. Falling back to Friedman test.")
            return self._run_friedman_test(dv, iv)
    
    def _run_kruskal_wallis(self, dv: str, iv: str) -> StatisticalResult:
        """Run Kruskal-Wallis H-test using pingouin for enhanced results.

        Author: PyADAP Development Team (Revised)
        """
        # Filter out groups with no data
        valid_groups_data = []
        group_names_list = []
        for name, group_df in self.data.groupby(iv):
            group_dv = group_df[dv].dropna()
            if not group_dv.empty:
                valid_groups_data.append(group_dv)
                group_names_list.append(name)
            else:
                self.logger.warning(f"Group '{name}' for IV '{iv}' has no data for Kruskal-Wallis (DV: '{dv}'). Skipping this group.")

        if len(valid_groups_data) < 2:
            raise ValueError(f"Kruskal-Wallis test requires at least 2 groups with data. Found {len(valid_groups_data)} valid groups for IV '{iv}'.")

        # Perform Kruskal-Wallis H-test using pingouin
        current_data_subset = self.data[[dv, iv]].dropna()
        # Ensure the 'between' variable is categorical for pingouin, if it's not already
        current_data_subset[iv] = current_data_subset[iv].astype('category')
        kruskal_results = pg.kruskal(data=current_data_subset, dv=dv, between=iv)
        
        statistic = kruskal_results['H'].iloc[0]
        p_value = kruskal_results['p-unc'].iloc[0]
        
        # Calculate Epsilon-squared as effect size for Kruskal-Wallis
        n_total = sum(len(g) for g in valid_groups_data)
        k_groups = len(valid_groups_data)
        if n_total > k_groups and n_total > 0 and (n_total**2 - 1) != 0 : # Avoid division by zero or invalid inputs
            # Epsilon-squared = H / ((n^2 - 1) / (n + 1)) which simplifies to H / (n - 1)
            # This is one form. Another common one is (H - k + 1) / (n - k)
            # Pingouin's documentation or source might specify if they provide one.
            # Let's use H / (n-1) as a common one if not directly provided by pingouin.
            # effect_size = statistic / (n_total - 1) # Eta-squared like for KW
            # Or, a more robust epsilon-squared:
            if (n_total - k_groups) > 0:
                 effect_size = (statistic - k_groups + 1) / (n_total - k_groups)
            else:
                 effect_size = np.nan
        else:
            effect_size = np.nan
        effect_interpretation = self._interpret_epsilon_squared(abs(effect_size))

        dof = kruskal_results['ddof1'].iloc[0]
        sample_size = n_total
        power = np.nan # Power for Kruskal-Wallis is complex and not directly provided.

        normality_notes_list = []
        for i, group_data in enumerate(valid_groups_data):
            # Construct a unique key for normality assumption results if needed, e.g., combining dv and group name
            # For simplicity, assuming group_names_list[i] can be part of a key
            norm_key = f"{dv}_{group_names_list[i]}" # Example key
            norm_status = self.assumptions_results.get('normality', {}).get(norm_key, {}).get('overall_normal', 'Not checked')
            normality_notes_list.append(f"Normality for group {group_names_list[i]} ('{dv}'): {norm_status}")
        assumption_notes = "; ".join(normality_notes_list)
        assumption_notes += ". Kruskal-Wallis assumes similar distribution shapes across groups for comparing medians."

        return StatisticalResult(
            test_name="Kruskal-Wallis H-test",
            test_type=TestType.NONPARAMETRIC_ANOVA,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            effect_size_interpretation=effect_interpretation,
            sample_size=sample_size,
            interpretation=self._interpret_p_value(p_value),
            power=power,
            notes=f"{assumption_notes} Power calculation for Kruskal-Wallis is not directly available.",
            additional_info={'degrees_of_freedom': dof}
        )
    
    def _run_friedman_test(self, dv: str, iv: str) -> StatisticalResult:
        """Run Friedman test using pingouin for enhanced results, including Kendall's W.

        Author: PyADAP Development Team (Revised)
        """
        if self.subject_column is None or self.subject_column not in self.data.columns:
            raise ValueError("Subject column is required for Friedman test and must be specified in Analyzer's config.")

        # Prepare data for pingouin.friedman
        # It expects data in long format: [subject, dv, iv]
        # Ensure no missing values in the relevant columns for the test itself
        current_data_subset = self.data[[self.subject_column, dv, iv]].dropna()

        # Check for sufficient data after dropping NaNs
        if current_data_subset.empty:
            raise ValueError(f"No complete cases for Friedman test after dropping NaNs for DV='{dv}', IV='{iv}', Subject='{self.subject_column}'.")

        # Check for duplicate entries for a subject within the same IV level, which shouldn't happen if data is structured correctly for repeated measures.
        # Pingouin's friedman expects one value per subject per condition.
        # If duplicates exist, this indicates an issue with data structure or prior aggregation needed.
        if current_data_subset.duplicated(subset=[self.subject_column, iv]).any():
            self.logger.warning(f"Duplicate entries found for subject and IV level ('{iv}') combination. Aggregating by taking the mean.")
            current_data_subset = current_data_subset.groupby([self.subject_column, iv], as_index=False)[dv].mean()
            
        # Check if each subject has observations for all levels of IV
        # Friedman test requires complete blocks (each subject measured on all conditions)
        # Pingouin handles this internally by dropping subjects with incomplete data, but we can check and log.
        n_conditions = current_data_subset[iv].nunique()
        subject_counts = current_data_subset.groupby(self.subject_column)[iv].nunique()
        incomplete_subjects = subject_counts[subject_counts < n_conditions].index.tolist()
        if incomplete_subjects:
            self.logger.warning(f"Subjects {incomplete_subjects} have incomplete data across IV levels and will be excluded by pingouin.friedman.")
            current_data_subset = current_data_subset[~current_data_subset[self.subject_column].isin(incomplete_subjects)]

        if current_data_subset[self.subject_column].nunique() < 2 or n_conditions < 2:
             raise ValueError(f"Friedman test requires at least 2 subjects and 2 conditions with complete data. After processing, found {current_data_subset[self.subject_column].nunique()} subjects and {n_conditions} conditions.")

        # Perform Friedman test using pingouin
        try:
            friedman_results = pg.friedman(data=current_data_subset, dv=dv, within=iv, subject=self.subject_column)
        except Exception as e:
            self.logger.error(f"Error during pg.friedman execution: {e}")
            # Fallback or re-raise with more context if necessary
            # For now, re-raise to indicate a problem with the test execution itself
            raise ValueError(f"Pingouin Friedman test failed. Original error: {e}. Check data structure and IV levels.")

        statistic = friedman_results['Q'].iloc[0] # Chi-squared statistic (Q)
        p_value = friedman_results['p-unc'].iloc[0]
        effect_size = friedman_results['W'].iloc[0]  # Kendall's W
        effect_interpretation = self._interpret_kendalls_w(abs(effect_size))
        dof = friedman_results['ddof1'].iloc[0]
        
        # Sample size is the number of subjects with complete data
        sample_size = current_data_subset[self.subject_column].nunique()
        power = np.nan # Power for Friedman test is complex and not directly provided by pingouin.

        # Note on assumptions for Friedman (non-parametric, so no normality of differences, but assumes sphericity is not an issue)
        assumption_notes = "Friedman test is a non-parametric alternative to one-way repeated measures ANOVA. It does not assume normality of differences but assumes that the distributions of the DV across conditions have similar shapes and variances (sphericity is not directly tested but violations can impact power)."

        return StatisticalResult(
            test_name="Friedman Test",
            test_type=TestType.NONPARAMETRIC_RM_ANOVA, # More specific type
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            effect_size_interpretation=effect_interpretation,
            sample_size=sample_size,
            interpretation=self._interpret_p_value(p_value),
            power=power,
            notes=f"{assumption_notes} Power calculation for Friedman test is not directly available.",
            additional_info={'degrees_of_freedom': dof, 'kendalls_w': effect_size}
        )
    
    def _run_pearson_correlation(self, var1: str, var2: str) -> StatisticalResult:
        """Run Pearson correlation."""
        clean_data = self.data[[var1, var2]].dropna()
        
        if len(clean_data) < 3:
            raise ValueError("Insufficient data for correlation")
        
        # Perform Pearson correlation
        statistic, p_value = pearsonr(clean_data[var1], clean_data[var2])
        
        # Calculate confidence interval
        ci = self._calculate_correlation_ci(statistic, len(clean_data))
        
        return StatisticalResult(
            test_name="Pearson Correlation",
            test_type=TestType.CORRELATION,
            statistic=statistic,
            p_value=p_value,
            effect_size=statistic,  # r is the effect size
            effect_size_interpretation=self._interpret_correlation(abs(statistic)),
            confidence_interval=ci,
            sample_size=len(clean_data),
            interpretation=self._interpret_p_value(p_value)
        )
    
    def _run_spearman_correlation(self, var1: str, var2: str) -> StatisticalResult:
        """Run Spearman correlation."""
        clean_data = self.data[[var1, var2]].dropna()
        
        if len(clean_data) < 3:
            raise ValueError("Insufficient data for correlation")
        
        # Perform Spearman correlation
        statistic, p_value = spearmanr(clean_data[var1], clean_data[var2])
        
        return StatisticalResult(
            test_name="Spearman Correlation",
            test_type=TestType.CORRELATION,
            statistic=statistic,
            p_value=p_value,
            effect_size=statistic,
            effect_size_interpretation=self._interpret_correlation(abs(statistic)),
            sample_size=len(clean_data),
            interpretation=self._interpret_p_value(p_value)
        )
    
    def _calculate_correlation_ci(self, r: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
        """Calculate confidence interval for correlation coefficient."""
        # Fisher's z-transformation
        z = 0.5 * np.log((1 + r) / (1 - r))
        se = 1 / np.sqrt(n - 3)
        
        # Critical z-value
        z_critical = stats.norm.ppf(1 - alpha/2)
        
        # Confidence interval in z-space
        z_lower = z - z_critical * se
        z_upper = z + z_critical * se
        
        # Transform back to correlation space
        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        
        return (r_lower, r_upper)
    
    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation coefficient magnitude."""
        if r < 0.1:
            return "negligible"
        elif r < 0.3:
            return "small"
        elif r < 0.5:
            return "medium"
        else:
            return "large"
    
    def _run_chi_square_test(self, var1: str, var2: str) -> StatisticalResult:
        """Run chi-square test of independence."""
        # Create contingency table
        contingency_table = pd.crosstab(self.data[var1], self.data[var2])
        
        # Perform chi-square test
        statistic, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Calculate effect size (Cramér's V)
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        effect_size = np.sqrt(statistic / (n * min_dim))
        
        return StatisticalResult(
            test_name="Chi-Square Test of Independence",
            test_type=TestType.COMPARISON,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            effect_size_interpretation=self._interpret_cramers_v(effect_size),
            sample_size=n,
            interpretation=self._interpret_p_value(p_value),
            additional_info={'degrees_of_freedom': dof, 'contingency_table': contingency_table.to_dict()}
        )
    
    def _interpret_cramers_v(self, v: float) -> str:
        """Interpret Cramér's V effect size."""
        if v < 0.1:
            return "negligible"
        elif v < 0.3:
            return "small"
        elif v < 0.5:
            return "medium"
        else:
            return "large"

    def _interpret_kendalls_w(self, w: float) -> str:
        """Interpret Kendall's W effect size (coefficient of concordance).

        Kendall's W ranges from 0 (no agreement) to 1 (complete agreement).
        Interpretations can vary, but a common guideline is:
        - 0.0 - <0.1: Negligible agreement
        - 0.1 - <0.3: Weak agreement
        - 0.3 - <0.5: Moderate agreement
        - 0.5 - <0.7: Strong agreement
        - 0.7 - 1.0: Very strong agreement
        """
        if w < 0.1:
            return "negligible agreement"
        elif w < 0.3:
            return "weak agreement"
        elif w < 0.5:
            return "moderate agreement"
        elif w < 0.7:
            return "strong agreement"
        else:
            return "very strong agreement"
    
    def _interpret_epsilon_squared(self, es: float) -> str:
        """Interpret Epsilon-squared effect size for Kruskal-Wallis.

        Epsilon-squared is an effect size measure for Kruskal-Wallis test,
        ranging from 0 to 1. Values closer to 1 indicate stronger effects.
        
        Common interpretation guidelines:
        - < 0.01: Negligible effect
        - < 0.04: Small effect
        - < 0.16: Medium effect
        - ≥ 0.16: Large effect
        """
        if es < 0.01:
            return "negligible"
        elif es < 0.04:
            return "small"
        elif es < 0.16:
            return "medium"
        else:
            return "large"
        
    def _execute_test(self, test_name: str, dv: pd.Series, iv: pd.Series) -> Optional[StatisticalResult]:
        """Execute a statistical test based on the test name.
        
        Args:
            test_name: Name of the test to execute
            dv: Dependent variable data
            iv: Independent variable data
            
        Returns:
            StatisticalResult object containing test results
        """
        test_method_name = f"_run_{test_name}"
        if hasattr(self, test_method_name):
            return getattr(self, test_method_name)(dv, iv)
        else:
            self.logger.warning(f"No method found for test: {test_name}")
            return self._run_descriptive_statistics(dv, iv)
        """Interpret Epsilon-squared effect size for Kruskal-Wallis.

        Epsilon-squared is an effect size for Kruskal-Wallis, similar to eta-squared.
        Ranges from 0 to 1.
        Common interpretation guidelines (Cohen's standards are often adapted):
        - 0.01 - <0.06: Small effect
        - 0.06 - <0.14: Medium effect
        - >= 0.14: Large effect
        Note: Some sources might use slightly different cutoffs.
        """
        if es < 0.01:
            return "negligible effect"
        elif es < 0.06:
            return "small effect"
        elif es < 0.14:
            return "medium effect"
        else:
            return "large effect"
    
    def _run_descriptive_statistics(self, dv: str, iv: str) -> StatisticalResult:
        """Run descriptive statistics."""
        # Group by independent variable
        grouped = self.data.groupby(iv)[dv]
        
        descriptives = {
            'group_means': grouped.mean().to_dict(),
            'group_stds': grouped.std().to_dict(),
            'group_sizes': grouped.size().to_dict(),
            'overall_mean': self.data[dv].mean(),
            'overall_std': self.data[dv].std()
        }
        
        return StatisticalResult(
            test_name="Descriptive Statistics",
            test_type=TestType.COMPARISON,
            statistic=0,  # No test statistic
            p_value=1,    # No p-value
            sample_size=len(self.data),
            interpretation="Descriptive analysis completed",
            additional_info=descriptives
        )
    
    def _interpret_p_value(self, p_value: float) -> str:
        """Interpret p-value based on significance level."""
        alpha = self.config.statistical.alpha_level
        
        if p_value < alpha:
            return f"Statistically significant (p < {alpha})"
        else:
            return f"Not statistically significant (p ≥ {alpha})"
    
    def _generate_recommendations(self, test_type: str, assumptions: Dict[str, bool], power: Optional[float]) -> List[str]:
        """Generate recommendations based on test results and assumptions."""
        recommendations = []
        
        # Check assumptions with specific recommendations (Author: PyADAP Development Team)
        if not assumptions.get('normality', True):
            recommendations.append("Normality assumption violated. Consider data transformation (log, Box-Cox, Yeo-Johnson) or non-parametric alternatives")
        
        if not assumptions.get('homogeneity', True):
            recommendations.append("Homogeneity of variance assumption violated. Consider Welch's correction, robust tests, or non-parametric alternatives")
        
        if not assumptions.get('sphericity', True):
            recommendations.append("Sphericity assumption violated. Apply Greenhouse-Geisser or Huynh-Feldt correction, or use mixed-effects models")
        
        if not assumptions.get('independence', True):
            recommendations.append("Independence assumption may be violated. Consider mixed-effects models or time-series analysis methods")
        
        # Check power with actionable recommendations (Author: PyADAP Development Team)
        if power is not None:
            if power < 0.8:
                recommendations.append(f"Statistical power is low ({power:.3f}). Consider increasing sample size, using more sensitive measures, or reducing measurement error")
            elif power > 0.95:
                recommendations.append(f"Statistical power is very high ({power:.3f}). Sample size may be larger than necessary for practical purposes")
        
        # Test-specific recommendations (Author: PyADAP Development Team)
        if 'transformation' in test_type:
            recommendations.append("Data transformation was applied. Interpret results in the context of transformed data and consider inverse transformation for reporting")
        
        if 'corrected' in test_type:
            recommendations.append("Statistical correction was applied due to assumption violations. Results are more robust but may have reduced power")
        
        return recommendations
    
    def generate_report(self) -> str:
        """Generate comprehensive statistical analysis report.
        
        Returns:
            Formatted report string
        """
        if not self.results:
            return "No analysis results available. Run analysis first."
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("STATISTICAL ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Data summary
        if self.data is not None:
            report_lines.append(f"Dataset: {len(self.data)} observations")
            report_lines.append(f"Dependent variables: {', '.join(self.dependent_vars)}")
            report_lines.append(f"Independent variables: {', '.join(self.independent_vars)}")
            report_lines.append("")
        
        # Assumption checks
        if self.assumptions_results:
            report_lines.append("ASSUMPTION CHECKS")
            report_lines.append("-" * 40)
            
            # Normality
            if 'normality' in self.assumptions_results:
                report_lines.append("Normality Tests:")
                for var, results in self.assumptions_results['normality'].items():
                    status = "✓" if results.get('overall_normal', False) else "✗"
                    report_lines.append(f"  {status} {var}: {'Normal' if results.get('overall_normal', False) else 'Non-normal'}")
                report_lines.append("")
            
            # Homogeneity
            if 'homogeneity' in self.assumptions_results:
                report_lines.append("Homogeneity of Variance:")
                for comparison, results in self.assumptions_results['homogeneity'].items():
                    status = "✓" if results.get('overall_homogeneous', True) else "✗"
                    report_lines.append(f"  {status} {comparison}: {'Homogeneous' if results.get('overall_homogeneous', True) else 'Heterogeneous'}")
                report_lines.append("")
        
        # Statistical tests
        report_lines.append("STATISTICAL TESTS")
        report_lines.append("-" * 40)
        
        for test_name, result in self.results.items():
            report_lines.append(f"\n{result.test_name} ({test_name}):")
            report_lines.append(f"  Statistic: {result.statistic:.4f}")
            report_lines.append(f"  p-value: {result.p_value:.4f}")
            
            if result.effect_size is not None:
                report_lines.append(f"  Effect size: {result.effect_size:.4f} ({result.effect_size_interpretation})")
            
            if result.confidence_interval:
                ci_lower, ci_upper = result.confidence_interval
                report_lines.append(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            
            if result.power is not None:
                report_lines.append(f"  Statistical power: {result.power:.4f}")
            
            report_lines.append(f"  Sample size: {result.sample_size}")
            report_lines.append(f"  Interpretation: {result.interpretation}")
            
            if result.recommendations:
                report_lines.append("  Recommendations:")
                for rec in result.recommendations:
                    report_lines.append(f"    • {rec}")
        
        report_lines.append("\n" + "=" * 80)
        
        return "\n".join(report_lines)