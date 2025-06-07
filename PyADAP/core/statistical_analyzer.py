"""Advanced Statistical Analysis for PyADAP 3.0

This module provides comprehensive statistical analysis with automatic test selection,
effect size calculations, power analysis, and robust reporting.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    shapiro, normaltest, jarque_bera, anderson, kstest,
    levene, bartlett, fligner,
    ttest_1samp, ttest_ind, ttest_rel,
    mannwhitneyu, wilcoxon, kruskal,
    f_oneway, friedmanchisquare,
    chi2_contingency, fisher_exact,
    pearsonr, spearmanr, kendalltau
)
import statsmodels.api as sm
from statsmodels.stats.power import (
    ttest_power, FTestAnovaPower, GofChisquarePower
)
from statsmodels.stats.contingency_tables import mcnemar
# Import effect size functions from utils.helpers
from ..utils.helpers import (
    calculate_effect_size, 
    calculate_correlation_effect_size
)
from statsmodels.stats.diagnostic import (
    het_breuschpagan, het_white
)
# jarque_bera is already imported from scipy.stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import OLSInfluence
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
                except:
                    pass
            
            # D'Agostino-Pearson test
            if len(data_col) >= 8:
                try:
                    stat, p = normaltest(data_col)
                    tests['dagostino_pearson'] = {
                        'statistic': stat,
                        'p_value': p,
                        'normal': p > self.config.statistical.alpha_level
                    }
                except:
                    pass
            
            # Jarque-Bera test
            if len(data_col) >= 2:
                try:
                    stat, p = jarque_bera(data_col)
                    tests['jarque_bera'] = {
                        'statistic': stat,
                        'p_value': p,
                        'normal': p > self.config.statistical.alpha_level
                    }
                except:
                    pass
            
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
            except:
                pass
            
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
            except:
                pass
            
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
        """Automatically select appropriate statistical test.
        
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
        
        # Check assumptions
        assumptions = self.assumptions_results.get('normality', {}).get(dependent_var, {})
        is_normal = assumptions.get('overall_normal', False)
        
        homogeneity = self.assumptions_results.get('homogeneity', {}).get(f"{dependent_var}_by_{independent_var}", {})
        is_homogeneous = homogeneity.get('overall_homogeneous', True)
        
        # Decision tree for test selection
        if dv_type == 'continuous':
            if iv_type == 'categorical':
                if n_groups == 2:
                    if is_paired:
                        return 'paired_t_test' if is_normal else 'wilcoxon_signed_rank'
                    else:
                        return 'independent_t_test' if (is_normal and is_homogeneous) else 'mann_whitney_u'
                elif n_groups > 2:
                    if is_paired:
                        return 'repeated_measures_anova' if is_normal else 'friedman_test'
                    else:
                        return 'one_way_anova' if (is_normal and is_homogeneous) else 'kruskal_wallis'
            elif iv_type == 'continuous':
                return 'pearson_correlation' if is_normal else 'spearman_correlation'
        
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
    
    def run_comprehensive_analysis(self) -> Dict[str, StatisticalResult]:
        """Run comprehensive statistical analysis.
        
        Returns:
            Dictionary of statistical results
        """
        if self.data is None:
            raise ValueError("No data set. Call set_data() first.")
        
        # Check assumptions first
        self.check_assumptions()
        
        # Run analyses for each DV-IV combination
        for dv in self.dependent_vars:
            for iv in self.independent_vars:
                test_name = self.auto_select_test(dv, iv)
                result_key = f"{dv}_by_{iv}"
                
                try:
                    if test_name == 'independent_t_test':
                        self.results[result_key] = self._run_independent_t_test(dv, iv)
                    elif test_name == 'paired_t_test':
                        self.results[result_key] = self._run_paired_t_test(dv, iv)
                    elif test_name == 'mann_whitney_u':
                        self.results[result_key] = self._run_mann_whitney_u(dv, iv)
                    elif test_name == 'wilcoxon_signed_rank':
                        self.results[result_key] = self._run_wilcoxon_signed_rank(dv, iv)
                    elif test_name == 'one_way_anova':
                        self.results[result_key] = self._run_one_way_anova(dv, iv)
                    elif test_name == 'repeated_measures_anova':
                        self.results[result_key] = self._run_repeated_measures_anova(dv, iv)
                    elif test_name == 'kruskal_wallis':
                        self.results[result_key] = self._run_kruskal_wallis(dv, iv)
                    elif test_name == 'friedman_test':
                        self.results[result_key] = self._run_friedman_test(dv, iv)
                    elif test_name == 'pearson_correlation':
                        self.results[result_key] = self._run_pearson_correlation(dv, iv)
                    elif test_name == 'spearman_correlation':
                        self.results[result_key] = self._run_spearman_correlation(dv, iv)
                    elif test_name == 'chi_square_test':
                        self.results[result_key] = self._run_chi_square_test(dv, iv)
                    else:
                        self.results[result_key] = self._run_descriptive_statistics(dv, iv)
                
                except Exception as e:
                    self.logger.error(f"Failed to run {test_name} for {dv} by {iv}: {str(e)}")
                    continue
        
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
        """Run paired samples t-test."""
        # This is a simplified implementation
        # In practice, you'd need to properly handle the pairing
        groups = [group[dv].dropna() for name, group in self.data.groupby(iv)]
        
        if len(groups) != 2 or len(groups[0]) != len(groups[1]):
            raise ValueError("Paired t-test requires exactly 2 groups of equal size")
        
        group1, group2 = groups
        
        # Perform paired t-test
        statistic, p_value = ttest_rel(group1, group2)
        
        # Calculate effect size
        differences = group1 - group2
        effect_size = differences.mean() / differences.std(ddof=1)
        effect_interpretation = self._interpret_cohens_d(effect_size)
        
        return StatisticalResult(
            test_name="Paired Samples t-test",
            test_type=TestType.COMPARISON,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            effect_size_interpretation=effect_interpretation,
            sample_size=len(group1),
            interpretation=self._interpret_p_value(p_value)
        )
    
    def _run_mann_whitney_u(self, dv: str, iv: str) -> StatisticalResult:
        """Run Mann-Whitney U test."""
        groups = [group[dv].dropna() for name, group in self.data.groupby(iv)]
        
        if len(groups) != 2:
            raise ValueError("Mann-Whitney U test requires exactly 2 groups")
        
        group1, group2 = groups
        
        # Perform Mann-Whitney U test
        statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
        
        # Calculate effect size (rank-biserial correlation)
        n1, n2 = len(group1), len(group2)
        effect_size = 1 - (2 * statistic) / (n1 * n2)
        
        return StatisticalResult(
            test_name="Mann-Whitney U test",
            test_type=TestType.NONPARAMETRIC,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            sample_size=n1 + n2,
            interpretation=self._interpret_p_value(p_value)
        )
    
    def _run_wilcoxon_signed_rank(self, dv: str, iv: str) -> StatisticalResult:
        """Run Wilcoxon signed-rank test."""
        groups = [group[dv].dropna() for name, group in self.data.groupby(iv)]
        
        if len(groups) != 2:
            raise ValueError("Wilcoxon signed-rank test requires exactly 2 groups")
        
        group1, group2 = groups
        
        # Perform Wilcoxon signed-rank test
        statistic, p_value = wilcoxon(group1, group2)
        
        return StatisticalResult(
            test_name="Wilcoxon Signed-Rank test",
            test_type=TestType.NONPARAMETRIC,
            statistic=statistic,
            p_value=p_value,
            sample_size=len(group1),
            interpretation=self._interpret_p_value(p_value)
        )
    
    def _run_one_way_anova(self, dv: str, iv: str) -> StatisticalResult:
        """Run one-way ANOVA."""
        groups = [group[dv].dropna() for name, group in self.data.groupby(iv)]
        
        if len(groups) < 2:
            raise ValueError("ANOVA requires at least 2 groups")
        
        # Perform one-way ANOVA
        statistic, p_value = f_oneway(*groups)
        
        # Calculate effect size (eta-squared)
        effect_size = self._calculate_eta_squared(groups)
        effect_interpretation = self._interpret_eta_squared(effect_size)
        
        return StatisticalResult(
            test_name="One-Way ANOVA",
            test_type=TestType.ANOVA,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            effect_size_interpretation=effect_interpretation,
            sample_size=sum(len(g) for g in groups),
            interpretation=self._interpret_p_value(p_value)
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
        """Run repeated measures ANOVA using pingouin."""
        try:
            # Use pingouin for repeated measures ANOVA
            result = pg.rm_anova(data=self.data, dv=dv, within=iv, subject=self.subject_column)
            
            return StatisticalResult(
                test_name="Repeated Measures ANOVA",
                test_type=TestType.ANOVA,
                statistic=result['F'].iloc[0],
                p_value=result['p-unc'].iloc[0],
                effect_size=result['np2'].iloc[0],  # partial eta-squared
                sample_size=len(self.data),
                interpretation=self._interpret_p_value(result['p-unc'].iloc[0])
            )
        except Exception as e:
            # Fallback to regular ANOVA if pingouin fails
            return self._run_one_way_anova(dv, iv)
    
    def _run_kruskal_wallis(self, dv: str, iv: str) -> StatisticalResult:
        """Run Kruskal-Wallis test."""
        groups = [group[dv].dropna() for name, group in self.data.groupby(iv)]
        
        if len(groups) < 2:
            raise ValueError("Kruskal-Wallis test requires at least 2 groups")
        
        # Perform Kruskal-Wallis test
        statistic, p_value = kruskal(*groups)
        
        # Calculate effect size (epsilon-squared)
        n_total = sum(len(g) for g in groups)
        effect_size = (statistic - len(groups) + 1) / (n_total - len(groups))
        
        return StatisticalResult(
            test_name="Kruskal-Wallis test",
            test_type=TestType.NONPARAMETRIC,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            sample_size=n_total,
            interpretation=self._interpret_p_value(p_value)
        )
    
    def _run_friedman_test(self, dv: str, iv: str) -> StatisticalResult:
        """Run Friedman test."""
        # Check if subject column exists
        if self.subject_column is None or self.subject_column not in self.data.columns:
            raise ValueError("Subject column is required for Friedman test")
        
        # Remove duplicates first to avoid pivot issues
        data_clean = self.data[[self.subject_column, iv, dv]].drop_duplicates()
        
        # Reshape data for Friedman test
        try:
            pivot_data = data_clean.pivot(index=self.subject_column, columns=iv, values=dv)
        except ValueError as e:
            if "duplicate entries" in str(e):
                # If still duplicates, take mean for each subject-condition combination
                pivot_data = data_clean.groupby([self.subject_column, iv])[dv].mean().unstack()
            else:
                raise e
        
        # Remove rows with missing values
        clean_data = pivot_data.dropna()
        
        if clean_data.empty:
            raise ValueError("No complete cases for Friedman test")
        
        # Perform Friedman test
        statistic, p_value = friedmanchisquare(*[clean_data[col] for col in clean_data.columns])
        
        return StatisticalResult(
            test_name="Friedman test",
            test_type=TestType.NONPARAMETRIC,
            statistic=statistic,
            p_value=p_value,
            sample_size=len(clean_data),
            interpretation=self._interpret_p_value(p_value)
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
        
        # Check assumptions
        if not assumptions.get('normality', True):
            recommendations.append("Consider using non-parametric alternatives due to non-normal distribution")
        
        if not assumptions.get('homogeneity', True):
            recommendations.append("Consider using Welch's t-test or non-parametric alternatives due to unequal variances")
        
        # Check power
        if power is not None:
            if power < 0.8:
                recommendations.append(f"Statistical power is low ({power:.3f}). Consider increasing sample size")
            elif power > 0.95:
                recommendations.append(f"Statistical power is very high ({power:.3f}). Sample size may be larger than necessary")
        
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