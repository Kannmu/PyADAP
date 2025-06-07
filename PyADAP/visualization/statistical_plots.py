"""Statistical plots for PyADAP 3.0

This module provides specialized plotting functions for statistical analysis results,
including plots for different types of statistical tests and their interpretations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler

from ..utils import Logger, get_logger, format_number, format_p_value
from ..config import Config


class StatisticalPlots:
    """Class for creating statistical analysis plots."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the statistical plots generator.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.logger = get_logger("PyADAP.StatisticalPlots")
    
    def create_analysis_plots(self, data: pd.DataFrame,
                            variables: Dict[str, List[str]],
                            results: Dict[str, Any]) -> Dict[str, Figure]:
        """Create plots for statistical analysis results.
        
        Args:
            data: Input DataFrame
            variables: Dictionary with variable assignments
            results: Statistical analysis results
            
        Returns:
            Dictionary of figure names and Figure objects
        """
        plots = {}
        
        try:
            # Determine analysis type and create appropriate plots
            if 'statistical_tests' in results:
                for test_name, test_result in results['statistical_tests'].items():
                    if isinstance(test_result, dict):
                        # Create plots based on test type
                        test_plots = self._create_test_specific_plots(
                            data, variables, test_name, test_result
                        )
                        plots.update(test_plots)
            
            # Create general comparison plots
            comparison_plots = self._create_comparison_plots(data, variables, results)
            plots.update(comparison_plots)
            
            # Create effect size visualization
            effect_plots = self._create_effect_size_plots(data, variables, results)
            plots.update(effect_plots)
            
            self.logger.info(f"Created {len(plots)} statistical analysis plots")
            
        except Exception as e:
            self.logger.error(f"Error creating statistical analysis plots: {str(e)}")
            raise
        
        return plots
    
    def _create_test_specific_plots(self, data: pd.DataFrame,
                                  variables: Dict[str, List[str]],
                                  test_name: str,
                                  test_result: Dict[str, Any]) -> Dict[str, Figure]:
        """Create plots specific to the type of statistical test.
        
        Args:
            data: Input DataFrame
            variables: Dictionary with variable assignments
            test_name: Name of the statistical test
            test_result: Test results
            
        Returns:
            Dictionary of figure names and Figure objects
        """
        plots = {}
        
        try:
            # Determine test type and create appropriate plots
            if 'ttest' in test_name.lower():
                plots.update(self._create_ttest_plots(data, variables, test_name, test_result))
            elif 'anova' in test_name.lower() or 'f_test' in test_name.lower():
                plots.update(self._create_anova_plots(data, variables, test_name, test_result))
            elif 'mann' in test_name.lower() or 'wilcoxon' in test_name.lower():
                plots.update(self._create_nonparametric_plots(data, variables, test_name, test_result))
            elif 'correlation' in test_name.lower() or 'pearson' in test_name.lower() or 'spearman' in test_name.lower():
                plots.update(self._create_correlation_plots(data, variables, test_name, test_result))
            elif 'chi' in test_name.lower():
                plots.update(self._create_chi_square_plots(data, variables, test_name, test_result))
            
        except Exception as e:
            self.logger.warning(f"Error creating plots for {test_name}: {str(e)}")
        
        return plots
    
    def _create_ttest_plots(self, data: pd.DataFrame,
                          variables: Dict[str, List[str]],
                          test_name: str,
                          test_result: Dict[str, Any]) -> Dict[str, Figure]:
        """Create plots for t-test results.
        
        Args:
            data: Input DataFrame
            variables: Dictionary with variable assignments
            test_name: Name of the test
            test_result: Test results
            
        Returns:
            Dictionary of plots
        """
        plots = {}
        
        dependent_vars = variables.get('dependent', [])
        independent_vars = variables.get('independent', [])
        
        if not dependent_vars or not independent_vars:
            return plots
        
        dep_var = dependent_vars[0]
        indep_var = independent_vars[0]
        
        if dep_var not in data.columns or indep_var not in data.columns:
            return plots
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'T-Test Results: {dep_var} by {indep_var}', fontsize=14, fontweight='bold')
        
        # Box plot comparison
        if data[indep_var].dtype in ['object', 'category'] or data[indep_var].nunique() <= 10:
            sns.boxplot(data=data, x=indep_var, y=dep_var, ax=ax1)
            ax1.set_title('Group Comparison (Box Plot)')
            ax1.tick_params(axis='x', rotation=45)
        else:
            # For continuous independent variable, create scatter plot
            ax1.scatter(data[indep_var], data[dep_var], alpha=0.6)
            ax1.set_xlabel(indep_var)
            ax1.set_ylabel(dep_var)
            ax1.set_title('Scatter Plot')
        
        # Distribution comparison
        if data[indep_var].dtype in ['object', 'category'] or data[indep_var].nunique() <= 10:
            groups = data.groupby(indep_var)[dep_var]
            for name, group in groups:
                if len(group.dropna()) > 0:
                    group.hist(alpha=0.6, label=f'{name} (n={len(group)})', ax=ax2, bins=20)
            
            ax2.set_xlabel(dep_var)
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution Comparison')
            ax2.legend()
        else:
            # For continuous variables, show distribution of dependent variable
            data[dep_var].hist(ax=ax2, bins=30, alpha=0.7)
            ax2.set_xlabel(dep_var)
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Dependent Variable')
        
        # Add test statistics
        stats_text = f"""Test Statistics:
Statistic: {format_number(test_result.get('statistic', 'N/A'))}
P-value: {format_p_value(test_result.get('p_value', 'N/A'))}
Effect Size: {format_number(test_result.get('effect_size', 'N/A'))}"""
        
        fig.text(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plots[f'{test_name}_comparison'] = fig
        
        return plots
    
    def _create_anova_plots(self, data: pd.DataFrame,
                          variables: Dict[str, List[str]],
                          test_name: str,
                          test_result: Dict[str, Any]) -> Dict[str, Figure]:
        """Create plots for ANOVA results.
        
        Args:
            data: Input DataFrame
            variables: Dictionary with variable assignments
            test_name: Name of the test
            test_result: Test results
            
        Returns:
            Dictionary of plots
        """
        plots = {}
        
        dependent_vars = variables.get('dependent', [])
        independent_vars = variables.get('independent', [])
        
        if not dependent_vars or not independent_vars:
            return plots
        
        dep_var = dependent_vars[0]
        indep_var = independent_vars[0]
        
        if dep_var not in data.columns or indep_var not in data.columns:
            return plots
        
        # Create ANOVA visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'ANOVA Results: {dep_var} by {indep_var}', fontsize=14, fontweight='bold')
        
        # Box plot
        sns.boxplot(data=data, x=indep_var, y=dep_var, ax=ax1)
        ax1.set_title('Group Comparison (Box Plot)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Violin plot
        sns.violinplot(data=data, x=indep_var, y=dep_var, ax=ax2)
        ax2.set_title('Distribution Shape Comparison')
        ax2.tick_params(axis='x', rotation=45)
        
        # Mean plot with error bars
        group_stats = data.groupby(indep_var)[dep_var].agg(['mean', 'std', 'count']).reset_index()
        group_stats['se'] = group_stats['std'] / np.sqrt(group_stats['count'])
        
        ax3.errorbar(range(len(group_stats)), group_stats['mean'], 
                    yerr=group_stats['se'], fmt='o-', capsize=5, capthick=2)
        ax3.set_xticks(range(len(group_stats)))
        ax3.set_xticklabels(group_stats[indep_var], rotation=45)
        ax3.set_ylabel(f'Mean {dep_var}')
        ax3.set_title('Group Means with Standard Error')
        ax3.grid(True, alpha=0.3)
        
        # Residuals plot (if available)
        if 'residuals' in test_result:
            residuals = test_result['residuals']
            fitted = test_result.get('fitted', range(len(residuals)))
            ax4.scatter(fitted, residuals, alpha=0.6)
            ax4.axhline(y=0, color='red', linestyle='--')
            ax4.set_xlabel('Fitted Values')
            ax4.set_ylabel('Residuals')
            ax4.set_title('Residuals vs Fitted')
        else:
            # Create a simple group comparison
            groups = data.groupby(indep_var)[dep_var]
            for i, (name, group) in enumerate(groups):
                if len(group.dropna()) > 0:
                    ax4.hist(group.dropna(), alpha=0.6, label=f'{name}', bins=15)
            ax4.set_xlabel(dep_var)
            ax4.set_ylabel('Frequency')
            ax4.set_title('Distribution by Group')
            ax4.legend()
        
        # Add test statistics
        stats_text = f"""ANOVA Results:
F-statistic: {format_number(test_result.get('statistic', 'N/A'))}
P-value: {format_p_value(test_result.get('p_value', 'N/A'))}
Effect Size (η²): {format_number(test_result.get('effect_size', 'N/A'))}"""
        
        fig.text(0.02, 0.02, stats_text, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plots[f'{test_name}_analysis'] = fig
        
        return plots
    
    def _create_nonparametric_plots(self, data: pd.DataFrame,
                                  variables: Dict[str, List[str]],
                                  test_name: str,
                                  test_result: Dict[str, Any]) -> Dict[str, Figure]:
        """Create plots for non-parametric test results.
        
        Args:
            data: Input DataFrame
            variables: Dictionary with variable assignments
            test_name: Name of the test
            test_result: Test results
            
        Returns:
            Dictionary of plots
        """
        plots = {}
        
        dependent_vars = variables.get('dependent', [])
        independent_vars = variables.get('independent', [])
        
        if not dependent_vars or not independent_vars:
            return plots
        
        dep_var = dependent_vars[0]
        indep_var = independent_vars[0]
        
        if dep_var not in data.columns or indep_var not in data.columns:
            return plots
        
        # Create non-parametric test visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Non-parametric Test: {dep_var} by {indep_var}', fontsize=14, fontweight='bold')
        
        # Box plot with individual points
        if data[indep_var].dtype in ['object', 'category'] or data[indep_var].nunique() <= 10:
            sns.boxplot(data=data, x=indep_var, y=dep_var, ax=ax1)
            sns.stripplot(data=data, x=indep_var, y=dep_var, ax=ax1, 
                         color='red', alpha=0.6, size=3)
            ax1.set_title('Group Comparison with Data Points')
            ax1.tick_params(axis='x', rotation=45)
        else:
            ax1.scatter(data[indep_var], data[dep_var], alpha=0.6)
            ax1.set_xlabel(indep_var)
            ax1.set_ylabel(dep_var)
            ax1.set_title('Scatter Plot')
        
        # Rank comparison (for non-parametric tests)
        if data[indep_var].dtype in ['object', 'category'] or data[indep_var].nunique() <= 10:
            # Calculate ranks
            data_copy = data.copy()
            data_copy[f'{dep_var}_rank'] = data_copy[dep_var].rank()
            
            sns.boxplot(data=data_copy, x=indep_var, y=f'{dep_var}_rank', ax=ax2)
            ax2.set_title('Rank Comparison')
            ax2.set_ylabel(f'Rank of {dep_var}')
            ax2.tick_params(axis='x', rotation=45)
        else:
            # Show distribution of dependent variable
            data[dep_var].hist(ax=ax2, bins=30, alpha=0.7)
            ax2.set_xlabel(dep_var)
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Dependent Variable')
        
        # Add test statistics
        stats_text = f"""Test Statistics:
Statistic: {format_number(test_result.get('statistic', 'N/A'))}
P-value: {format_p_value(test_result.get('p_value', 'N/A'))}
Effect Size: {format_number(test_result.get('effect_size', 'N/A'))}"""
        
        fig.text(0.02, 0.02, stats_text, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plots[f'{test_name}_nonparametric'] = fig
        
        return plots
    
    def _create_correlation_plots(self, data: pd.DataFrame,
                                variables: Dict[str, List[str]],
                                test_name: str,
                                test_result: Dict[str, Any]) -> Dict[str, Figure]:
        """Create plots for correlation analysis.
        
        Args:
            data: Input DataFrame
            variables: Dictionary with variable assignments
            test_name: Name of the test
            test_result: Test results
            
        Returns:
            Dictionary of plots
        """
        plots = {}
        
        dependent_vars = variables.get('dependent', [])
        independent_vars = variables.get('independent', [])
        
        if not dependent_vars or not independent_vars:
            return plots
        
        # Create correlation plots for all combinations
        all_vars = dependent_vars + independent_vars
        numeric_vars = [var for var in all_vars if var in data.columns and 
                       data[var].dtype in ['int64', 'float64']]
        
        if len(numeric_vars) < 2:
            return plots
        
        # Create correlation analysis plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Correlation Analysis', fontsize=14, fontweight='bold')
        
        # Scatter plot with regression line
        if len(numeric_vars) >= 2:
            var1, var2 = numeric_vars[0], numeric_vars[1]
            
            # Scatter plot
            ax1.scatter(data[var1], data[var2], alpha=0.6)
            
            # Add regression line
            try:
                z = np.polyfit(data[var1].dropna(), data[var2].dropna(), 1)
                p = np.poly1d(z)
                ax1.plot(data[var1], p(data[var1]), "r--", alpha=0.8)
            except:
                pass
            
            ax1.set_xlabel(var1)
            ax1.set_ylabel(var2)
            ax1.set_title(f'Scatter Plot: {var1} vs {var2}')
            ax1.grid(True, alpha=0.3)
            
            # Residuals plot
            try:
                from scipy import stats as scipy_stats
                slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                    data[var1].dropna(), data[var2].dropna()
                )
                predicted = slope * data[var1] + intercept
                residuals = data[var2] - predicted
                
                ax2.scatter(predicted, residuals, alpha=0.6)
                ax2.axhline(y=0, color='red', linestyle='--')
                ax2.set_xlabel('Predicted Values')
                ax2.set_ylabel('Residuals')
                ax2.set_title('Residuals Plot')
                ax2.grid(True, alpha=0.3)
            except:
                ax2.text(0.5, 0.5, 'Could not create residuals plot', 
                        ha='center', va='center', transform=ax2.transAxes)
        
        # Correlation matrix heatmap
        if len(numeric_vars) > 2:
            corr_matrix = data[numeric_vars].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                       square=True, ax=ax3, fmt='.3f')
            ax3.set_title('Correlation Matrix')
        else:
            ax3.text(0.5, 0.5, 'Need more variables for correlation matrix', 
                    ha='center', va='center', transform=ax3.transAxes)
        
        # Distribution plots
        if len(numeric_vars) >= 2:
            var1, var2 = numeric_vars[0], numeric_vars[1]
            
            # Joint distribution
            try:
                # Create a simple joint plot
                ax4.hist2d(data[var1].dropna(), data[var2].dropna(), bins=20, alpha=0.7)
                ax4.set_xlabel(var1)
                ax4.set_ylabel(var2)
                ax4.set_title('Joint Distribution')
            except:
                ax4.text(0.5, 0.5, 'Could not create joint distribution', 
                        ha='center', va='center', transform=ax4.transAxes)
        
        # Add correlation statistics
        correlation = test_result.get('statistic', 'N/A')
        p_value = test_result.get('p_value', 'N/A')
        
        stats_text = f"""Correlation Results:
Correlation: {format_number(correlation)}
P-value: {format_p_value(p_value)}
Test: {test_name}"""
        
        fig.text(0.02, 0.02, stats_text, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plots[f'{test_name}_correlation'] = fig
        
        return plots
    
    def _create_chi_square_plots(self, data: pd.DataFrame,
                               variables: Dict[str, List[str]],
                               test_name: str,
                               test_result: Dict[str, Any]) -> Dict[str, Figure]:
        """Create plots for chi-square test results.
        
        Args:
            data: Input DataFrame
            variables: Dictionary with variable assignments
            test_name: Name of the test
            test_result: Test results
            
        Returns:
            Dictionary of plots
        """
        plots = {}
        
        dependent_vars = variables.get('dependent', [])
        independent_vars = variables.get('independent', [])
        
        if not dependent_vars or not independent_vars:
            return plots
        
        dep_var = dependent_vars[0]
        indep_var = independent_vars[0]
        
        if dep_var not in data.columns or indep_var not in data.columns:
            return plots
        
        # Create chi-square visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Chi-Square Test: {dep_var} by {indep_var}', fontsize=14, fontweight='bold')
        
        # Contingency table heatmap
        try:
            contingency_table = pd.crosstab(data[indep_var], data[dep_var])
            sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', ax=ax1)
            ax1.set_title('Observed Frequencies')
            ax1.set_xlabel(dep_var)
            ax1.set_ylabel(indep_var)
        except Exception as e:
            ax1.text(0.5, 0.5, f'Error creating contingency table: {str(e)}', 
                    ha='center', va='center', transform=ax1.transAxes)
        
        # Expected frequencies (if available)
        if 'expected' in test_result:
            try:
                expected = test_result['expected']
                if isinstance(expected, np.ndarray):
                    expected_df = pd.DataFrame(expected, 
                                             index=contingency_table.index,
                                             columns=contingency_table.columns)
                    sns.heatmap(expected_df, annot=True, fmt='.1f', cmap='Reds', ax=ax2)
                    ax2.set_title('Expected Frequencies')
                    ax2.set_xlabel(dep_var)
                    ax2.set_ylabel(indep_var)
                else:
                    ax2.text(0.5, 0.5, 'Expected frequencies not available', 
                            ha='center', va='center', transform=ax2.transAxes)
            except:
                ax2.text(0.5, 0.5, 'Could not display expected frequencies', 
                        ha='center', va='center', transform=ax2.transAxes)
        else:
            ax2.text(0.5, 0.5, 'Expected frequencies not available', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        # Stacked bar chart
        try:
            contingency_pct = pd.crosstab(data[indep_var], data[dep_var], normalize='index') * 100
            contingency_pct.plot(kind='bar', stacked=True, ax=ax3, alpha=0.8)
            ax3.set_title('Percentage Distribution')
            ax3.set_xlabel(indep_var)
            ax3.set_ylabel('Percentage')
            ax3.legend(title=dep_var, bbox_to_anchor=(1.05, 1), loc='upper left')
            ax3.tick_params(axis='x', rotation=45)
        except:
            ax3.text(0.5, 0.5, 'Could not create percentage distribution', 
                    ha='center', va='center', transform=ax3.transAxes)
        
        # Residuals heatmap (if available)
        if 'residuals' in test_result:
            try:
                residuals = test_result['residuals']
                if isinstance(residuals, np.ndarray):
                    residuals_df = pd.DataFrame(residuals,
                                              index=contingency_table.index,
                                              columns=contingency_table.columns)
                    sns.heatmap(residuals_df, annot=True, fmt='.2f', 
                               cmap='RdBu_r', center=0, ax=ax4)
                    ax4.set_title('Standardized Residuals')
                    ax4.set_xlabel(dep_var)
                    ax4.set_ylabel(indep_var)
                else:
                    ax4.text(0.5, 0.5, 'Residuals not available', 
                            ha='center', va='center', transform=ax4.transAxes)
            except:
                ax4.text(0.5, 0.5, 'Could not display residuals', 
                        ha='center', va='center', transform=ax4.transAxes)
        else:
            # Create a simple grouped bar chart
            try:
                contingency_table.plot(kind='bar', ax=ax4, alpha=0.8)
                ax4.set_title('Grouped Bar Chart')
                ax4.set_xlabel(indep_var)
                ax4.set_ylabel('Count')
                ax4.legend(title=dep_var, bbox_to_anchor=(1.05, 1), loc='upper left')
                ax4.tick_params(axis='x', rotation=45)
            except:
                ax4.text(0.5, 0.5, 'Could not create grouped bar chart', 
                        ha='center', va='center', transform=ax4.transAxes)
        
        # Add test statistics
        chi2_stat = test_result.get('statistic', 'N/A')
        p_value = test_result.get('p_value', 'N/A')
        dof = test_result.get('dof', 'N/A')
        
        stats_text = f"""Chi-Square Results:
χ² = {format_number(chi2_stat)}
df = {dof}
P-value: {format_p_value(p_value)}
Effect Size: {format_number(test_result.get('effect_size', 'N/A'))}"""
        
        fig.text(0.02, 0.02, stats_text, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plots[f'{test_name}_chi_square'] = fig
        
        return plots
    
    def _create_comparison_plots(self, data: pd.DataFrame,
                               variables: Dict[str, List[str]],
                               results: Dict[str, Any]) -> Dict[str, Figure]:
        """Create general comparison plots.
        
        Args:
            data: Input DataFrame
            variables: Dictionary with variable assignments
            results: Analysis results
            
        Returns:
            Dictionary of plots
        """
        plots = {}
        
        dependent_vars = variables.get('dependent', [])
        independent_vars = variables.get('independent', [])
        
        if not dependent_vars or not independent_vars:
            return plots
        
        # Create comprehensive comparison plot
        try:
            dep_var = dependent_vars[0]
            indep_var = independent_vars[0]
            
            if dep_var not in data.columns or indep_var not in data.columns:
                return plots
            
            # Determine plot type based on variable types
            dep_is_numeric = data[dep_var].dtype in ['int64', 'float64']
            indep_is_numeric = data[indep_var].dtype in ['int64', 'float64']
            indep_is_categorical = (data[indep_var].dtype in ['object', 'category'] or 
                                  data[indep_var].nunique() <= 10)
            
            if dep_is_numeric and indep_is_categorical:
                # Numeric dependent, categorical independent
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(f'Group Comparison: {dep_var} by {indep_var}', 
                           fontsize=14, fontweight='bold')
                
                # Box plot
                sns.boxplot(data=data, x=indep_var, y=dep_var, ax=ax1)
                ax1.set_title('Box Plot Comparison')
                ax1.tick_params(axis='x', rotation=45)
                
                # Violin plot
                sns.violinplot(data=data, x=indep_var, y=dep_var, ax=ax2)
                ax2.set_title('Distribution Shape Comparison')
                ax2.tick_params(axis='x', rotation=45)
                
                # Strip plot
                sns.stripplot(data=data, x=indep_var, y=dep_var, ax=ax3, alpha=0.6)
                ax3.set_title('Individual Data Points')
                ax3.tick_params(axis='x', rotation=45)
                
                # Histogram by group
                groups = data.groupby(indep_var)[dep_var]
                for name, group in groups:
                    if len(group.dropna()) > 0:
                        group.hist(alpha=0.6, label=f'{name} (n={len(group)})', 
                                 ax=ax4, bins=20)
                ax4.set_xlabel(dep_var)
                ax4.set_ylabel('Frequency')
                ax4.set_title('Distribution by Group')
                ax4.legend()
                
                plt.tight_layout()
                plots['group_comparison'] = fig
                
            elif dep_is_numeric and indep_is_numeric:
                # Both numeric - correlation/regression plot
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(f'Relationship: {dep_var} vs {indep_var}', 
                           fontsize=14, fontweight='bold')
                
                # Scatter plot
                ax1.scatter(data[indep_var], data[dep_var], alpha=0.6)
                ax1.set_xlabel(indep_var)
                ax1.set_ylabel(dep_var)
                ax1.set_title('Scatter Plot')
                ax1.grid(True, alpha=0.3)
                
                # Scatter with regression line
                sns.regplot(data=data, x=indep_var, y=dep_var, ax=ax2, scatter_kws={'alpha': 0.6})
                ax2.set_title('Regression Line')
                
                # Hexbin plot for density
                ax3.hexbin(data[indep_var], data[dep_var], gridsize=20, cmap='Blues')
                ax3.set_xlabel(indep_var)
                ax3.set_ylabel(dep_var)
                ax3.set_title('Density Plot')
                
                # Residuals (if regression was performed)
                try:
                    from scipy import stats as scipy_stats
                    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                        data[indep_var].dropna(), data[dep_var].dropna()
                    )
                    predicted = slope * data[indep_var] + intercept
                    residuals = data[dep_var] - predicted
                    
                    ax4.scatter(predicted, residuals, alpha=0.6)
                    ax4.axhline(y=0, color='red', linestyle='--')
                    ax4.set_xlabel('Predicted Values')
                    ax4.set_ylabel('Residuals')
                    ax4.set_title('Residuals Plot')
                    ax4.grid(True, alpha=0.3)
                except:
                    ax4.text(0.5, 0.5, 'Could not create residuals plot', 
                            ha='center', va='center', transform=ax4.transAxes)
                
                plt.tight_layout()
                plots['relationship_analysis'] = fig
            
        except Exception as e:
            self.logger.warning(f"Error creating comparison plots: {str(e)}")
        
        return plots
    
    def _create_effect_size_plots(self, data: pd.DataFrame,
                                variables: Dict[str, List[str]],
                                results: Dict[str, Any]) -> Dict[str, Figure]:
        """Create effect size visualization plots.
        
        Args:
            data: Input DataFrame
            variables: Dictionary with variable assignments
            results: Analysis results
            
        Returns:
            Dictionary of plots
        """
        plots = {}
        
        if 'statistical_tests' not in results:
            return plots
        
        # Extract effect sizes
        effect_data = []
        for test_name, test_result in results['statistical_tests'].items():
            if isinstance(test_result, dict) and 'effect_size' in test_result:
                effect_size = test_result['effect_size']
                if isinstance(effect_size, (int, float)) and not np.isnan(effect_size):
                    effect_data.append({
                        'Test': test_name,
                        'Effect Size': abs(effect_size),
                        'Direction': 'Positive' if effect_size >= 0 else 'Negative'
                    })
        
        if not effect_data:
            return plots
        
        # Create effect size plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Effect Size Analysis', fontsize=14, fontweight='bold')
        
        # Bar plot of effect sizes
        effect_df = pd.DataFrame(effect_data)
        colors = ['green' if d == 'Positive' else 'red' for d in effect_df['Direction']]
        
        bars = ax1.bar(range(len(effect_df)), effect_df['Effect Size'], color=colors, alpha=0.7)
        ax1.set_xticks(range(len(effect_df)))
        ax1.set_xticklabels(effect_df['Test'], rotation=45, ha='right')
        ax1.set_ylabel('Effect Size (absolute)')
        ax1.set_title('Effect Sizes by Test')
        ax1.grid(True, alpha=0.3)
        
        # Add effect size interpretation lines
        ax1.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='Small (0.2)')
        ax1.axhline(y=0.5, color='blue', linestyle='--', alpha=0.7, label='Medium (0.5)')
        ax1.axhline(y=0.8, color='purple', linestyle='--', alpha=0.7, label='Large (0.8)')
        ax1.legend()
        
        # Effect size interpretation
        interpretations = []
        for _, row in effect_df.iterrows():
            es = row['Effect Size']
            if es < 0.2:
                interp = 'Negligible'
            elif es < 0.5:
                interp = 'Small'
            elif es < 0.8:
                interp = 'Medium'
            else:
                interp = 'Large'
            interpretations.append(interp)
        
        effect_df['Interpretation'] = interpretations
        
        # Pie chart of effect size categories
        interp_counts = effect_df['Interpretation'].value_counts()
        colors_pie = ['red', 'orange', 'yellow', 'green']
        ax2.pie(interp_counts.values, labels=interp_counts.index, autopct='%1.1f%%',
               colors=colors_pie[:len(interp_counts)], startangle=90)
        ax2.set_title('Effect Size Distribution')
        
        plt.tight_layout()
        plots['effect_sizes'] = fig
        
        return plots