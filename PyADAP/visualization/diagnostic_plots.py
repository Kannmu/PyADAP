"""Diagnostic plots for PyADAP 3.0

This module provides diagnostic plotting functions for assumption checking,
data quality assessment, and model validation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from matplotlib.figure import Figure
import scipy.stats as stats
from scipy.stats import shapiro, normaltest, jarque_bera
from sklearn.preprocessing import StandardScaler
import warnings

from ..utils import get_logger, format_number, format_p_value
from ..config import Config

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


class DiagnosticPlots:
    """Class for creating diagnostic and assumption checking plots."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the diagnostic plots generator.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.logger = get_logger("PyADAP.DiagnosticPlots")
    
    def create_diagnostic_plots(self, data: pd.DataFrame,
                              variables: Dict[str, List[str]],
                              results: Optional[Dict[str, Any]] = None) -> Dict[str, Figure]:
        """Create comprehensive diagnostic plots.
        
        Args:
            data: Input DataFrame
            variables: Dictionary with variable assignments
            results: Optional analysis results
            
        Returns:
            Dictionary of figure names and Figure objects
        """
        plots = {}
        
        try:
            # Normality diagnostic plots
            plots.update(self._create_normality_plots(data, variables))
            
            # Homogeneity of variance plots
            plots.update(self._create_homogeneity_plots(data, variables))
            
            # Linearity diagnostic plots
            plots.update(self._create_linearity_plots(data, variables))
            
            # Independence diagnostic plots
            plots.update(self._create_independence_plots(data, variables))
            
            # Outlier diagnostic plots
            plots.update(self._create_outlier_diagnostic_plots(data, variables))
            
            # Multicollinearity diagnostic plots
            plots.update(self._create_multicollinearity_plots(data, variables))
            
            # Model diagnostic plots (if results available)
            if results:
                plots.update(self._create_model_diagnostic_plots(data, variables, results))
            
            self.logger.info(f"Created {len(plots)} diagnostic plots")
            
        except Exception as e:
            self.logger.error(f"Error creating diagnostic plots: {str(e)}")
            raise
        
        return plots
    
    def _create_normality_plots(self, data: pd.DataFrame,
                              variables: Dict[str, List[str]]) -> Dict[str, Figure]:
        """Create normality diagnostic plots.
        
        Args:
            data: Input DataFrame
            variables: Dictionary with variable assignments
            
        Returns:
            Dictionary of plots
        """
        plots = {}
        
        # Get numeric variables
        all_vars = variables.get('dependent', []) + variables.get('independent', [])
        numeric_vars = [var for var in all_vars if var in data.columns and 
                       data[var].dtype in ['int64', 'float64']]
        
        if not numeric_vars:
            return plots
        
        # Limit to reasonable number of variables
        if len(numeric_vars) > 6:
            numeric_vars = numeric_vars[:6]
            self.logger.warning("Limited normality plots to first 6 numeric variables")
        
        # Create normality diagnostic plots
        n_vars = len(numeric_vars)
        n_cols = min(3, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        fig.suptitle('Normality Diagnostics', fontsize=16, fontweight='bold')
        
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, var in enumerate(numeric_vars):
            ax = axes[i]
            
            try:
                # Get clean data
                clean_data = data[var].dropna()
                
                if len(clean_data) < 3:
                    ax.text(0.5, 0.5, f'Insufficient data for {var}', 
                           ha='center', va='center', transform=ax.transAxes)
                    continue
                
                # Q-Q plot
                stats.probplot(clean_data, dist="norm", plot=ax)
                ax.set_title(f'Q-Q Plot: {var}')
                ax.grid(True, alpha=0.3)
                
                # Perform normality tests
                test_results = []
                
                # Shapiro-Wilk test (for n < 5000)
                if len(clean_data) < 5000:
                    try:
                        shapiro_stat, shapiro_p = shapiro(clean_data)
                        test_results.append(f'Shapiro: p={format_p_value(shapiro_p)}')
                    except Exception as e:
                        self.logger.warning(f"Shapiro-Wilk test failed for {var}: {e}")
                
                # D'Agostino's normality test
                if len(clean_data) >= 8:
                    try:
                        dagostino_stat, dagostino_p = normaltest(clean_data)
                        test_results.append(f"D'Agostino: p={format_p_value(dagostino_p)}")
                    except Exception as e:
                        self.logger.warning(f"D'Agostino's test failed for {var}: {e}")
                
                # Jarque-Bera test
                if len(clean_data) >= 2:
                    try:
                        jb_stat, jb_p = jarque_bera(clean_data)
                        test_results.append(f'Jarque-Bera: p={format_p_value(jb_p)}')
                    except Exception as e:
                        self.logger.warning(f"Jarque-Bera test failed for {var}: {e}")
                
                # Add test results to plot
                if test_results:
                    test_text = '\n'.join(test_results)
                    ax.text(0.02, 0.98, test_text, transform=ax.transAxes, 
                           verticalalignment='top', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error plotting {var}\n{str(e)}', 
                       ha='center', va='center', transform=ax.transAxes)
                self.logger.warning(f"Error creating normality plot for {var}: {str(e)}")
        
        # Hide empty subplots
        for i in range(len(numeric_vars), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plots['normality_diagnostics'] = fig
        
        # Create histogram with normal overlay
        fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        fig2.suptitle('Distribution vs Normal Curve', fontsize=16, fontweight='bold')
        
        if n_rows == 1 and n_cols == 1:
            axes2 = [axes2]
        elif n_rows == 1 or n_cols == 1:
            axes2 = axes2.flatten()
        else:
            axes2 = axes2.flatten()
        
        for i, var in enumerate(numeric_vars):
            ax = axes2[i]
            
            try:
                clean_data = data[var].dropna()
                
                if len(clean_data) < 3:
                    ax.text(0.5, 0.5, f'Insufficient data for {var}', 
                           ha='center', va='center', transform=ax.transAxes)
                    continue
                
                # Histogram
                n_bins = min(30, max(10, int(np.sqrt(len(clean_data)))))
                counts, bins, patches = ax.hist(clean_data, bins=n_bins, density=True, 
                                               alpha=0.7, color='skyblue', edgecolor='black')
                
                # Overlay normal distribution
                mu, sigma = clean_data.mean(), clean_data.std()
                x = np.linspace(clean_data.min(), clean_data.max(), 100)
                normal_curve = stats.norm.pdf(x, mu, sigma)
                ax.plot(x, normal_curve, 'r-', linewidth=2, label='Normal Distribution')
                
                # Add KDE
                try:
                    clean_data.plot.kde(ax=ax, color='green', linewidth=2, label='KDE')
                except Exception as e:
                    self.logger.warning(f"KDE plot failed for {var}: {e}")
                
                ax.set_title(f'Distribution: {var}')
                ax.set_xlabel(var)
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add statistics
                stats_text = f'μ = {mu:.3f}\nσ = {sigma:.3f}\nSkew = {stats.skew(clean_data):.3f}\nKurt = {stats.kurtosis(clean_data):.3f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error plotting {var}\n{str(e)}', 
                       ha='center', va='center', transform=ax.transAxes)
        
        # Hide empty subplots
        for i in range(len(numeric_vars), len(axes2)):
            axes2[i].set_visible(False)
        
        plt.tight_layout()
        plots['distribution_comparison'] = fig2
        
        return plots
    
    def _create_homogeneity_plots(self, data: pd.DataFrame,
                                variables: Dict[str, List[str]]) -> Dict[str, Figure]:
        """Create homogeneity of variance diagnostic plots.
        
        Args:
            data: Input DataFrame
            variables: Dictionary with variable assignments
            
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
        
        # Check if independent variable is categorical
        if not (data[indep_var].dtype in ['object', 'category'] or data[indep_var].nunique() <= 10):
            return plots
        
        # Create homogeneity diagnostic plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Homogeneity of Variance: {dep_var} by {indep_var}', 
                    fontsize=14, fontweight='bold')
        
        try:
            # Box plot to visualize variance differences
            sns.boxplot(data=data, x=indep_var, y=dep_var, ax=ax1)
            ax1.set_title('Box Plot - Variance Comparison')
            ax1.tick_params(axis='x', rotation=45)
            
            # Violin plot to show distribution shapes
            sns.violinplot(data=data, x=indep_var, y=dep_var, ax=ax2)
            ax2.set_title('Violin Plot - Distribution Shapes')
            ax2.tick_params(axis='x', rotation=45)
            
            # Spread vs level plot
            groups = data.groupby(indep_var)[dep_var]
            group_means = groups.mean()
            group_stds = groups.std()
            
            ax3.scatter(group_means, group_stds, s=100, alpha=0.7)
            for i, (mean, std) in enumerate(zip(group_means, group_stds)):
                ax3.annotate(group_means.index[i], (mean, std), 
                           xytext=(5, 5), textcoords='offset points')
            
            ax3.set_xlabel('Group Mean')
            ax3.set_ylabel('Group Standard Deviation')
            ax3.set_title('Spread vs Level Plot')
            ax3.grid(True, alpha=0.3)
            
            # Residuals vs fitted (if we can compute them)
            try:
                # Simple ANOVA-style residuals
                overall_mean = data[dep_var].mean()
                residuals = []
                fitted = []
                
                for group_name, group_data in groups:
                    group_mean = group_data.mean()
                    group_residuals = group_data - group_mean
                    residuals.extend(group_residuals)
                    fitted.extend([group_mean] * len(group_data))
                
                ax4.scatter(fitted, residuals, alpha=0.6)
                ax4.axhline(y=0, color='red', linestyle='--')
                ax4.set_xlabel('Fitted Values (Group Means)')
                ax4.set_ylabel('Residuals')
                ax4.set_title('Residuals vs Fitted')
                ax4.grid(True, alpha=0.3)
                
            except Exception as e:
                ax4.text(0.5, 0.5, f'Could not create residuals plot\n{str(e)}', 
                        ha='center', va='center', transform=ax4.transAxes)
            
            # Perform Levene's test
            try:
                from scipy.stats import levene
                group_data = [group.dropna() for name, group in groups if len(group.dropna()) > 0]
                if len(group_data) >= 2:
                    levene_stat, levene_p = levene(*group_data)
                    
                    # Add test result
                    test_text = f"Levene's Test:\nStatistic: {format_number(levene_stat)}\nP-value: {format_p_value(levene_p)}"
                    if levene_p < 0.05:
                        test_text += "\nResult: Variances differ significantly"
                    else:
                        test_text += "\nResult: Variances are homogeneous"
                    
                    fig.text(0.02, 0.02, test_text, fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            except Exception as e:
                self.logger.warning(f"Could not perform Levene's test: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error creating homogeneity plots: {str(e)}")
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        plots['homogeneity_diagnostics'] = fig
        
        return plots
    
    def _create_linearity_plots(self, data: pd.DataFrame,
                              variables: Dict[str, List[str]]) -> Dict[str, Figure]:
        """Create linearity diagnostic plots.
        
        Args:
            data: Input DataFrame
            variables: Dictionary with variable assignments
            
        Returns:
            Dictionary of plots
        """
        plots = {}
        
        # Get numeric variables
        all_vars = variables.get('dependent', []) + variables.get('independent', [])
        numeric_vars = [var for var in all_vars if var in data.columns and 
                       data[var].dtype in ['int64', 'float64']]
        
        if len(numeric_vars) < 2:
            return plots
        
        # Create linearity diagnostic plots
        dependent_vars = variables.get('dependent', [])
        independent_vars = variables.get('independent', [])
        
        if dependent_vars and independent_vars:
            dep_var = dependent_vars[0]
            
            # Get numeric independent variables
            numeric_indep = [var for var in independent_vars if var in data.columns and 
                           data[var].dtype in ['int64', 'float64']]
            
            if numeric_indep:
                n_vars = len(numeric_indep)
                n_cols = min(3, n_vars)
                n_rows = (n_vars + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
                fig.suptitle(f'Linearity Diagnostics: {dep_var}', fontsize=14, fontweight='bold')
                
                if n_rows == 1 and n_cols == 1:
                    axes = [axes]
                elif n_rows == 1 or n_cols == 1:
                    axes = axes.flatten()
                else:
                    axes = axes.flatten()
                
                for i, indep_var in enumerate(numeric_indep):
                    ax = axes[i]
                    
                    try:
                        # Scatter plot with regression line and LOWESS
                        clean_data = data[[dep_var, indep_var]].dropna()
                        
                        if len(clean_data) < 3:
                            ax.text(0.5, 0.5, 'Insufficient data', 
                                   ha='center', va='center', transform=ax.transAxes)
                            continue
                        
                        x = clean_data[indep_var]
                        y = clean_data[dep_var]
                        
                        # Scatter plot
                        ax.scatter(x, y, alpha=0.6, s=30)
                        
                        # Linear regression line
                        try:
                            z = np.polyfit(x, y, 1)
                            p = np.poly1d(z)
                            ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2, label='Linear')
                        except Exception as e:
                            self.logger.warning(f"Linear regression line failed for {indep_var} vs {dep_var}: {e}")
                        
                        # LOWESS smooth line
                        try:
                            from scipy.signal import savgol_filter
                            # Sort data for smooth line
                            sorted_indices = np.argsort(x)
                            x_sorted = x.iloc[sorted_indices]
                            y_sorted = y.iloc[sorted_indices]
                            
                            # Apply smoothing if we have enough points
                            if len(x_sorted) > 10:
                                window_length = min(len(x_sorted) // 3, 51)
                                if window_length % 2 == 0:
                                    window_length += 1
                                if window_length >= 3:
                                    y_smooth = savgol_filter(y_sorted, window_length, 2)
                                    ax.plot(x_sorted, y_smooth, 'g-', linewidth=2, label='LOWESS')
                        except Exception as e:
                            self.logger.warning(f"LOWESS smoothing failed for {indep_var} vs {dep_var}: {e}")
                        
                        ax.set_xlabel(indep_var)
                        ax.set_ylabel(dep_var)
                        ax.set_title(f'{dep_var} vs {indep_var}')
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        
                        # Calculate correlation
                        try:
                            correlation = np.corrcoef(x, y)[0, 1]
                            ax.text(0.02, 0.98, f'r = {correlation:.3f}', 
                                   transform=ax.transAxes, verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                        except Exception as e:
                            self.logger.warning(f"Correlation calculation failed for {indep_var} vs {dep_var}: {e}")
                        
                    except Exception as e:
                        ax.text(0.5, 0.5, f'Error plotting {indep_var}\n{str(e)}', 
                               ha='center', va='center', transform=ax.transAxes)
                
                # Hide empty subplots
                for i in range(len(numeric_indep), len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                plots['linearity_diagnostics'] = fig
        
        return plots
    
    def _create_independence_plots(self, data: pd.DataFrame,
                                 variables: Dict[str, List[str]]) -> Dict[str, Figure]:
        """Create independence diagnostic plots.
        
        Args:
            data: Input DataFrame
            variables: Dictionary with variable assignments
            
        Returns:
            Dictionary of plots
        """
        plots = {}
        
        # Get numeric variables
        all_vars = variables.get('dependent', []) + variables.get('independent', [])
        numeric_vars = [var for var in all_vars if var in data.columns and 
                       data[var].dtype in ['int64', 'float64']]
        
        if not numeric_vars:
            return plots
        
        # Create independence diagnostic plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Independence Diagnostics', fontsize=14, fontweight='bold')
        
        try:
            # Plot 1: Autocorrelation plot (if we have a time-like index)
            if len(data) > 10:
                # Use the first numeric variable
                var = numeric_vars[0]
                clean_data = data[var].dropna()
                
                if len(clean_data) > 10:
                    # Simple autocorrelation
                    lags = range(1, min(20, len(clean_data) // 4))
                    autocorrs = []
                    
                    for lag in lags:
                        if lag < len(clean_data):
                            corr = np.corrcoef(clean_data[:-lag], clean_data[lag:])[0, 1]
                            autocorrs.append(corr if not np.isnan(corr) else 0)
                        else:
                            autocorrs.append(0)
                    
                    ax1.bar(lags, autocorrs, alpha=0.7)
                    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                    ax1.axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='±0.2')
                    ax1.axhline(y=-0.2, color='red', linestyle='--', alpha=0.7)
                    ax1.set_xlabel('Lag')
                    ax1.set_ylabel('Autocorrelation')
                    ax1.set_title(f'Autocorrelation: {var}')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                else:
                    ax1.text(0.5, 0.5, 'Insufficient data for autocorrelation', 
                            ha='center', va='center', transform=ax1.transAxes)
            else:
                ax1.text(0.5, 0.5, 'Insufficient data for autocorrelation', 
                        ha='center', va='center', transform=ax1.transAxes)
            
            # Plot 2: Residuals vs order (if we have residuals or can compute them)
            if len(numeric_vars) >= 2:
                var1, var2 = numeric_vars[0], numeric_vars[1]
                clean_data = data[[var1, var2]].dropna()
                
                if len(clean_data) > 3:
                    # Simple residuals from linear regression
                    try:
                        from scipy import stats as scipy_stats
                        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                            clean_data[var2], clean_data[var1]
                        )
                        predicted = slope * clean_data[var2] + intercept
                        residuals = clean_data[var1] - predicted
                        
                        ax2.plot(range(len(residuals)), residuals, 'o-', alpha=0.7)
                        ax2.axhline(y=0, color='red', linestyle='--')
                        ax2.set_xlabel('Observation Order')
                        ax2.set_ylabel('Residuals')
                        ax2.set_title('Residuals vs Order')
                        ax2.grid(True, alpha=0.3)
                    except Exception as e:
                        self.logger.warning(f"Could not compute residuals for {var1} vs {var2}: {e}")
                        ax2.text(0.5, 0.5, 'Could not compute residuals', 
                                ha='center', va='center', transform=ax2.transAxes)
                else:
                    ax2.text(0.5, 0.5, 'Insufficient data', 
                            ha='center', va='center', transform=ax2.transAxes)
            else:
                ax2.text(0.5, 0.5, 'Need at least 2 numeric variables', 
                        ha='center', va='center', transform=ax2.transAxes)
            
            # Plot 3: Runs test visualization
            if len(numeric_vars) >= 1:
                var = numeric_vars[0]
                clean_data = data[var].dropna()
                
                if len(clean_data) > 10:
                    # Simple runs test - above/below median
                    median_val = clean_data.median()
                    above_median = (clean_data > median_val).astype(int)
                    
                    ax3.plot(range(len(above_median)), above_median, 'o-', alpha=0.7)
                    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Median')
                    ax3.set_xlabel('Observation Order')
                    ax3.set_ylabel('Above Median (1) / Below Median (0)')
                    ax3.set_title(f'Runs Test: {var}')
                    ax3.set_ylim(-0.1, 1.1)
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
                    
                    # Count runs
                    runs = 1
                    for i in range(1, len(above_median)):
                        if above_median.iloc[i] != above_median.iloc[i-1]:
                            runs += 1
                    
                    ax3.text(0.02, 0.98, f'Number of runs: {runs}', 
                            transform=ax3.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    ax3.text(0.5, 0.5, 'Insufficient data for runs test', 
                            ha='center', va='center', transform=ax3.transAxes)
            else:
                ax3.text(0.5, 0.5, 'No numeric variables', 
                        ha='center', va='center', transform=ax3.transAxes)
            
            # Plot 4: Durbin-Watson test visualization (if applicable)
            if len(numeric_vars) >= 2:
                var1, var2 = numeric_vars[0], numeric_vars[1]
                clean_data = data[[var1, var2]].dropna()
                
                if len(clean_data) > 3:
                    try:
                        # Compute residuals
                        from scipy import stats as scipy_stats
                        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                            clean_data[var2], clean_data[var1]
                        )
                        predicted = slope * clean_data[var2] + intercept
                        residuals = clean_data[var1] - predicted
                        
                        # Plot residuals vs lagged residuals
                        if len(residuals) > 1:
                            lagged_residuals = residuals.shift(1).dropna()
                            current_residuals = residuals[1:]
                            
                            ax4.scatter(lagged_residuals, current_residuals, alpha=0.7)
                            ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
                            ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7)
                            ax4.set_xlabel('Lagged Residuals (t-1)')
                            ax4.set_ylabel('Current Residuals (t)')
                            ax4.set_title('Residuals vs Lagged Residuals')
                            ax4.grid(True, alpha=0.3)
                            
                            # Calculate simple Durbin-Watson statistic
                            diff_residuals = np.diff(residuals)
                            dw_stat = np.sum(diff_residuals**2) / np.sum(residuals**2)
                            
                            ax4.text(0.02, 0.98, f'DW ≈ {dw_stat:.3f}', 
                                    transform=ax4.transAxes, verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                        else:
                            ax4.text(0.5, 0.5, 'Insufficient residuals', 
                                    ha='center', va='center', transform=ax4.transAxes)
                    except Exception as e:
                        self.logger.warning(f"Could not compute residuals for Durbin-Watson test ({var1} vs {var2}): {e}")
                        ax4.text(0.5, 0.5, 'Could not compute residuals', 
                                ha='center', va='center', transform=ax4.transAxes)
                else:
                    ax4.text(0.5, 0.5, 'Insufficient data', 
                            ha='center', va='center', transform=ax4.transAxes)
            else:
                ax4.text(0.5, 0.5, 'Need at least 2 numeric variables', 
                        ha='center', va='center', transform=ax4.transAxes)
            
        except Exception as e:
            self.logger.error(f"Error creating independence plots: {str(e)}")
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        plots['independence_diagnostics'] = fig
        
        return plots
    
    def _create_outlier_diagnostic_plots(self, data: pd.DataFrame,
                                       variables: Dict[str, List[str]]) -> Dict[str, Figure]:
        """Create outlier diagnostic plots.
        
        Args:
            data: Input DataFrame
            variables: Dictionary with variable assignments
            
        Returns:
            Dictionary of plots
        """
        plots = {}
        
        # Get numeric variables
        all_vars = variables.get('dependent', []) + variables.get('independent', [])
        numeric_vars = [var for var in all_vars if var in data.columns and 
                       data[var].dtype in ['int64', 'float64']]
        
        if not numeric_vars:
            return plots
        
        # Limit to reasonable number of variables
        if len(numeric_vars) > 6:
            numeric_vars = numeric_vars[:6]
            self.logger.warning("Limited outlier plots to first 6 numeric variables")
        
        # Create outlier diagnostic plots
        n_vars = len(numeric_vars)
        n_cols = min(3, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        fig.suptitle('Outlier Diagnostics', fontsize=16, fontweight='bold')
        
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, var in enumerate(numeric_vars):
            ax = axes[i]
            
            try:
                clean_data = data[var].dropna()
                
                if len(clean_data) < 3:
                    ax.text(0.5, 0.5, f'Insufficient data for {var}', 
                           ha='center', va='center', transform=ax.transAxes)
                    continue
                
                # Box plot with outlier identification
                bp = ax.boxplot(clean_data, patch_artist=True, 
                               boxprops=dict(facecolor='lightblue', alpha=0.7),
                               medianprops=dict(color='red', linewidth=2))
                
                # Calculate outlier statistics
                Q1 = clean_data.quantile(0.25)
                Q3 = clean_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
                outlier_pct = len(outliers) / len(clean_data) * 100
                
                # Z-score outliers
                z_scores = np.abs(stats.zscore(clean_data))
                z_outliers = clean_data[z_scores > 3]
                z_outlier_pct = len(z_outliers) / len(clean_data) * 100
                
                ax.set_title(f'{var}')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
                
                # Add statistics
                stats_text = f'IQR Outliers: {len(outliers)} ({outlier_pct:.1f}%)\nZ-score Outliers: {len(z_outliers)} ({z_outlier_pct:.1f}%)'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error plotting {var}\n{str(e)}', 
                       ha='center', va='center', transform=ax.transAxes)
        
        # Hide empty subplots
        for i in range(len(numeric_vars), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plots['outlier_diagnostics'] = fig
        
        return plots
    
    def _create_multicollinearity_plots(self, data: pd.DataFrame,
                                       variables: Dict[str, List[str]]) -> Dict[str, Figure]:
        """Create multicollinearity diagnostic plots.
        
        Args:
            data: Input DataFrame
            variables: Dictionary with variable assignments
            
        Returns:
            Dictionary of plots
        """
        plots = {}
        
        # Get numeric independent variables
        independent_vars = variables.get('independent', [])
        numeric_indep = [var for var in independent_vars if var in data.columns and 
                        data[var].dtype in ['int64', 'float64']]
        
        if len(numeric_indep) < 2:
            return plots
        
        # Create multicollinearity diagnostic plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Multicollinearity Diagnostics', fontsize=14, fontweight='bold')
        
        try:
            # Correlation matrix heatmap
            corr_matrix = data[numeric_indep].corr()
            
            # Mask for upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                       square=True, linewidths=0.5, ax=ax1, fmt='.3f')
            ax1.set_title('Correlation Matrix')
            
            # Correlation magnitude plot
            # Get upper triangle correlations
            upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
            correlations = corr_matrix.values[upper_tri_indices]
            abs_correlations = np.abs(correlations)
            
            # Create labels for variable pairs
            var_pairs = []
            for i, j in zip(upper_tri_indices[0], upper_tri_indices[1]):
                var_pairs.append(f'{numeric_indep[i]}\nvs\n{numeric_indep[j]}')
            
            # Bar plot of absolute correlations
            colors = ['red' if abs_corr > 0.8 else 'orange' if abs_corr > 0.6 else 'green' 
                     for abs_corr in abs_correlations]
            
            bars = ax2.bar(range(len(abs_correlations)), abs_correlations, color=colors, alpha=0.7)
            ax2.set_xticks(range(len(var_pairs)))
            ax2.set_xticklabels(var_pairs, rotation=45, ha='right', fontsize=8)
            ax2.set_ylabel('Absolute Correlation')
            ax2.set_title('Pairwise Correlations')
            ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='High (0.8)')
            ax2.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='Moderate (0.6)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # VIF calculation (simplified)
            try:
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import r2_score
                
                vif_data = []
                clean_data = data[numeric_indep].dropna()
                
                if len(clean_data) > len(numeric_indep) + 1:
                    for i, var in enumerate(numeric_indep):
                        # Regress var on all other variables
                        X = clean_data.drop(columns=[var])
                        y = clean_data[var]
                        
                        if len(X.columns) > 0:
                            model = LinearRegression()
                            model.fit(X, y)
                            y_pred = model.predict(X)
                            r2 = r2_score(y, y_pred)
                            
                            # VIF = 1 / (1 - R²)
                            vif = 1 / (1 - r2) if r2 < 0.999 else float('inf')
                            vif_data.append({'Variable': var, 'VIF': vif})
                    
                    if vif_data:
                        vif_df = pd.DataFrame(vif_data)
                        
                        # VIF bar plot
                        colors = ['red' if vif > 10 else 'orange' if vif > 5 else 'green' 
                                 for vif in vif_df['VIF']]
                        
                        bars = ax3.bar(range(len(vif_df)), vif_df['VIF'], color=colors, alpha=0.7)
                        ax3.set_xticks(range(len(vif_df)))
                        ax3.set_xticklabels(vif_df['Variable'], rotation=45, ha='right')
                        ax3.set_ylabel('VIF')
                        ax3.set_title('Variance Inflation Factors')
                        ax3.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='High (10)')
                        ax3.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Moderate (5)')
                        ax3.legend()
                        ax3.grid(True, alpha=0.3)
                        
                        # Add VIF values on bars
                        for i, (bar, vif) in enumerate(zip(bars, vif_df['VIF'])):
                            height = bar.get_height()
                            if not np.isinf(height):
                                ax3.text(bar.get_x() + bar.get_width()/2., height + max(vif_df['VIF']) * 0.01,
                                        f'{vif:.2f}', ha='center', va='bottom', fontsize=8)
                    else:
                        ax3.text(0.5, 0.5, 'Could not calculate VIF', 
                                ha='center', va='center', transform=ax3.transAxes)
                else:
                    ax3.text(0.5, 0.5, 'Insufficient data for VIF calculation', 
                            ha='center', va='center', transform=ax3.transAxes)
                    
            except Exception as e:
                self.logger.warning(f"VIF calculation failed: {e}")
                ax3.text(0.5, 0.5, f'VIF calculation error:\n{str(e)}', 
                        ha='center', va='center', transform=ax3.transAxes)
            
            # Condition number and eigenvalues
            try:
                # Standardize the data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(clean_data)
                
                # Calculate correlation matrix eigenvalues
                eigenvalues = np.linalg.eigvals(corr_matrix)
                condition_number = np.max(eigenvalues) / np.min(eigenvalues)
                
                # Plot eigenvalues
                ax4.bar(range(len(eigenvalues)), sorted(eigenvalues, reverse=True), alpha=0.7)
                ax4.set_xlabel('Eigenvalue Index')
                ax4.set_ylabel('Eigenvalue')
                ax4.set_title('Correlation Matrix Eigenvalues')
                ax4.grid(True, alpha=0.3)
                
                # Add condition number
                ax4.text(0.02, 0.98, f'Condition Number: {condition_number:.2f}', 
                        transform=ax4.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                if condition_number > 30:
                    ax4.text(0.02, 0.88, 'High multicollinearity detected', 
                            transform=ax4.transAxes, verticalalignment='top', color='red',
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
                
            except Exception as e:
                self.logger.warning(f"Eigenvalue calculation failed: {e}")
                ax4.text(0.5, 0.5, f'Eigenvalue calculation error:\n{str(e)}', 
                        ha='center', va='center', transform=ax4.transAxes)
            
        except Exception as e:
            self.logger.error(f"Error creating multicollinearity plots: {str(e)}")
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        plots['multicollinearity_diagnostics'] = fig
        
        return plots
    
    def _create_model_diagnostic_plots(self, data: pd.DataFrame,
                                     variables: Dict[str, List[str]],
                                     results: Dict[str, Any]) -> Dict[str, Figure]:
        """Create model-specific diagnostic plots.
        
        Args:
            data: Input DataFrame
            variables: Dictionary with variable assignments
            results: Analysis results
            
        Returns:
            Dictionary of plots
        """
        plots = {}
        
        # Check if we have residuals in results
        if 'residuals' not in results:
            return plots
        
        residuals = results['residuals']
        fitted = results.get('fitted', range(len(residuals)))
        
        # Create model diagnostic plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Diagnostics', fontsize=14, fontweight='bold')
        
        try:
            # Residuals vs fitted
            ax1.scatter(fitted, residuals, alpha=0.6)
            ax1.axhline(y=0, color='red', linestyle='--')
            ax1.set_xlabel('Fitted Values')
            ax1.set_ylabel('Residuals')
            ax1.set_title('Residuals vs Fitted')
            ax1.grid(True, alpha=0.3)
            
            # Q-Q plot of residuals
            stats.probplot(residuals, dist="norm", plot=ax2)
            ax2.set_title('Q-Q Plot of Residuals')
            ax2.grid(True, alpha=0.3)
            
            # Scale-location plot
            sqrt_abs_residuals = np.sqrt(np.abs(residuals))
            ax3.scatter(fitted, sqrt_abs_residuals, alpha=0.6)
            ax3.set_xlabel('Fitted Values')
            ax3.set_ylabel('√|Residuals|')
            ax3.set_title('Scale-Location Plot')
            ax3.grid(True, alpha=0.3)
            
            # Residuals vs leverage (if available)
            if 'leverage' in results:
                leverage = results['leverage']
                ax4.scatter(leverage, residuals, alpha=0.6)
                ax4.axhline(y=0, color='red', linestyle='--')
                ax4.set_xlabel('Leverage')
                ax4.set_ylabel('Residuals')
                ax4.set_title('Residuals vs Leverage')
                ax4.grid(True, alpha=0.3)
            else:
                # Histogram of residuals
                ax4.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
                ax4.set_xlabel('Residuals')
                ax4.set_ylabel('Frequency')
                ax4.set_title('Distribution of Residuals')
                ax4.grid(True, alpha=0.3)
            
        except Exception as e:
            self.logger.error(f"Error creating model diagnostic plots: {str(e)}")
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        plots['model_diagnostics'] = fig
        
        return plots