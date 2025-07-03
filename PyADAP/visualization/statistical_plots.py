"""Statistical plots for PyADAP 3.0

This module provides specialized plotting functions for statistical analysis results,
including plots for different types of statistical tests and their interpretations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from matplotlib.figure import Figure
import scipy.stats as stats

from ..utils import get_logger, format_number, format_p_value
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

        # Set global plot style and font
        sns.set_theme(style="whitegrid", font="Arial")
        plt.rcParams['font.family'] = 'Arial'
        # Ensure Arial is available and used, especially for sans-serif
        plt.rcParams['font.sans-serif'] = ['Arial'] 
        # Ensure all text is rendered in English by default if not specified otherwise
        # Matplotlib typically defaults to English, but this reinforces it.
        plt.rcParams['axes.titlesize'] = 'large' # Example: Adjust title size
        plt.rcParams['axes.labelsize'] = 'medium' # Example: Adjust label size
        plt.rcParams['xtick.labelsize'] = 'small'
        plt.rcParams['ytick.labelsize'] = 'small'
        plt.rcParams['legend.fontsize'] = 'medium'
        plt.rcParams['figure.titlesize'] = 'x-large'

    def create_analysis_plots(
        self,
        data: pd.DataFrame,
        variables: Dict[str, List[str]],
        results: Dict[str, Any],
    ) -> Dict[str, Figure]:
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
            if "statistical_tests" in results:
                for test_name, test_result in results["statistical_tests"].items():
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

    def _create_test_specific_plots(
        self,
        data: pd.DataFrame,
        variables: Dict[str, List[str]],
        test_name: str,
        test_result: Dict[str, Any],
    ) -> Dict[str, Figure]:
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
            if "ttest" in test_name.lower():
                plots.update(
                    self._create_ttest_plots(data, variables, test_name, test_result)
                )
            elif "anova" in test_name.lower() or "f_test" in test_name.lower():
                plots.update(
                    self._create_anova_plots(data, variables, test_name, test_result)
                )
            elif "mann" in test_name.lower() or "wilcoxon" in test_name.lower():
                plots.update(
                    self._create_nonparametric_plots(
                        data, variables, test_name, test_result
                    )
                )
            elif (
                "correlation" in test_name.lower()
                or "pearson" in test_name.lower()
                or "spearman" in test_name.lower()
            ):
                plots.update(
                    self._create_correlation_plots(
                        data, variables, test_name, test_result
                    )
                )
            elif "chi" in test_name.lower():
                plots.update(
                    self._create_chi_square_plots(
                        data, variables, test_name, test_result
                    )
                )

        except Exception as e:
            self.logger.warning(f"Error creating plots for {test_name}: {str(e)}")

        return plots

    def _create_ttest_plots(
        self,
        data: pd.DataFrame,
        variables: Dict[str, List[str]],
        test_name: str,
        test_result: Dict[str, Any],
    ) -> Dict[str, Figure]:
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

        dependent_vars = variables.get("dependent", [])
        independent_vars = variables.get("independent", [])

        if not dependent_vars or not independent_vars:
            self.logger.warning("Dependent or independent variables not provided for t-test plot.")
            return plots

        dep_var = dependent_vars[0]
        indep_var = independent_vars[0]

        if dep_var not in data.columns or indep_var not in data.columns:
            self.logger.warning(f"Variables '{dep_var}' or '{indep_var}' not found in data for t-test plot.")
            return plots

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7)) # Increased figure size
        fig.suptitle(
            f"T-Test Analysis: {dep_var} by {indep_var}", 
            #fontsize=16, # Controlled by rcParams
            fontweight="bold"
        )

        # Box plot comparison
        if (
            data[indep_var].dtype in ["object", "category"]
            or data[indep_var].nunique() <= 10
        ):
            # Use Greens palette and ensure solid fill with consistent outlier style
            sns.boxplot(data=data, x=indep_var, y=dep_var, ax=ax1, palette="Greens",
                       flierprops=dict(marker='o', markerfacecolor='darkgreen', markeredgecolor='darkgreen'))
            ax1.set_title("Group Comparison (Box Plot)")
            ax1.set_xlabel(indep_var)
            ax1.set_ylabel(dep_var)
            ax1.tick_params(axis="x", rotation=30, ha="right") # Improved rotation
        else:
            # For continuous independent variable, create scatter plot with regression line
            sns.regplot(data=data, x=indep_var, y=dep_var, ax=ax1, scatter_kws={'alpha':0.6}, line_kws={'color':'red'}) # Added regression line
            ax1.set_xlabel(indep_var)
            ax1.set_ylabel(dep_var)
            ax1.set_title(f"Scatter Plot: {dep_var} vs {indep_var}")

        # Distribution comparison
        if (
            data[indep_var].dtype in ["object", "category"]
            or data[indep_var].nunique() <= 10
        ):
            groups = data.groupby(indep_var)[dep_var]
            for name, group in groups:
                if len(group.dropna()) > 0:
                    sns.histplot(
                        group.dropna(), 
                        label=f"{name} (n={len(group.dropna())})", 
                        ax=ax2, 
                        kde=True, # Added KDE
                        bins=20,
                        alpha=0.7
                    )
            ax2.set_xlabel(dep_var)
            ax2.set_ylabel("Density") # Changed to Density due to KDE
            ax2.set_title("Distribution Comparison with Density")
            ax2.legend(title=indep_var)
        else:
            # For continuous variables, show distribution of dependent variable with KDE
            sns.histplot(data[dep_var].dropna(), ax=ax2, bins=30, kde=True, alpha=0.7)
            ax2.set_xlabel(dep_var)
            ax2.set_ylabel("Density")
            ax2.set_title(f"Distribution of {dep_var}")

        # Add test statistics
        effect_size_name = "Cohen's d" if "cohen" in test_result.get('effect_size_type', '').lower() else "Effect Size"
        stats_text = f"""T-Test Statistics:
        Statistic (t): {format_number(test_result.get('statistic', 'N/A'))}
        P-value: {format_p_value(test_result.get('p_value', 'N/A'))}
        Degrees of Freedom: {test_result.get('dof', 'N/A')}
        {effect_size_name}: {format_number(test_result.get('effect_size', 'N/A'))}
        Confidence Interval: {test_result.get('confidence_interval', 'N/A')} """

        fig.text(
            0.5, # Centered horizontally
            0.01, # Lower position
            stats_text,
            ha="center", # Horizontal alignment
            va="bottom", # Vertical alignment
            #fontsize=10, # Controlled by rcParams
            bbox=dict(boxstyle="round,pad=0.5", facecolor="aliceblue", alpha=0.9), # Adjusted style
        )

        plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout to make space for text
        plots[f"{test_name}_comparison_plot"] = fig # More descriptive name

        self.logger.info(f"Created t-test plot for {dep_var} by {indep_var}")
        return plots

    def _create_anova_plots(
        self,
        data: pd.DataFrame,
        variables: Dict[str, List[str]],
        test_name: str,
        test_result: Dict[str, Any],
    ) -> Dict[str, Figure]:
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

        dependent_vars = variables.get("dependent", [])
        # For ANOVA, independent_vars can be one (One-Way) or two (Two-Way for interaction)
        independent_vars = variables.get("independent", []) 
        between_subjects = variables.get("between_subjects", []) # For mixed ANOVA or more complex designs

        if not dependent_vars or not (independent_vars or between_subjects):
            self.logger.warning("Dependent or independent/between_subjects variables not provided for ANOVA plot.")
            return plots

        dep_var = dependent_vars[0]
        
        # Determine the main factors for plotting
        # If 'independent_vars' is a list of lists (e.g. for interaction term from formula)
        # flatten it or take the first one for simplicity in some plots.
        # For interaction plots, we'll handle multiple factors specifically.
        if independent_vars:
            if isinstance(independent_vars[0], list): # handles [['factor1', 'factor2']]
                 main_plot_indep_var = independent_vars[0][0] if independent_vars[0] else None
            else: # handles ['factor1'] or ['factor1', 'factor2']
                 main_plot_indep_var = independent_vars[0]
        elif between_subjects: # Fallback to between_subjects if independent_vars is empty
            main_plot_indep_var = between_subjects[0]
        else:
            self.logger.warning("No suitable independent/between_subjects variable found for ANOVA plot titles.")
            return plots

        if dep_var not in data.columns or (main_plot_indep_var and main_plot_indep_var not in data.columns):
            self.logger.warning(f"Variables '{dep_var}' or '{main_plot_indep_var}' not found in data for ANOVA plot.")
            return plots

        fig_title = f"ANOVA Analysis: {dep_var}"
        if main_plot_indep_var:
            fig_title += f" by {main_plot_indep_var}"
        if len(independent_vars) > 1 or (independent_vars and len(independent_vars[0]) > 1 if isinstance(independent_vars[0], list) else False) :
             fig_title += " (and other factors)"

        # Create ANOVA visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14)) # Increased size
        fig.suptitle(fig_title, fontweight="bold")

        # Box plot
        if main_plot_indep_var:
            sns.boxplot(data=data, x=main_plot_indep_var, y=dep_var, ax=ax1, palette="Greens",
                       flierprops=dict(marker='o', markerfacecolor='darkgreen', markeredgecolor='darkgreen'))
            ax1.set_title(f"Group Comparison (Box Plot): {main_plot_indep_var}")
            ax1.set_xlabel(main_plot_indep_var)
            ax1.set_ylabel(dep_var)
            ax1.tick_params("x", rotation=30, ha="right")
        else:
            ax1.text(0.5, 0.5, "Main factor for boxplot not identified", ha="center", va="center", transform=ax1.transAxes)

        # Violin plot
        if main_plot_indep_var:
            sns.violinplot(data=data, x=main_plot_indep_var, y=dep_var, ax=ax2, palette="Greens", inner="quartile") # Changed palette
            ax2.set_title(f"Distribution Shape: {main_plot_indep_var}")
            ax2.set_xlabel(main_plot_indep_var)
            ax2.set_ylabel(dep_var)
            ax2.tick_params("x", rotation=30, ha="right")
        else:
            ax2.text(0.5, 0.5, "Main factor for violinplot not identified", ha="center", va="center", transform=ax2.transAxes)

        # Mean plot with error bars (confidence intervals if available, else SE)
        if main_plot_indep_var:
            try:
                group_stats = data.groupby(main_plot_indep_var)[dep_var].agg(['mean', 'std', 'count']).reset_index()
                # Calculate 95% CI for the mean: mean +/- t_critical * (std / sqrt(count))
                # For simplicity, using 1.96 for large samples, or calculate t_critical if scipy is available
                t_critical = stats.t.ppf(0.975, group_stats['count'] - 1) 
                group_stats['ci_margin'] = t_critical * (group_stats['std'] / np.sqrt(group_stats['count']))

                ax3.errorbar(
                    x=group_stats[main_plot_indep_var], # Use actual group names if categorical
                    y=group_stats['mean'],
                    yerr=group_stats['ci_margin'],
                    fmt='o-', # More distinct marker and line
                    capsize=5,
                    capthick=2,
                    elinewidth=2, # Thicker error bars
                    color='dodgerblue' # Professional color
                )
                ax3.set_xticks(range(len(group_stats)) if data[main_plot_indep_var].nunique() > 10 else group_stats[main_plot_indep_var])
                ax3.set_xticklabels(group_stats[main_plot_indep_var], rotation=30, ha="right")
                ax3.set_ylabel(f"Mean {dep_var} (with 95% CI)")
                ax3.set_xlabel(main_plot_indep_var)
                ax3.set_title(f"Group Means: {main_plot_indep_var}")
                ax3.grid(True, linestyle='--', alpha=0.7)
            except Exception as e:
                self.logger.warning(f"Could not create mean plot for ANOVA: {e}")
                ax3.text(0.5, 0.5, "Error creating mean plot", ha="center", va="center", transform=ax3.transAxes)
        else:
            ax3.text(0.5, 0.5, "Main factor for mean plot not identified", ha="center", va="center", transform=ax3.transAxes)

        # Residuals vs Fitted plot (if available)
        if "residuals" in test_result and "fitted_values" in test_result:
            residuals = test_result["residuals"]
            fitted = test_result["fitted_values"]
            if residuals is not None and fitted is not None and len(residuals) == len(fitted):
                sns.scatterplot(x=fitted, y=residuals, ax=ax4, alpha=0.6, color='firebrick')
                ax4.axhline(y=0, color="black", linestyle="--", linewidth=1)
                ax4.set_xlabel("Fitted Values")
                ax4.set_ylabel("Residuals")
                ax4.set_title("Residuals vs. Fitted Values")
                ax4.grid(True, linestyle='--', alpha=0.5)
            else:
                ax4.text(0.5, 0.5, "Residuals or fitted values data incomplete", ha="center", va="center", transform=ax4.transAxes)
        else:
            # Fallback: Q-Q plot of residuals if only residuals are available
            if "residuals" in test_result and test_result["residuals"] is not None:
                stats.probplot(test_result["residuals"], dist="norm", plot=ax4)
                ax4.get_lines()[0].set_markerfacecolor('cornflowerblue') # Customize points
                ax4.get_lines()[0].set_markeredgecolor('cornflowerblue')
                ax4.get_lines()[1].set_color('red') # Customize line
                ax4.set_title("Q-Q Plot of Residuals")
                ax4.set_xlabel("Theoretical Quantiles")
                ax4.set_ylabel("Sample Quantiles")
            else:
                ax4.text(0.5, 0.5, "Residuals data not available for plot", ha="center", va="center", transform=ax4.transAxes)

        # Add test statistics
        # Assuming test_result['anova_table'] is a DataFrame from statsmodels or similar
        anova_summary_str = "ANOVA Table not available or in unexpected format."
        if 'anova_table' in test_result and isinstance(test_result['anova_table'], pd.DataFrame):
            try:
                # Format the anova table for display
                anova_df = test_result['anova_table'].copy()
                # Round numeric columns to reasonable precision
                for col in anova_df.select_dtypes(include=np.number).columns:
                    if 'p_value' in col.lower() or 'pr(>f)' in col.lower():
                        anova_df[col] = anova_df[col].apply(format_p_value)
                    else:
                        anova_df[col] = anova_df[col].apply(lambda x: format_number(x, precision=3))
                anova_summary_str = "ANOVA Summary:\n" + anova_df.to_string(index=True)
            except Exception as e:
                self.logger.warning(f"Error formatting ANOVA table for display: {e}")
                anova_summary_str = f"ANOVA F-statistic: {format_number(test_result.get('statistic', 'N/A'))}\n"
                anova_summary_str += f"P-value: {format_p_value(test_result.get('p_value', 'N/A'))}\n"
                anova_summary_str += f"Effect Size (η²p): {format_number(test_result.get('effect_size_eta_sq_partial', 'N/A'))}"
        else:
            stats_text_parts = []
            if 'statistic' in test_result: stats_text_parts.append(f"F-statistic: {format_number(test_result.get('statistic'))}")
            if 'p_value' in test_result: stats_text_parts.append(f"P-value: {format_p_value(test_result.get('p_value'))}")
            if 'df' in test_result: stats_text_parts.append(f"DF: {test_result.get('df')}") # (DF_num, DF_den)
            if 'effect_size_eta_sq_partial' in test_result: 
                stats_text_parts.append(f"Partial η²: {format_number(test_result.get('effect_size_eta_sq_partial'))}")
            elif 'effect_size' in test_result: # Generic effect size
                stats_text_parts.append(f"Effect Size (η²): {format_number(test_result.get('effect_size'))}")
            anova_summary_str = "ANOVA Main Results:\n" + "\n".join(stats_text_parts)

        fig.text(
            0.5, 
            0.01, 
            anova_summary_str,
            ha="center", 
            va="bottom", 
            #fontsize=9, # Smaller for potentially long table, or adjust dynamically
            linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="ghostwhite", alpha=0.95, edgecolor='lightgray'),
            fontfamily='monospace' # Better for table-like text
        )

        plt.tight_layout(rect=[0, 0.10, 1, 0.95]) # Adjust layout for stats text
        plots[f"{test_name}_main_effects_plot"] = fig
        self.logger.info(f"Created ANOVA main effects plot for {dep_var}.")

        # Check for interaction effects and plot if present
        # This requires at least two independent variables
        interaction_factors = []
        if independent_vars and isinstance(independent_vars[0], list) and len(independent_vars[0]) >=2:
            interaction_factors = independent_vars[0] # e.g. ['factorA', 'factorB']
        elif independent_vars and len(independent_vars) >= 2 and all(isinstance(f, str) for f in independent_vars):
            interaction_factors = independent_vars[:2] # Take first two if multiple simple factors
        
        # Check if an interaction term was significant in the anova_table
        significant_interaction = False
        if 'anova_table' in test_result and isinstance(test_result['anova_table'], pd.DataFrame) and interaction_factors:
            interaction_term_name = ":".join(interaction_factors) # e.g., 'factorA:factorB'
            if interaction_term_name in test_result['anova_table'].index:
                p_val_col = next((col for col in ['PR(>F)', 'p-value', 'P>F'] if col in test_result['anova_table'].columns), None)
                if p_val_col and test_result['anova_table'].loc[interaction_term_name, p_val_col] < 0.05:
                    significant_interaction = True
            # Also check for reversed order, e.g. 'factorB:factorA'
            interaction_term_name_reversed = ":".join(interaction_factors[::-1])
            if not significant_interaction and interaction_term_name_reversed in test_result['anova_table'].index:
                 if p_val_col and test_result['anova_table'].loc[interaction_term_name_reversed, p_val_col] < 0.05:
                    significant_interaction = True

        if interaction_factors and significant_interaction:
            factor1, factor2 = interaction_factors[0], interaction_factors[1]
            if factor1 in data.columns and factor2 in data.columns:
                try:
                    interaction_fig, ax_int = plt.subplots(figsize=(10, 7))
                    # Ensure factor1 and factor2 are treated as categorical for the interaction plot
                    data_copy = data.copy()
                    data_copy[factor1] = data_copy[factor1].astype('category')
                    data_copy[factor2] = data_copy[factor2].astype('category')

                    sns.pointplot(data=data_copy, x=factor1, y=dep_var, hue=factor2, ax=ax_int, 
                                  dodge=True, errorbar=('ci', 95), palette='viridis', capsize=.1)
                    ax_int.set_title(f"Interaction Plot: {dep_var} by {factor1} and {factor2}", fontweight="bold")
                    ax_int.set_xlabel(factor1)
                    ax_int.set_ylabel(f"Mean {dep_var}")
                    ax_int.legend(title=factor2)
                    ax_int.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    plots[f"{test_name}_interaction_{factor1}_{factor2}"] = interaction_fig
                    self.logger.info(f"Created interaction plot for {factor1} and {factor2} on {dep_var}.")
                except Exception as e:
                    self.logger.warning(f"Could not create interaction plot for {factor1} and {factor2}: {e}")
            else:
                self.logger.warning(f"Interaction factors {factor1} or {factor2} not in data columns for interaction plot.")
        elif interaction_factors and not significant_interaction:
            self.logger.info(f"Interaction between {interaction_factors} was not statistically significant (p >= 0.05), skipping interaction plot.")

        return plots

    def _create_nonparametric_plots(
        self,
        data: pd.DataFrame,
        variables: Dict[str, List[str]],
        test_name: str,
        test_result: Dict[str, Any],
    ) -> Dict[str, Figure]:
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

        dependent_vars = variables.get("dependent", [])
        independent_vars = variables.get("independent", [])

        if not dependent_vars or not independent_vars:
            self.logger.warning("Dependent or independent variables not provided for non-parametric plot.")
            return plots

        dep_var = dependent_vars[0]
        indep_var = independent_vars[0]

        if dep_var not in data.columns or indep_var not in data.columns:
            self.logger.warning(f"Variables '{dep_var}' or '{indep_var}' not found in data for non-parametric plot.")
            return plots

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7)) # Increased figure size
        fig.suptitle(
            f"Non-Parametric Test Analysis: {dep_var} by {indep_var}",
            #fontsize=16, # Controlled by rcParams
            fontweight="bold",
        )

        # Box plot with individual points (strip plot or swarm plot for better visualization)
        if (
            data[indep_var].dtype in ["object", "category"]
            or data[indep_var].nunique() <= 10
        ):
            sns.boxplot(data=data, x=indep_var, y=dep_var, ax=ax1, palette="Greens", showfliers=False) # Hide outliers for clarity with stripplot
            sns.stripplot(
                data=data,
                x=indep_var,
                y=dep_var,
                ax=ax1,
                color="darkgreen", # Changed to match Greens theme
                alpha=0.6,
                size=4, # Slightly larger points
                jitter=True # Add jitter to avoid overlap
            )
            ax1.set_title("Group Comparison (Box Plot with Data Points)")
            ax1.set_xlabel(indep_var)
            ax1.set_ylabel(dep_var)
            ax1.tick_params(axis="x", rotation=30, ha="right")
        else:
            # Scatter plot for continuous independent variable
            sns.scatterplot(data=data, x=indep_var, y=dep_var, ax=ax1, alpha=0.7, color='teal')
            ax1.set_xlabel(indep_var)
            ax1.set_ylabel(dep_var)
            ax1.set_title(f"Scatter Plot: {dep_var} vs {indep_var}")

        # Rank comparison or distribution of ranks
        if (
            data[indep_var].dtype in ["object", "category"]
            or data[indep_var].nunique() <= 10
        ):
            data_copy = data.copy()
            data_copy[f"{dep_var}_rank"] = data_copy[dep_var].rank(method='average') # Use average for ties

            sns.boxplot(data=data_copy, x=indep_var, y=f"{dep_var}_rank", ax=ax2, palette="Greens")
            ax2.set_title(f"Rank Comparison of {dep_var} by {indep_var}")
            ax2.set_xlabel(indep_var)
            ax2.set_ylabel(f"Rank of {dep_var}")
            ax2.tick_params(axis="x", rotation=30, ha="right")
        else:
            # Distribution of dependent variable (as ranks might not be meaningful for continuous indep_var here)
            sns.histplot(data[dep_var].dropna(), ax=ax2, bins=30, kde=True, alpha=0.7, color='mediumseagreen')
            ax2.set_xlabel(dep_var)
            ax2.set_ylabel("Density")
            ax2.set_title(f"Distribution of {dep_var}")

        # Add test statistics
        # Determine effect size name (e.g., Rank-Biserial Correlation, Cliff's Delta, etc.)
        effect_size_name = test_result.get('effect_size_type', 'Effect Size')
        if not effect_size_name or effect_size_name == 'Effect Size': # Default if not specified
            if 'mannwhitneyu' in test_name.lower() or 'wilcoxon' in test_name.lower() and 'rank' in test_name.lower():
                effect_size_name = "Rank-Biserial Correlation"
            elif 'kruskal' in test_name.lower():
                effect_size_name = "Epsilon-squared (ε²)"
        
        stats_text = f"""Non-Parametric Test Statistics ({test_name.replace('_', ' ').title()}):
        Statistic: {format_number(test_result.get('statistic', 'N/A'))}
        P-value: {format_p_value(test_result.get('p_value', 'N/A'))}
        {effect_size_name}: {format_number(test_result.get('effect_size', 'N/A'))}"""
        if 'df' in test_result: # For Kruskal-Wallis
            stats_text += f"\n        Degrees of Freedom: {test_result.get('df', 'N/A')}"

        fig.text(
            0.5, 
            0.01, 
            stats_text,
            ha="center", 
            va="bottom", 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9, edgecolor='darkkhaki'),
        )

        plt.tight_layout(rect=[0, 0.08, 1, 0.95]) # Adjust layout for stats text
        plots[f"{test_name}_comparison_plot"] = fig # More descriptive name
        self.logger.info(f"Created non-parametric plot for {dep_var} by {indep_var}.")

        return plots

    def _create_correlation_plots(
        self,
        data: pd.DataFrame,
        variables: Dict[str, List[str]],
        test_name: str,
        test_result: Dict[str, Any],
    ) -> Dict[str, Figure]:
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

        # Correlation typically involves numeric variables.
        # 'dependent' and 'independent' might be less relevant here, consider all numeric vars.
        all_vars_from_spec = variables.get("dependent", []) + variables.get("independent", [])
        
        numeric_vars = [
            var
            for var in all_vars_from_spec # Use variables specified for the test
            if var in data.columns and pd.api.types.is_numeric_dtype(data[var])
        ]

        if len(numeric_vars) < 2:
            self.logger.warning(
                f"Correlation plot for '{test_name}' requires at least two numeric variables. Found: {numeric_vars}"
            )
            return plots

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14)) # Increased size
        fig.suptitle(f"Correlation Analysis: {test_name.replace('_', ' ').title()}", fontweight="bold")

        var1, var2 = numeric_vars[0], numeric_vars[1] # For pairwise plots, use the first two

        # Scatter plot with regression line using seaborn for better aesthetics
        try:
            sns.regplot(data=data, x=var1, y=var2, ax=ax1, 
                        scatter_kws={'alpha':0.6, 'color': 'seagreen'}, # Changed scatter color
                        line_kws={'color':'darkgreen', 'linestyle':'--'}) # Changed line color
            ax1.set_title(f"Scatter Plot with Regression: {var1} vs {var2}")
            ax1.set_xlabel(var1)
            ax1.set_ylabel(var2)
            ax1.grid(True, linestyle='--', alpha=0.7)
        except Exception as e:
            self.logger.warning(f"Could not create scatter/regression plot for {var1} vs {var2}: {e}")
            ax1.text(0.5, 0.5, "Error creating scatter plot", ha="center", va="center", transform=ax1.transAxes)

        # Residuals plot (associated with the regression on ax1)
        try:
            # Calculate residuals manually if not directly available from sns.regplot
            # This requires fitting a model, e.g. OLS from statsmodels, or simple linregress
            slope, intercept, r_val, p_val, std_err = stats.linregress(data[var1].dropna(), data[var2].dropna())
            predicted_values = slope * data[var1] + intercept
            residuals = data[var2] - predicted_values
            
            sns.scatterplot(x=predicted_values, y=residuals, ax=ax2, alpha=0.6, color='mediumseagreen') # Changed color
            ax2.axhline(y=0, color="black", linestyle="--", linewidth=1)
            ax2.set_xlabel(f"Predicted {var2}")
            ax2.set_ylabel("Residuals")
            ax2.set_title(f"Residuals vs. Predicted Plot ({var1} & {var2})")
            ax2.grid(True, linestyle='--', alpha=0.5)
        except Exception as e:
            self.logger.warning(f"Could not create residuals plot for {var1} vs {var2}: {e}")
            ax2.text(0.5, 0.5, "Error creating residuals plot", ha="center", va="center", transform=ax2.transAxes)

        # Correlation matrix heatmap (for all specified numeric variables)
        if len(numeric_vars) >= 2: # Ensure there's enough for a matrix
            try:
                corr_matrix = data[numeric_vars].corr(method=test_name.split('_')[0] if 'pearson' in test_name or 'spearman' in test_name else 'pearson') # Infer method
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Mask for upper triangle
                sns.heatmap(
                    corr_matrix,
                    annot=True,
                    cmap="Greens", # Changed to Greens palette
                    center=0,
                    square=True,
                    ax=ax3,
                    fmt=".2f", # Consistent formatting
                    linewidths=.5, # Add lines between cells
                    cbar_kws={"shrink": .8}, # Adjust colorbar size
                    mask=mask # Apply mask
                )
                ax3.set_title(f"Correlation Matrix ({test_name.split('_')[0].title()})")
                ax3.tick_params(axis='x', rotation=30, ha='right')
                ax3.tick_params(axis='y', rotation=0)
            except Exception as e:
                self.logger.warning(f"Could not create correlation matrix heatmap: {e}")
                ax3.text(0.5, 0.5, "Error creating heatmap", ha="center", va="center", transform=ax3.transAxes)
        else:
            ax3.text(0.5, 0.5, "Not enough numeric variables for correlation matrix", ha="center", va="center", transform=ax3.transAxes)

        # Joint distribution plot (e.g., KDE plot or hexbin for density)
        if len(numeric_vars) >= 2:
            try:
                sns.kdeplot(data=data, x=var1, y=var2, ax=ax4, cmap="Greens", fill=True, thresh=0.05) # Changed to Greens cmap
                # sns.hist2d(data=data, x=var1, y=var2, ax=ax4, bins=30, cmap='Greens') # Alternative: 2D Histogram with Greens
                ax4.set_title(f"Joint Distribution Plot: {var1} & {var2}")
                ax4.set_xlabel(var1)
                ax4.set_ylabel(var2)
                ax4.grid(True, linestyle='--', alpha=0.5)
            except Exception as e:
                self.logger.warning(f"Could not create joint distribution plot for {var1} vs {var2}: {e}")
                ax4.text(0.5, 0.5, "Error creating joint distribution plot", ha="center", va="center", transform=ax4.transAxes)
        else:
             ax4.text(0.5, 0.5, "Not enough variables for joint plot", ha="center", va="center", transform=ax4.transAxes)

        # Add correlation statistics from the test result
        correlation_coefficient = test_result.get('statistic', test_result.get('correlation_coefficient', 'N/A'))
        p_value_corr = test_result.get('p_value', 'N/A')
        df_corr = test_result.get('df', 'N/A') # Degrees of freedom if available

        stats_text = f"""{test_name.replace('_', ' ').title()} Statistics:
        Correlation ({'r' if 'pearson' in test_name else 'ρ' if 'spearman' in test_name else 'coeff.'}): {format_number(correlation_coefficient)}
        P-value: {format_p_value(p_value_corr)}"""
        if df_corr != 'N/A': 
            stats_text += f"\n        Degrees of Freedom: {df_corr}"
        if 'confidence_interval' in test_result:
             stats_text += f"\n        95% CI: {test_result.get('confidence_interval', 'N/A')}"

        fig.text(
            0.5, 
            0.01, 
            stats_text,
            ha="center", 
            va="bottom", 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="honeydew", alpha=0.95, edgecolor='darkseagreen'),
        )

        plt.tight_layout(rect=[0, 0.08, 1, 0.95]) # Adjust layout for stats text
        plots[f"{test_name}_analysis_plot"] = fig # More descriptive name
        self.logger.info(f"Created correlation analysis plot for '{test_name}'.")

        return plots

    def _create_chi_square_plots(
        self,
        data: pd.DataFrame,
        variables: Dict[str, List[str]],
        test_name: str,
        test_result: Dict[str, Any],
    ) -> Dict[str, Figure]:
        """Create plots for Chi-square test results.

        Args:
            data: Input DataFrame
            variables: Dictionary with variable assignments
            test_name: Name of the test
            test_result: Test results

        Returns:
            Dictionary of plots
        """
        plots = {}
        dependent_vars = variables.get("dependent", [])
        independent_vars = variables.get("independent", [])

        if not dependent_vars or not independent_vars:
            self.logger.warning(
                f"Chi-square plots for '{test_name}' require dependent and independent variables."
            )
            return plots

        var1 = independent_vars[0]
        var2 = dependent_vars[0]

        # Ensure variables are categorical or can be treated as such
        try:
            data[var1] = data[var1].astype('category')
            data[var2] = data[var2].astype('category')
        except KeyError as e:
            self.logger.error(f"Variable not found for Chi-square plot: {e}")
            return plots
        except Exception as e:
            self.logger.warning(f"Could not convert variables to category for Chi-square plot: {e}")
            # Continue if conversion fails, but plots might not be ideal

        # Create contingency table
        contingency_table = pd.crosstab(data[var1], data[var2])
        expected_freq = test_result.get("expected_freq", None)
        if expected_freq is not None and not isinstance(expected_freq, pd.DataFrame):
            try:
                expected_freq = pd.DataFrame(expected_freq, index=contingency_table.index, columns=contingency_table.columns)
            except Exception as e:
                self.logger.warning(f"Could not convert expected_freq to DataFrame: {e}")
                expected_freq = None

        fig, axes = plt.subplots(2, 2, figsize=(18, 16)) # Increased size
        fig.suptitle(f"Chi-square Test Analysis: {var1} vs {var2}", fontweight="bold")

        # 1. Contingency Table Heatmap (Observed Frequencies)
        try:
            sns.heatmap(contingency_table, annot=True, fmt="d", cmap="Greens", ax=axes[0, 0], cbar=True, linewidths=.5)
            axes[0, 0].set_title(f"Observed Frequencies: {var1} by {var2}")
            axes[0, 0].set_xlabel(var2)
            axes[0, 0].set_ylabel(var1)
            axes[0,0].tick_params(axis='x', rotation=30, ha='right')
            axes[0,0].tick_params(axis='y', rotation=0)
        except Exception as e:
            self.logger.warning(f"Could not create observed frequencies heatmap: {e}")
            axes[0,0].text(0.5, 0.5, "Error: Observed Freq. Heatmap", ha="center", va="center", transform=axes[0,0].transAxes)

        # 2. Expected Frequencies Heatmap
        if expected_freq is not None:
            try:
                sns.heatmap(expected_freq, annot=True, fmt=".2f", cmap="Greens", ax=axes[0, 1], cbar=True, linewidths=.5)
                axes[0, 1].set_title(f"Expected Frequencies: {var1} by {var2}")
                axes[0, 1].set_xlabel(var2)
                axes[0, 1].set_ylabel(var1)
                axes[0,1].tick_params(axis='x', rotation=30, ha='right')
                axes[0,1].tick_params(axis='y', rotation=0)
            except Exception as e:
                self.logger.warning(f"Could not create expected frequencies heatmap: {e}")
                axes[0,1].text(0.5, 0.5, "Error: Expected Freq. Heatmap", ha="center", va="center", transform=axes[0,1].transAxes)
        else:
            axes[0, 1].text(0.5, 0.5, "Expected frequencies not available", ha="center", va="center", transform=axes[0,1].transAxes)
            axes[0, 1].set_title("Expected Frequencies")

        # 3. Proportional Stacked Bar Chart
        try:
            # Calculate percentages for stacking
            percentage_table = contingency_table.apply(lambda x: x / x.sum() * 100, axis=1)
            percentage_table.plot(kind='bar', stacked=True, ax=axes[1, 0], colormap='Greens', alpha=0.85)
            axes[1, 0].set_title(f"Proportional Stacked Bar Chart: {var2} by {var1}")
            axes[1, 0].set_xlabel(var1)
            axes[1, 0].set_ylabel("Percentage (%)")
            axes[1, 0].legend(title=var2, bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1,0].tick_params(axis='x', rotation=30, ha='right')
            axes[1,0].grid(axis='y', linestyle='--', alpha=0.7)
        except Exception as e:
            self.logger.warning(f"Could not create percentage stacked bar chart: {e}")
            axes[1,0].text(0.5, 0.5, "Error: Stacked Bar Chart", ha="center", va="center", transform=axes[1,0].transAxes)

        # 4. Standardized Residuals Heatmap or Grouped Bar Chart
        residuals = test_result.get("residuals", None)
        if residuals is not None and expected_freq is not None:
            try:
                # Standardized residuals: (Observed - Expected) / sqrt(Expected)
                standardized_residuals = (contingency_table - expected_freq) / np.sqrt(expected_freq)
                sns.heatmap(standardized_residuals, annot=True, fmt=".2f", cmap="Greens", center=0, ax=axes[1, 1], cbar=True, linewidths=.5)
                axes[1, 1].set_title(f"Standardized Residuals: {var1} by {var2}")
                axes[1, 1].set_xlabel(var2)
                axes[1, 1].set_ylabel(var1)
                axes[1,1].tick_params(axis='x', rotation=30, ha='right')
                axes[1,1].tick_params(axis='y', rotation=0)
            except Exception as e:
                self.logger.warning(f"Could not create standardized residuals heatmap: {e}")
                axes[1,1].text(0.5, 0.5, "Error: Residuals Heatmap", ha="center", va="center", transform=axes[1,1].transAxes)
        else:
            # Fallback: Grouped bar chart of observed frequencies if residuals are not available
            try:
                contingency_table.plot(kind='bar', ax=axes[1, 1], colormap='Greens', alpha=0.8)
                axes[1, 1].set_title(f"Grouped Bar Chart (Observed): {var2} by {var1}")
                axes[1, 1].set_xlabel(var1)
                axes[1, 1].set_ylabel("Frequency")
                axes[1, 1].legend(title=var2, bbox_to_anchor=(1.05, 1), loc='upper left')
                axes[1,1].tick_params(axis='x', rotation=30, ha='right')
                axes[1,1].grid(axis='y', linestyle='--', alpha=0.7)
            except Exception as e:
                self.logger.warning(f"Could not create fallback grouped bar chart: {e}")
                axes[1,1].text(0.5, 0.5, "Residuals not available / Error in Fallback", ha="center", va="center", transform=axes[1,1].transAxes)

        # Add Chi-square statistics
        chi2_statistic = test_result.get("statistic", "N/A")
        p_value_chi2 = test_result.get("p_value", "N/A")
        df_chi2 = test_result.get("df", "N/A")
        effect_size_name = test_result.get("effect_size_name", "Effect Size")
        effect_size_value = test_result.get("effect_size", "N/A")

        stats_text = f"""Chi-square Test Results ({var1} & {var2}):
        Chi-square (χ²): {format_number(chi2_statistic)}
        P-value: {format_p_value(p_value_chi2)}
        Degrees of Freedom (df): {df_chi2}
        {effect_size_name}: {format_number(effect_size_value)}"""

        fig.text(
            0.5, 
            0.01, 
            stats_text, 
            ha="center", 
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.95, edgecolor='khaki'),
        )

        plt.tight_layout(rect=[0, 0.08, 1, 0.95]) # Adjust layout for stats text
        plots[f"{test_name}_analysis_plot"] = fig # More descriptive name
        self.logger.info(f"Created Chi-square analysis plot for '{test_name}'.")

        return plots

    def _create_comparison_plot(
        self,
        data: pd.DataFrame,
        variables: Dict[str, List[str]],
        results: Dict[str, Any],
    ) -> Dict[str, Figure]:
        """Create general comparison plots.

        Args:
            data: Input DataFrame
            variables: Dictionary with variable assignments
            results: Analysis results

        Returns:
            Dictionary of plots
        """
        plots = {}

        dependent_vars = variables.get("dependent", [])
        independent_vars = variables.get("independent", [])

        if not dependent_vars or not independent_vars:
            return plots

        # Create comprehensive comparison plot
        try:
            dep_var = dependent_vars[0]
            indep_var = independent_vars[0]

            if dep_var not in data.columns or indep_var not in data.columns:
                return plots

            # Determine plot type based on variable types
            dep_is_numeric = data[dep_var].dtype in ["int64", "float64"]
            indep_is_numeric = data[indep_var].dtype in ["int64", "float64"]
            indep_is_categorical = (
                data[indep_var].dtype in ["object", "category"]
                or data[indep_var].nunique() <= 10
            )

            if dep_is_numeric and indep_is_categorical:
                # Numeric dependent, categorical independent
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(
                    f"Group Comparison: {dep_var} by {indep_var}",
                    fontsize=14,
                    fontweight="bold",
                )

                # Box plot
                sns.boxplot(data=data, x=indep_var, y=dep_var, ax=ax1, palette="Greens", flierprops=dict(marker='o', markerfacecolor='darkgreen', markersize=4))
                ax1.set_title("Box Plot Comparison")
                ax1.tick_params(axis="x", rotation=45)

                # Violin plot
                sns.violinplot(data=data, x=indep_var, y=dep_var, ax=ax2, palette="Greens")
                ax2.set_title("Distribution Shape Comparison")
                ax2.tick_params(axis="x", rotation=45)

                # Strip plot
                sns.stripplot(data=data, x=indep_var, y=dep_var, ax=ax3, alpha=0.6, color="seagreen")
                ax3.set_title("Individual Data Points")
                ax3.tick_params(axis="x", rotation=45)

                # Histogram by group
                groups = data.groupby(indep_var)[dep_var]
                for name, group in groups:
                    if len(group.dropna()) > 0:
                        group.hist(
                            alpha=0.6, label=f"{name} (n={len(group)})", ax=ax4, bins=20
                        )
                ax4.set_xlabel(dep_var)
                ax4.set_ylabel("Frequency")
                ax4.set_title("Distribution by Group")
                ax4.legend()

                plt.tight_layout()
                plots["group_comparison"] = fig

            elif dep_is_numeric and indep_is_numeric:
                # Both numeric - correlation/regression plot
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(
                    f"Relationship: {dep_var} vs {indep_var}",
                    fontsize=14,
                    fontweight="bold",
                )

                # Scatter plot
                ax1.scatter(data[indep_var], data[dep_var], alpha=0.6, color="seagreen")
                ax1.set_xlabel(indep_var)
                ax1.set_ylabel(dep_var)
                ax1.set_title("Scatter Plot")
                ax1.grid(True, alpha=0.3)

                # Scatter with regression line
                sns.regplot(
                    data=data,
                    x=indep_var,
                    y=dep_var,
                    ax=ax2,
                    scatter_kws={"alpha": 0.6, "color": "seagreen"},
                    line_kws={"color": "darkgreen"}
                )
                ax2.set_title("Regression Line")

                # Hexbin plot for density
                ax3.hexbin(data[indep_var], data[dep_var], gridsize=20, cmap="Greens")
                ax3.set_xlabel(indep_var)
                ax3.set_ylabel(dep_var)
                ax3.set_title("Density Plot")

                # Residuals (if regression was performed)
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        data[indep_var].dropna(), data[dep_var].dropna()
                    )
                    predicted = slope * data[indep_var] + intercept
                    residuals = data[dep_var] - predicted

                    ax4.scatter(predicted, residuals, alpha=0.6, color="seagreen")
                    ax4.axhline(y=0, color="red", linestyle="--")
                    ax4.set_xlabel("Predicted Values")
                    ax4.set_ylabel("Residuals")
                    ax4.set_title("Residuals Plot")
                    ax4.grid(True, alpha=0.3)
                except:
                    ax4.text(
                        0.5,
                        0.5,
                        "Could not create residuals plot",
                        ha="center",
                        va="center",
                        transform=ax4.transAxes,
                    )

                plt.tight_layout()
                plots["relationship_analysis"] = fig

        except Exception as e:
            self.logger.warning(f"Error creating comparison plots: {str(e)}")

        return plots

    def _create_effect_size_plots(
        self,
        data: pd.DataFrame,
        variables: Dict[str, List[str]],
        results: Dict[str, Any],
    ) -> Dict[str, Figure]:
        """Create effect size visualization plots.

        Args:
            data: Input DataFrame
            variables: Dictionary with variable assignments
            results: Analysis results

        Returns:
            Dictionary of plots
        """
        plots = {}

        if "statistical_tests" not in results:
            return plots

        # Extract effect sizes
        effect_data = []
        for test_name, test_result in results["statistical_tests"].items():
            if isinstance(test_result, dict) and "effect_size" in test_result:
                effect_size = test_result["effect_size"]
                if isinstance(effect_size, (int, float)) and not np.isnan(effect_size):
                    effect_data.append(
                        {
                            "Test": test_name,
                            "Effect Size": abs(effect_size),
                            "Direction": "Positive" if effect_size >= 0 else "Negative",
                        }
                    )

        if not effect_data:
            return plots

        # Create effect size plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle("Effect Size Analysis", fontsize=14, fontweight="bold")

        # Bar plot of effect sizes
        effect_df = pd.DataFrame(effect_data)
        # Use shades of green: darkgreen for Positive, mediumseagreen for Negative
        colors = ["darkgreen" if d == "Positive" else "mediumseagreen" for d in effect_df["Direction"]]

        bars = ax1.bar(
            range(len(effect_df)), effect_df["Effect Size"], color=colors, alpha=0.8 # Increased alpha for better visibility
        )
        ax1.set_xticks(range(len(effect_df)))
        ax1.set_xticklabels(effect_df["Test"], rotation=45, ha="right")
        ax1.set_ylabel("Effect Size (absolute)")
        ax1.set_title("Effect Sizes by Test")
        ax1.grid(True, alpha=0.3)

        # Add effect size interpretation lines
        ax1.axhline(
            y=0.2, color="orange", linestyle="--", alpha=0.7, label="Small (0.2)"
        )
        ax1.axhline(
            y=0.5, color="blue", linestyle="--", alpha=0.7, label="Medium (0.5)"
        )
        ax1.axhline(
            y=0.8, color="purple", linestyle="--", alpha=0.7, label="Large (0.8)"
        )
        ax1.legend()

        # Effect size interpretation
        interpretations = []
        for _, row in effect_df.iterrows():
            es = row["Effect Size"]
            if es < 0.2:
                interp = "Negligible"
            elif es < 0.5:
                interp = "Small"
            elif es < 0.8:
                interp = "Medium"
            else:
                interp = "Large"
            interpretations.append(interp)

        effect_df["Interpretation"] = interpretations

        # Pie chart of effect size categories
        interp_counts = effect_df["Interpretation"].value_counts()
        # Use 'Greens' palette for pie chart
        num_categories = len(interp_counts)
        # Ensure we have enough colors from the Greens palette, repeating if necessary
        palette = sns.color_palette("Greens", num_categories if num_categories > 0 else 1)
        ax2.pie(
            interp_counts.values,
            labels=interp_counts.index,
            autopct="%1.1f%%",
            colors=palette,
            startangle=90,
        )
        ax2.set_title("Effect Size Distribution")

        plt.tight_layout()
        plots["effect_sizes"] = fig

        return plots

    def _create_effect_size_visualization(
        self,
        test_name: str,
        test_result: Dict[str, Any],
    ) -> Dict[str, Figure]:
        """Create visualization for effect sizes.

        Args:
            test_name: Name of the test
            test_result: Test results containing effect size information

        Returns:
            Dictionary of plots
        """
        plots = {}
        effect_size = test_result.get("effect_size", None)
        effect_size_name = test_result.get("effect_size_name", "Effect Size")

        if effect_size is None:
            self.logger.warning(
                f"Effect size visualization for '{test_name}' requires effect size value."
            )
            return plots

        # Create effect size visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f"Effect Size Analysis: {test_name.replace('_', ' ').title()}", fontweight="bold")

        # Bar plot showing effect size magnitude
        try:
            effect_size_value = float(effect_size)  # Convert to float for comparison
            # Use shades of green: lightgreen for negative, mediumseagreen for positive
            bar_colors = ['lightgreen' if effect_size_value < 0 else 'mediumseagreen']
            ax1.bar([effect_size_name], [abs(effect_size_value)], color=bar_colors, alpha=0.85)
            ax1.set_title("Effect Size Magnitude")
            ax1.set_ylabel("Absolute Value")
            ax1.grid(True, linestyle='--', alpha=0.7, axis='y')

            # Add value annotation on top of the bar
            ax1.text(0, abs(effect_size_value), 
                    f"{effect_size_value:+.3f}",
                    ha='center', va='bottom')

            # Add direction indicator if applicable
            if effect_size_value != 0:
                direction = "Negative" if effect_size_value < 0 else "Positive"
                ax1.text(0, abs(effect_size_value)/2,
                        f"{direction}\nEffect",
                        ha='center', va='center',
                        color='white', fontweight='bold')
        except Exception as e:
            self.logger.warning(f"Could not create effect size bar plot: {e}")
            ax1.text(0.5, 0.5, "Error creating bar plot", ha="center", va="center", transform=ax1.transAxes)

        # Pie chart showing effect size interpretation
        try:
            # Determine effect size category based on common guidelines
            # Note: These thresholds can vary by test type and field
            abs_effect = abs(float(effect_size))
            
            if 'cohen_d' in test_name.lower() or 'd' in effect_size_name.lower():
                # Cohen's d interpretation - Use Greens palette
                greens_palette = sns.color_palette("Greens_r", 4) # _r for reversed to get darker shades first
                if abs_effect < 0.2:
                    category = "Small"
                    colors = [greens_palette[3], '#CCCCCC'] # Lightest green
                    sizes = [abs_effect/0.2 * 100, 100 - (abs_effect/0.2 * 100)]
                elif abs_effect < 0.5:
                    category = "Medium"
                    colors = [greens_palette[2], '#CCCCCC']
                    sizes = [abs_effect/0.5 * 100, 100 - (abs_effect/0.5 * 100)]
                elif abs_effect < 0.8:
                    category = "Large"
                    colors = [greens_palette[1], '#CCCCCC']
                    sizes = [abs_effect/0.8 * 100, 100 - (abs_effect/0.8 * 100)]
                else:
                    category = "Very Large"
                    colors = [greens_palette[0], '#CCCCCC'] # Darkest green
                    sizes = [100, 0]
            elif 'eta_squared' in test_name.lower() or 'η²' in effect_size_name.lower():
                # Eta-squared interpretation - Use Greens palette
                greens_palette_eta = sns.color_palette("Greens_r", 3)
                if abs_effect < 0.01:
                    category = "Small"
                    colors = [greens_palette_eta[2], '#CCCCCC'] # Lightest green
                    sizes = [abs_effect/0.01 * 100, 100 - (abs_effect/0.01 * 100)]
                elif abs_effect < 0.06:
                    category = "Medium"
                    colors = [greens_palette_eta[1], '#CCCCCC']
                    sizes = [abs_effect/0.06 * 100, 100 - (abs_effect/0.06 * 100)]
                else:
                    category = "Large"
                    colors = [greens_palette_eta[0], '#CCCCCC'] # Darkest green
                    sizes = [100, 0]
            elif 'cramer_v' in test_name.lower() or 'v' in effect_size_name.lower():
                # Cramer's V interpretation - Use Greens palette
                greens_palette_cramer = sns.color_palette("Greens_r", 4)
                if abs_effect < 0.1:
                    category = "Small"
                    colors = [greens_palette_cramer[3], '#CCCCCC'] # Lightest green
                    sizes = [abs_effect/0.1 * 100, 100 - (abs_effect/0.1 * 100)]
                elif abs_effect < 0.3:
                    category = "Medium"
                    colors = [greens_palette_cramer[2], '#CCCCCC']
                    sizes = [abs_effect/0.3 * 100, 100 - (abs_effect/0.3 * 100)]
                elif abs_effect < 0.5:
                    category = "Large"
                    colors = [greens_palette_cramer[1], '#CCCCCC']
                    sizes = [abs_effect/0.5 * 100, 100 - (abs_effect/0.5 * 100)]
                else:
                    category = "Very Large"
                    colors = [greens_palette_cramer[0], '#CCCCCC'] # Darkest green
                    sizes = [100, 0]
            else:
                # Generic interpretation (based on correlation coefficient-like metrics) - Use Greens palette
                greens_palette_generic = sns.color_palette("Greens_r", 4)
                if abs_effect < 0.1:
                    category = "Negligible"
                    colors = [greens_palette_generic[3], '#CCCCCC'] # Lightest green
                    sizes = [abs_effect/0.1 * 100, 100 - (abs_effect/0.1 * 100)]
                elif abs_effect < 0.3:
                    category = "Small"
                    colors = [greens_palette_generic[2], '#CCCCCC']
                    sizes = [abs_effect/0.3 * 100, 100 - (abs_effect/0.3 * 100)]
                elif abs_effect < 0.5:
                    category = "Medium"
                    colors = [greens_palette_generic[1], '#CCCCCC']
                    sizes = [abs_effect/0.5 * 100, 100 - (abs_effect/0.5 * 100)]
                else:
                    category = "Large"
                    colors = [greens_palette_generic[0], '#CCCCCC'] # Darkest green
                    sizes = [100, 0]

            # Create pie chart
            wedges, texts, autotexts = ax2.pie(sizes, colors=colors, autopct='%1.1f%%',
                                              startangle=90, counterclock=False)
            # Customize pie chart appearance
            plt.setp(autotexts, size=9, weight="bold")
            plt.setp(texts, size=0)  # Hide default labels
            
            ax2.set_title(f"Effect Size Category: {category}")
            
            # Add interpretation text below the pie chart
            interpretation_text = f"Interpretation:\n{effect_size_name} = {effect_size:+.3f}\nCategory: {category}"
            ax2.text(0.5, -0.1, interpretation_text,
                    ha='center', va='center',
                    transform=ax2.transAxes,
                    bbox=dict(boxstyle='round,pad=0.5',
                             facecolor='white',
                             alpha=0.8,
                             edgecolor='gray'))

        except Exception as e:
            self.logger.warning(f"Could not create effect size pie chart: {e}")
            ax2.text(0.5, 0.5, "Error creating pie chart", ha="center", va="center", transform=ax2.transAxes)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plots["effect_size_visualization"] = fig
        self.logger.info(f"Created effect size visualization for '{test_name}'.")

        return plots
