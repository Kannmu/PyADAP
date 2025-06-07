"""Main plotter class for PyADAP 3.0

This module provides the main Plotter class that coordinates all visualization
functionalities and provides a unified interface for creating plots.
"""

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import warnings
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .statistical_plots import StatisticalPlots
from .diagnostic_plots import DiagnosticPlots
from ..utils import Logger, get_logger
from ..config import Config

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


class Plotter:
    """Main plotter class that coordinates all visualization functionalities."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the plotter.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.logger = get_logger("PyADAP.Plotter")
        
        # Initialize sub-plotters
        self.statistical_plots = StatisticalPlots(self.config)
        self.diagnostic_plots = DiagnosticPlots(self.config)
        
        # Set up plotting style
        self._setup_style()
        
        # Store created figures
        self.figures: Dict[str, Figure] = {}
        
        self.logger.info("Plotter initialized")
    
    def _setup_style(self) -> None:
        """Set up the plotting style based on configuration."""
        try:
            # Set matplotlib style
            plt.style.use('default')
            
            # Configure seaborn
            sns.set_theme(
                style="whitegrid",
                palette="husl",
                font_scale=1.0,
                rc={
                    "figure.figsize": (10, 6),
                    "axes.spines.right": False,
                    "axes.spines.top": False,
                    "font.family": "sans-serif",
                    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
                }
            )
            
            # Apply config-specific settings
            if hasattr(self.config, 'visualization'):
                viz_config = self.config.visualization
                
                # Update figure size
                if hasattr(viz_config, 'figure_size'):
                    plt.rcParams['figure.figsize'] = viz_config.figure_size
                
                # Update DPI
                if hasattr(viz_config, 'dpi'):
                    plt.rcParams['figure.dpi'] = viz_config.dpi
                
                # Update font size
                if hasattr(viz_config, 'font_size'):
                    plt.rcParams['font.size'] = viz_config.font_size
            
            self.logger.debug("Plotting style configured")
            
        except Exception as e:
            self.logger.warning(f"Failed to set up plotting style: {str(e)}")
    
    def create_data_overview_plots(self, data: pd.DataFrame, 
                                 variables: Optional[Dict[str, List[str]]] = None) -> Dict[str, Figure]:
        """Create comprehensive data overview plots.
        
        Args:
            data: Input DataFrame
            variables: Dictionary with variable assignments
            
        Returns:
            Dictionary of figure names and Figure objects
        """
        plots = {}
        
        try:
            # Data distribution plots
            plots.update(self._create_distribution_plots(data, variables))
            
            # Correlation plots
            plots.update(self._create_correlation_plots(data, variables))
            
            # Missing data plots
            plots.update(self._create_missing_data_plots(data))
            
            # Outlier detection plots
            plots.update(self._create_outlier_plots(data, variables))
            
            self.figures.update(plots)
            self.logger.info(f"Created {len(plots)} data overview plots")
            
        except Exception as e:
            self.logger.error(f"Error creating data overview plots: {str(e)}")
            raise
        
        return plots
    
    def create_statistical_analysis_plots(self, data: pd.DataFrame,
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
            # Delegate to statistical plots
            plots.update(self.statistical_plots.create_analysis_plots(data, variables, results))
            
            self.figures.update(plots)
            self.logger.info(f"Created {len(plots)} statistical analysis plots")
            
        except Exception as e:
            self.logger.error(f"Error creating statistical analysis plots: {str(e)}")
            raise
        
        return plots
    
    def create_comparison_plot(self, data: pd.DataFrame, dependent_var: str, 
                             independent_var: str, plot_type: str = 'box',
                             save_path: Optional[Path] = None) -> Optional[Path]:
        """Create a comparison plot for statistical analysis.
        
        Args:
            data: Input DataFrame
            dependent_var: Name of dependent variable
            independent_var: Name of independent variable
            plot_type: Type of plot ('box', 'violin', 'bar')
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot or None if failed
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if plot_type == 'box':
                data.boxplot(column=dependent_var, by=independent_var, ax=ax)
            elif plot_type == 'violin':
                import seaborn as sns
                sns.violinplot(data=data, x=independent_var, y=dependent_var, ax=ax)
            elif plot_type == 'bar':
                grouped = data.groupby(independent_var)[dependent_var].mean()
                grouped.plot(kind='bar', ax=ax)
            else:
                # Default to box plot
                data.boxplot(column=dependent_var, by=independent_var, ax=ax)
            
            ax.set_title(f'{dependent_var} by {independent_var}')
            ax.set_xlabel(independent_var)
            ax.set_ylabel(dependent_var)
            
            plt.tight_layout()
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                return save_path
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to create comparison plot: {str(e)}")
            return None
    
    def create_scatter_plot(self, data: pd.DataFrame, x_var: str, y_var: str,
                          group_var: Optional[str] = None, save_path: Optional[Path] = None) -> Optional[Path]:
        """Create a scatter plot.
        
        Args:
            data: Input DataFrame
            x_var: X-axis variable name
            y_var: Y-axis variable name
            group_var: Optional grouping variable for color coding
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot or None if failed
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if group_var and group_var in data.columns:
                # Color by group
                groups = data[group_var].unique()
                colors = plt.cm.Set1(np.linspace(0, 1, len(groups)))
                
                for i, group in enumerate(groups):
                    group_data = data[data[group_var] == group]
                    ax.scatter(group_data[x_var], group_data[y_var], 
                             c=[colors[i]], label=str(group), alpha=0.7)
                
                ax.legend(title=group_var)
            else:
                # Simple scatter plot
                ax.scatter(data[x_var], data[y_var], alpha=0.7)
            
            ax.set_xlabel(x_var)
            ax.set_ylabel(y_var)
            ax.set_title(f'{y_var} vs {x_var}')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                return save_path
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to create scatter plot: {str(e)}")
            return None
    
    def create_correlation_matrix(self, data: pd.DataFrame, save_path: Optional[Path] = None) -> Optional[Path]:
        """Create a correlation matrix heatmap.
        
        Args:
            data: Input DataFrame
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot or None if failed
        """
        try:
            # Get numeric columns only
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                self.logger.warning("Not enough numeric columns for correlation matrix")
                return None
            
            # Calculate correlation matrix
            corr_matrix = data[numeric_cols].corr()
            
            # Create figure
            fig, ax = plt.subplots(figsize=(max(8, len(numeric_cols)*0.6), max(6, len(numeric_cols)*0.5)))
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # Create heatmap
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax,
                       fmt='.2f', annot_kws={'size': 8})
            
            ax.set_title('Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                return save_path
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to create correlation matrix: {str(e)}")
            return None
    
    def create_statistical_plots(self, data: pd.DataFrame,
                              variables: Dict[str, List[str]],
                              results: Optional[Dict[str, Any]] = None) -> Dict[str, Figure]:
        """Create diagnostic plots for assumption checking.
        
        Args:
            data: Input DataFrame
            variables: Dictionary with variable assignments
            results: Optional analysis results
            
        Returns:
            Dictionary of figure names and Figure objects
        """
        plots = {}
        
        try:
            # Delegate to diagnostic plots
            plots.update(self.diagnostic_plots.create_diagnostic_plots(data, variables, results))
            
            self.figures.update(plots)
            self.logger.info(f"Created {len(plots)} diagnostic plots")
            
        except Exception as e:
            self.logger.error(f"Error creating diagnostic plots: {str(e)}")
            raise
        
        return plots
    
    def _create_distribution_plots(self, data: pd.DataFrame,
                                 variables: Optional[Dict[str, List[str]]] = None) -> Dict[str, Figure]:
        """Create distribution plots for numeric variables.
        
        Args:
            data: Input DataFrame
            variables: Dictionary with variable assignments
            
        Returns:
            Dictionary of figure names and Figure objects
        """
        plots = {}
        
        # Get numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return plots
        
        # Limit to reasonable number of variables
        if len(numeric_cols) > 12:
            numeric_cols = numeric_cols[:12]
            self.logger.warning(f"Limited distribution plots to first 12 numeric variables")
        
        # Create distribution plot
        n_cols = min(4, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        fig.suptitle('Variable Distributions', fontsize=16, fontweight='bold')
        
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            ax = axes[i]
            
            # Create histogram with KDE
            try:
                data[col].hist(ax=ax, bins=30, alpha=0.7, density=True, color='skyblue')
                
                # Add KDE if possible
                if len(data[col].dropna()) > 1:
                    data[col].plot.kde(ax=ax, color='red', linewidth=2)
                
                ax.set_title(f'{col}', fontweight='bold')
                ax.set_xlabel(col)
                ax.set_ylabel('Density')
                ax.grid(True, alpha=0.3)
                
                # Add statistics text
                mean_val = data[col].mean()
                std_val = data[col].std()
                ax.text(0.02, 0.98, f'μ = {mean_val:.2f}\nσ = {std_val:.2f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error plotting {col}', 
                       ha='center', va='center', transform=ax.transAxes)
                self.logger.warning(f"Error plotting distribution for {col}: {str(e)}")
        
        # Hide empty subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plots['distributions'] = fig
        
        return plots
    
    def _create_correlation_plots(self, data: pd.DataFrame,
                                variables: Optional[Dict[str, List[str]]] = None) -> Dict[str, Figure]:
        """Create correlation plots.
        
        Args:
            data: Input DataFrame
            variables: Dictionary with variable assignments
            
        Returns:
            Dictionary of figure names and Figure objects
        """
        plots = {}
        
        # Get numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return plots
        
        # Limit to reasonable number of variables
        if len(numeric_cols) > 20:
            numeric_cols = numeric_cols[:20]
            self.logger.warning(f"Limited correlation plot to first 20 numeric variables")
        
        # Calculate correlation matrix
        corr_matrix = data[numeric_cols].corr()
        
        # Create correlation heatmap
        fig, ax = plt.subplots(figsize=(max(8, len(numeric_cols)*0.6), max(6, len(numeric_cols)*0.5)))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax,
                   fmt='.2f', annot_kws={'size': 8})
        
        ax.set_title('Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plots['correlation_matrix'] = fig
        
        # Create pairplot for key variables (if not too many)
        if len(numeric_cols) <= 6:
            try:
                # Create pairplot
                pair_fig = plt.figure(figsize=(12, 10))
                
                # Use seaborn pairplot
                g = sns.pairplot(data[numeric_cols], diag_kind='hist', plot_kws={'alpha': 0.6})
                g.fig.suptitle('Pairwise Relationships', y=1.02, fontsize=16, fontweight='bold')
                
                plots['pairplot'] = g.fig
                
            except Exception as e:
                self.logger.warning(f"Error creating pairplot: {str(e)}")
        
        return plots
    
    def _create_missing_data_plots(self, data: pd.DataFrame) -> Dict[str, Figure]:
        """Create missing data visualization plots.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary of figure names and Figure objects
        """
        plots = {}
        
        # Check if there's any missing data
        missing_counts = data.isnull().sum()
        if missing_counts.sum() == 0:
            return plots
        
        # Create missing data plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Missing Data Analysis', fontsize=16, fontweight='bold')
        
        # Missing data by column
        missing_data = missing_counts[missing_counts > 0].sort_values(ascending=True)
        missing_pct = (missing_data / len(data) * 100)
        
        # Bar plot of missing counts
        bars = ax1.barh(range(len(missing_data)), missing_data.values, color='coral')
        ax1.set_yticks(range(len(missing_data)))
        ax1.set_yticklabels(missing_data.index)
        ax1.set_xlabel('Number of Missing Values')
        ax1.set_title('Missing Values by Column')
        ax1.grid(True, alpha=0.3)
        
        # Add percentage labels
        for i, (count, pct) in enumerate(zip(missing_data.values, missing_pct.values)):
            ax1.text(count + max(missing_data.values) * 0.01, i, f'{pct:.1f}%', 
                    va='center', fontsize=9)
        
        # Missing data pattern (heatmap)
        if len(data.columns) <= 20:  # Only for reasonable number of columns
            missing_matrix = data.isnull().astype(int)
            
            # Sample data if too many rows
            if len(missing_matrix) > 1000:
                missing_matrix = missing_matrix.sample(n=1000, random_state=42)
            
            im = ax2.imshow(missing_matrix.T, cmap='RdYlBu_r', aspect='auto', interpolation='nearest')
            ax2.set_title('Missing Data Pattern')
            ax2.set_xlabel('Observations')
            ax2.set_ylabel('Variables')
            ax2.set_yticks(range(len(data.columns)))
            ax2.set_yticklabels(data.columns, fontsize=8)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
            cbar.set_label('Missing (1) / Present (0)')
        else:
            ax2.text(0.5, 0.5, 'Too many variables\nfor pattern visualization', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Missing Data Pattern')
        
        plt.tight_layout()
        plots['missing_data'] = fig
        
        return plots
    
    def _create_outlier_plots(self, data: pd.DataFrame,
                            variables: Optional[Dict[str, List[str]]] = None) -> Dict[str, Figure]:
        """Create outlier detection plots.
        
        Args:
            data: Input DataFrame
            variables: Dictionary with variable assignments
            
        Returns:
            Dictionary of figure names and Figure objects
        """
        plots = {}
        
        # Get numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return plots
        
        # Limit to reasonable number of variables
        if len(numeric_cols) > 12:
            numeric_cols = numeric_cols[:12]
            self.logger.warning(f"Limited outlier plots to first 12 numeric variables")
        
        # Create box plots
        n_cols = min(4, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        fig.suptitle('Outlier Detection (Box Plots)', fontsize=16, fontweight='bold')
        
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            ax = axes[i]
            
            try:
                # Create box plot
                box_data = data[col].dropna()
                if len(box_data) > 0:
                    bp = ax.boxplot(box_data, patch_artist=True, 
                                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                                   medianprops=dict(color='red', linewidth=2))
                    
                    ax.set_title(f'{col}', fontweight='bold')
                    ax.set_ylabel('Value')
                    ax.grid(True, alpha=0.3)
                    
                    # Calculate and display outlier statistics
                    Q1 = box_data.quantile(0.25)
                    Q3 = box_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = box_data[(box_data < lower_bound) | (box_data > upper_bound)]
                    outlier_pct = len(outliers) / len(box_data) * 100
                    
                    ax.text(0.02, 0.98, f'Outliers: {len(outliers)} ({outlier_pct:.1f}%)', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error plotting {col}', 
                       ha='center', va='center', transform=ax.transAxes)
                self.logger.warning(f"Error creating box plot for {col}: {str(e)}")
        
        # Hide empty subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plots['outlier_detection'] = fig
        
        return plots
    
    def save_plots(self, output_dir: Union[str, Path], 
                  plots: Optional[Dict[str, Figure]] = None,
                  format: str = 'png', dpi: int = 300) -> List[str]:
        """Save plots to files.
        
        Args:
            output_dir: Output directory
            plots: Dictionary of plots to save (if None, saves all stored plots)
            format: File format ('png', 'pdf', 'svg')
            dpi: Resolution for raster formats
            
        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plots_to_save = plots or self.figures
        saved_files = []
        
        for plot_name, fig in plots_to_save.items():
            try:
                filename = f"{plot_name}.{format}"
                filepath = output_dir / filename
                
                fig.savefig(filepath, format=format, dpi=dpi, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                
                saved_files.append(str(filepath))
                self.logger.debug(f"Saved plot: {filepath}")
                
            except Exception as e:
                self.logger.error(f"Error saving plot {plot_name}: {str(e)}")
        
        self.logger.info(f"Saved {len(saved_files)} plots to {output_dir}")
        return saved_files
    
    def close_all_figures(self) -> None:
        """Close all stored figures to free memory."""
        for fig in self.figures.values():
            plt.close(fig)
        
        self.figures.clear()
        self.logger.debug("Closed all figures")
    
    def get_figure(self, name: str) -> Optional[Figure]:
        """Get a specific figure by name.
        
        Args:
            name: Figure name
            
        Returns:
            Figure object or None if not found
        """
        return self.figures.get(name)
    
    def list_figures(self) -> List[str]:
        """Get list of available figure names.
        
        Returns:
            List of figure names
        """
        return list(self.figures.keys())
    
    def create_summary_plot(self, results: Dict[str, Any]) -> Figure:
        """Create a summary plot of key results.
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            Summary figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Analysis Summary', fontsize=16, fontweight='bold')
        
        try:
            # Plot 1: P-values of statistical tests
            if 'statistical_tests' in results:
                test_names = []
                p_values = []
                
                for test_name, test_result in results['statistical_tests'].items():
                    if isinstance(test_result, dict) and 'p_value' in test_result:
                        test_names.append(test_name)
                        p_values.append(test_result['p_value'])
                
                if test_names:
                    colors = ['red' if p < 0.05 else 'blue' for p in p_values]
                    bars = ax1.bar(range(len(test_names)), p_values, color=colors, alpha=0.7)
                    ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.8, label='α = 0.05')
                    ax1.set_xticks(range(len(test_names)))
                    ax1.set_xticklabels(test_names, rotation=45, ha='right')
                    ax1.set_ylabel('P-value')
                    ax1.set_title('Statistical Test Results')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                else:
                    ax1.text(0.5, 0.5, 'No statistical tests', ha='center', va='center', transform=ax1.transAxes)
            else:
                ax1.text(0.5, 0.5, 'No statistical tests', ha='center', va='center', transform=ax1.transAxes)
            
            # Plot 2: Effect sizes
            if 'statistical_tests' in results:
                effect_names = []
                effect_sizes = []
                
                for test_name, test_result in results['statistical_tests'].items():
                    if isinstance(test_result, dict) and 'effect_size' in test_result:
                        effect_val = test_result['effect_size']
                        if isinstance(effect_val, (int, float)) and not np.isnan(effect_val):
                            effect_names.append(test_name)
                            effect_sizes.append(abs(effect_val))
                
                if effect_names:
                    bars = ax2.bar(range(len(effect_names)), effect_sizes, color='green', alpha=0.7)
                    ax2.set_xticks(range(len(effect_names)))
                    ax2.set_xticklabels(effect_names, rotation=45, ha='right')
                    ax2.set_ylabel('Effect Size (absolute)')
                    ax2.set_title('Effect Sizes')
                    ax2.grid(True, alpha=0.3)
                else:
                    ax2.text(0.5, 0.5, 'No effect sizes', ha='center', va='center', transform=ax2.transAxes)
            else:
                ax2.text(0.5, 0.5, 'No effect sizes', ha='center', va='center', transform=ax2.transAxes)
            
            # Plot 3: Assumption test results
            if 'assumption_tests' in results:
                assumption_names = []
                assumption_results = []
                
                for test_name, test_result in results['assumption_tests'].items():
                    if isinstance(test_result, dict) and 'p_value' in test_result:
                        assumption_names.append(test_name)
                        p_val = test_result['p_value']
                        # Assumption is met if p > 0.05 (generally)
                        assumption_results.append(1 if p_val > 0.05 else 0)
                
                if assumption_names:
                    colors = ['green' if result else 'red' for result in assumption_results]
                    bars = ax3.bar(range(len(assumption_names)), assumption_results, color=colors, alpha=0.7)
                    ax3.set_xticks(range(len(assumption_names)))
                    ax3.set_xticklabels(assumption_names, rotation=45, ha='right')
                    ax3.set_ylabel('Assumption Met (1=Yes, 0=No)')
                    ax3.set_title('Assumption Test Results')
                    ax3.set_ylim(-0.1, 1.1)
                    ax3.grid(True, alpha=0.3)
                else:
                    ax3.text(0.5, 0.5, 'No assumption tests', ha='center', va='center', transform=ax3.transAxes)
            else:
                ax3.text(0.5, 0.5, 'No assumption tests', ha='center', va='center', transform=ax3.transAxes)
            
            # Plot 4: Data quality summary
            if 'data_info' in results:
                info = results['data_info']
                
                # Create a simple data quality overview
                quality_metrics = []
                quality_values = []
                
                if 'missing_percentage' in info:
                    quality_metrics.append('Missing Data %')
                    quality_values.append(info['missing_percentage'])
                
                if 'duplicate_rows' in info:
                    total_rows = info.get('rows', 1)
                    dup_pct = (info['duplicate_rows'] / total_rows) * 100 if total_rows > 0 else 0
                    quality_metrics.append('Duplicate Rows %')
                    quality_values.append(dup_pct)
                
                if 'outlier_percentage' in info:
                    quality_metrics.append('Outliers %')
                    quality_values.append(info['outlier_percentage'])
                
                if quality_metrics:
                    colors = ['red' if val > 10 else 'orange' if val > 5 else 'green' for val in quality_values]
                    bars = ax4.bar(range(len(quality_metrics)), quality_values, color=colors, alpha=0.7)
                    ax4.set_xticks(range(len(quality_metrics)))
                    ax4.set_xticklabels(quality_metrics, rotation=45, ha='right')
                    ax4.set_ylabel('Percentage')
                    ax4.set_title('Data Quality Metrics')
                    ax4.grid(True, alpha=0.3)
                else:
                    ax4.text(0.5, 0.5, 'No quality metrics', ha='center', va='center', transform=ax4.transAxes)
            else:
                ax4.text(0.5, 0.5, 'No data info', ha='center', va='center', transform=ax4.transAxes)
            
        except Exception as e:
            self.logger.error(f"Error creating summary plot: {str(e)}")
            # Create error message plot
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        self.figures['summary'] = fig
        
        return fig
    
    def create_distribution_plot(self, data: pd.DataFrame, variable: str, 
                               group_by: Optional[str] = None, 
                               save_path: Optional[Path] = None) -> Optional[Path]:
        """Create a distribution plot for a specific variable.
        
        Args:
            data: Input DataFrame
            variable: Variable to plot
            group_by: Optional grouping variable
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot or None if failed
        """
        try:
            if variable not in data.columns:
                self.logger.warning(f"Variable '{variable}' not found in data")
                return None
            
            # Create figure
            if group_by and group_by in data.columns:
                # Grouped distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Check if grouping variable has reasonable number of groups
                n_groups = data[group_by].nunique()
                if n_groups > 10:
                    self.logger.warning(f"Too many groups ({n_groups}) for distribution plot")
                    return None
                
                # Create distribution plot by group
                for group_name in data[group_by].unique():
                    if pd.isna(group_name):
                        continue
                    group_data = data[data[group_by] == group_name][variable].dropna()
                    if len(group_data) > 0:
                        group_data.hist(alpha=0.6, label=f'{group_name} (n={len(group_data)})', 
                                      bins=20, ax=ax, density=True)
                
                ax.set_xlabel(variable)
                ax.set_ylabel('Density')
                ax.set_title(f'Distribution of {variable} by {group_by}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            else:
                # Single variable distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Create histogram with KDE
                clean_data = data[variable].dropna()
                if len(clean_data) == 0:
                    self.logger.warning(f"No valid data for variable '{variable}'")
                    return None
                
                clean_data.hist(bins=30, alpha=0.7, density=True, color='skyblue', ax=ax)
                
                # Add KDE if enough data points
                if len(clean_data) > 10:
                    try:
                        clean_data.plot.kde(ax=ax, color='red', linewidth=2)
                    except Exception as e:
                        self.logger.debug(f"Could not add KDE: {str(e)}")
                
                ax.set_xlabel(variable)
                ax.set_ylabel('Density')
                ax.set_title(f'Distribution of {variable}')
                ax.grid(True, alpha=0.3)
                
                # Add statistics text
                stats_text = f"""Statistics:
Mean: {clean_data.mean():.3f}
Std: {clean_data.std():.3f}
Median: {clean_data.median():.3f}
N: {len(clean_data)}"""
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            # Save plot if path provided
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Distribution plot saved: {save_path}")
                plt.close(fig)
                return save_path
            else:
                # Store figure for later use
                plot_name = f"distribution_{variable}"
                if group_by:
                    plot_name += f"_by_{group_by}"
                self.figures[plot_name] = fig
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating distribution plot: {str(e)}")
            return None