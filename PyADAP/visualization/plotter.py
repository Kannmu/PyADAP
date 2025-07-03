"""Main plotter class for PyADAP 3.0

This module provides the main Plotter class that coordinates all visualization
functionalities and provides a unified interface for creating plots.
"""

import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import warnings
from matplotlib.figure import Figure

from .statistical_plots import StatisticalPlots
from .diagnostic_plots import DiagnosticPlots
from ..utils import get_logger
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
        self.statistical_plots = StatisticalPlots(self.config) # This line should already be correct based on view
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
            palette = 'Greens'  # Default palette set to Greens
            if hasattr(self.config, 'visualization') and hasattr(self.config.visualization, 'palette'):
                palette = self.config.visualization.palette
            
            sns.set_theme(
                style="whitegrid",
                palette=palette,
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

            # Advanced plots
            plots.update(self.create_advanced_plots(data, variables))
            
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
                             save_path: Optional[Path] = None) -> Union[Figure, Path, None]:
        """Create a comparison plot for statistical analysis.
        
        Args:
            data: Input DataFrame
            dependent_var: Name of dependent variable
            independent_var: Name of independent variable
            plot_type: Type of plot ('box', 'violin', 'bar')
            save_path: Path to save the plot
            
        Returns:
            Figure object if save_path is None, otherwise path to saved plot.
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if plot_type == 'box':
                # Use patch_artist=True for fill, and specify colors for solid fill
                sns.boxplot(data=data, x=independent_var, y=dependent_var, ax=ax, palette=palette, flierprops=dict(marker='o', markersize=5, markerfacecolor='gray'))
            elif plot_type == 'violin':
                sns.violinplot(data=data, x=independent_var, y=dependent_var, ax=ax, palette=palette)
            elif plot_type == 'bar':
                # Ensure bars are solid
                grouped = data.groupby(independent_var)[dependent_var].mean().reset_index()
                sns.barplot(data=grouped, x=independent_var, y=dependent_var, ax=ax, palette=palette)
            else:
                # Default to box plot with solid fill
                sns.boxplot(data=data, x=independent_var, y=dependent_var, ax=ax, palette=palette, flierprops=dict(marker='o', markersize=5, markerfacecolor='gray'))
            
            ax.set_title(f'{dependent_var} by {independent_var}')
            ax.set_xlabel(independent_var)
            ax.set_ylabel(dependent_var)
            
            plt.tight_layout()
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                # plt.close(fig) # Keep figure open if we need to return it
                return fig  # Return the figure object even when saving
            else:
                return fig # Return the figure object if not saving
        
        except Exception as e:
            self.logger.error(f"Error creating comparison plot for {dependent_var} by {independent_var}: {str(e)}")
            # Optionally re-raise or return None/empty Figure
            # For now, let's return None to indicate failure clearly in this context
            return None
    
    def create_scatter_plot(self, data: pd.DataFrame, x_var: str, y_var: str,
                          group_var: Optional[str] = None, save_path: Optional[Path] = None) -> Union[Figure, None]:
        """Create a scatter plot.
        
        Args:
            data: Input DataFrame
            x_var: X-axis variable name
            y_var: Y-axis variable name
            group_var: Optional grouping variable for color coding
            save_path: Path to save the plot
            
        Returns:
            Figure object or None if failed
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if group_var and group_var in data.columns:
                # Color by group
                groups = data[group_var].unique()
                # Use a qualitative colormap that is robust to colorblindness if possible
                # and has enough distinct colors for the number of groups.
                # If too many groups, consider alternative visualization or summarization.
                if len(groups) <= 10:
                    colors = sns.color_palette("colorblind", n_colors=len(groups))
                else:
                    # Fallback for more groups, though this might not be ideal for distinction
                    colors = plt.cm.get_cmap('tab20', len(groups))
                    if len(groups) > 20:
                        self.logger.warning(f"Scatter plot has {len(groups)} groups, colors may not be distinct.")
                
                for i, group in enumerate(groups):
                    group_data = data[data[group_var] == group]
                    ax.scatter(group_data[x_var], group_data[y_var], 
                             color=colors[i], label=str(group), alpha=0.7)
                
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
                # plt.close(fig) # Keep figure open if we need to return it
            return fig # Always return the figure object
                
        except Exception as e:
            self.logger.error(f"Failed to create scatter plot for {y_var} vs {x_var}: {str(e)}")
            return None
    
    def create_correlation_matrix(self, data: pd.DataFrame, save_path: Optional[Path] = None) -> Union[Figure, None]:
        """Create a correlation matrix heatmap.
        
        Args:
            data: Input DataFrame
            save_path: Path to save the plot
            
        Returns:
            Figure object or None if failed
        """
        try:
            # Get numeric columns only
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                self.logger.warning("No numeric columns found for correlation matrix.")
                return None
            
            correlation_matrix = data[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(max(8, len(numeric_cols) * 0.8), max(6, len(numeric_cols) * 0.6)))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
            ax.set_title('Correlation Matrix')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                # plt.close(fig) # Keep figure open if we need to return it
            return fig # Always return the figure object

        except Exception as e:
            self.logger.error(f"Failed to create correlation matrix: {str(e)}")
            return None
    
    def create_statistical_plots(self, data: pd.DataFrame,
                              variables: Dict[str, List[str]],
                              results: Optional[Dict[str, Any]] = None) -> Dict[str, Figure]:
        """Create statistical analysis plots.

        This method delegates the creation of statistical plots to the StatisticalPlots class.

        Args:
            data: Input DataFrame.
            variables: Dictionary with variable assignments (e.g., dependent, independent).
            results: Dictionary containing the results from statistical analyses.

        Returns:
            Dictionary of figure names and Matplotlib Figure objects.
        """
        self.logger.info("Starting creation of statistical plots via Plotter facade...")
        plots = {}

        if results is None:
            self.logger.warning("No analysis results provided to Plotter for statistical plots. Skipping.")
            return plots

        try:
            # Delegate to StatisticalPlots instance for creating plots based on analysis results
            statistical_analysis_plots = self.statistical_plots.create_analysis_plots(
                data, variables, results
            )
            plots.update(statistical_analysis_plots)
            self.figures.update(plots) # Add to the main figures dictionary
            self.logger.info(f"Plotter: Successfully created {len(statistical_analysis_plots)} statistical analysis plots.")

        except Exception as e:
            self.logger.error(f"Plotter: Error during statistical plot generation: {str(e)}", exc_info=True)
            # Optionally, create a placeholder error plot or return empty
            # For now, returning empty plots on error after logging

        return plots
    
    def create_advanced_plots(self, data: pd.DataFrame,
                            variables: Optional[Dict[str, List[str]]] = None) -> Dict[str, Figure]:
        """Create advanced visualization plots.
        
        Args:
            data: Input DataFrame
            variables: Dictionary with variable assignments
            
        Returns:
            Dictionary of figure names and Figure objects
        """
        plots = {}
        
        try:
            # 1. Violin plots for distribution comparison
            if variables and 'dependent' in variables and 'independent' in variables:
                fig, ax = plt.subplots(figsize=(12, 6))
                for dep_var in variables['dependent']:
                    for ind_var in variables['independent']:
                        sns.violinplot(data=data, x=ind_var, y=dep_var, ax=ax)
                        ax.set_title(f'Distribution of {dep_var} by {ind_var}')
                        plt.tight_layout()
                        plots[f'violin_{dep_var}_{ind_var}'] = fig
                        fig, ax = plt.subplots(figsize=(12, 6))  # New figure for next plot
            
            # 2. Joint plots for relationship visualization
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                for i in range(min(len(numeric_cols)-1, 3)):
                    for j in range(i+1, min(len(numeric_cols), 4)):
                        g = sns.jointplot(data=data, x=numeric_cols[i], y=numeric_cols[j],
                                        kind='reg', height=8)
                        g.fig.suptitle(f'Joint Distribution: {numeric_cols[i]} vs {numeric_cols[j]}',
                                    y=1.02)
                        plots[f'joint_{numeric_cols[i]}_{numeric_cols[j]}'] = g.fig
            
            # 3. Pair plots for selected variables
            if len(numeric_cols) > 2:
                selected_cols = numeric_cols[:min(5, len(numeric_cols))]
                g = sns.pairplot(data[selected_cols], diag_kind='kde')
                g.fig.suptitle('Pair-wise Relationships', y=1.02)
                plots['pairplot'] = g.fig
            
            # 4. Enhanced box plots with swarm overlay
            if variables and 'dependent' in variables and 'independent' in variables:
                for dep_var in variables['dependent']:
                    for ind_var in variables['independent']:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        sns.boxplot(data=data, x=ind_var, y=dep_var, ax=ax, alpha=0.6)
                        sns.swarmplot(data=data, x=ind_var, y=dep_var, ax=ax, color='0.25', alpha=0.5)
                        ax.set_title(f'Box Plot with Data Points: {dep_var} by {ind_var}')
                        plt.tight_layout()
                        plots[f'boxswarm_{dep_var}_{ind_var}'] = fig
            
            self.logger.info(f"Created {len(plots)} advanced visualization plots")
            
        except Exception as e:
            self.logger.error(f"Error creating advanced plots: {str(e)}")
        
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
            self.logger.warning("Limited distribution plots to first 12 numeric variables")
        
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
                data[col].hist(ax=ax, bins=30, alpha=0.7, density=True, color='mediumseagreen') # Changed to 'mediumseagreen' for Greens theme
                
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
        
        # 1. Correlation Matrix Heatmap (already handled by create_correlation_matrix, but can be a summary here)
        # We can call the public method if we want to ensure saving logic is also handled, 
        # or replicate a simpler version here if it's just for internal overview.
        # For now, let's assume create_correlation_matrix is the primary one and this is a quick overview.
        try:
            corr_matrix_fig = self.create_correlation_matrix(data) # This now returns a Figure
            if corr_matrix_fig:
                plots['correlation_matrix_overview'] = corr_matrix_fig
        except Exception as e:
            self.logger.warning(f"Could not generate overview correlation matrix: {e}")

        # 2. Scatter matrix for top N correlated pairs (optional, can be intensive)
        # This is similar to what's in create_advanced_plots (pairplot)
        # We might want to avoid duplication or make this more specific.
        # For simplicity, we'll rely on the pairplot in create_advanced_plots for now.

        # 3. Individual scatter plots for highly correlated pairs (if not covered by advanced plots)
        # This could be useful if advanced_plots are disabled or for specific focus.
        # Example: Plot top 3 positive and negative correlations
        if len(numeric_cols) >=2:
            correlation_matrix = data[numeric_cols].corr().abs()
            upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
            top_corr_pairs = upper_tri.unstack().sort_values(ascending=False)
            
            num_scatter_to_plot = min(3, len(top_corr_pairs[top_corr_pairs > 0.5])) # Plot up to 3 pairs with |corr| > 0.5

            if num_scatter_to_plot > 0:
                fig_scatter_corr, axes_scatter_corr = plt.subplots(1, num_scatter_to_plot, figsize=(5 * num_scatter_to_plot, 4))
                if num_scatter_to_plot == 1:
                    axes_scatter_corr = [axes_scatter_corr] # Make it iterable
                
                for i, ((var1, var2), corr_val) in enumerate(top_corr_pairs.head(num_scatter_to_plot).items()):
                    ax = axes_scatter_corr[i]
                    sns.scatterplot(data=data, x=var1, y=var2, ax=ax, alpha=0.7)
                    ax.set_title(f'{var1} vs {var2}\nCorr: {data[var1].corr(data[var2]):.2f}')
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plots['top_correlated_scatter'] = fig_scatter_corr

        return plots
    
    def _create_missing_data_plots(self, data: pd.DataFrame) -> Dict[str, Figure]:
        """Create missing data visualization plots.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary of figure names and Figure objects
        """
        self.logger.info("Creating missing data plots...")
        plots = {}

        missing_counts = data.isnull().sum()
        if missing_counts.sum() == 0:
            self.logger.info("No missing data found.")
            return plots

        try:
            # Try using missingno if available
            import missingno as msno
            self.logger.info("Using 'missingno' library for missing data plots.")

            # Matrix plot
            fig_matrix = plt.figure(figsize=self.config.visualization.figure_size)
            ax_matrix = fig_matrix.gca()
            msno.matrix(data, ax=ax_matrix, sparkline=False)
            ax_matrix.set_title('Missing Data Matrix (missingno)', fontsize=14, fontweight='bold')
            plots['missing_data_matrix_msno'] = fig_matrix
            self.logger.debug("Created missing data matrix plot with missingno.")

            # Bar plot
            fig_bar = plt.figure(figsize=(max(10, len(data.columns) * 0.5), 6))
            ax_bar = fig_bar.gca()
            msno.bar(data, ax=ax_bar, color=self.config.visualization.palette[0] if isinstance(self.config.visualization.palette, list) else self.config.visualization.palette)
            ax_bar.set_title('Missing Data Bar Chart (missingno)', fontsize=14, fontweight='bold')
            plots['missing_data_bar_msno'] = fig_bar
            self.logger.debug("Created missing data bar chart with missingno.")

            # Heatmap (optional, can be intensive)
            if len(data.columns) <= self.config.visualization.max_heatmap_columns_missingno:
                fig_heatmap = plt.figure(figsize=self.config.visualization.figure_size)
                ax_heatmap = fig_heatmap.gca()
                msno.heatmap(data, ax=ax_heatmap, cmap=self.config.visualization.cmap)
                ax_heatmap.set_title('Missing Data Correlation Heatmap (missingno)', fontsize=14, fontweight='bold')
                plots['missing_data_heatmap_msno'] = fig_heatmap
                self.logger.debug("Created missing data heatmap with missingno.")
            else:
                self.logger.info(f"Skipping missingno heatmap due to high number of columns ({len(data.columns)} > {self.config.visualization.max_heatmap_columns_missingno}).")

        except ImportError:
            self.logger.warning("'missingno' library not found. Falling back to seaborn/matplotlib for missing data plots.")
            # Fallback to seaborn/matplotlib plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Missing Data Analysis', fontsize=16, fontweight='bold')

            missing_data = missing_counts[missing_counts > 0].sort_values(ascending=True)
            missing_pct = (missing_data / len(data) * 100)

            bars = ax1.barh(range(len(missing_data)), missing_data.values, color='mediumseagreen') # Changed to 'mediumseagreen' for Greens theme
            ax1.set_yticks(range(len(missing_data)))
            ax1.set_yticklabels(missing_data.index)
            ax1.set_xlabel('Number of Missing Values')
            ax1.set_title('Missing Values by Column')
            ax1.grid(True, alpha=0.3)

            for i, (count, pct) in enumerate(zip(missing_data.values, missing_pct.values)):
                ax1.text(count + max(missing_data.values) * 0.01, i, f'{pct:.1f}%',
                         va='center', fontsize=9)

            if len(data.columns) <= self.config.visualization.max_heatmap_columns_fallback:
                missing_matrix = data.isnull().astype(int)
                if len(missing_matrix) > 1000:
                    missing_matrix = missing_matrix.sample(n=1000, random_state=self.config.visualization.random_seed)
                im = ax2.imshow(missing_matrix.T, cmap=self.config.visualization.cmap_fallback_missing, aspect='auto', interpolation='nearest')
                ax2.set_title('Missing Data Pattern')
                ax2.set_xlabel('Observations')
                ax2.set_ylabel('Variables')
                ax2.set_yticks(range(len(data.columns)))
                ax2.set_yticklabels(data.columns, fontsize=8)
                cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
                cbar.set_label('Missing (1) / Present (0)')
            else:
                ax2.text(0.5, 0.5, f'Too many variables ({len(data.columns)})\nfor fallback pattern visualization.',
                         ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                ax2.set_title('Missing Data Pattern (Fallback)')
            
            plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
            plots['missing_data_fallback'] = fig
            self.logger.debug("Created fallback missing data plots.")
        except Exception as e:
            self.logger.error(f"Error creating missing data plots: {str(e)}", exc_info=True)

        self.logger.info(f"Finished creating missing data plots. Generated {len(plots)} plot(s).")
        return plots
    
    def _create_outlier_plots(self, data: pd.DataFrame,
                            variables: Optional[Dict[str, List[str]]] = None) -> Dict[str, Figure]:
        """Create outlier detection plots.

        Args:
            data: Input DataFrame
            variables: Dictionary with variable assignments (currently unused but kept for API consistency)

        Returns:
            Dictionary of figure names and Figure objects
        """
        self.logger.info("Creating outlier detection plots...")
        plots = {}

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            self.logger.info("No numeric columns found for outlier plots.")
            return plots

        max_outlier_plots = self.config.visualization.max_outlier_plots
        if len(numeric_cols) > max_outlier_plots:
            self.logger.warning(f"Number of numeric columns ({len(numeric_cols)}) exceeds max_outlier_plots ({max_outlier_plots}). Plotting first {max_outlier_plots} columns.")
            numeric_cols_to_plot = numeric_cols[:max_outlier_plots]
        else:
            numeric_cols_to_plot = numeric_cols

        if not numeric_cols_to_plot:
            self.logger.info("No numeric columns selected for outlier plotting after filtering.")
            return plots

        # Determine subplot layout
        n_plots = len(numeric_cols_to_plot)
        n_cols_subplot = self.config.visualization.outlier_plot_ncols
        n_rows_subplot = (n_plots + n_cols_subplot - 1) // n_cols_subplot

        fig_width = n_cols_subplot * self.config.visualization.outlier_plot_base_width
        fig_height = n_rows_subplot * self.config.visualization.outlier_plot_base_height

        try:
            fig, axes = plt.subplots(n_rows_subplot, n_cols_subplot, figsize=(fig_width, fig_height), squeeze=False)
            fig.suptitle('Outlier Detection (Box Plots)', fontsize=16, fontweight='bold')
            axes_flat = axes.flatten()

            for i, col in enumerate(numeric_cols_to_plot):
                ax = axes_flat[i]
                try:
                    box_data = data[col].dropna()
                    if not box_data.empty:
                        # Ensure solid fill by using palette and flierprops for consistency
                        sns.boxplot(y=box_data, ax=ax, palette="Greens", 
                                    width=0.5, flierprops=dict(marker='o', markersize=self.config.visualization.flier_size, markerfacecolor='darkgreen', markeredgecolor='darkgreen'))
                        ax.set_title(f'{col}', fontweight='bold', fontsize=10)
                        ax.set_ylabel('Value', fontsize=9)
                        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) # Hide x-axis ticks and labels
                        ax.grid(True, linestyle='--', alpha=0.6)

                        Q1 = box_data.quantile(0.25)
                        Q3 = box_data.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers = box_data[(box_data < lower_bound) | (box_data > upper_bound)]
                        outlier_pct = (len(outliers) / len(box_data) * 100) if len(box_data) > 0 else 0

                        ax.text(0.02, 0.98, f'Outliers: {len(outliers)} ({outlier_pct:.1f}%)',
                                transform=ax.transAxes, verticalalignment='top',
                                fontsize=8, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
                        self.logger.debug(f"Created box plot for {col}. Outliers: {len(outliers)} ({outlier_pct:.1f}%).")
                    else:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=10)
                        ax.set_title(f'{col} (No Data)', fontweight='bold', fontsize=10)
                        self.logger.debug(f"No data for box plot for {col}.")
                except Exception as e_col:
                    self.logger.error(f"Error creating box plot for column {col}: {str(e_col)}", exc_info=True)
                    ax.text(0.5, 0.5, f'Error plotting {col}', ha='center', va='center', transform=ax.transAxes, color='red', fontsize=10)
                    ax.set_title(f'{col} (Error)', fontweight='bold', fontsize=10)

            # Hide unused subplots
            for j in range(n_plots, n_rows_subplot * n_cols_subplot):
                fig.delaxes(axes_flat[j])

            plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
            plots['outlier_detection_summary'] = fig
            self.logger.debug("Created summary outlier detection plot.")

        except Exception as e_fig:
            self.logger.error(f"Error creating the main figure for outlier plots: {str(e_fig)}", exc_info=True)
            # Create a fallback figure indicating an error
            fig_err = plt.figure()
            ax_err = fig_err.gca()
            ax_err.text(0.5, 0.5, 'Error generating outlier plots.', ha='center', va='center', color='red', fontsize=14)
            plots['outlier_detection_error'] = fig_err

        self.logger.info(f"Finished creating outlier plots. Generated {len(plots)} plot(s).")
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
    
    def create_statistical_plots(self, data: pd.DataFrame,
                              variables: Dict[str, List[str]],
                              results: Optional[Dict[str, Any]] = None) -> Dict[str, Figure]:
        """Create statistical analysis plots.

        This method delegates the creation of statistical plots to the StatisticalPlots class.

        Args:
            data: Input DataFrame.
            variables: Dictionary with variable assignments (e.g., dependent, independent).
            results: Dictionary containing the results from statistical analyses.

        Returns:
            Dictionary of figure names and Matplotlib Figure objects.
        """
        self.logger.info("Starting creation of statistical plots via Plotter facade...")
        plots = {}

        if results is None:
            self.logger.warning("No analysis results provided to Plotter for statistical plots. Skipping.")
            return plots

        try:
            # Delegate to StatisticalPlots instance for creating plots based on analysis results
            statistical_analysis_plots = self.statistical_plots.create_analysis_plots(
                data, variables, results
            )
            plots.update(statistical_analysis_plots)
            self.figures.update(plots) # Add to the main figures dictionary
            self.logger.info(f"Plotter: Successfully created {len(statistical_analysis_plots)} statistical analysis plots.")

        except Exception as e:
            self.logger.error(f"Plotter: Error during statistical plot generation: {str(e)}", exc_info=True)
            # Optionally, create a placeholder error plot or return empty
            # For now, returning empty plots on error after logging

        return plots
    
    def create_advanced_plots(self, data: pd.DataFrame,
                            variables: Optional[Dict[str, List[str]]] = None) -> Dict[str, Figure]:
        """Create advanced visualization plots.
        
        Args:
            data: Input DataFrame
            variables: Dictionary with variable assignments
            
        Returns:
            Dictionary of figure names and Figure objects
        """
        plots = {}
        
        try:
            # 1. Violin plots for distribution comparison
            if variables and 'dependent' in variables and 'independent' in variables:
                fig, ax = plt.subplots(figsize=(12, 6))
                for dep_var in variables['dependent']:
                    for ind_var in variables['independent']:
                        sns.violinplot(data=data, x=ind_var, y=dep_var, ax=ax)
                        ax.set_title(f'Distribution of {dep_var} by {ind_var}')
                        plt.tight_layout()
                        plots[f'violin_{dep_var}_{ind_var}'] = fig
                        fig, ax = plt.subplots(figsize=(12, 6))  # New figure for next plot
            
            # 2. Joint plots for relationship visualization
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                for i in range(min(len(numeric_cols)-1, 3)):
                    for j in range(i+1, min(len(numeric_cols), 4)):
                        g = sns.jointplot(data=data, x=numeric_cols[i], y=numeric_cols[j],
                                        kind='reg', height=8)
                        g.fig.suptitle(f'Joint Distribution: {numeric_cols[i]} vs {numeric_cols[j]}',
                                    y=1.02)
                        plots[f'joint_{numeric_cols[i]}_{numeric_cols[j]}'] = g.fig
            
            # 3. Pair plots for selected variables
            if len(numeric_cols) > 2:
                selected_cols = numeric_cols[:min(5, len(numeric_cols))]
                g = sns.pairplot(data[selected_cols], diag_kind='kde')
                g.fig.suptitle('Pair-wise Relationships', y=1.02)
                plots['pairplot'] = g.fig
            
            # 4. Enhanced box plots with swarm overlay
            if variables and 'dependent' in variables and 'independent' in variables:
                for dep_var in variables['dependent']:
                    for ind_var in variables['independent']:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        sns.boxplot(data=data, x=ind_var, y=dep_var, ax=ax, alpha=0.6)
                        sns.swarmplot(data=data, x=ind_var, y=dep_var, ax=ax, color='0.25', alpha=0.5)
                        ax.set_title(f'Box Plot with Data Points: {dep_var} by {ind_var}')
                        plt.tight_layout()
                        plots[f'boxswarm_{dep_var}_{ind_var}'] = fig
            
            self.logger.info(f"Created {len(plots)} advanced visualization plots")
            
        except Exception as e:
            self.logger.error(f"Error creating advanced plots: {str(e)}")
        
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
            self.logger.warning("Limited distribution plots to first 12 numeric variables")
        
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
                data[col].hist(ax=ax, bins=30, alpha=0.7, density=True, color='mediumseagreen') # Changed to 'mediumseagreen' for Greens theme
                
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
        
        # 1. Correlation Matrix Heatmap (already handled by create_correlation_matrix, but can be a summary here)
        # We can call the public method if we want to ensure saving logic is also handled, 
        # or replicate a simpler version here if it's just for internal overview.
        # For now, let's assume create_correlation_matrix is the primary one and this is a quick overview.
        try:
            corr_matrix_fig = self.create_correlation_matrix(data) # This now returns a Figure
            if corr_matrix_fig:
                plots['correlation_matrix_overview'] = corr_matrix_fig
        except Exception as e:
            self.logger.warning(f"Could not generate overview correlation matrix: {e}")

        # 2. Scatter matrix for top N correlated pairs (optional, can be intensive)
        # This is similar to what's in create_advanced_plots (pairplot)
        # We might want to avoid duplication or make this more specific.
        # For simplicity, we'll rely on the pairplot in create_advanced_plots for now.

        # 3. Individual scatter plots for highly correlated pairs (if not covered by advanced plots)
        # This could be useful if advanced_plots are disabled or for specific focus.
        # Example: Plot top 3 positive and negative correlations
        if len(numeric_cols) >=2:
            correlation_matrix = data[numeric_cols].corr().abs()
            upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
            top_corr_pairs = upper_tri.unstack().sort_values(ascending=False)
            
            num_scatter_to_plot = min(3, len(top_corr_pairs[top_corr_pairs > 0.5])) # Plot up to 3 pairs with |corr| > 0.5

            if num_scatter_to_plot > 0:
                fig_scatter_corr, axes_scatter_corr = plt.subplots(1, num_scatter_to_plot, figsize=(5 * num_scatter_to_plot, 4))
                if num_scatter_to_plot == 1:
                    axes_scatter_corr = [axes_scatter_corr] # Make it iterable
                
                for i, ((var1, var2), corr_val) in enumerate(top_corr_pairs.head(num_scatter_to_plot).items()):
                    ax = axes_scatter_corr[i]
                    sns.scatterplot(data=data, x=var1, y=var2, ax=ax, alpha=0.7)
                    ax.set_title(f'{var1} vs {var2}\nCorr: {data[var1].corr(data[var2]):.2f}')
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plots['top_correlated_scatter'] = fig_scatter_corr

        return plots
    
    def _create_missing_data_plots(self, data: pd.DataFrame) -> Dict[str, Figure]:
        """Create missing data visualization plots.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary of figure names and Figure objects
        """
        self.logger.info("Creating missing data plots...")
        plots = {}

        missing_counts = data.isnull().sum()
        if missing_counts.sum() == 0:
            self.logger.info("No missing data found.")
            return plots

        try:
            # Try using missingno if available
            import missingno as msno
            self.logger.info("Using 'missingno' library for missing data plots.")

            # Matrix plot
            fig_matrix = plt.figure(figsize=self.config.visualization.figure_size)
            ax_matrix = fig_matrix.gca()
            msno.matrix(data, ax=ax_matrix, sparkline=False)
            ax_matrix.set_title('Missing Data Matrix (missingno)', fontsize=14, fontweight='bold')
            plots['missing_data_matrix_msno'] = fig_matrix
            self.logger.debug("Created missing data matrix plot with missingno.")

            # Bar plot
            fig_bar = plt.figure(figsize=(max(10, len(data.columns) * 0.5), 6))
            ax_bar = fig_bar.gca()
            msno.bar(data, ax=ax_bar, color=self.config.visualization.palette[0] if isinstance(self.config.visualization.palette, list) else self.config.visualization.palette)
            ax_bar.set_title('Missing Data Bar Chart (missingno)', fontsize=14, fontweight='bold')
            plots['missing_data_bar_msno'] = fig_bar
            self.logger.debug("Created missing data bar chart with missingno.")

            # Heatmap (optional, can be intensive)
            if len(data.columns) <= self.config.visualization.max_heatmap_columns_missingno:
                fig_heatmap = plt.figure(figsize=self.config.visualization.figure_size)
                ax_heatmap = fig_heatmap.gca()
                msno.heatmap(data, ax=ax_heatmap, cmap=self.config.visualization.cmap)
                ax_heatmap.set_title('Missing Data Correlation Heatmap (missingno)', fontsize=14, fontweight='bold')
                plots['missing_data_heatmap_msno'] = fig_heatmap
                self.logger.debug("Created missing data heatmap with missingno.")
            else:
                self.logger.info(f"Skipping missingno heatmap due to high number of columns ({len(data.columns)} > {self.config.visualization.max_heatmap_columns_missingno}).")

        except ImportError:
            self.logger.warning("'missingno' library not found. Falling back to seaborn/matplotlib for missing data plots.")
            # Fallback to seaborn/matplotlib plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Missing Data Analysis', fontsize=16, fontweight='bold')

            missing_data = missing_counts[missing_counts > 0].sort_values(ascending=True)
            missing_pct = (missing_data / len(data) * 100)

            bars = ax1.barh(range(len(missing_data)), missing_data.values, color='mediumseagreen') # Changed to 'mediumseagreen' for Greens theme
            ax1.set_yticks(range(len(missing_data)))
            ax1.set_yticklabels(missing_data.index)
            ax1.set_xlabel('Number of Missing Values')
            ax1.set_title('Missing Values by Column')
            ax1.grid(True, alpha=0.3)

            for i, (count, pct) in enumerate(zip(missing_data.values, missing_pct.values)):
                ax1.text(count + max(missing_data.values) * 0.01, i, f'{pct:.1f}%',
                         va='center', fontsize=9)

            if len(data.columns) <= self.config.visualization.max_heatmap_columns_fallback:
                missing_matrix = data.isnull().astype(int)
                if len(missing_matrix) > 1000:
                    missing_matrix = missing_matrix.sample(n=1000, random_state=self.config.visualization.random_seed)
                im = ax2.imshow(missing_matrix.T, cmap=self.config.visualization.cmap_fallback_missing, aspect='auto', interpolation='nearest')
                ax2.set_title('Missing Data Pattern')
                ax2.set_xlabel('Observations')
                ax2.set_ylabel('Variables')
                ax2.set_yticks(range(len(data.columns)))
                ax2.set_yticklabels(data.columns, fontsize=8)
                cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
                cbar.set_label('Missing (1) / Present (0)')
            else:
                ax2.text(0.5, 0.5, f'Too many variables ({len(data.columns)})\nfor fallback pattern visualization.',
                         ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                ax2.set_title('Missing Data Pattern (Fallback)')
            
            plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
            plots['missing_data_fallback'] = fig
            self.logger.debug("Created fallback missing data plots.")
        except Exception as e:
            self.logger.error(f"Error creating missing data plots: {str(e)}", exc_info=True)

        self.logger.info(f"Finished creating missing data plots. Generated {len(plots)} plot(s).")
        return plots
    
    def _create_outlier_plots(self, data: pd.DataFrame,
                            variables: Optional[Dict[str, List[str]]] = None) -> Dict[str, Figure]:
        """Create outlier detection plots.

        Args:
            data: Input DataFrame
            variables: Dictionary with variable assignments (currently unused but kept for API consistency)

        Returns:
            Dictionary of figure names and Figure objects
        """
        self.logger.info("Creating outlier detection plots...")
        plots = {}

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            self.logger.info("No numeric columns found for outlier plots.")
            return plots

        max_outlier_plots = self.config.visualization.max_outlier_plots
        if len(numeric_cols) > max_outlier_plots:
            self.logger.warning(f"Number of numeric columns ({len(numeric_cols)}) exceeds max_outlier_plots ({max_outlier_plots}). Plotting first {max_outlier_plots} columns.")
            numeric_cols_to_plot = numeric_cols[:max_outlier_plots]
        else:
            numeric_cols_to_plot = numeric_cols

        if not numeric_cols_to_plot:
            self.logger.info("No numeric columns selected for outlier plotting after filtering.")
            return plots

        # Determine subplot layout
        n_plots = len(numeric_cols_to_plot)
        n_cols_subplot = self.config.visualization.outlier_plot_ncols
        n_rows_subplot = (n_plots + n_cols_subplot - 1) // n_cols_subplot

        fig_width = n_cols_subplot * self.config.visualization.outlier_plot_base_width
        fig_height = n_rows_subplot * self.config.visualization.outlier_plot_base_height

        try:
            fig, axes = plt.subplots(n_rows_subplot, n_cols_subplot, figsize=(fig_width, fig_height), squeeze=False)
            fig.suptitle('Outlier Detection (Box Plots)', fontsize=16, fontweight='bold')
            axes_flat = axes.flatten()

            for i, col in enumerate(numeric_cols_to_plot):
                ax = axes_flat[i]
                try:
                    box_data = data[col].dropna()
                    if not box_data.empty:
                        # Ensure solid fill by using palette and flierprops for consistency
                        sns.boxplot(y=box_data, ax=ax, palette="Greens", 
                                    width=0.5, flierprops=dict(marker='o', markersize=self.config.visualization.flier_size, markerfacecolor='darkgreen', markeredgecolor='darkgreen'))
                        ax.set_title(f'{col}', fontweight='bold', fontsize=10)
                        ax.set_ylabel('Value', fontsize=9)
                        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) # Hide x-axis ticks and labels
                        ax.grid(True, linestyle='--', alpha=0.6)

                        Q1 = box_data.quantile(0.25)
                        Q3 = box_data.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers = box_data[(box_data < lower_bound) | (box_data > upper_bound)]
                        outlier_pct = (len(outliers) / len(box_data) * 100) if len(box_data) > 0 else 0

                        ax.text(0.02, 0.98, f'Outliers: {len(outliers)} ({outlier_pct:.1f}%)',
                                transform=ax.transAxes, verticalalignment='top',
                                fontsize=8, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
                        self.logger.debug(f"Created box plot for {col}. Outliers: {len(outliers)} ({outlier_pct:.1f}%).")
                    else:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=10)
                        ax.set_title(f'{col} (No Data)', fontweight='bold', fontsize=10)
                        self.logger.debug(f"No data for box plot for {col}.")
                except Exception as e_col:
                    self.logger.error(f"Error creating box plot for column {col}: {str(e_col)}", exc_info=True)
                    ax.text(0.5, 0.5, f'Error plotting {col}', ha='center', va='center', transform=ax.transAxes, color='red', fontsize=10)
                    ax.set_title(f'{col} (Error)', fontweight='bold', fontsize=10)

            # Hide unused subplots
            for j in range(n_plots, n_rows_subplot * n_cols_subplot):
                fig.delaxes(axes_flat[j])

            plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
            plots['outlier_detection_summary'] = fig
            self.logger.debug("Created summary outlier detection plot.")

        except Exception as e_fig:
            self.logger.error(f"Error creating the main figure for outlier plots: {str(e_fig)}", exc_info=True)
            # Create a fallback figure indicating an error
            fig_err = plt.figure()
            ax_err = fig_err.gca()
            ax_err.text(0.5, 0.5, 'Error generating outlier plots.', ha='center', va='center', color='red', fontsize=14)
            plots['outlier_detection_error'] = fig_err

        self.logger.info(f"Finished creating outlier plots. Generated {len(plots)} plot(s).")
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
    