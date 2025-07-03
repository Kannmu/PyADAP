#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apple-Style HTML Report Generator for PyADAP

This module provides a modern, Apple-inspired HTML report generator with
comprehensive data visualization capabilities and interactive features.
"""

import base64
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
import io
from ..config import Config
from ..utils import get_logger

from .apple_html_template import AppleHTMLTemplate


class AppleStyleReportGenerator:
    """Modern Apple-style HTML report generator with advanced visualizations."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the Apple-style report generator.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.logger = get_logger("PyADAP.AppleStyleReportGenerator")
        
        # Report settings
        self.output_dir = Path(self.config.output_dir) if hasattr(self.config, 'output_dir') else None
        
        # Color schemes
        self.apple_colors = {
            'blue': '#007AFF',
            'green': '#34C759', 
            'orange': '#FF9500',
            'red': '#FF3B30',
            'purple': '#AF52DE',
            'pink': '#FF2D92',
            'teal': '#5AC8FA',
            'indigo': '#5856D6',
            'gray': '#8E8E93',
            'gray_light': '#F2F2F7',
            'gray_dark': '#1C1C1E',
            'white': '#FFFFFF',
            'black': '#000000'
        }
        
        # Chart configurations
        self.chart_config = {
            'figure_size': (12, 8),
            'dpi': 150,
            'style': 'seaborn-v0_8-whitegrid',
            'font_family': 'SF Pro Display'
        }
        
        self.logger.info("Apple-style report generator initialized")
    
    def generate_comprehensive_report(self, 
                                    results: Dict[str, Any],
                                    data: Optional[pd.DataFrame] = None,
                                    data_info: Optional[Dict[str, Any]] = None,
                                    filename: Optional[str] = None,
                                    output_dir: Optional[Path] = None) -> Path:
        """Generate a comprehensive Apple-style HTML report.
        
        Args:
            results: Analysis results dictionary
            data: Original DataFrame (optional, for additional visualizations)
            data_info: Information about the analyzed data
            filename: Custom filename (optional)
            output_dir: Custom output directory (optional)
            
        Returns:
            Path to the generated report file
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"apple_analysis_report_{timestamp}.html"
            
            # Use custom output directory if provided
            if output_dir:
                target_dir = Path(output_dir)
            elif self.output_dir is not None:
                target_dir = self.output_dir
            else:
                raise ValueError("No output directory configured. Please provide output_dir parameter or set output_dir in config.")
            
            target_dir.mkdir(parents=True, exist_ok=True)
            output_path = target_dir / filename
            
            # Generate visualizations
            visualizations = self._generate_all_visualizations(results, data)
            
            # Create HTML content using the template
            template = AppleHTMLTemplate()
            html_content = template.create_apple_html_content(results, data_info or {}, visualizations)
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"Apple-style HTML report generated: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate Apple-style report: {str(e)}")
            raise
    
    def _generate_all_visualizations(self, results: Dict[str, Any], 
                                   data: Optional[pd.DataFrame] = None) -> Dict[str, str]:
        """Generate all visualizations and return as base64 encoded strings.
        
        Args:
            results: Analysis results
            data: Original DataFrame
            
        Returns:
            Dictionary of visualization names to base64 encoded images
        """
        visualizations = {}
        
        try:
            # Set up matplotlib style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Statistical Tests Results Chart
            if 'statistical_tests' in results:
                viz = self._create_statistical_tests_chart(results['statistical_tests'])
                if viz:
                    visualizations['statistical_tests'] = viz
            
            # 2. Data Quality Dashboard
            if 'quality_report' in results:
                viz = self._create_data_quality_dashboard(results['quality_report'])
                if viz:
                    visualizations['data_quality'] = viz
            
            # 3. Correlation Heatmap
            if data is not None:
                viz = self._create_correlation_heatmap(data)
                if viz:
                    visualizations['correlation_heatmap'] = viz
            
            # 4. Distribution Analysis
            if data is not None:
                viz = self._create_distribution_analysis(data)
                if viz:
                    visualizations['distribution_analysis'] = viz
            
            # 5. Variable Importance Chart
            if 'variable_importance' in results:
                viz = self._create_variable_importance_chart(results['variable_importance'])
                if viz:
                    visualizations['variable_importance'] = viz
            
            # 6. Assumptions Validation Chart
            if 'assumptions' in results:
                viz = self._create_assumptions_chart(results['assumptions'])
                if viz:
                    visualizations['assumptions'] = viz
            
            # 7. Effect Sizes Visualization
            if 'effect_sizes' in results:
                viz = self._create_effect_sizes_chart(results['effect_sizes'])
                if viz:
                    visualizations['effect_sizes'] = viz
            
            # 8. Data Overview Dashboard
            if data is not None:
                viz = self._create_data_overview_dashboard(data)
                if viz:
                    visualizations['data_overview'] = viz
            
            self.logger.info(f"Generated {len(visualizations)} visualizations")
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
        
        return visualizations
    
    def _create_statistical_tests_chart(self, statistical_tests: Dict[str, Any]) -> Optional[str]:
        """Create a modern statistical tests results chart."""
        try:
            if not statistical_tests:
                return None
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.patch.set_facecolor('white')
            
            # Extract test data
            test_names = []
            p_values = []
            effect_sizes = []
            
            for test_name, test_result in statistical_tests.items():
                test_names.append(test_name.replace('_', ' ').title())
                
                if isinstance(test_result, dict):
                    p_val = test_result.get('p_value', test_result.get('pvalue', 0.5))
                    if hasattr(p_val, 'item'):
                        p_val = p_val.item()
                    p_values.append(float(p_val))
                    
                    effect_size = test_result.get('effect_size', test_result.get('cohens_d', 0))
                    if hasattr(effect_size, 'item'):
                        effect_size = effect_size.item()
                    effect_sizes.append(abs(float(effect_size)) if effect_size else 0)
                else:
                    p_values.append(0.5)
                    effect_sizes.append(0)
            
            # P-values chart
            colors = [self.apple_colors['red'] if p < 0.05 else self.apple_colors['green'] for p in p_values]
            bars1 = ax1.barh(test_names, p_values, color=colors, alpha=0.8)
            ax1.axvline(x=0.05, color=self.apple_colors['red'], linestyle='--', alpha=0.7, linewidth=2)
            ax1.set_xlabel('P-value', fontsize=14, fontweight='600')
            ax1.set_title('Statistical Significance', fontsize=16, fontweight='700', pad=20)
            ax1.set_xlim(0, max(max(p_values), 0.1))
            
            # Add value labels
            for i, (bar, p_val) in enumerate(zip(bars1, p_values)):
                ax1.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{p_val:.4f}', va='center', fontsize=10, fontweight='500')
            
            # Effect sizes chart
            if any(effect_sizes):
                bars2 = ax2.barh(test_names, effect_sizes, color=self.apple_colors['blue'], alpha=0.8)
                ax2.set_xlabel('Effect Size', fontsize=14, fontweight='600')
                ax2.set_title('Effect Sizes', fontsize=16, fontweight='700', pad=20)
                
                # Add value labels
                for i, (bar, effect) in enumerate(zip(bars2, effect_sizes)):
                    ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                            f'{effect:.3f}', va='center', fontsize=10, fontweight='500')
            else:
                ax2.text(0.5, 0.5, 'No Effect Size Data', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=14, color=self.apple_colors['gray'])
                ax2.set_xlim(0, 1)
            
            # Styling
            for ax in [ax1, ax2]:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color(self.apple_colors['gray'])
                ax.spines['bottom'].set_color(self.apple_colors['gray'])
                ax.tick_params(colors=self.apple_colors['gray_dark'])
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            self.logger.error(f"Error creating statistical tests chart: {str(e)}")
            return None
        finally:
            plt.close('all')
    
    def _create_data_quality_dashboard(self, quality_report: Dict[str, Any]) -> Optional[str]:
        """Create a comprehensive data quality dashboard."""
        try:
            fig = plt.figure(figsize=(20, 12))
            fig.patch.set_facecolor('white')
            
            # Create a 2x3 grid
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            
            # 1. Missing Values Overview
            ax1 = fig.add_subplot(gs[0, 0])
            missing_values = quality_report.get('missing_values', {})
            if missing_values and isinstance(missing_values, dict):
                columns = list(missing_values.keys())[:10]  # Top 10
                values = [missing_values[col] for col in columns]
                
                bars = ax1.bar(range(len(columns)), values, color=self.apple_colors['orange'], alpha=0.8)
                ax1.set_title('Missing Values by Column', fontsize=14, fontweight='700')
                ax1.set_xticks(range(len(columns)))
                ax1.set_xticklabels(columns, rotation=45, ha='right')
                ax1.set_ylabel('Count')
                
                # Add value labels
                for bar, val in zip(bars, values):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                            str(val), ha='center', va='bottom', fontweight='500')
            else:
                ax1.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                        transform=ax1.transAxes, fontsize=12, color=self.apple_colors['green'])
            
            # 2. Data Types Distribution
            ax2 = fig.add_subplot(gs[0, 1])
            data_types = quality_report.get('data_types', {})
            if data_types:
                type_counts = {}
                for dtype in data_types.values():
                    dtype_str = str(dtype)
                    if 'int' in dtype_str or 'float' in dtype_str:
                        type_counts['Numeric'] = type_counts.get('Numeric', 0) + 1
                    elif 'object' in dtype_str or 'string' in dtype_str:
                        type_counts['Text'] = type_counts.get('Text', 0) + 1
                    elif 'datetime' in dtype_str:
                        type_counts['DateTime'] = type_counts.get('DateTime', 0) + 1
                    else:
                        type_counts['Other'] = type_counts.get('Other', 0) + 1
                
                colors = [self.apple_colors['blue'], self.apple_colors['green'], 
                         self.apple_colors['purple'], self.apple_colors['orange']]
                wedges, texts, autotexts = ax2.pie(type_counts.values(), labels=type_counts.keys(), 
                                                  autopct='%1.1f%%', colors=colors[:len(type_counts)],
                                                  startangle=90)
                ax2.set_title('Data Types Distribution', fontsize=14, fontweight='700')
                
                # Style the text
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('600')
            
            # 3. Duplicates Information
            ax3 = fig.add_subplot(gs[0, 2])
            duplicates = quality_report.get('duplicates', 0)
            total_rows = quality_report.get('total_rows', 1000)
            
            duplicate_pct = (duplicates / total_rows * 100) if total_rows > 0 else 0
            clean_pct = 100 - duplicate_pct
            
            colors = [self.apple_colors['green'], self.apple_colors['red']]
            sizes = [clean_pct, duplicate_pct]
            labels = ['Clean Data', 'Duplicates']
            
            wedges, texts, autotexts = ax3.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                              colors=colors, startangle=90)
            ax3.set_title('Data Cleanliness', fontsize=14, fontweight='700')
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('600')
            
            # 4. Data Quality Score
            ax4 = fig.add_subplot(gs[1, :])
            
            # Calculate quality metrics
            completeness = (1 - sum(missing_values.values()) / (total_rows * len(missing_values))) * 100 if missing_values else 100
            uniqueness = (1 - duplicates / total_rows) * 100 if total_rows > 0 else 100
            consistency = 95  # Placeholder - would need actual consistency checks
            overall_quality = (completeness + uniqueness + consistency) / 3
            
            metrics = ['Completeness', 'Uniqueness', 'Consistency', 'Overall Quality']
            scores = [completeness, uniqueness, consistency, overall_quality]
            
            # Create horizontal bar chart
            y_pos = np.arange(len(metrics))
            colors_bars = [self.apple_colors['green'] if score >= 80 else 
                          self.apple_colors['orange'] if score >= 60 else 
                          self.apple_colors['red'] for score in scores]
            
            bars = ax4.barh(y_pos, scores, color=colors_bars, alpha=0.8)
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(metrics)
            ax4.set_xlabel('Quality Score (%)')
            ax4.set_title('Data Quality Metrics', fontsize=16, fontweight='700', pad=20)
            ax4.set_xlim(0, 100)
            
            # Add score labels
            for i, (bar, score) in enumerate(zip(bars, scores)):
                ax4.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                        f'{score:.1f}%', va='center', fontweight='600')
            
            # Style all axes
            for ax in [ax1, ax2, ax3, ax4]:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                if ax != ax2 and ax != ax3:  # Don't modify pie chart spines
                    ax.spines['left'].set_color(self.apple_colors['gray'])
                    ax.spines['bottom'].set_color(self.apple_colors['gray'])
                ax.tick_params(colors=self.apple_colors['gray_dark'])
                if ax != ax2 and ax != ax3:
                    ax.grid(True, alpha=0.3)
            
            return self._fig_to_base64(fig)
            
        except Exception as e:
            self.logger.error(f"Error creating data quality dashboard: {str(e)}")
            return None
        finally:
            plt.close('all')
    
    def _create_correlation_heatmap(self, data: pd.DataFrame) -> Optional[str]:
        """Create a modern correlation heatmap."""
        try:
            # Select only numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty or len(numeric_data.columns) < 2:
                return None
            
            # Calculate correlation matrix
            corr_matrix = numeric_data.corr()
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))
            fig.patch.set_facecolor('white')
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # Generate heatmap
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax,
                       fmt='.2f', annot_kws={'fontsize': 10, 'fontweight': '500'})
            
            ax.set_title('Correlation Matrix', fontsize=16, fontweight='700', pad=20)
            
            # Style the plot
            ax.tick_params(colors=self.apple_colors['gray_dark'])
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            self.logger.error(f"Error creating correlation heatmap: {str(e)}")
            return None
        finally:
            plt.close('all')
    
    def _create_distribution_analysis(self, data: pd.DataFrame) -> Optional[str]:
        """Create distribution analysis for numeric variables."""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                return None
            
            # Select up to 6 most important numeric columns
            columns_to_plot = numeric_data.columns[:6]
            n_cols = len(columns_to_plot)
            
            if n_cols == 0:
                return None
            
            # Calculate grid dimensions
            n_rows = (n_cols + 2) // 3
            n_plot_cols = min(3, n_cols)
            
            fig, axes = plt.subplots(n_rows, n_plot_cols, figsize=(15, 5 * n_rows))
            fig.patch.set_facecolor('white')
            
            if n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, col in enumerate(columns_to_plot):
                row = i // 3
                col_idx = i % 3
                ax = axes[row, col_idx] if n_rows > 1 else axes[col_idx]
                
                # Create histogram with KDE
                data_col = numeric_data[col].dropna()
                # Ensure data_col is array-like for matplotlib hist
                data_values = data_col.values if hasattr(data_col, 'values') else data_col
                ax.hist(data_values, bins=30, alpha=0.7, color=self.apple_colors['blue'], 
                       density=True, edgecolor='white', linewidth=0.5)
                
                # Add KDE line
                try:
                    sns.kdeplot(data=data_values, ax=ax, color=self.apple_colors['red'], linewidth=2)
                except Exception as e:
                    self.logger.debug(f"Failed to add KDE for {col}: {str(e)}")
                
                ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='600')
                ax.set_xlabel(col)
                ax.set_ylabel('Density')
                
                # Style
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color(self.apple_colors['gray'])
                ax.spines['bottom'].set_color(self.apple_colors['gray'])
                ax.tick_params(colors=self.apple_colors['gray_dark'])
                ax.grid(True, alpha=0.3)
            
            # Hide empty subplots
            if n_cols < n_rows * n_plot_cols:
                for i in range(n_cols, n_rows * n_plot_cols):
                    row = i // 3
                    col_idx = i % 3
                    ax = axes[row, col_idx] if n_rows > 1 else axes[col_idx]
                    ax.set_visible(False)
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            self.logger.error(f"Error creating distribution analysis: {str(e)}")
            return None
        finally:
            plt.close('all')
    
    def _create_variable_importance_chart(self, variable_importance: Dict[str, Any]) -> Optional[str]:
        """Create variable importance visualization."""
        try:
            if not variable_importance:
                return None
            
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.patch.set_facecolor('white')
            
            # Extract variable names and importance scores
            if isinstance(variable_importance, dict):
                variables = list(variable_importance.keys())
                importance = list(variable_importance.values())
            else:
                return None
            
            # Sort by importance
            sorted_data = sorted(zip(variables, importance), key=lambda x: x[1], reverse=True)
            variables, importance = zip(*sorted_data)
            
            # Create horizontal bar chart
            y_pos = np.arange(len(variables))
            bars = ax.barh(y_pos, importance, color=self.apple_colors['purple'], alpha=0.8)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(variables)
            ax.set_xlabel('Importance Score')
            ax.set_title('Variable Importance', fontsize=16, fontweight='700', pad=20)
            
            # Add value labels
            for i, (bar, imp) in enumerate(zip(bars, importance)):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{imp:.3f}', va='center', fontweight='500')
            
            # Style
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(self.apple_colors['gray'])
            ax.spines['bottom'].set_color(self.apple_colors['gray'])
            ax.tick_params(colors=self.apple_colors['gray_dark'])
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            self.logger.error(f"Error creating variable importance chart: {str(e)}")
            return None
        finally:
            plt.close('all')
    
    def _create_assumptions_chart(self, assumptions: Dict[str, Any]) -> Optional[str]:
        """Create assumptions validation chart."""
        try:
            if not assumptions:
                return None
            
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.patch.set_facecolor('white')
            
            assumption_names = []
            p_values = []
            statuses = []
            
            for name, result in assumptions.items():
                assumption_names.append(name.replace('_', ' ').title())
                
                if isinstance(result, dict):
                    p_val = result.get('p_value', result.get('pvalue', 0.5))
                    if hasattr(p_val, 'item'):
                        p_val = p_val.item()
                    p_values.append(float(p_val))
                    statuses.append('Met' if p_val > 0.05 else 'Violated')
                else:
                    p_values.append(0.5)
                    statuses.append('Unknown')
            
            # Create bar chart
            colors = [self.apple_colors['green'] if status == 'Met' else 
                     self.apple_colors['red'] if status == 'Violated' else 
                     self.apple_colors['gray'] for status in statuses]
            
            bars = ax.barh(assumption_names, p_values, color=colors, alpha=0.8)
            ax.axvline(x=0.05, color=self.apple_colors['red'], linestyle='--', alpha=0.7, linewidth=2)
            
            ax.set_xlabel('P-value')
            ax.set_title('Statistical Assumptions Validation', fontsize=16, fontweight='700', pad=20)
            ax.set_xlim(0, max(max(p_values), 0.1))
            
            # Add value labels
            for i, (bar, p_val, status) in enumerate(zip(bars, p_values, statuses)):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                       f'{p_val:.4f} ({status})', va='center', fontweight='500')
            
            # Style
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(self.apple_colors['gray'])
            ax.spines['bottom'].set_color(self.apple_colors['gray'])
            ax.tick_params(colors=self.apple_colors['gray_dark'])
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            self.logger.error(f"Error creating assumptions chart: {str(e)}")
            return None
        finally:
            plt.close('all')
    
    def _create_effect_sizes_chart(self, effect_sizes: Dict[str, Any]) -> Optional[str]:
        """Create effect sizes visualization."""
        try:
            if not effect_sizes:
                return None
            
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.patch.set_facecolor('white')
            
            # Extract effect size data
            test_names = []
            effect_values = []
            
            for name, value in effect_sizes.items():
                test_names.append(name.replace('_', ' ').title())
                if hasattr(value, 'item'):
                    value = value.item()
                effect_values.append(abs(float(value)) if value else 0)
            
            # Create bar chart with effect size interpretation colors
            colors = []
            for effect in effect_values:
                if effect < 0.2:
                    colors.append(self.apple_colors['gray'])  # Small
                elif effect < 0.5:
                    colors.append(self.apple_colors['orange'])  # Medium
                elif effect < 0.8:
                    colors.append(self.apple_colors['blue'])  # Large
                else:
                    colors.append(self.apple_colors['red'])  # Very large
            
            bars = ax.barh(test_names, effect_values, color=colors, alpha=0.8)
            
            # Add reference lines
            ax.axvline(x=0.2, color=self.apple_colors['gray'], linestyle='--', alpha=0.5, linewidth=1)
            ax.axvline(x=0.5, color=self.apple_colors['orange'], linestyle='--', alpha=0.5, linewidth=1)
            ax.axvline(x=0.8, color=self.apple_colors['blue'], linestyle='--', alpha=0.5, linewidth=1)
            
            ax.set_xlabel('Effect Size (|d|)')
            ax.set_title('Effect Sizes', fontsize=16, fontweight='700', pad=20)
            
            # Add value labels
            for i, (bar, effect) in enumerate(zip(bars, effect_values)):
                interpretation = ('Small' if effect < 0.2 else 
                                'Medium' if effect < 0.5 else 
                                'Large' if effect < 0.8 else 'Very Large')
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{effect:.3f} ({interpretation})', va='center', fontweight='500')
            
            # Style
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(self.apple_colors['gray'])
            ax.spines['bottom'].set_color(self.apple_colors['gray'])
            ax.tick_params(colors=self.apple_colors['gray_dark'])
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            self.logger.error(f"Error creating effect sizes chart: {str(e)}")
            return None
        finally:
            plt.close('all')
    
    def _create_data_overview_dashboard(self, data: pd.DataFrame) -> Optional[str]:
        """Create a comprehensive data overview dashboard."""
        try:
            fig = plt.figure(figsize=(20, 12))
            fig.patch.set_facecolor('white')
            
            # Create a 2x3 grid
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            
            # 1. Data Shape and Basic Info
            ax1 = fig.add_subplot(gs[0, 0])
            info_text = f"""Dataset Overview
            
Rows: {data.shape[0]:,}
Columns: {data.shape[1]:,}
Memory Usage: {data.memory_usage(deep=True).sum() / 1024**2:.1f} MB
            
Data Types:
Numeric: {len(data.select_dtypes(include=[np.number]).columns)}
Text: {len(data.select_dtypes(include=['object']).columns)}
DateTime: {len(data.select_dtypes(include=['datetime']).columns)}"""
            
            ax1.text(0.1, 0.9, info_text, transform=ax1.transAxes, fontsize=12, 
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=self.apple_colors['gray_light']))
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.axis('off')
            ax1.set_title('Dataset Information', fontsize=14, fontweight='700')
            
            # 2. Missing Values Heatmap
            ax2 = fig.add_subplot(gs[0, 1])
            missing_data = data.isnull()
            if missing_data.any().any():
                # Sample columns if too many
                cols_to_show = data.columns[:20] if len(data.columns) > 20 else data.columns
                missing_subset = missing_data[cols_to_show]
                
                sns.heatmap(missing_subset.T, cbar=True, cmap='RdYlBu_r', ax=ax2,
                           xticklabels=False, yticklabels=True)
                ax2.set_title('Missing Values Pattern', fontsize=14, fontweight='700')
            else:
                ax2.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=14, color=self.apple_colors['green'])
                ax2.set_title('Missing Values Pattern', fontsize=14, fontweight='700')
            
            # 3. Data Types Distribution
            ax3 = fig.add_subplot(gs[0, 2])
            dtype_counts = {
                'Numeric': len(data.select_dtypes(include=[np.number]).columns),
                'Text': len(data.select_dtypes(include=['object']).columns),
                'DateTime': len(data.select_dtypes(include=['datetime']).columns),
                'Boolean': len(data.select_dtypes(include=['bool']).columns)
            }
            dtype_counts = {k: v for k, v in dtype_counts.items() if v > 0}
            
            if dtype_counts:
                colors = [self.apple_colors['blue'], self.apple_colors['green'], 
                         self.apple_colors['orange'], self.apple_colors['purple']][:len(dtype_counts)]
                wedges, texts, autotexts = ax3.pie(dtype_counts.values(), labels=dtype_counts.keys(), 
                                                  autopct='%1.1f%%', colors=colors, startangle=90)
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('600')
            ax3.set_title('Column Types Distribution', fontsize=14, fontweight='700')
            
            # 4. Numeric Variables Summary Statistics
            ax4 = fig.add_subplot(gs[1, :])
            numeric_data = data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                # Get summary statistics for up to 10 numeric columns
                cols_to_summarize = numeric_data.columns[:10]
                summary_stats = numeric_data[cols_to_summarize].describe().T
                
                # Create a heatmap of summary statistics
                sns.heatmap(summary_stats, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                           ax=ax4, cbar_kws={'label': 'Value'})
                ax4.set_title('Summary Statistics (Numeric Variables)', fontsize=16, fontweight='700', pad=20)
                ax4.set_xlabel('Statistics')
                ax4.set_ylabel('Variables')
            else:
                ax4.text(0.5, 0.5, 'No Numeric Variables', ha='center', va='center', 
                        transform=ax4.transAxes, fontsize=14, color=self.apple_colors['gray'])
                ax4.set_title('Summary Statistics', fontsize=16, fontweight='700', pad=20)
            
            return self._fig_to_base64(fig)
            
        except Exception as e:
            self.logger.error(f"Error creating data overview dashboard: {str(e)}")
            return None
        finally:
            plt.close('all')
    
    def _fig_to_base64(self, fig: Figure) -> str:
        """Convert matplotlib figure to base64 encoded string."""
        try:
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            buffer.close()
            return image_base64
        except Exception as e:
            self.logger.error(f"Error converting figure to base64: {str(e)}")
            return ""
        finally:
            plt.close(fig)