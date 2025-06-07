"""Results viewer for PyADAP 3.0

This module provides a comprehensive results viewer for displaying
statistical analysis results, plots, and reports.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import webbrowser
import tempfile

from ..utils import Logger, get_logger, format_number, format_p_value


class ResultsViewer:
    """Comprehensive results viewer for statistical analysis results."""
    
    def __init__(self, parent: tk.Widget):
        """Initialize the results viewer.
        
        Args:
            parent: Parent widget
        """
        self.parent = parent
        self.results: Optional[Dict[str, Any]] = None
        self.logger = get_logger("PyADAP.ResultsViewer")
        
        # Create main frame
        self.main_frame = ttk.Frame(parent)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create UI
        self._create_ui()
    
    def _create_ui(self) -> None:
        """Create the results viewer UI."""
        # Create toolbar
        self._create_toolbar()
        
        # Create main content area
        self._create_content_area()
        
        # Create status bar
        self._create_status_bar()
    
    def _create_toolbar(self) -> None:
        """Create the toolbar."""
        toolbar_frame = ttk.Frame(self.main_frame)
        toolbar_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # File operations
        ttk.Button(toolbar_frame, text="Open Results", command=self._open_results).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="Save Results", command=self._save_results).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="Export Report", command=self._export_report).pack(side=tk.LEFT, padx=2)
        
        # Separator
        ttk.Separator(toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # View options
        ttk.Button(toolbar_frame, text="Refresh", command=self._refresh_view).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="Clear", command=self._clear_results).pack(side=tk.LEFT, padx=2)
        
        # Help
        ttk.Button(toolbar_frame, text="Help", command=self._show_help).pack(side=tk.RIGHT, padx=2)
    
    def _create_content_area(self) -> None:
        """Create the main content area."""
        # Create notebook for different result views
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self._create_summary_tab()
        self._create_statistics_tab()
        self._create_plots_tab()
        self._create_assumptions_tab()
        self._create_data_tab()
        self._create_raw_tab()
    
    def _create_summary_tab(self) -> None:
        """Create the summary tab."""
        self.summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_frame, text="Summary")
        
        # Create scrollable text widget
        text_frame = ttk.Frame(self.summary_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.summary_text = tk.Text(text_frame, wrap=tk.WORD, state=tk.DISABLED, font=('Consolas', 10))
        summary_scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scroll.set)
        
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        summary_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _create_statistics_tab(self) -> None:
        """Create the statistics tab."""
        self.stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_frame, text="Statistics")
        
        # Create treeview for statistical results
        tree_frame = ttk.Frame(self.stats_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Statistics treeview
        self.stats_tree = ttk.Treeview(tree_frame, show="tree headings")
        
        # Scrollbars
        stats_v_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.stats_tree.yview)
        self.stats_tree.configure(yscrollcommand=stats_v_scroll.set)
        
        stats_h_scroll = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.stats_tree.xview)
        self.stats_tree.configure(xscrollcommand=stats_h_scroll.set)
        
        # Pack scrollbars and treeview
        stats_v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        stats_h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.stats_tree.pack(fill=tk.BOTH, expand=True)
        
        # Context menu for statistics
        self.stats_menu = tk.Menu(self.stats_tree, tearoff=0)
        self.stats_menu.add_command(label="Copy", command=self._copy_stats_selection)
        self.stats_menu.add_command(label="Export to CSV", command=self._export_stats_csv)
        
        self.stats_tree.bind("<Button-3>", self._show_stats_context_menu)
    
    def _create_plots_tab(self) -> None:
        """Create the plots tab."""
        self.plots_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plots_frame, text="Plots")
        
        # Plot selection frame
        plot_control_frame = ttk.Frame(self.plots_frame)
        plot_control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(plot_control_frame, text="Select Plot:").pack(side=tk.LEFT, padx=5)
        
        self.plot_var = tk.StringVar()
        self.plot_combo = ttk.Combobox(plot_control_frame, textvariable=self.plot_var, state="readonly")
        self.plot_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.plot_combo.bind("<<ComboboxSelected>>", self._on_plot_selected)
        
        # Plot control buttons
        ttk.Button(plot_control_frame, text="Save Plot", command=self._save_current_plot).pack(side=tk.RIGHT, padx=2)
        ttk.Button(plot_control_frame, text="Refresh", command=self._refresh_current_plot).pack(side=tk.RIGHT, padx=2)
        
        # Plot display frame
        self.plot_display_frame = ttk.Frame(self.plots_frame)
        self.plot_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Initialize with empty plot
        self._create_empty_plot()
    
    def _create_assumptions_tab(self) -> None:
        """Create the assumptions checking tab."""
        self.assumptions_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.assumptions_frame, text="Assumptions")
        
        # Create treeview for assumption results
        assumptions_tree_frame = ttk.Frame(self.assumptions_frame)
        assumptions_tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.assumptions_tree = ttk.Treeview(assumptions_tree_frame, 
                                           columns=("Test", "Statistic", "P-value", "Result", "Interpretation"), 
                                           show="headings")
        
        # Configure columns
        self.assumptions_tree.heading("Test", text="Test")
        self.assumptions_tree.heading("Statistic", text="Statistic")
        self.assumptions_tree.heading("P-value", text="P-value")
        self.assumptions_tree.heading("Result", text="Result")
        self.assumptions_tree.heading("Interpretation", text="Interpretation")
        
        self.assumptions_tree.column("Test", width=150)
        self.assumptions_tree.column("Statistic", width=100)
        self.assumptions_tree.column("P-value", width=100)
        self.assumptions_tree.column("Result", width=100)
        self.assumptions_tree.column("Interpretation", width=200)
        
        # Scrollbar for assumptions
        assumptions_scroll = ttk.Scrollbar(assumptions_tree_frame, orient=tk.VERTICAL, command=self.assumptions_tree.yview)
        self.assumptions_tree.configure(yscrollcommand=assumptions_scroll.set)
        
        assumptions_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.assumptions_tree.pack(fill=tk.BOTH, expand=True)
    
    def _create_data_tab(self) -> None:
        """Create the data tab for processed data."""
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="Data")
        
        # Data control frame
        data_control_frame = ttk.Frame(self.data_frame)
        data_control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(data_control_frame, text="Data View:").pack(side=tk.LEFT, padx=5)
        
        self.data_view_var = tk.StringVar(value="Original")
        data_view_combo = ttk.Combobox(data_control_frame, textvariable=self.data_view_var,
                                     values=["Original", "Processed", "Residuals"], state="readonly")
        data_view_combo.pack(side=tk.LEFT, padx=5)
        data_view_combo.bind("<<ComboboxSelected>>", self._on_data_view_changed)
        
        # Export button
        ttk.Button(data_control_frame, text="Export Data", command=self._export_data).pack(side=tk.RIGHT, padx=2)
        
        # Data display frame
        data_display_frame = ttk.Frame(self.data_frame)
        data_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Data treeview
        self.data_tree = ttk.Treeview(data_display_frame, show="tree headings")
        
        # Scrollbars for data
        data_v_scroll = ttk.Scrollbar(data_display_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=data_v_scroll.set)
        
        data_h_scroll = ttk.Scrollbar(data_display_frame, orient=tk.HORIZONTAL, command=self.data_tree.xview)
        self.data_tree.configure(xscrollcommand=data_h_scroll.set)
        
        # Pack scrollbars and treeview
        data_v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        data_h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.data_tree.pack(fill=tk.BOTH, expand=True)
    
    def _create_raw_tab(self) -> None:
        """Create the raw results tab."""
        self.raw_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.raw_frame, text="Raw Results")
        
        # Create text widget for raw JSON
        raw_text_frame = ttk.Frame(self.raw_frame)
        raw_text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.raw_text = tk.Text(raw_text_frame, wrap=tk.NONE, state=tk.DISABLED, font=('Consolas', 9))
        
        # Scrollbars for raw text
        raw_v_scroll = ttk.Scrollbar(raw_text_frame, orient=tk.VERTICAL, command=self.raw_text.yview)
        self.raw_text.configure(yscrollcommand=raw_v_scroll.set)
        
        raw_h_scroll = ttk.Scrollbar(raw_text_frame, orient=tk.HORIZONTAL, command=self.raw_text.xview)
        self.raw_text.configure(xscrollcommand=raw_h_scroll.set)
        
        # Pack scrollbars and text
        raw_v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        raw_h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.raw_text.pack(fill=tk.BOTH, expand=True)
    
    def _create_status_bar(self) -> None:
        """Create the status bar."""
        self.status_frame = ttk.Frame(self.main_frame)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_var = tk.StringVar(value="No results loaded")
        ttk.Label(self.status_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=5, pady=2)
        
        # Progress bar for operations
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.status_frame, variable=self.progress_var, mode='determinate')
        self.progress_bar.pack(side=tk.RIGHT, padx=5, pady=2, fill=tk.X, expand=True)
        self.progress_bar.pack_forget()  # Hide initially
    
    def _create_empty_plot(self) -> None:
        """Create an empty plot placeholder."""
        # Clear existing plot
        for widget in self.plot_display_frame.winfo_children():
            widget.destroy()
        
        # Create empty figure
        self.current_figure = Figure(figsize=(8, 6), dpi=100)
        self.current_axes = self.current_figure.add_subplot(111)
        self.current_axes.text(0.5, 0.5, 'No plot available', 
                              horizontalalignment='center', verticalalignment='center',
                              transform=self.current_axes.transAxes, fontsize=16, color='gray')
        self.current_axes.set_xticks([])
        self.current_axes.set_yticks([])
        
        # Create canvas
        self.plot_canvas = FigureCanvasTkAgg(self.current_figure, self.plot_display_frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create toolbar
        self.plot_toolbar = NavigationToolbar2Tk(self.plot_canvas, self.plot_display_frame)
        self.plot_toolbar.update()
    
    def set_results(self, results: Dict[str, Any]) -> None:
        """Set the results to display.
        
        Args:
            results: Dictionary containing analysis results
        """
        self.results = results
        self._refresh_view()
        self.logger.info("Results loaded into viewer")
    
    def load_results(self, results: Dict[str, Any]) -> None:
        """Load results into the viewer (alias for set_results).
        
        Args:
            results: Dictionary containing analysis results
        """
        self.set_results(results)
    
    def show(self) -> None:
        """Show the results viewer window."""
        # Create a new top-level window
        self.window = tk.Toplevel(self.parent)
        self.window.title("PyADAP Results Viewer")
        self.window.geometry("1200x800")
        
        # Move the main frame to the new window
        self.main_frame.pack_forget()
        self.main_frame = ttk.Frame(self.window)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Recreate UI in the new window
        self._create_ui()
        
        # Refresh view if results are already loaded
        if self.results:
            self._refresh_view()
    
    def _refresh_view(self) -> None:
        """Refresh all views with current results."""
        if self.results is None:
            self._clear_results()
            return
        
        try:
            self._update_summary_view()
            self._update_statistics_view()
            self._update_plots_view()
            self._update_assumptions_view()
            self._update_data_view()
            self._update_raw_view()
            
            self.status_var.set("Results loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error refreshing results view: {str(e)}")
            messagebox.showerror("Error", f"Failed to refresh results view: {str(e)}")
    
    def _update_summary_view(self) -> None:
        """Update the summary view."""
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        
        if 'summary' in self.results:
            self.summary_text.insert(tk.END, self.results['summary'])
        else:
            # Generate summary from available results
            summary_lines = []
            summary_lines.append("PyADAP Analysis Results")
            summary_lines.append("=" * 50)
            summary_lines.append("")
            
            # Basic info
            if 'data_info' in self.results:
                info = self.results['data_info']
                summary_lines.append(f"Dataset: {info.get('rows', 'N/A')} rows × {info.get('columns', 'N/A')} columns")
                summary_lines.append("")
            
            # Statistical tests performed
            if 'statistical_tests' in self.results:
                summary_lines.append("Statistical Tests Performed:")
                summary_lines.append("-" * 30)
                for test_name, test_result in self.results['statistical_tests'].items():
                    if isinstance(test_result, dict) and 'p_value' in test_result:
                        p_val = test_result['p_value']
                        summary_lines.append(f"{test_name}: p = {format_p_value(p_val)}")
                summary_lines.append("")
            
            # Key findings
            if 'key_findings' in self.results:
                summary_lines.append("Key Findings:")
                summary_lines.append("-" * 15)
                for finding in self.results['key_findings']:
                    summary_lines.append(f"• {finding}")
                summary_lines.append("")
            
            # Recommendations
            if 'recommendations' in self.results:
                summary_lines.append("Recommendations:")
                summary_lines.append("-" * 17)
                for rec in self.results['recommendations']:
                    summary_lines.append(f"• {rec}")
            
            self.summary_text.insert(tk.END, "\n".join(summary_lines))
        
        self.summary_text.config(state=tk.DISABLED)
    
    def _update_statistics_view(self) -> None:
        """Update the statistics view."""
        # Clear existing items
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)
        
        if 'statistical_tests' not in self.results:
            return
        
        # Set up columns
        columns = ["Test", "Statistic", "P-value", "Effect Size", "Confidence Interval", "Interpretation"]
        self.stats_tree["columns"] = columns
        self.stats_tree.column("#0", width=0, stretch=False)  # Hide tree column
        
        for col in columns:
            self.stats_tree.heading(col, text=col)
            self.stats_tree.column(col, width=120, anchor=tk.W)
        
        # Add statistical test results
        for test_name, test_result in self.results['statistical_tests'].items():
            if isinstance(test_result, dict):
                statistic = format_number(test_result.get('statistic', 'N/A'))
                p_value = format_p_value(test_result.get('p_value', 'N/A'))
                effect_size = format_number(test_result.get('effect_size', 'N/A'))
                
                # Format confidence interval
                ci = test_result.get('confidence_interval')
                if ci and isinstance(ci, (list, tuple)) and len(ci) == 2:
                    ci_str = f"[{format_number(ci[0])}, {format_number(ci[1])}]"
                else:
                    ci_str = "N/A"
                
                # Interpretation
                interpretation = test_result.get('interpretation', 'N/A')
                
                self.stats_tree.insert("", tk.END, values=(
                    test_name, statistic, p_value, effect_size, ci_str, interpretation
                ))
    
    def _update_plots_view(self) -> None:
        """Update the plots view."""
        # Clear plot combo
        self.plot_combo['values'] = ()
        self.plot_var.set('')
        
        if 'plots' in self.results and self.results['plots']:
            plot_names = list(self.results['plots'].keys())
            self.plot_combo['values'] = plot_names
            if plot_names:
                self.plot_var.set(plot_names[0])
                self._display_plot(plot_names[0])
        else:
            self._create_empty_plot()
    
    def _update_assumptions_view(self) -> None:
        """Update the assumptions view."""
        # Clear existing items
        for item in self.assumptions_tree.get_children():
            self.assumptions_tree.delete(item)
        
        if 'assumption_tests' not in self.results:
            return
        
        # Add assumption test results
        for test_name, test_result in self.results['assumption_tests'].items():
            if isinstance(test_result, dict):
                statistic = format_number(test_result.get('statistic', 'N/A'))
                p_value = format_p_value(test_result.get('p_value', 'N/A'))
                
                # Determine result
                alpha = 0.05  # Default alpha
                if 'alpha' in self.results:
                    alpha = self.results['alpha']
                
                if isinstance(test_result.get('p_value'), (int, float)):
                    result = "Violated" if test_result['p_value'] < alpha else "Met"
                else:
                    result = "Unknown"
                
                interpretation = test_result.get('interpretation', 'N/A')
                
                self.assumptions_tree.insert("", tk.END, values=(
                    test_name, statistic, p_value, result, interpretation
                ))
    
    def _update_data_view(self) -> None:
        """Update the data view."""
        self._display_data_view()
    
    def _update_raw_view(self) -> None:
        """Update the raw results view."""
        self.raw_text.config(state=tk.NORMAL)
        self.raw_text.delete(1.0, tk.END)
        
        if self.results:
            try:
                # Pretty print JSON
                json_str = json.dumps(self.results, indent=2, default=str)
                self.raw_text.insert(tk.END, json_str)
            except Exception as e:
                self.raw_text.insert(tk.END, f"Error formatting results: {str(e)}")
        
        self.raw_text.config(state=tk.DISABLED)
    
    def _display_data_view(self) -> None:
        """Display data based on current view selection."""
        # Clear existing data
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        view_type = self.data_view_var.get()
        data_key = {
            "Original": "original_data",
            "Processed": "processed_data", 
            "Residuals": "residuals"
        }.get(view_type)
        
        if not data_key or data_key not in self.results:
            return
        
        data = self.results[data_key]
        if not isinstance(data, pd.DataFrame):
            return
        
        # Set up columns
        columns = ["Index"] + list(data.columns)
        self.data_tree["columns"] = columns
        self.data_tree.column("#0", width=0, stretch=False)
        
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100, anchor=tk.W)
        
        # Add data (limit to first 1000 rows for performance)
        display_data = data.head(1000)
        for idx, (row_idx, row) in enumerate(display_data.iterrows()):
            values = [str(row_idx)] + [self._format_cell_value(val) for val in row]
            self.data_tree.insert("", tk.END, values=values)
    
    def _format_cell_value(self, value: Any) -> str:
        """Format a cell value for display.
        
        Args:
            value: Value to format
            
        Returns:
            Formatted string
        """
        if pd.isnull(value):
            return "NaN"
        
        if isinstance(value, (int, float)):
            if isinstance(value, float) and abs(value) < 1e-3:
                return f"{value:.2e}"
            elif isinstance(value, float):
                return f"{value:.4f}"
            else:
                return str(value)
        
        # Truncate long strings
        str_val = str(value)
        if len(str_val) > 30:
            return str_val[:27] + "..."
        
        return str_val
    
    def _display_plot(self, plot_name: str) -> None:
        """Display a specific plot.
        
        Args:
            plot_name: Name of the plot to display
        """
        if 'plots' not in self.results or plot_name not in self.results['plots']:
            self._create_empty_plot()
            return
        
        try:
            plot_data = self.results['plots'][plot_name]
            
            # Clear existing plot
            for widget in self.plot_display_frame.winfo_children():
                widget.destroy()
            
            # Create new figure
            self.current_figure = Figure(figsize=(8, 6), dpi=100)
            
            # If plot_data is a matplotlib figure, copy it
            if hasattr(plot_data, 'axes'):
                # Copy the plot
                for i, ax in enumerate(plot_data.axes):
                    new_ax = self.current_figure.add_subplot(len(plot_data.axes), 1, i+1)
                    # Copy basic properties
                    new_ax.set_title(ax.get_title())
                    new_ax.set_xlabel(ax.get_xlabel())
                    new_ax.set_ylabel(ax.get_ylabel())
                    
                    # Copy plot elements (simplified)
                    for line in ax.get_lines():
                        new_ax.plot(line.get_xdata(), line.get_ydata(), 
                                   color=line.get_color(), label=line.get_label())
                    
                    if ax.get_legend():
                        new_ax.legend()
            else:
                # Create a simple plot if data format is not recognized
                ax = self.current_figure.add_subplot(111)
                ax.text(0.5, 0.5, f'Plot: {plot_name}', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14)
            
            self.current_figure.tight_layout()
            
            # Create canvas
            self.plot_canvas = FigureCanvasTkAgg(self.current_figure, self.plot_display_frame)
            self.plot_canvas.draw()
            self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Create toolbar
            self.plot_toolbar = NavigationToolbar2Tk(self.plot_canvas, self.plot_display_frame)
            self.plot_toolbar.update()
            
        except Exception as e:
            self.logger.error(f"Error displaying plot {plot_name}: {str(e)}")
            self._create_empty_plot()
    
    def _on_plot_selected(self, event=None) -> None:
        """Handle plot selection change."""
        plot_name = self.plot_var.get()
        if plot_name:
            self._display_plot(plot_name)
    
    def _on_data_view_changed(self, event=None) -> None:
        """Handle data view change."""
        self._display_data_view()
    
    def _clear_results(self) -> None:
        """Clear all results."""
        self.results = None
        
        # Clear all views
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.config(state=tk.DISABLED)
        
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)
        
        for item in self.assumptions_tree.get_children():
            self.assumptions_tree.delete(item)
        
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        self.raw_text.config(state=tk.NORMAL)
        self.raw_text.delete(1.0, tk.END)
        self.raw_text.config(state=tk.DISABLED)
        
        self.plot_combo['values'] = ()
        self.plot_var.set('')
        self._create_empty_plot()
        
        self.status_var.set("No results loaded")
    
    def _open_results(self) -> None:
        """Open results from file."""
        filetypes = [
            ("JSON files", "*.json"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Open Results",
            filetypes=filetypes
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                self.set_results(results)
                self.status_var.set(f"Results loaded from {Path(filename).name}")
                
            except Exception as e:
                self.logger.error(f"Error opening results: {str(e)}")
                messagebox.showerror("Error", f"Failed to open results: {str(e)}")
    
    def _save_results(self) -> None:
        """Save current results to file."""
        if self.results is None:
            messagebox.showwarning("No Results", "No results to save.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.results, f, indent=2, default=str)
                
                self.status_var.set(f"Results saved to {Path(filename).name}")
                messagebox.showinfo("Success", "Results saved successfully.")
                
            except Exception as e:
                self.logger.error(f"Error saving results: {str(e)}")
                messagebox.showerror("Error", f"Failed to save results: {str(e)}")
    
    def _export_report(self) -> None:
        """Export results as HTML report."""
        if self.results is None:
            messagebox.showwarning("No Results", "No results to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Report",
            defaultextension=".html",
            filetypes=[("HTML files", "*.html"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self._generate_html_report(filename)
                self.status_var.set(f"Report exported to {Path(filename).name}")
                messagebox.showinfo("Success", "Report exported successfully.")
                
                # Ask if user wants to open the report
                if messagebox.askyesno("Open Report", "Would you like to open the report in your browser?"):
                    webbrowser.open(f"file://{Path(filename).absolute()}")
                
            except Exception as e:
                self.logger.error(f"Error exporting report: {str(e)}")
                messagebox.showerror("Error", f"Failed to export report: {str(e)}")
    
    def _generate_html_report(self, filename: str) -> None:
        """Generate HTML report.
        
        Args:
            filename: Output filename
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PyADAP Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #f9f9f9; padding: 20px; border-radius: 5px; }}
                .significant {{ color: #d9534f; font-weight: bold; }}
                .not-significant {{ color: #5cb85c; }}
            </style>
        </head>
        <body>
            <h1>PyADAP Analysis Report</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <pre>{self._get_summary_text()}</pre>
            </div>
            
            <h2>Statistical Tests</h2>
            {self._generate_stats_table()}
            
            <h2>Assumption Tests</h2>
            {self._generate_assumptions_table()}
            
        </body>
        </html>
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _get_summary_text(self) -> str:
        """Get summary text for HTML report."""
        self.summary_text.config(state=tk.NORMAL)
        summary = self.summary_text.get(1.0, tk.END)
        self.summary_text.config(state=tk.DISABLED)
        return summary
    
    def _generate_stats_table(self) -> str:
        """Generate HTML table for statistical tests."""
        if 'statistical_tests' not in self.results:
            return "<p>No statistical tests performed.</p>"
        
        html = "<table><tr><th>Test</th><th>Statistic</th><th>P-value</th><th>Effect Size</th><th>Interpretation</th></tr>"
        
        for test_name, test_result in self.results['statistical_tests'].items():
            if isinstance(test_result, dict):
                statistic = format_number(test_result.get('statistic', 'N/A'))
                p_value = test_result.get('p_value', 'N/A')
                p_value_str = format_p_value(p_value)
                effect_size = format_number(test_result.get('effect_size', 'N/A'))
                interpretation = test_result.get('interpretation', 'N/A')
                
                # Color code p-values
                p_class = "significant" if isinstance(p_value, (int, float)) and p_value < 0.05 else "not-significant"
                
                html += f"<tr><td>{test_name}</td><td>{statistic}</td><td class='{p_class}'>{p_value_str}</td><td>{effect_size}</td><td>{interpretation}</td></tr>"
        
        html += "</table>"
        return html
    
    def _generate_assumptions_table(self) -> str:
        """Generate HTML table for assumption tests."""
        if 'assumption_tests' not in self.results:
            return "<p>No assumption tests performed.</p>"
        
        html = "<table><tr><th>Test</th><th>Statistic</th><th>P-value</th><th>Result</th><th>Interpretation</th></tr>"
        
        for test_name, test_result in self.results['assumption_tests'].items():
            if isinstance(test_result, dict):
                statistic = format_number(test_result.get('statistic', 'N/A'))
                p_value = test_result.get('p_value', 'N/A')
                p_value_str = format_p_value(p_value)
                
                # Determine result
                alpha = 0.05
                if isinstance(p_value, (int, float)):
                    result = "Violated" if p_value < alpha else "Met"
                    result_class = "significant" if p_value < alpha else "not-significant"
                else:
                    result = "Unknown"
                    result_class = ""
                
                interpretation = test_result.get('interpretation', 'N/A')
                
                html += f"<tr><td>{test_name}</td><td>{statistic}</td><td>{p_value_str}</td><td class='{result_class}'>{result}</td><td>{interpretation}</td></tr>"
        
        html += "</table>"
        return html
    
    def _save_current_plot(self) -> None:
        """Save the currently displayed plot."""
        if not hasattr(self, 'current_figure'):
            messagebox.showwarning("No Plot", "No plot to save.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Plot",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg")]
        )
        
        if filename:
            try:
                self.current_figure.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", "Plot saved successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save plot: {str(e)}")
    
    def _refresh_current_plot(self) -> None:
        """Refresh the current plot."""
        plot_name = self.plot_var.get()
        if plot_name:
            self._display_plot(plot_name)
    
    def _copy_stats_selection(self) -> None:
        """Copy selected statistics to clipboard."""
        selection = self.stats_tree.selection()
        if not selection:
            return
        
        # Get selected data
        data = []
        for item in selection:
            values = self.stats_tree.item(item, 'values')
            data.append('\t'.join(str(v) for v in values))
        
        # Copy to clipboard
        self.stats_tree.clipboard_clear()
        self.stats_tree.clipboard_append('\n'.join(data))
    
    def _export_stats_csv(self) -> None:
        """Export statistics to CSV."""
        if 'statistical_tests' not in self.results:
            messagebox.showwarning("No Data", "No statistical tests to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Statistics",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if filename:
            try:
                # Create DataFrame from statistical tests
                data = []
                for test_name, test_result in self.results['statistical_tests'].items():
                    if isinstance(test_result, dict):
                        row = {
                            'Test': test_name,
                            'Statistic': test_result.get('statistic', 'N/A'),
                            'P-value': test_result.get('p_value', 'N/A'),
                            'Effect Size': test_result.get('effect_size', 'N/A'),
                            'Interpretation': test_result.get('interpretation', 'N/A')
                        }
                        data.append(row)
                
                df = pd.DataFrame(data)
                df.to_csv(filename, index=False)
                messagebox.showinfo("Success", "Statistics exported successfully.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export statistics: {str(e)}")
    
    def _export_data(self) -> None:
        """Export current data view."""
        view_type = self.data_view_var.get()
        data_key = {
            "Original": "original_data",
            "Processed": "processed_data", 
            "Residuals": "residuals"
        }.get(view_type)
        
        if not data_key or data_key not in self.results:
            messagebox.showwarning("No Data", f"No {view_type.lower()} data to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            title=f"Export {view_type} Data",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")]
        )
        
        if filename:
            try:
                data = self.results[data_key]
                if filename.endswith('.xlsx'):
                    data.to_excel(filename, index=True)
                else:
                    data.to_csv(filename, index=True)
                
                messagebox.showinfo("Success", f"{view_type} data exported successfully.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export data: {str(e)}")
    
    def _show_stats_context_menu(self, event) -> None:
        """Show context menu for statistics tree."""
        try:
            self.stats_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.stats_menu.grab_release()
    
    def _show_help(self) -> None:
        """Show help dialog."""
        help_text = """
PyADAP Results Viewer Help

Tabs:
• Summary: Overview of analysis results
• Statistics: Detailed statistical test results
• Plots: Generated visualizations
• Assumptions: Assumption test results
• Data: Original and processed data
• Raw Results: Complete results in JSON format

Features:
• Export results to various formats
• Save individual plots
• Copy statistics to clipboard
• Generate HTML reports

Tips:
• Right-click on statistics for context menu
• Use the toolbar buttons for quick actions
• Switch between data views in the Data tab
        """
        
        messagebox.showinfo("Help", help_text)