"""Configuration dialog for PyADAP 3.0

This module provides a comprehensive configuration dialog for adjusting
analysis parameters, data processing options, and visualization settings.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path

from ..config import Config
from ..utils import get_logger


class ConfigDialog:
    """Configuration dialog for PyADAP settings."""
    
    def __init__(self, parent: tk.Tk, config: Config):
        """Initialize the configuration dialog.
        
        Args:
            parent: Parent window
            config: Current configuration object
        """
        self.parent = parent
        self.config = config.copy()  # Work with a copy
        self.result = None
        self.logger = get_logger("PyADAP.ConfigDialog")
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("PyADAP Configuration")
        self.dialog.geometry("600x700")
        self.dialog.resizable(True, True)
        
        # Make dialog modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self._center_dialog()
        
        # Create UI
        self._create_ui()
        
        # Bind events
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_cancel)
        
        # Wait for dialog to close
        self.dialog.wait_window()
    
    def _center_dialog(self) -> None:
        """Center the dialog on the parent window."""
        self.dialog.update_idletasks()
        
        # Get parent window position and size
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        # Get dialog size
        dialog_width = self.dialog.winfo_reqwidth()
        dialog_height = self.dialog.winfo_reqheight()
        
        # Calculate center position
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        self.dialog.geometry(f"+{x}+{y}")
    
    def _create_ui(self) -> None:
        """Create the dialog UI."""
        # Create main frame with scrollbar
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for different configuration sections
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create configuration tabs
        self._create_statistical_tab()
        self._create_data_tab()
        self._create_visualization_tab()
        self._create_output_tab()
        
        # Create button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Buttons
        ttk.Button(button_frame, text="OK", command=self._on_ok).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="Cancel", command=self._on_cancel).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="Apply", command=self._on_apply).pack(side=tk.RIGHT, padx=(0, 5))
        ttk.Button(button_frame, text="Reset to Defaults", command=self._on_reset).pack(side=tk.LEFT)
        
        # Load current configuration
        self._load_config()
    
    def _create_statistical_tab(self) -> None:
        """Create the statistical configuration tab."""
        self.stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_frame, text="Statistical Analysis")
        
        # Create scrollable frame
        canvas = tk.Canvas(self.stats_frame)
        scrollbar = ttk.Scrollbar(self.stats_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Statistical parameters
        params_frame = ttk.LabelFrame(scrollable_frame, text="Statistical Parameters")
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Alpha level
        alpha_frame = ttk.Frame(params_frame)
        alpha_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(alpha_frame, text="Significance level (α):").pack(side=tk.LEFT)
        self.alpha_var = tk.DoubleVar()
        ttk.Spinbox(alpha_frame, from_=0.001, to=0.1, increment=0.001, 
                   textvariable=self.alpha_var, width=10).pack(side=tk.RIGHT)
        
        # Power
        power_frame = ttk.Frame(params_frame)
        power_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(power_frame, text="Desired power:").pack(side=tk.LEFT)
        self.power_var = tk.DoubleVar()
        ttk.Spinbox(power_frame, from_=0.5, to=0.99, increment=0.01, 
                   textvariable=self.power_var, width=10).pack(side=tk.RIGHT)
        
        # Effect size threshold
        effect_frame = ttk.Frame(params_frame)
        effect_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(effect_frame, text="Minimum effect size:").pack(side=tk.LEFT)
        self.effect_size_var = tk.DoubleVar()
        ttk.Spinbox(effect_frame, from_=0.1, to=2.0, increment=0.1, 
                   textvariable=self.effect_size_var, width=10).pack(side=tk.RIGHT)
        
        # Test selection options
        test_frame = ttk.LabelFrame(scrollable_frame, text="Test Selection")
        test_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.auto_test_var = tk.BooleanVar()
        ttk.Checkbutton(test_frame, text="Automatic test selection", 
                       variable=self.auto_test_var).pack(anchor=tk.W, padx=5, pady=2)
        
        self.prefer_nonparametric_var = tk.BooleanVar()
        ttk.Checkbutton(test_frame, text="Prefer non-parametric tests", 
                       variable=self.prefer_nonparametric_var).pack(anchor=tk.W, padx=5, pady=2)
        
        self.robust_tests_var = tk.BooleanVar()
        ttk.Checkbutton(test_frame, text="Use robust statistical tests", 
                       variable=self.robust_tests_var).pack(anchor=tk.W, padx=5, pady=2)
        
        # Multiple comparisons
        mc_frame = ttk.LabelFrame(scrollable_frame, text="Multiple Comparisons")
        mc_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(mc_frame, text="Correction method:").pack(anchor=tk.W, padx=5, pady=2)
        self.mc_method_var = tk.StringVar()
        mc_combo = ttk.Combobox(mc_frame, textvariable=self.mc_method_var, 
                               values=["bonferroni", "holm", "fdr_bh", "fdr_by", "none"],
                               state="readonly")
        mc_combo.pack(fill=tk.X, padx=5, pady=2)
        
        # Assumption checking
        assumption_frame = ttk.LabelFrame(scrollable_frame, text="Assumption Checking")
        assumption_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.check_normality_var = tk.BooleanVar()
        ttk.Checkbutton(assumption_frame, text="Check normality", 
                       variable=self.check_normality_var).pack(anchor=tk.W, padx=5, pady=2)
        
        self.check_homogeneity_var = tk.BooleanVar()
        ttk.Checkbutton(assumption_frame, text="Check homogeneity of variance", 
                       variable=self.check_homogeneity_var).pack(anchor=tk.W, padx=5, pady=2)
        
        self.check_independence_var = tk.BooleanVar()
        ttk.Checkbutton(assumption_frame, text="Check independence", 
                       variable=self.check_independence_var).pack(anchor=tk.W, padx=5, pady=2)
        
        # Bootstrap options
        bootstrap_frame = ttk.LabelFrame(scrollable_frame, text="Bootstrap Options")
        bootstrap_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.use_bootstrap_var = tk.BooleanVar()
        ttk.Checkbutton(bootstrap_frame, text="Use bootstrap confidence intervals", 
                       variable=self.use_bootstrap_var).pack(anchor=tk.W, padx=5, pady=2)
        
        bootstrap_n_frame = ttk.Frame(bootstrap_frame)
        bootstrap_n_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(bootstrap_n_frame, text="Bootstrap samples:").pack(side=tk.LEFT)
        self.bootstrap_n_var = tk.IntVar()
        ttk.Spinbox(bootstrap_n_frame, from_=100, to=10000, increment=100, 
                   textvariable=self.bootstrap_n_var, width=10).pack(side=tk.RIGHT)
    
    def _create_data_tab(self) -> None:
        """Create the data processing configuration tab."""
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="Data Processing")
        
        # Missing data handling
        missing_frame = ttk.LabelFrame(self.data_frame, text="Missing Data Handling")
        missing_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(missing_frame, text="Missing data strategy:").pack(anchor=tk.W, padx=5, pady=2)
        self.missing_strategy_var = tk.StringVar()
        missing_combo = ttk.Combobox(missing_frame, textvariable=self.missing_strategy_var,
                                   values=["drop", "mean", "median", "mode", "forward_fill", "backward_fill"],
                                   state="readonly")
        missing_combo.pack(fill=tk.X, padx=5, pady=2)
        
        missing_threshold_frame = ttk.Frame(missing_frame)
        missing_threshold_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(missing_threshold_frame, text="Missing data threshold:").pack(side=tk.LEFT)
        self.missing_threshold_var = tk.DoubleVar()
        ttk.Spinbox(missing_threshold_frame, from_=0.0, to=1.0, increment=0.05, 
                   textvariable=self.missing_threshold_var, width=10).pack(side=tk.RIGHT)
        
        # Outlier detection
        outlier_frame = ttk.LabelFrame(self.data_frame, text="Outlier Detection")
        outlier_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.detect_outliers_var = tk.BooleanVar()
        ttk.Checkbutton(outlier_frame, text="Enable outlier detection", 
                       variable=self.detect_outliers_var).pack(anchor=tk.W, padx=5, pady=2)
        
        ttk.Label(outlier_frame, text="Detection method:").pack(anchor=tk.W, padx=5, pady=2)
        self.outlier_method_var = tk.StringVar()
        outlier_combo = ttk.Combobox(outlier_frame, textvariable=self.outlier_method_var,
                                   values=["iqr", "zscore", "modified_zscore", "isolation_forest"],
                                   state="readonly")
        outlier_combo.pack(fill=tk.X, padx=5, pady=2)
        
        outlier_threshold_frame = ttk.Frame(outlier_frame)
        outlier_threshold_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(outlier_threshold_frame, text="Outlier threshold:").pack(side=tk.LEFT)
        self.outlier_threshold_var = tk.DoubleVar()
        ttk.Spinbox(outlier_threshold_frame, from_=1.0, to=5.0, increment=0.1, 
                   textvariable=self.outlier_threshold_var, width=10).pack(side=tk.RIGHT)
        
        ttk.Label(outlier_frame, text="Outlier action:").pack(anchor=tk.W, padx=5, pady=2)
        self.outlier_action_var = tk.StringVar()
        action_combo = ttk.Combobox(outlier_frame, textvariable=self.outlier_action_var,
                                  values=["remove", "cap", "transform", "flag"],
                                  state="readonly")
        action_combo.pack(fill=tk.X, padx=5, pady=2)
        
        # Data transformations
        transform_frame = ttk.LabelFrame(self.data_frame, text="Data Transformations")
        transform_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.auto_transform_var = tk.BooleanVar()
        ttk.Checkbutton(transform_frame, text="Automatic transformation selection", 
                       variable=self.auto_transform_var).pack(anchor=tk.W, padx=5, pady=2)
        
        ttk.Label(transform_frame, text="Preferred transformation:").pack(anchor=tk.W, padx=5, pady=2)
        self.transform_method_var = tk.StringVar()
        transform_combo = ttk.Combobox(transform_frame, textvariable=self.transform_method_var,
                                     values=["none", "log", "sqrt", "box_cox", "yeo_johnson"],
                                     state="readonly")
        transform_combo.pack(fill=tk.X, padx=5, pady=2)
        
        # Scaling
        scaling_frame = ttk.LabelFrame(self.data_frame, text="Data Scaling")
        scaling_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.auto_scaling_var = tk.BooleanVar()
        ttk.Checkbutton(scaling_frame, text="Automatic scaling", 
                       variable=self.auto_scaling_var).pack(anchor=tk.W, padx=5, pady=2)
        
        ttk.Label(scaling_frame, text="Scaling method:").pack(anchor=tk.W, padx=5, pady=2)
        self.scaling_method_var = tk.StringVar()
        scaling_combo = ttk.Combobox(scaling_frame, textvariable=self.scaling_method_var,
                                   values=["none", "standard", "minmax", "robust"],
                                   state="readonly")
        scaling_combo.pack(fill=tk.X, padx=5, pady=2)
    
    def _create_visualization_tab(self) -> None:
        """Create the visualization configuration tab."""
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="Visualization")
        
        # Plot settings
        plot_frame = ttk.LabelFrame(self.viz_frame, text="Plot Settings")
        plot_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Figure size
        figsize_frame = ttk.Frame(plot_frame)
        figsize_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(figsize_frame, text="Figure width:").pack(side=tk.LEFT)
        self.fig_width_var = tk.DoubleVar()
        ttk.Spinbox(figsize_frame, from_=4, to=20, increment=0.5, 
                   textvariable=self.fig_width_var, width=8).pack(side=tk.LEFT, padx=(5, 10))
        ttk.Label(figsize_frame, text="Height:").pack(side=tk.LEFT)
        self.fig_height_var = tk.DoubleVar()
        ttk.Spinbox(figsize_frame, from_=3, to=15, increment=0.5, 
                   textvariable=self.fig_height_var, width=8).pack(side=tk.RIGHT)
        
        # DPI
        dpi_frame = ttk.Frame(plot_frame)
        dpi_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(dpi_frame, text="DPI (resolution):").pack(side=tk.LEFT)
        self.dpi_var = tk.IntVar()
        ttk.Spinbox(dpi_frame, from_=72, to=300, increment=12, 
                   textvariable=self.dpi_var, width=10).pack(side=tk.RIGHT)
        
        # Style
        style_frame = ttk.Frame(plot_frame)
        style_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(style_frame, text="Plot style:").pack(side=tk.LEFT)
        self.plot_style_var = tk.StringVar()
        style_combo = ttk.Combobox(style_frame, textvariable=self.plot_style_var,
                                 values=["default", "seaborn", "ggplot", "bmh", "classic"],
                                 state="readonly")
        style_combo.pack(side=tk.RIGHT)
        
        # Color palette
        color_frame = ttk.Frame(plot_frame)
        color_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(color_frame, text="Color palette:").pack(side=tk.LEFT)
        self.color_palette_var = tk.StringVar()
        color_combo = ttk.Combobox(color_frame, textvariable=self.color_palette_var,
                                 values=["Set1", "Set2", "tab10", "viridis", "plasma", "husl"],
                                 state="readonly")
        color_combo.pack(side=tk.RIGHT)
        
        # Plot options
        options_frame = ttk.LabelFrame(self.viz_frame, text="Plot Options")
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.show_grid_var = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="Show grid", 
                       variable=self.show_grid_var).pack(anchor=tk.W, padx=5, pady=2)
        
        self.show_legend_var = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="Show legend", 
                       variable=self.show_legend_var).pack(anchor=tk.W, padx=5, pady=2)
        
        self.tight_layout_var = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="Use tight layout", 
                       variable=self.tight_layout_var).pack(anchor=tk.W, padx=5, pady=2)
        
        self.save_plots_var = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="Automatically save plots", 
                       variable=self.save_plots_var).pack(anchor=tk.W, padx=5, pady=2)
        
        # Plot format
        format_frame = ttk.Frame(options_frame)
        format_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(format_frame, text="Save format:").pack(side=tk.LEFT)
        self.plot_format_var = tk.StringVar()
        format_combo = ttk.Combobox(format_frame, textvariable=self.plot_format_var,
                                  values=["png", "pdf", "svg", "eps", "jpg"],
                                  state="readonly")
        format_combo.pack(side=tk.RIGHT)
    
    def _create_output_tab(self) -> None:
        """Create the output configuration tab."""
        self.output_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.output_frame, text="Output")
        
        # Output directory
        dir_frame = ttk.LabelFrame(self.output_frame, text="Output Directory")
        dir_frame.pack(fill=tk.X, padx=5, pady=5)
        
        dir_select_frame = ttk.Frame(dir_frame)
        dir_select_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.output_dir_var = tk.StringVar()
        ttk.Entry(dir_select_frame, textvariable=self.output_dir_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(dir_select_frame, text="Browse...", command=self._browse_output_dir).pack(side=tk.RIGHT, padx=(5, 0))
        
        self.create_subdir_var = tk.BooleanVar()
        ttk.Checkbutton(dir_frame, text="Create subdirectory for each analysis", 
                       variable=self.create_subdir_var).pack(anchor=tk.W, padx=5, pady=2)
        
        # File naming
        naming_frame = ttk.LabelFrame(self.output_frame, text="File Naming")
        naming_frame.pack(fill=tk.X, padx=5, pady=5)
        
        prefix_frame = ttk.Frame(naming_frame)
        prefix_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(prefix_frame, text="File prefix:").pack(side=tk.LEFT)
        self.file_prefix_var = tk.StringVar()
        ttk.Entry(prefix_frame, textvariable=self.file_prefix_var, width=20).pack(side=tk.RIGHT)
        
        self.include_timestamp_var = tk.BooleanVar()
        ttk.Checkbutton(naming_frame, text="Include timestamp in filenames", 
                       variable=self.include_timestamp_var).pack(anchor=tk.W, padx=5, pady=2)
        
        # Export options
        export_frame = ttk.LabelFrame(self.output_frame, text="Export Options")
        export_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.export_excel_var = tk.BooleanVar()
        ttk.Checkbutton(export_frame, text="Export to Excel", 
                       variable=self.export_excel_var).pack(anchor=tk.W, padx=5, pady=2)
        
        self.export_csv_var = tk.BooleanVar()
        ttk.Checkbutton(export_frame, text="Export to CSV", 
                       variable=self.export_csv_var).pack(anchor=tk.W, padx=5, pady=2)
        
        self.export_html_var = tk.BooleanVar()
        ttk.Checkbutton(export_frame, text="Export HTML report", 
                       variable=self.export_html_var).pack(anchor=tk.W, padx=5, pady=2)
        
        self.export_pdf_var = tk.BooleanVar()
        ttk.Checkbutton(export_frame, text="Export PDF report", 
                       variable=self.export_pdf_var).pack(anchor=tk.W, padx=5, pady=2)
        
        # Logging
        logging_frame = ttk.LabelFrame(self.output_frame, text="Logging")
        logging_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.enable_logging_var = tk.BooleanVar()
        ttk.Checkbutton(logging_frame, text="Enable logging", 
                       variable=self.enable_logging_var).pack(anchor=tk.W, padx=5, pady=2)
        
        level_frame = ttk.Frame(logging_frame)
        level_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(level_frame, text="Log level:").pack(side=tk.LEFT)
        self.log_level_var = tk.StringVar()
        level_combo = ttk.Combobox(level_frame, textvariable=self.log_level_var,
                                 values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                                 state="readonly")
        level_combo.pack(side=tk.RIGHT)
        
        self.log_to_file_var = tk.BooleanVar()
        ttk.Checkbutton(logging_frame, text="Log to file", 
                       variable=self.log_to_file_var).pack(anchor=tk.W, padx=5, pady=2)
    
    def _browse_output_dir(self) -> None:
        """Browse for output directory."""
        directory = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.output_dir_var.get()
        )
        if directory:
            self.output_dir_var.set(directory)
    
    def _load_config(self) -> None:
        """Load current configuration into the dialog."""
        # Statistical configuration
        self.alpha_var.set(self.config.statistical.alpha)
        self.power_var.set(self.config.statistical.power)
        self.effect_size_var.set(self.config.statistical.min_effect_size)
        self.auto_test_var.set(self.config.statistical.auto_test_selection)
        self.prefer_nonparametric_var.set(self.config.statistical.prefer_nonparametric)
        self.robust_tests_var.set(self.config.statistical.use_robust_tests)
        self.mc_method_var.set(self.config.statistical.multiple_comparisons_method)
        self.check_normality_var.set(self.config.statistical.check_normality)
        self.check_homogeneity_var.set(self.config.statistical.check_homogeneity)
        self.check_independence_var.set(self.config.statistical.check_independence)
        self.use_bootstrap_var.set(self.config.statistical.use_bootstrap)
        self.bootstrap_n_var.set(self.config.statistical.bootstrap_n_samples)
        
        # Data configuration
        self.missing_strategy_var.set(self.config.data.missing_data_strategy)
        self.missing_threshold_var.set(self.config.data.missing_data_threshold)
        self.detect_outliers_var.set(self.config.data.detect_outliers)
        self.outlier_method_var.set(self.config.data.outlier_method)
        self.outlier_threshold_var.set(self.config.data.outlier_threshold)
        self.outlier_action_var.set(self.config.data.outlier_action)
        self.auto_transform_var.set(self.config.data.auto_transform)
        self.transform_method_var.set(self.config.data.transform_method)
        self.auto_scaling_var.set(self.config.data.auto_scaling)
        self.scaling_method_var.set(self.config.data.scaling_method)
        
        # Visualization configuration
        self.fig_width_var.set(self.config.visualization.figure_size[0])
        self.fig_height_var.set(self.config.visualization.figure_size[1])
        self.dpi_var.set(self.config.visualization.dpi)
        self.plot_style_var.set(self.config.visualization.style)
        self.color_palette_var.set(self.config.visualization.color_palette)
        self.show_grid_var.set(self.config.visualization.show_grid)
        self.show_legend_var.set(self.config.visualization.show_legend)
        self.tight_layout_var.set(self.config.visualization.tight_layout)
        self.save_plots_var.set(self.config.visualization.save_plots)
        self.plot_format_var.set(self.config.visualization.plot_format)
        
        # Output configuration
        self.output_dir_var.set(str(self.config.output.output_directory))
        self.create_subdir_var.set(self.config.output.create_subdirectory)
        self.file_prefix_var.set(self.config.output.file_prefix)
        self.include_timestamp_var.set(self.config.output.include_timestamp)
        self.export_excel_var.set(self.config.output.export_excel)
        self.export_csv_var.set(self.config.output.export_csv)
        self.export_html_var.set(self.config.output.export_html)
        self.export_pdf_var.set(self.config.output.export_pdf)
        self.enable_logging_var.set(self.config.output.enable_logging)
        self.log_level_var.set(self.config.output.log_level)
        self.log_to_file_var.set(self.config.output.log_to_file)
    
    def _save_config(self) -> None:
        """Save dialog values to configuration."""
        try:
            # Statistical configuration
            self.config.statistical.alpha = self.alpha_var.get()
            self.config.statistical.power = self.power_var.get()
            self.config.statistical.min_effect_size = self.effect_size_var.get()
            self.config.statistical.auto_test_selection = self.auto_test_var.get()
            self.config.statistical.prefer_nonparametric = self.prefer_nonparametric_var.get()
            self.config.statistical.use_robust_tests = self.robust_tests_var.get()
            self.config.statistical.multiple_comparisons_method = self.mc_method_var.get()
            self.config.statistical.check_normality = self.check_normality_var.get()
            self.config.statistical.check_homogeneity = self.check_homogeneity_var.get()
            self.config.statistical.check_independence = self.check_independence_var.get()
            self.config.statistical.use_bootstrap = self.use_bootstrap_var.get()
            self.config.statistical.bootstrap_n_samples = self.bootstrap_n_var.get()
            
            # Data configuration
            self.config.data.missing_data_strategy = self.missing_strategy_var.get()
            self.config.data.missing_data_threshold = self.missing_threshold_var.get()
            self.config.data.detect_outliers = self.detect_outliers_var.get()
            self.config.data.outlier_method = self.outlier_method_var.get()
            self.config.data.outlier_threshold = self.outlier_threshold_var.get()
            self.config.data.outlier_action = self.outlier_action_var.get()
            self.config.data.auto_transform = self.auto_transform_var.get()
            self.config.data.transform_method = self.transform_method_var.get()
            self.config.data.auto_scaling = self.auto_scaling_var.get()
            self.config.data.scaling_method = self.scaling_method_var.get()
            
            # Visualization configuration
            self.config.visualization.figure_size = (self.fig_width_var.get(), self.fig_height_var.get())
            self.config.visualization.dpi = self.dpi_var.get()
            self.config.visualization.style = self.plot_style_var.get()
            self.config.visualization.color_palette = self.color_palette_var.get()
            self.config.visualization.show_grid = self.show_grid_var.get()
            self.config.visualization.show_legend = self.show_legend_var.get()
            self.config.visualization.tight_layout = self.tight_layout_var.get()
            self.config.visualization.save_plots = self.save_plots_var.get()
            self.config.visualization.plot_format = self.plot_format_var.get()
            
            # Output configuration
            self.config.output.output_directory = Path(self.output_dir_var.get())
            self.config.output.create_subdirectory = self.create_subdir_var.get()
            self.config.output.file_prefix = self.file_prefix_var.get()
            self.config.output.include_timestamp = self.include_timestamp_var.get()
            self.config.output.export_excel = self.export_excel_var.get()
            self.config.output.export_csv = self.export_csv_var.get()
            self.config.output.export_html = self.export_html_var.get()
            self.config.output.export_pdf = self.export_pdf_var.get()
            self.config.output.enable_logging = self.enable_logging_var.get()
            self.config.output.log_level = self.log_level_var.get()
            self.config.output.log_to_file = self.log_to_file_var.get()
            
            # Validate configuration
            self.config.validate()
            
        except Exception as e:
            raise ValueError(f"Invalid configuration: {str(e)}")
    
    def _on_ok(self) -> None:
        """Handle OK button click."""
        try:
            self._save_config()
            self.result = self.config
            self.dialog.destroy()
        except Exception as e:
            messagebox.showerror("Configuration Error", str(e))
    
    def _on_cancel(self) -> None:
        """Handle Cancel button click."""
        self.result = None
        self.dialog.destroy()
    
    def _on_apply(self) -> None:
        """Handle Apply button click."""
        try:
            self._save_config()
            messagebox.showinfo("Success", "Configuration applied successfully.")
        except Exception as e:
            messagebox.showerror("Configuration Error", str(e))
    
    def _on_reset(self) -> None:
        """Handle Reset to Defaults button click."""
        if messagebox.askyesno("Confirm Reset", "Reset all settings to defaults?"):
            self.config.reset_to_defaults()
            self._load_config()
            messagebox.showinfo("Success", "Configuration reset to defaults.")