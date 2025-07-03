"""Main window for PyADAP 3.0 GUI

This module provides the main application window with modern UI design
and comprehensive functionality for data analysis.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Optional, Dict, Any, List
import threading
from datetime import datetime

from ..core import DataManager, AnalysisPipeline
from ..config import Config
from ..utils import get_logger
from .config_dialog import ConfigDialog
from .data_preview import DataPreviewWidget
from .analysis_wizard import AnalysisWizard
from .results_viewer import ResultsViewer

from traceback import print_exc
class MainWindow:
    """Main application window for PyADAP 3.0."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the main window.
        
        Args:
            config: Configuration object (optional)
        """
        # Initialize logger first
        self.logger = get_logger("PyADAP.GUI")
        
        if config:
            self.config = config
            self.logger.info("Using provided configuration")
        else:
            # Try to load configuration from default location
            self.config = Config()
            try:
                config_path = self.config.get_default_config_path()
                if Path(config_path).exists():
                    self.config.load_from_file(config_path)
                    self.logger.info(f"Configuration loaded from {config_path}")
                    
                    # Validate configuration
                    validation_errors = self.config.validate()
                    if validation_errors:
                        self.logger.warning(f"Configuration validation errors: {validation_errors}")
                        self.logger.info("Using default values for invalid settings")
                else:
                    self.logger.info("No configuration file found, using default configuration")
            except FileNotFoundError:
                self.logger.info("Configuration file not found, using defaults")
            except PermissionError:
                self.logger.warning("Permission denied accessing configuration file, using defaults")
            except Exception as e:
                self.logger.error(f"Unexpected error loading configuration: {e}")
                self.logger.info("Falling back to default configuration")
                # Reset to ensure clean state
                self.config = Config()
        
        # Initialize components
        self.data_manager = None
        self.pipeline = None
        self.current_data = None
        self.current_data_file = None  # Store current data file path
        self.analysis_results = None
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("PyADAP 3.0 - Advanced Data Analysis Platform")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Create UI
        self._create_menu()
        self._create_toolbar()
        self._create_main_layout()
        self._create_status_bar()
        
        # Bind events
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Initialize UI state
        self._update_ui_state()
        
        self.logger.info("PyADAP 3.0 GUI initialized")
    
    def _create_menu(self) -> None:
        """Create the main menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Data...", command=self._open_data, accelerator="Ctrl+O")
        file_menu.add_command(label="Save Results...", command=self._save_results, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Export Report...", command=self._export_report)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_closing, accelerator="Ctrl+Q")
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Preferences...", command=self._open_preferences)
        edit_menu.add_command(label="Reset Configuration", command=self._reset_config)
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Run Analysis Wizard...", command=self._open_analysis_wizard)
        analysis_menu.add_command(label="Quick Analysis", command=self._quick_analysis)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="Data Quality Report", command=self._generate_quality_report)
        analysis_menu.add_command(label="Assumption Checks", command=self._check_assumptions)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Data Preview", command=self._show_data_preview)
        view_menu.add_command(label="Results Viewer", command=self._show_results_viewer)
        view_menu.add_separator()
        view_menu.add_command(label="Log Window", command=self._show_log_window)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self._show_user_guide)
        help_menu.add_command(label="Statistical Tests Guide", command=self._show_stats_guide)
        help_menu.add_separator()
        help_menu.add_command(label="About PyADAP", command=self._show_about)
        
        # Keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self._open_data())
        self.root.bind('<Control-s>', lambda e: self._save_results())
        self.root.bind('<Control-q>', lambda e: self._on_closing())
    
    def _create_toolbar(self) -> None:
        """Create the toolbar."""
        toolbar_frame = ttk.Frame(self.root)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        # File operations
        ttk.Button(toolbar_frame, text="📁 Open Data", command=self._open_data).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="💾 Save Results", command=self._save_results).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # Analysis operations
        ttk.Button(toolbar_frame, text="🔍 Analysis Wizard", command=self._open_analysis_wizard).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="⚡ Quick Analysis", command=self._quick_analysis).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # View operations
        ttk.Button(toolbar_frame, text="📊 Data Preview", command=self._show_data_preview).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="📈 Results", command=self._show_results_viewer).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # Settings
        ttk.Button(toolbar_frame, text="⚙️ Settings", command=self._open_preferences).pack(side=tk.LEFT, padx=2)
    
    def _create_main_layout(self) -> None:
        """Create the main layout with panels."""
        # Create main paned window
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Data and Variables
        self.left_panel = ttk.Frame(self.main_paned)
        self.main_paned.add(self.left_panel, weight=1)
        
        # Right panel - Analysis and Results
        self.right_panel = ttk.Frame(self.main_paned)
        self.main_paned.add(self.right_panel, weight=2)
        
        self._create_left_panel()
        self._create_right_panel()
    
    def _create_left_panel(self) -> None:
        """Create the left panel with data information and variable selection."""
        # Data Information
        data_frame = ttk.LabelFrame(self.left_panel, text="Data Information")
        data_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.data_info_text = tk.Text(data_frame, height=6, wrap=tk.WORD)
        scrollbar1 = ttk.Scrollbar(data_frame, orient=tk.VERTICAL, command=self.data_info_text.yview)
        self.data_info_text.configure(yscrollcommand=scrollbar1.set)
        
        self.data_info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar1.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Variable Selection
        var_frame = ttk.LabelFrame(self.left_panel, text="Variable Selection")
        var_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Available variables
        ttk.Label(var_frame, text="Available Variables:").pack(anchor=tk.W, padx=5, pady=(5,0))
        
        available_frame = ttk.Frame(var_frame)
        available_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.available_vars = tk.Listbox(available_frame, selectmode=tk.EXTENDED)
        scrollbar2 = ttk.Scrollbar(available_frame, orient=tk.VERTICAL, command=self.available_vars.yview)
        self.available_vars.configure(yscrollcommand=scrollbar2.set)
        
        self.available_vars.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Buttons for variable assignment
        button_frame = ttk.Frame(var_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="→ Dependent", command=self._add_dependent_var).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="→ Independent", command=self._add_independent_var).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="→ Subject ID", command=self._add_subject_var).pack(side=tk.LEFT, padx=2)
        
        # Selected variables
        selected_frame = ttk.Frame(var_frame)
        selected_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Dependent variables
        dep_frame = ttk.LabelFrame(selected_frame, text="Dependent Variables")
        dep_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        
        self.dependent_vars = tk.Listbox(dep_frame, height=4)
        self.dependent_vars.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Independent variables
        indep_frame = ttk.LabelFrame(selected_frame, text="Independent Variables")
        indep_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        
        self.independent_vars = tk.Listbox(indep_frame, height=4)
        self.independent_vars.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Subject ID
        subject_frame = ttk.LabelFrame(var_frame, text="Subject ID")
        subject_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.subject_var = ttk.Combobox(subject_frame, state="readonly")
        self.subject_var.pack(fill=tk.X, padx=5, pady=5)
    
    def _create_right_panel(self) -> None:
        """Create the right panel with analysis options and results."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Analysis tab
        self.analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_tab, text="Analysis")
        
        # Results tab
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="Results")
        
        # Visualization tab
        self.viz_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_tab, text="Visualization")
        
        self._create_analysis_tab()
        self._create_results_tab()
        self._create_visualization_tab()
    
    def _create_analysis_tab(self) -> None:
        """Create the analysis configuration tab."""
        # Analysis options
        options_frame = ttk.LabelFrame(self.analysis_tab, text="Analysis Options")
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Statistical tests
        tests_frame = ttk.LabelFrame(options_frame, text="Statistical Tests")
        tests_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.auto_test_selection = tk.BooleanVar(value=True)
        ttk.Checkbutton(tests_frame, text="Automatic test selection", 
                       variable=self.auto_test_selection).pack(anchor=tk.W, padx=5, pady=2)
        
        self.run_descriptive = tk.BooleanVar(value=True)
        ttk.Checkbutton(tests_frame, text="Descriptive statistics", 
                       variable=self.run_descriptive).pack(anchor=tk.W, padx=5, pady=2)
        
        self.check_assumptions = tk.BooleanVar(value=True)
        ttk.Checkbutton(tests_frame, text="Check statistical assumptions", 
                       variable=self.check_assumptions).pack(anchor=tk.W, padx=5, pady=2)
        
        self.calculate_effect_sizes = tk.BooleanVar(value=True)
        ttk.Checkbutton(tests_frame, text="Calculate effect sizes", 
                       variable=self.calculate_effect_sizes).pack(anchor=tk.W, padx=5, pady=2)
        
        # Data preprocessing
        preprocess_frame = ttk.LabelFrame(options_frame, text="Data Preprocessing")
        preprocess_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.handle_missing = tk.BooleanVar(value=True)
        ttk.Checkbutton(preprocess_frame, text="Handle missing values", 
                       variable=self.handle_missing).pack(anchor=tk.W, padx=5, pady=2)
        
        self.detect_outliers = tk.BooleanVar(value=True)
        ttk.Checkbutton(preprocess_frame, text="Detect and handle outliers", 
                       variable=self.detect_outliers).pack(anchor=tk.W, padx=5, pady=2)
        
        self.apply_transformations = tk.BooleanVar(value=False)
        ttk.Checkbutton(preprocess_frame, text="Apply data transformations", 
                       variable=self.apply_transformations).pack(anchor=tk.W, padx=5, pady=2)
        
        # Analysis parameters
        params_frame = ttk.LabelFrame(options_frame, text="Parameters")
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Alpha level
        alpha_frame = ttk.Frame(params_frame)
        alpha_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(alpha_frame, text="Significance level (α):").pack(side=tk.LEFT)
        self.alpha_var = tk.DoubleVar(value=0.05)
        alpha_spinbox = ttk.Spinbox(alpha_frame, from_=0.001, to=0.1, increment=0.001, 
                                   textvariable=self.alpha_var, width=10)
        alpha_spinbox.pack(side=tk.RIGHT)
        
        # Run analysis button
        button_frame = ttk.Frame(self.analysis_tab)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.run_button = ttk.Button(button_frame, text="🚀 Run Analysis", 
                                    command=self._run_analysis, style="Accent.TButton")
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="⏹️ Stop", 
                                     command=self._stop_analysis, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(button_frame, variable=self.progress_var, 
                                          mode='determinate')
        self.progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
    
    def _create_results_tab(self) -> None:
        """Create the results display tab."""
        # Results text area
        self.results_text = tk.Text(self.results_tab, wrap=tk.WORD, font=('Consolas', 10))
        results_scrollbar = ttk.Scrollbar(self.results_tab, orient=tk.VERTICAL, 
                                        command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _create_visualization_tab(self) -> None:
        """Create the visualization tab."""
        # Placeholder for matplotlib canvas
        viz_label = ttk.Label(self.viz_tab, text="Visualization will be displayed here")
        viz_label.pack(expand=True)
    
    def _create_status_bar(self) -> None:
        """Create the status bar."""
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.status_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT, padx=5, pady=2)
        
        # Memory usage
        self.memory_var = tk.StringVar(value="Memory: 0 MB")
        self.memory_label = ttk.Label(self.status_frame, textvariable=self.memory_var)
        self.memory_label.pack(side=tk.RIGHT, padx=5, pady=2)
    
    def _open_data(self) -> None:
        """Open data file dialog and load data."""
        filetypes = [
            ('Excel files', '*.xlsx *.xls'),
            ('CSV files', '*.csv'),
            ('TSV files', '*.tsv'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=filetypes
        )
        
        if filename:
            self._load_data(filename)
    
    def _load_data(self, filename: str) -> None:
        """Load data from file.
        
        Args:
            filename: Path to data file
        """
        try:
            # Initialize data manager
            self.data_manager = DataManager()
            
            # Load data
            self.current_data = self.data_manager.load_data(filename)
            
            # Store the data file path
            self.current_data_file = filename
            
            # Update UI
            self._update_data_info()
            self._update_variable_lists()
            self._update_ui_state()
            
            self._update_status(f"Data loaded: {filename}")
            self.logger.info(f"Data loaded from {filename}")
            
        except Exception as e:
            error_msg = f"Error loading data: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.logger.error(error_msg)
            self._update_status("Error loading data")
            
            # Reset data state and update UI
            self.current_data = None
            self.current_data_file = None
            self._update_ui_state()
    
    def _update_data_info(self) -> None:
        """Update data information display."""
        if self.current_data is None:
            return
        
        info_text = f"""Dataset Information:
• Shape: {self.current_data.shape[0]} rows × {self.current_data.shape[1]} columns
• Memory usage: {self.current_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB
• Missing values: {self.current_data.isnull().sum().sum()}
• Data types: {self.current_data.dtypes.value_counts().to_dict()}
"""
        
        self.data_info_text.delete(1.0, tk.END)
        self.data_info_text.insert(1.0, info_text)
        
        # Update memory usage in status bar
        memory_mb = self.current_data.memory_usage(deep=True).sum() / 1024**2
        self.memory_var.set(f"Memory: {memory_mb:.1f} MB")
    
    def _update_variable_lists(self) -> None:
        """Update variable selection lists."""
        if self.current_data is None:
            return
        
        # Clear existing lists
        self.available_vars.delete(0, tk.END)
        self.dependent_vars.delete(0, tk.END)
        self.independent_vars.delete(0, tk.END)
        
        # Populate available variables
        for col in self.current_data.columns:
            self.available_vars.insert(tk.END, col)
        
        # Update subject variable combobox
        self.subject_var['values'] = [''] + list(self.current_data.columns)
    
    def _update_ui_state(self) -> None:
        """Update UI state based on data availability."""
        data_loaded = self.current_data is not None and not self.current_data.empty
        
        # Enable/disable analysis buttons based on data availability
        if hasattr(self, 'run_button'):
            self.run_button.config(state=tk.NORMAL if data_loaded else tk.DISABLED)
        
        # Update toolbar buttons if they exist
        # Note: Toolbar buttons are created dynamically, so we need to find them
        try:
            for widget in self.toolbar_frame.winfo_children():
                if isinstance(widget, ttk.Button):
                    button_text = widget.cget('text')
                    if '⚡ Quick Analysis' in button_text or '🔍 Analysis Wizard' in button_text:
                        widget.config(state=tk.NORMAL if data_loaded else tk.DISABLED)
        except (AttributeError, tk.TclError):
            # Toolbar might not be created yet or widgets might not exist
            pass
    
    def _add_dependent_var(self) -> None:
        """Add selected variables to dependent list."""
        selected = self.available_vars.curselection()
        for idx in selected:
            var_name = self.available_vars.get(idx)
            if var_name not in self.dependent_vars.get(0, tk.END):
                self.dependent_vars.insert(tk.END, var_name)
    
    def _add_independent_var(self) -> None:
        """Add selected variables to independent list."""
        selected = self.available_vars.curselection()
        for idx in selected:
            var_name = self.available_vars.get(idx)
            if var_name not in self.independent_vars.get(0, tk.END):
                self.independent_vars.insert(tk.END, var_name)
    
    def _add_subject_var(self) -> None:
        """Set selected variable as subject ID."""
        selected = self.available_vars.curselection()
        if selected:
            var_name = self.available_vars.get(selected[0])
            self.subject_var.set(var_name)
    
    def _run_analysis(self) -> None:
        """Run the statistical analysis."""
        # Enhanced data validation
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return
            
        if self.current_data.empty:
            messagebox.showwarning("Warning", "The loaded dataset is empty.")
            return
        
        # Get selected variables
        dependent_vars = list(self.dependent_vars.get(0, tk.END))
        independent_vars = list(self.independent_vars.get(0, tk.END))
        subject_var = self.subject_var.get() if self.subject_var.get() else None
        
        if not dependent_vars:
            messagebox.showwarning("Warning", "Please select at least one dependent variable.")
            return
        
        # Run analysis in separate thread
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        analysis_thread = threading.Thread(
            target=self._run_analysis_thread,
            args=(dependent_vars, independent_vars, subject_var)
        )
        analysis_thread.daemon = True
        analysis_thread.start()
    
    def _run_analysis_thread(self, dependent_vars: List[str], 
                           independent_vars: List[str], 
                           subject_var: Optional[str]) -> None:
        """Run analysis in separate thread.
        
        Args:
            dependent_vars: List of dependent variable names
            independent_vars: List of independent variable names
            subject_var: Subject ID variable name
        """
        try:
            self._update_status("Running analysis...")
            
            # Initialize pipeline
            self.pipeline = AnalysisPipeline(
                config=self.config
            )
            
            # Set data first with file path
            self.pipeline.set_data(self.current_data, self.current_data_file)
            
            # Set variables
            self.pipeline.set_variables(
                dependent_vars=dependent_vars,
                independent_vars=independent_vars,
                subject_var=subject_var
            )
            
            # Configure analysis options
            analysis_config = {
                'alpha': self.alpha_var.get(),
                'auto_test_selection': self.auto_test_selection.get(),
                'run_descriptive': self.run_descriptive.get(),
                'check_assumptions': self.check_assumptions.get(),
                'calculate_effect_sizes': self.calculate_effect_sizes.get(),
                'handle_missing': self.handle_missing.get(),
                'detect_outliers': self.detect_outliers.get(),
                'apply_transformations': self.apply_transformations.get()
            }
            
            # Run pipeline
            self.analysis_results = self.pipeline.run_analysis(
                dependent_vars=dependent_vars,
                independent_vars=independent_vars,
                subject_var=subject_var,
                **analysis_config)
            
            # Update UI in main thread
            self.root.after(0, self._analysis_completed)
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            self.logger.error(error_msg)
            self.root.after(0, lambda: self._analysis_failed(error_msg))
    
    def _analysis_completed(self) -> None:
        """Handle analysis completion."""
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress_var.set(100)
        
        # Display results
        self._display_results()
        
        # Update visualization tab
        self._update_visualization()
        
        # Switch to results tab
        self.notebook.select(self.results_tab)
        
        self._update_status("Analysis completed")
        self.logger.info("Analysis completed successfully")
    
    def _update_visualization(self) -> None:
        """Update the visualization tab with plots from analysis results."""
        # Clear existing content
        for widget in self.viz_tab.winfo_children():
            widget.destroy()
        
        if self.analysis_results is None or 'plots' not in self.analysis_results or not self.analysis_results['plots']:
            viz_label = ttk.Label(self.viz_tab, text="No visualizations available")
            viz_label.pack(expand=True)
            self.logger.warning("No plots available in analysis results")
            return
        
        # Create a frame for the plot selection
        control_frame = ttk.Frame(self.viz_tab)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add plot selection combo
        ttk.Label(control_frame, text="Select Plot:").pack(side=tk.LEFT, padx=5)
        plot_names = list(self.analysis_results['plots'].keys())
        
        if not plot_names:
            viz_label = ttk.Label(self.viz_tab, text="No plots generated")
            viz_label.pack(expand=True)
            self.logger.warning("Plot dictionary is empty")
            return
            
        plot_var = tk.StringVar(value=plot_names[0])
        plot_combo = ttk.Combobox(control_frame, textvariable=plot_var, values=plot_names, state="readonly")
        plot_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Create a frame for the plot
        plot_frame = ttk.Frame(self.viz_tab)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        def display_plot(*args):
            try:
                # Clear existing plot
                for widget in plot_frame.winfo_children():
                    widget.destroy()
                
                plot_name = plot_var.get()
                if plot_name and plot_name in self.analysis_results['plots']:
                    plot_data = self.analysis_results['plots'][plot_name]
                    
                    if plot_data is None:
                        self.logger.warning(f"Plot data is None for {plot_name}")
                        return
                    
                    # Create canvas
                    canvas = FigureCanvasTkAgg(plot_data, plot_frame)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                    
                    # Add toolbar
                    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
                    toolbar = NavigationToolbar2Tk(canvas, plot_frame)
                    toolbar.update()
                    
                    self.logger.info(f"Successfully displayed plot: {plot_name}")
            except Exception as e:
                self.logger.error(f"Error displaying plot {plot_name}: {str(e)}")
                error_label = ttk.Label(plot_frame, text=f"Error displaying plot: {str(e)}")
                error_label.pack(expand=True)
        
        # Bind plot selection change
        plot_combo.bind('<<ComboboxSelected>>', display_plot)
        
        # Display initial plot
        display_plot()
    
    def _analysis_failed(self, error_msg: str) -> None:
        """Handle analysis failure.
        
        Args:
            error_msg: Error message
        """
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress_var.set(0)
        
        messagebox.showerror("Analysis Error", error_msg)
        self._update_status("Analysis failed")
    
    def _stop_analysis(self) -> None:
        """Stop the running analysis."""
        # Implementation would depend on how the analysis can be interrupted
        self._update_status("Stopping analysis...")
        self.logger.info("Analysis stop requested")
    
    def _display_results(self) -> None:
        """Display analysis results."""
        if self.analysis_results is None:
            return
        
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        
        # Format and display results
        results_text = self._format_results(self.analysis_results)
        self.results_text.insert(1.0, results_text)
    
    def _format_results(self, results: Dict[str, Any]) -> str:
        """Format analysis results for display.
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            Formatted results string
        """
        output = []
        output.append("=" * 80)
        output.append("PyADAP 3.0 - Statistical Analysis Results")
        output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("=" * 80)
        output.append("")
        
        # Add formatted results sections
        for section, content in results.items():
            output.append(f"[{section.upper()}]")
            output.append("-" * 40)
            
            if isinstance(content, dict):
                for key, value in content.items():
                    output.append(f"{key}: {value}")
            elif isinstance(content, str):
                output.append(content)
            else:
                output.append(str(content))
            
            output.append("")
        
        return "\n".join(output)
    
    def _update_status(self, message: str) -> None:
        """Update status bar message.
        
        Args:
            message: Status message
        """
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def _save_results(self) -> None:
        """Save analysis results to file."""
        if self.analysis_results is None:
            messagebox.showwarning("Warning", "No results to save.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".xlsx",
            filetypes=[
                ('Excel files', '*.xlsx'),
                ('Text files', '*.txt'),
                ('All files', '*.*')
            ]
        )
        
        if filename:
            try:
                # Save results using pipeline
                self.pipeline.export_results(filename)
                self._update_status(f"Results saved: {filename}")
                messagebox.showinfo("Success", "Results saved successfully.")
            except Exception as e:
                error_msg = f"Error saving results: {str(e)}"
                messagebox.showerror("Error", error_msg)
                self.logger.error(error_msg)
    
    def _export_report(self) -> None:
        """Export comprehensive analysis report."""
        if self.pipeline is None:
            messagebox.showwarning("Warning", "No analysis to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Report",
            defaultextension=".html",
            filetypes=[
                ('HTML files', '*.html'),
                ('PDF files', '*.pdf'),
                ('All files', '*.*')
            ]
        )
        
        if filename:
            try:
                # Generate and save report
                self.pipeline.generate_report(filename)
                self._update_status(f"Report exported: {filename}")
                messagebox.showinfo("Success", "Report exported successfully.")
            except Exception as e:
                error_msg = f"Error exporting report: {str(e)}"
                messagebox.showerror("Error", error_msg)
                self.logger.error(error_msg)
    
    def _open_preferences(self) -> None:
        """Open preferences dialog."""
        dialog = ConfigDialog(self.root, self.config)
        if dialog.result:
            self.config = dialog.result
            # Save configuration to file
            try:
                config_path = self.config.get_default_config_path()
                self.config.save_to_file(config_path)
                self.logger.info(f"Configuration updated and saved to {config_path}")
            except Exception as e:
                self.logger.error(f"Failed to save configuration: {e}")
                messagebox.showwarning("Warning", f"Configuration updated but could not be saved: {e}")
    
    def _reset_config(self) -> None:
        """Reset configuration to defaults."""
        if messagebox.askyesno("Confirm", "Reset all settings to defaults?"):
            self.config.reset_to_defaults()
            self.logger.info("Configuration reset to defaults")
            messagebox.showinfo("Success", "Configuration reset successfully.")
    
    def _open_analysis_wizard(self) -> None:
        """Open analysis wizard."""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return
        
        wizard = AnalysisWizard(self.root, self.config, self.current_data)
        if wizard.result:
            # Apply wizard results
            self.logger.info("Analysis wizard completed")
    
    def _quick_analysis(self) -> None:
        """Run quick automatic analysis."""
        # Enhanced data validation
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return
            
        if self.current_data.empty:
            messagebox.showwarning("Warning", "The loaded dataset is empty.")
            return
        
        self.logger.info("Quick analysis requested")
        
        try:
            # Auto-select variables for quick analysis
            numeric_cols = self.current_data.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = self.current_data.select_dtypes(include=['object', 'category']).columns.tolist()
        except Exception as e:
            error_msg = f"Error accessing data columns: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.logger.error(error_msg)
            return
        
        if not numeric_cols:
            messagebox.showwarning("Warning", "No numeric variables found for analysis.")
            return
        
        # For quick analysis, use first numeric column as dependent variable
        # and first categorical column (if exists) as independent variable
        dependent_vars = [numeric_cols[0]]
        independent_vars = categorical_cols[:1] if categorical_cols else []
        subject_var = None
        
        # Run analysis in separate thread
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        analysis_thread = threading.Thread(
            target=self._quick_analysis_thread,
            args=(dependent_vars, independent_vars, subject_var)
        )
        analysis_thread.daemon = True
        analysis_thread.start()
    
    def _quick_analysis_thread(self, dependent_vars: List[str], 
                              independent_vars: List[str], 
                              subject_var: Optional[str]) -> None:
        """Run quick analysis in separate thread.
        
        Args:
            dependent_vars: List of dependent variable names
            independent_vars: List of independent variable names
            subject_var: Subject ID variable name
        """
        try:
            self._update_status("Running quick analysis...")
            
            # Initialize pipeline
            self.pipeline = AnalysisPipeline(
                config=self.config
            )
            
            # Set data first
            self.pipeline.set_data(self.current_data, self.current_data_file)
            
            # Set variables
            self.pipeline.set_variables(
                dependent_vars=dependent_vars,
                independent_vars=independent_vars,
                subject_var=subject_var
            )
            
            # Configure analysis options for quick analysis
            analysis_config = {
                'alpha': 0.05,
                'auto_test_selection': True,
                'run_descriptive': True,
                'check_assumptions': True,
                'calculate_effect_sizes': True,
                'handle_missing': True,
                'detect_outliers': False,
                'apply_transformations': False
            }
            
            # Run pipeline with required parameters
            self.analysis_results = self.pipeline.run_analysis(
                dependent_vars=dependent_vars,
                independent_vars=independent_vars,
                subject_var=subject_var,
                **analysis_config
            )
            
            # Update UI in main thread
            self.root.after(0, self._quick_analysis_completed)
            
        except Exception as e:
            print_exc()
            error_msg = f"Quick analysis failed: {str(e)}"
            self.logger.error(error_msg)
            self.root.after(0, lambda: self._analysis_failed(error_msg))
    
    def _quick_analysis_completed(self) -> None:
        """Handle quick analysis completion."""
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress_var.set(100)
        
        # Display results
        self._display_results()
        
        # Switch to results tab
        self.notebook.select(self.results_tab)
        
        self._update_status("Quick analysis completed")
        self.logger.info("Quick analysis completed successfully")
        
        # Show completion message
        messagebox.showinfo("Quick Analysis", "Quick analysis completed! Results are displayed in the Results tab.")
    
    def _generate_quality_report(self) -> None:
        """Generate data quality report."""
        if self.data_manager is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return
        
        try:
            quality_report = self.data_manager.get_data_quality_report()
            # Display quality report
            self.logger.info("Data quality report generated")
        except Exception as e:
            error_msg = f"Error generating quality report: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.logger.error(error_msg)
    
    def _check_assumptions(self) -> None:
        """Check statistical assumptions."""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return
        
        # Implement assumption checking
        self.logger.info("Assumption checking requested")
    
    def _show_data_preview(self) -> None:
        """Show data preview window."""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return
        
        preview_window = DataPreviewWidget(self.root, self.current_data)
        preview_window.show()
    
    def _show_results_viewer(self) -> None:
        """Show results viewer window."""
        if self.analysis_results is None:
            messagebox.showwarning("Warning", "No results to display.")
            return
        
        results_window = ResultsViewer(self.root)
        results_window.load_results(self.analysis_results)
        results_window.show()
    
    def _show_log_window(self) -> None:
        """Show log window."""
        # Implement log window
        messagebox.showinfo("Info", "Log window not yet implemented.")
    
    def _show_user_guide(self) -> None:
        """Show user guide."""
        messagebox.showinfo("User Guide", "User guide will be available in future versions.")
    
    def _show_stats_guide(self) -> None:
        """Show statistical tests guide."""
        messagebox.showinfo("Statistical Tests Guide", "Statistical tests guide will be available in future versions.")
    
    def _show_about(self) -> None:
        """Show about dialog."""
        about_text = """PyADAP 3.0
Advanced Data Analysis Platform

A comprehensive statistical analysis tool with modern GUI
and automated analysis capabilities.

Version: 3.0.0
Author: Kannmu
License: MIT"""
        
        messagebox.showinfo("About PyADAP", about_text)
    
    def _on_closing(self) -> None:
        """Handle window closing event."""
        if messagebox.askokcancel("Quit", "Do you want to quit PyADAP?"):
            # Save configuration before closing
            try:
                config_path = self.config.get_default_config_path()
                self.config.save_to_file(config_path)
                self.logger.info(f"Configuration saved to {config_path}")
            except Exception as e:
                self.logger.error(f"Failed to save configuration on exit: {e}")
            
            self.logger.info("PyADAP GUI closing")
            self.root.destroy()
    
    def run(self) -> None:
        """Start the GUI application."""
        self.logger.info("Starting PyADAP 3.0 GUI")
        self.root.mainloop()


def main():
    """Main entry point for GUI application."""
    app = MainWindow()
    app.run()


if __name__ == "__main__":
    main()