"""Analysis wizard for PyADAP 3.0

This module provides a step-by-step wizard to guide users through
the statistical analysis process.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path

from ..config import Config
from ..core import DataManager, StatisticalAnalyzer, AnalysisPipeline
from ..utils import Logger, get_logger


class AnalysisWizard:
    """Step-by-step analysis wizard."""
    
    def __init__(self, parent: tk.Tk, config: Config, on_complete: Optional[Callable] = None):
        """Initialize the analysis wizard.
        
        Args:
            parent: Parent window
            config: Configuration object
            on_complete: Callback function when analysis is complete
        """
        self.parent = parent
        self.config = config
        self.on_complete = on_complete
        self.logger = get_logger("PyADAP.AnalysisWizard")
        
        # Data and analysis objects
        self.data_manager: Optional[DataManager] = None
        self.analyzer: Optional[StatisticalAnalyzer] = None
        self.pipeline: Optional[AnalysisPipeline] = None
        self.data: Optional[pd.DataFrame] = None
        self.result: Optional[Dict[str, Any]] = None
        
        # Wizard state
        self.current_step = 0
        self.steps = [
            ("Data Loading", self._create_data_step),
            ("Variable Selection", self._create_variables_step),
            ("Data Preprocessing", self._create_preprocessing_step),
            ("Analysis Options", self._create_analysis_step),
            ("Review & Run", self._create_review_step)
        ]
        
        # Create wizard window
        self.wizard = tk.Toplevel(parent)
        self.wizard.title("PyADAP Analysis Wizard")
        self.wizard.geometry("800x600")
        self.wizard.resizable(True, True)
        
        # Make wizard modal
        self.wizard.transient(parent)
        self.wizard.grab_set()
        
        # Center the wizard
        self._center_wizard()
        
        # Create UI
        self._create_ui()
        
        # Bind events
        self.wizard.protocol("WM_DELETE_WINDOW", self._on_cancel)
        
        # Start with first step
        self._show_step(0)
    
    def _center_wizard(self) -> None:
        """Center the wizard on the parent window."""
        self.wizard.update_idletasks()
        
        # Get parent window position and size
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        # Get wizard size
        wizard_width = self.wizard.winfo_reqwidth()
        wizard_height = self.wizard.winfo_reqheight()
        
        # Calculate center position
        x = parent_x + (parent_width - wizard_width) // 2
        y = parent_y + (parent_height - wizard_height) // 2
        
        self.wizard.geometry(f"+{x}+{y}")
    
    def _create_ui(self) -> None:
        """Create the wizard UI."""
        # Main frame
        main_frame = ttk.Frame(self.wizard)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header frame
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Step indicator
        self.step_var = tk.StringVar()
        step_label = ttk.Label(header_frame, textvariable=self.step_var, font=('TkDefaultFont', 12, 'bold'))
        step_label.pack(side=tk.LEFT)
        
        # Progress bar
        self.progress = ttk.Progressbar(header_frame, mode='determinate')
        self.progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(20, 0))
        
        # Content frame
        self.content_frame = ttk.Frame(main_frame)
        self.content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Navigation buttons
        self.back_button = ttk.Button(button_frame, text="< Back", command=self._go_back)
        self.back_button.pack(side=tk.LEFT)
        
        self.cancel_button = ttk.Button(button_frame, text="Cancel", command=self._on_cancel)
        self.cancel_button.pack(side=tk.LEFT, padx=(10, 0))
        
        self.next_button = ttk.Button(button_frame, text="Next >", command=self._go_next)
        self.next_button.pack(side=tk.RIGHT)
        
        # Status label
        self.status_var = tk.StringVar()
        ttk.Label(button_frame, textvariable=self.status_var, foreground="blue").pack(side=tk.RIGHT, padx=(0, 10))
    
    def _show_step(self, step_index: int) -> None:
        """Show a specific step.
        
        Args:
            step_index: Index of step to show
        """
        if step_index < 0 or step_index >= len(self.steps):
            return
        
        self.current_step = step_index
        
        # Update header
        step_name, step_func = self.steps[step_index]
        self.step_var.set(f"Step {step_index + 1} of {len(self.steps)}: {step_name}")
        
        # Update progress
        progress_value = ((step_index + 1) / len(self.steps)) * 100
        self.progress['value'] = progress_value
        
        # Clear content frame
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Create step content
        try:
            step_func()
        except Exception as e:
            self.logger.error(f"Error creating step {step_index}: {str(e)}")
            error_label = ttk.Label(self.content_frame, text=f"Error creating step: {str(e)}")
            error_label.pack(pady=20)
        
        # Update button states
        self.back_button['state'] = 'normal' if step_index > 0 else 'disabled'
        
        if step_index == len(self.steps) - 1:
            self.next_button['text'] = "Finish"
        else:
            self.next_button['text'] = "Next >"
        
        self.status_var.set("")
    
    def _create_data_step(self) -> None:
        """Create the data loading step."""
        # Title
        title_label = ttk.Label(self.content_frame, text="Load Your Data", font=('TkDefaultFont', 14, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Instructions
        instructions = (
            "Select a data file to begin your analysis. PyADAP supports Excel, CSV, TSV, and other common formats.\n"
            "Make sure your data is properly formatted with column headers in the first row."
        )
        ttk.Label(self.content_frame, text=instructions, wraplength=600).pack(pady=(0, 20))
        
        # File selection frame
        file_frame = ttk.LabelFrame(self.content_frame, text="Data File")
        file_frame.pack(fill=tk.X, pady=(0, 20))
        
        # File path entry
        path_frame = ttk.Frame(file_frame)
        path_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.file_path_var = tk.StringVar()
        ttk.Entry(path_frame, textvariable=self.file_path_var, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(path_frame, text="Browse...", command=self._browse_file).pack(side=tk.RIGHT, padx=(10, 0))
        
        # Data preview frame
        self.data_preview_frame = ttk.LabelFrame(self.content_frame, text="Data Preview")
        self.data_preview_frame.pack(fill=tk.BOTH, expand=True)
        
        # Preview text
        preview_text = ttk.Label(self.data_preview_frame, text="No data loaded", foreground="gray")
        preview_text.pack(expand=True)
    
    def _create_variables_step(self) -> None:
        """Create the variable selection step."""
        # Title
        title_label = ttk.Label(self.content_frame, text="Select Variables", font=('TkDefaultFont', 14, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Instructions
        instructions = (
            "Assign roles to your variables. This helps PyADAP choose the appropriate statistical tests.\n"
            "• Independent Variables: Factors or predictors\n"
            "• Dependent Variables: Outcomes or responses\n"
            "• Subject ID: Identifier for repeated measures (optional)\n"
            "• Covariates: Control variables (optional)"
        )
        ttk.Label(self.content_frame, text=instructions, wraplength=600).pack(pady=(0, 20))
        
        if self.data is None:
            ttk.Label(self.content_frame, text="Please load data first", foreground="red").pack()
            return
        
        # Variable assignment frame
        var_frame = ttk.Frame(self.content_frame)
        var_frame.pack(fill=tk.BOTH, expand=True)
        
        # Available variables
        available_frame = ttk.LabelFrame(var_frame, text="Available Variables")
        available_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.available_listbox = tk.Listbox(available_frame, selectmode=tk.EXTENDED)
        available_scroll = ttk.Scrollbar(available_frame, orient=tk.VERTICAL, command=self.available_listbox.yview)
        self.available_listbox.configure(yscrollcommand=available_scroll.set)
        
        self.available_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        available_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Populate available variables
        for col in self.data.columns:
            self.available_listbox.insert(tk.END, col)
        
        # Assignment buttons
        button_frame = ttk.Frame(var_frame)
        button_frame.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(button_frame, text="→ Independent", command=lambda: self._assign_variable('independent')).pack(pady=5)
        ttk.Button(button_frame, text="→ Dependent", command=lambda: self._assign_variable('dependent')).pack(pady=5)
        ttk.Button(button_frame, text="→ Subject ID", command=lambda: self._assign_variable('subject')).pack(pady=5)
        ttk.Button(button_frame, text="→ Covariate", command=lambda: self._assign_variable('covariate')).pack(pady=5)
        ttk.Button(button_frame, text="← Remove", command=self._remove_variable).pack(pady=20)
        
        # Assigned variables
        assigned_frame = ttk.LabelFrame(var_frame, text="Assigned Variables")
        assigned_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Create notebook for different variable types
        self.var_notebook = ttk.Notebook(assigned_frame)
        self.var_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Independent variables
        self.independent_frame = ttk.Frame(self.var_notebook)
        self.var_notebook.add(self.independent_frame, text="Independent")
        self.independent_listbox = tk.Listbox(self.independent_frame)
        self.independent_listbox.pack(fill=tk.BOTH, expand=True)
        
        # Dependent variables
        self.dependent_frame = ttk.Frame(self.var_notebook)
        self.var_notebook.add(self.dependent_frame, text="Dependent")
        self.dependent_listbox = tk.Listbox(self.dependent_frame)
        self.dependent_listbox.pack(fill=tk.BOTH, expand=True)
        
        # Subject ID
        self.subject_frame = ttk.Frame(self.var_notebook)
        self.var_notebook.add(self.subject_frame, text="Subject ID")
        self.subject_listbox = tk.Listbox(self.subject_frame)
        self.subject_listbox.pack(fill=tk.BOTH, expand=True)
        
        # Covariates
        self.covariate_frame = ttk.Frame(self.var_notebook)
        self.var_notebook.add(self.covariate_frame, text="Covariates")
        self.covariate_listbox = tk.Listbox(self.covariate_frame)
        self.covariate_listbox.pack(fill=tk.BOTH, expand=True)
    
    def _create_preprocessing_step(self) -> None:
        """Create the data preprocessing step."""
        # Title
        title_label = ttk.Label(self.content_frame, text="Data Preprocessing", font=('TkDefaultFont', 14, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Instructions
        instructions = (
            "Configure how PyADAP should preprocess your data. These settings help ensure "
            "your data meets the assumptions of statistical tests."
        )
        ttk.Label(self.content_frame, text=instructions, wraplength=600).pack(pady=(0, 20))
        
        # Create scrollable frame
        canvas = tk.Canvas(self.content_frame)
        scrollbar = ttk.Scrollbar(self.content_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Missing data handling
        missing_frame = ttk.LabelFrame(scrollable_frame, text="Missing Data")
        missing_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(missing_frame, text="Strategy:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.missing_strategy_var = tk.StringVar(value=self.config.data.missing_data_strategy)
        missing_combo = ttk.Combobox(missing_frame, textvariable=self.missing_strategy_var,
                                   values=["drop", "mean", "median", "mode"], state="readonly")
        missing_combo.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # Outlier detection
        outlier_frame = ttk.LabelFrame(scrollable_frame, text="Outlier Detection")
        outlier_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.detect_outliers_var = tk.BooleanVar(value=self.config.data.detect_outliers)
        ttk.Checkbutton(outlier_frame, text="Detect and handle outliers", 
                       variable=self.detect_outliers_var).pack(anchor=tk.W, padx=5, pady=5)
        
        outlier_method_frame = ttk.Frame(outlier_frame)
        outlier_method_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(outlier_method_frame, text="Method:").pack(side=tk.LEFT)
        self.outlier_method_var = tk.StringVar(value=self.config.data.outlier_method)
        outlier_combo = ttk.Combobox(outlier_method_frame, textvariable=self.outlier_method_var,
                                   values=["iqr", "zscore", "modified_zscore"], state="readonly")
        outlier_combo.pack(side=tk.RIGHT)
        
        # Data transformation
        transform_frame = ttk.LabelFrame(scrollable_frame, text="Data Transformation")
        transform_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.auto_transform_var = tk.BooleanVar(value=self.config.data.auto_transform)
        ttk.Checkbutton(transform_frame, text="Automatic transformation selection", 
                       variable=self.auto_transform_var).pack(anchor=tk.W, padx=5, pady=5)
        
        transform_method_frame = ttk.Frame(transform_frame)
        transform_method_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(transform_method_frame, text="Preferred method:").pack(side=tk.LEFT)
        self.transform_method_var = tk.StringVar(value=self.config.data.transform_method)
        transform_combo = ttk.Combobox(transform_method_frame, textvariable=self.transform_method_var,
                                     values=["none", "log", "sqrt", "box_cox"], state="readonly")
        transform_combo.pack(side=tk.RIGHT)
        
        # Data scaling
        scaling_frame = ttk.LabelFrame(scrollable_frame, text="Data Scaling")
        scaling_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.auto_scaling_var = tk.BooleanVar(value=self.config.data.auto_scaling)
        ttk.Checkbutton(scaling_frame, text="Automatic scaling", 
                       variable=self.auto_scaling_var).pack(anchor=tk.W, padx=5, pady=5)
        
        scaling_method_frame = ttk.Frame(scaling_frame)
        scaling_method_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(scaling_method_frame, text="Method:").pack(side=tk.LEFT)
        self.scaling_method_var = tk.StringVar(value=self.config.data.scaling_method)
        scaling_combo = ttk.Combobox(scaling_method_frame, textvariable=self.scaling_method_var,
                                   values=["none", "standard", "minmax", "robust"], state="readonly")
        scaling_combo.pack(side=tk.RIGHT)
    
    def _create_analysis_step(self) -> None:
        """Create the analysis options step."""
        # Title
        title_label = ttk.Label(self.content_frame, text="Analysis Options", font=('TkDefaultFont', 14, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Instructions
        instructions = (
            "Configure the statistical analysis settings. PyADAP can automatically select "
            "appropriate tests based on your data and variables."
        )
        ttk.Label(self.content_frame, text=instructions, wraplength=600).pack(pady=(0, 20))
        
        # Analysis options
        options_frame = ttk.LabelFrame(self.content_frame, text="Statistical Options")
        options_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Significance level
        alpha_frame = ttk.Frame(options_frame)
        alpha_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(alpha_frame, text="Significance level (α):").pack(side=tk.LEFT)
        self.alpha_var = tk.DoubleVar(value=self.config.statistical.alpha)
        ttk.Spinbox(alpha_frame, from_=0.001, to=0.1, increment=0.001, 
                   textvariable=self.alpha_var, width=10).pack(side=tk.RIGHT)
        
        # Test selection
        self.auto_test_var = tk.BooleanVar(value=self.config.statistical.auto_test_selection)
        ttk.Checkbutton(options_frame, text="Automatic test selection", 
                       variable=self.auto_test_var).pack(anchor=tk.W, padx=10, pady=5)
        
        self.robust_tests_var = tk.BooleanVar(value=self.config.statistical.use_robust_tests)
        ttk.Checkbutton(options_frame, text="Use robust statistical tests", 
                       variable=self.robust_tests_var).pack(anchor=tk.W, padx=10, pady=5)
        
        # Assumption checking
        assumption_frame = ttk.LabelFrame(self.content_frame, text="Assumption Checking")
        assumption_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.check_normality_var = tk.BooleanVar(value=self.config.statistical.check_normality)
        ttk.Checkbutton(assumption_frame, text="Check normality", 
                       variable=self.check_normality_var).pack(anchor=tk.W, padx=10, pady=5)
        
        self.check_homogeneity_var = tk.BooleanVar(value=self.config.statistical.check_homogeneity)
        ttk.Checkbutton(assumption_frame, text="Check homogeneity of variance", 
                       variable=self.check_homogeneity_var).pack(anchor=tk.W, padx=10, pady=5)
        
        # Output options
        output_frame = ttk.LabelFrame(self.content_frame, text="Output Options")
        output_frame.pack(fill=tk.X)
        
        self.save_plots_var = tk.BooleanVar(value=self.config.visualization.save_plots)
        ttk.Checkbutton(output_frame, text="Save plots automatically", 
                       variable=self.save_plots_var).pack(anchor=tk.W, padx=10, pady=5)
        
        self.export_excel_var = tk.BooleanVar(value=self.config.output.export_excel)
        ttk.Checkbutton(output_frame, text="Export results to Excel", 
                       variable=self.export_excel_var).pack(anchor=tk.W, padx=10, pady=5)
    
    def _create_review_step(self) -> None:
        """Create the review and run step."""
        # Title
        title_label = ttk.Label(self.content_frame, text="Review & Run Analysis", font=('TkDefaultFont', 14, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Instructions
        instructions = (
            "Review your analysis configuration below. Click 'Finish' to run the analysis."
        )
        ttk.Label(self.content_frame, text=instructions, wraplength=600).pack(pady=(0, 20))
        
        # Review frame
        review_frame = ttk.LabelFrame(self.content_frame, text="Analysis Summary")
        review_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create text widget for review
        text_frame = ttk.Frame(review_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.review_text = tk.Text(text_frame, wrap=tk.WORD, state=tk.DISABLED)
        review_scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.review_text.yview)
        self.review_text.configure(yscrollcommand=review_scroll.set)
        
        self.review_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        review_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Generate review content
        self._update_review()
    
    def _update_review(self) -> None:
        """Update the review content."""
        self.review_text.config(state=tk.NORMAL)
        self.review_text.delete(1.0, tk.END)
        
        # Data information
        if self.data is not None:
            rows, cols = self.data.shape
            self.review_text.insert(tk.END, f"Data: {rows:,} rows × {cols} columns\n\n")
        
        # Variables
        self.review_text.insert(tk.END, "Variables:\n")
        if hasattr(self, 'independent_listbox'):
            independent_vars = list(self.independent_listbox.get(0, tk.END))
            dependent_vars = list(self.dependent_listbox.get(0, tk.END))
            subject_vars = list(self.subject_listbox.get(0, tk.END))
            covariate_vars = list(self.covariate_listbox.get(0, tk.END))
            
            self.review_text.insert(tk.END, f"  Independent: {', '.join(independent_vars) if independent_vars else 'None'}\n")
            self.review_text.insert(tk.END, f"  Dependent: {', '.join(dependent_vars) if dependent_vars else 'None'}\n")
            self.review_text.insert(tk.END, f"  Subject ID: {', '.join(subject_vars) if subject_vars else 'None'}\n")
            self.review_text.insert(tk.END, f"  Covariates: {', '.join(covariate_vars) if covariate_vars else 'None'}\n\n")
        
        # Preprocessing options
        self.review_text.insert(tk.END, "Preprocessing:\n")
        if hasattr(self, 'missing_strategy_var'):
            self.review_text.insert(tk.END, f"  Missing data: {self.missing_strategy_var.get()}\n")
            self.review_text.insert(tk.END, f"  Outlier detection: {'Yes' if self.detect_outliers_var.get() else 'No'}\n")
            if self.detect_outliers_var.get():
                self.review_text.insert(tk.END, f"  Outlier method: {self.outlier_method_var.get()}\n")
            self.review_text.insert(tk.END, f"  Auto transformation: {'Yes' if self.auto_transform_var.get() else 'No'}\n")
            if not self.auto_transform_var.get():
                self.review_text.insert(tk.END, f"  Transform method: {self.transform_method_var.get()}\n")
            self.review_text.insert(tk.END, f"  Auto scaling: {'Yes' if self.auto_scaling_var.get() else 'No'}\n")
            if not self.auto_scaling_var.get():
                self.review_text.insert(tk.END, f"  Scaling method: {self.scaling_method_var.get()}\n")
        self.review_text.insert(tk.END, "\n")
        
        # Analysis options
        self.review_text.insert(tk.END, "Analysis Options:\n")
        if hasattr(self, 'alpha_var'):
            self.review_text.insert(tk.END, f"  Significance level: {self.alpha_var.get()}\n")
            self.review_text.insert(tk.END, f"  Auto test selection: {'Yes' if self.auto_test_var.get() else 'No'}\n")
            self.review_text.insert(tk.END, f"  Robust tests: {'Yes' if self.robust_tests_var.get() else 'No'}\n")
            self.review_text.insert(tk.END, f"  Check normality: {'Yes' if self.check_normality_var.get() else 'No'}\n")
            self.review_text.insert(tk.END, f"  Check homogeneity: {'Yes' if self.check_homogeneity_var.get() else 'No'}\n")
        self.review_text.insert(tk.END, "\n")
        
        # Output options
        self.review_text.insert(tk.END, "Output Options:\n")
        if hasattr(self, 'save_plots_var'):
            self.review_text.insert(tk.END, f"  Save plots: {'Yes' if self.save_plots_var.get() else 'No'}\n")
            self.review_text.insert(tk.END, f"  Export to Excel: {'Yes' if self.export_excel_var.get() else 'No'}\n")
        
        self.review_text.config(state=tk.DISABLED)
    
    def _browse_file(self) -> None:
        """Browse for data file."""
        filetypes = [
            ("Excel files", "*.xlsx *.xls"),
            ("CSV files", "*.csv"),
            ("TSV files", "*.tsv"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=filetypes
        )
        
        if filename:
            self.file_path_var.set(filename)
            self._load_data(filename)
    
    def _load_data(self, filename: str) -> None:
        """Load data from file.
        
        Args:
            filename: Path to data file
        """
        try:
            # Clear preview frame
            for widget in self.data_preview_frame.winfo_children():
                widget.destroy()
            
            # Create data manager and load data
            self.data_manager = DataManager(self.config)
            self.data = self.data_manager.load_data(filename)
            
            if self.data is not None:
                # Show preview
                preview_label = ttk.Label(self.data_preview_frame, text="Data Preview:", font=('TkDefaultFont', 10, 'bold'))
                preview_label.pack(anchor=tk.W, padx=5, pady=5)
                
                # Create treeview for preview
                tree_frame = ttk.Frame(self.data_preview_frame)
                tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                
                preview_tree = ttk.Treeview(tree_frame, show="headings", height=8)
                
                # Set up columns (limit to first 10 columns)
                display_cols = list(self.data.columns)[:10]
                preview_tree["columns"] = display_cols
                
                for col in display_cols:
                    preview_tree.heading(col, text=col)
                    preview_tree.column(col, width=100)
                
                # Add data (first 10 rows)
                for idx, (_, row) in enumerate(self.data.head(10).iterrows()):
                    values = [str(row[col])[:20] for col in display_cols]  # Truncate long values
                    preview_tree.insert("", tk.END, values=values)
                
                preview_tree.pack(fill=tk.BOTH, expand=True)
                
                # Add scrollbar
                scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=preview_tree.yview)
                preview_tree.configure(yscrollcommand=scrollbar.set)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                
                # Show data info
                rows, cols = self.data.shape
                info_text = f"Loaded: {rows:,} rows × {cols} columns"
                if cols > 10:
                    info_text += f" (showing first 10 columns)"
                
                info_label = ttk.Label(self.data_preview_frame, text=info_text, foreground="green")
                info_label.pack(pady=5)
                
                self.status_var.set("Data loaded successfully")
                self.logger.info(f"Data loaded: {filename} ({rows} rows, {cols} columns)")
            
        except Exception as e:
            error_msg = f"Error loading data: {str(e)}"
            self.logger.error(error_msg)
            
            error_label = ttk.Label(self.data_preview_frame, text=error_msg, foreground="red")
            error_label.pack(pady=20)
            
            self.status_var.set("Error loading data")
    
    def _assign_variable(self, var_type: str) -> None:
        """Assign selected variables to a type.
        
        Args:
            var_type: Type of variable ('independent', 'dependent', 'subject', 'covariate')
        """
        selected_indices = self.available_listbox.curselection()
        if not selected_indices:
            return
        
        # Get target listbox
        target_listbox = getattr(self, f"{var_type}_listbox")
        
        # Move variables
        for idx in reversed(selected_indices):  # Reverse to maintain indices
            var_name = self.available_listbox.get(idx)
            target_listbox.insert(tk.END, var_name)
            self.available_listbox.delete(idx)
    
    def _remove_variable(self) -> None:
        """Remove variable from assigned lists back to available."""
        # Check all assigned listboxes
        listboxes = [self.independent_listbox, self.dependent_listbox, 
                    self.subject_listbox, self.covariate_listbox]
        
        for listbox in listboxes:
            selected_indices = listbox.curselection()
            for idx in reversed(selected_indices):
                var_name = listbox.get(idx)
                self.available_listbox.insert(tk.END, var_name)
                listbox.delete(idx)
    
    def _validate_step(self) -> bool:
        """Validate current step.
        
        Returns:
            True if step is valid, False otherwise
        """
        if self.current_step == 0:  # Data loading
            if self.data is None:
                messagebox.showerror("Error", "Please load a data file.")
                return False
        
        elif self.current_step == 1:  # Variable selection
            if not hasattr(self, 'dependent_listbox'):
                return True  # Skip validation if widgets not created
            
            dependent_vars = list(self.dependent_listbox.get(0, tk.END))
            if not dependent_vars:
                messagebox.showerror("Error", "Please select at least one dependent variable.")
                return False
        
        return True
    
    def _go_back(self) -> None:
        """Go to previous step."""
        if self.current_step > 0:
            self._show_step(self.current_step - 1)
    
    def _go_next(self) -> None:
        """Go to next step or finish."""
        if not self._validate_step():
            return
        
        if self.current_step < len(self.steps) - 1:
            self._show_step(self.current_step + 1)
        else:
            self._finish_wizard()
    
    def _finish_wizard(self) -> None:
        """Finish the wizard and run analysis."""
        try:
            # Update configuration with wizard settings
            self._update_config()
            
            # Prepare variable assignments
            variable_assignments = {}
            if hasattr(self, 'independent_listbox'):
                variable_assignments['independent'] = list(self.independent_listbox.get(0, tk.END))
                variable_assignments['dependent'] = list(self.dependent_listbox.get(0, tk.END))
                variable_assignments['subject'] = list(self.subject_listbox.get(0, tk.END))
                variable_assignments['covariates'] = list(self.covariate_listbox.get(0, tk.END))
            
            # Set result
            self.result = {
                'data': self.data,
                'data_manager': self.data_manager,
                'config': self.config,
                'variables': variable_assignments
            }
            
            # Close wizard
            self.wizard.destroy()
            
            # Call completion callback
            if self.on_complete:
                self.on_complete(self.result)
            
            self.logger.info("Analysis wizard completed successfully")
        
        except Exception as e:
            self.logger.error(f"Error finishing wizard: {str(e)}")
            messagebox.showerror("Error", f"Failed to complete wizard: {str(e)}")
    
    def _update_config(self) -> None:
        """Update configuration with wizard settings."""
        if hasattr(self, 'missing_strategy_var'):
            self.config.data.missing_data_strategy = self.missing_strategy_var.get()
            self.config.data.detect_outliers = self.detect_outliers_var.get()
            self.config.data.outlier_method = self.outlier_method_var.get()
            self.config.data.auto_transform = self.auto_transform_var.get()
            self.config.data.transform_method = self.transform_method_var.get()
            self.config.data.auto_scaling = self.auto_scaling_var.get()
            self.config.data.scaling_method = self.scaling_method_var.get()
        
        if hasattr(self, 'alpha_var'):
            self.config.statistical.alpha = self.alpha_var.get()
            self.config.statistical.auto_test_selection = self.auto_test_var.get()
            self.config.statistical.use_robust_tests = self.robust_tests_var.get()
            self.config.statistical.check_normality = self.check_normality_var.get()
            self.config.statistical.check_homogeneity = self.check_homogeneity_var.get()
        
        if hasattr(self, 'save_plots_var'):
            self.config.visualization.save_plots = self.save_plots_var.get()
            self.config.output.export_excel = self.export_excel_var.get()
    
    def _on_cancel(self) -> None:
        """Handle cancel button or window close."""
        if messagebox.askyesno("Confirm", "Are you sure you want to cancel the wizard?"):
            self.wizard.destroy()