"""Data preview widget for PyADAP 3.0

This module provides a comprehensive data preview widget that allows users
to view, explore, and understand their data before analysis.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path

from ..utils import Logger, get_logger, format_number


class DataPreviewWidget:
    """Widget for previewing and exploring data."""
    
    def __init__(self, parent: tk.Widget, data: Optional[pd.DataFrame] = None):
        """Initialize the data preview widget.
        
        Args:
            parent: Parent widget
            data: DataFrame to preview (optional)
        """
        self.parent = parent
        self.data = data
        self.logger = get_logger("PyADAP.DataPreview")
        
        # Create main window
        self.window = tk.Toplevel(parent)
        self.window.title("Data Preview")
        self.window.geometry("1000x700")
        
        # Create main frame
        self.main_frame = ttk.Frame(self.window)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create UI
        self._create_ui()
        
        # Load data if provided
        if self.data is not None:
            self.load_data(self.data)
    
    def _create_ui(self) -> None:
        """Create the preview UI."""
        # Create notebook for different views
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self._create_data_tab()
        self._create_summary_tab()
        self._create_missing_tab()
        self._create_types_tab()
        
        # Create control frame
        self._create_controls()
    
    def _create_data_tab(self) -> None:
        """Create the data view tab."""
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="Data View")
        
        # Create toolbar
        toolbar_frame = ttk.Frame(self.data_frame)
        toolbar_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Navigation controls
        nav_frame = ttk.Frame(toolbar_frame)
        nav_frame.pack(side=tk.LEFT)
        
        ttk.Button(nav_frame, text="First", command=self._go_first).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Previous", command=self._go_previous).pack(side=tk.LEFT, padx=2)
        
        # Page info
        self.page_var = tk.StringVar(value="Page 1 of 1")
        ttk.Label(nav_frame, textvariable=self.page_var).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(nav_frame, text="Next", command=self._go_next).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Last", command=self._go_last).pack(side=tk.LEFT, padx=2)
        
        # Rows per page
        rows_frame = ttk.Frame(toolbar_frame)
        rows_frame.pack(side=tk.RIGHT)
        
        ttk.Label(rows_frame, text="Rows per page:").pack(side=tk.LEFT, padx=5)
        self.rows_per_page_var = tk.IntVar(value=100)
        rows_combo = ttk.Combobox(rows_frame, textvariable=self.rows_per_page_var,
                                 values=[50, 100, 200, 500, 1000], width=8, state="readonly")
        rows_combo.pack(side=tk.LEFT)
        rows_combo.bind("<<ComboboxSelected>>", self._on_rows_per_page_changed)
        
        # Create treeview for data display
        tree_frame = ttk.Frame(self.data_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Treeview with scrollbars
        self.data_tree = ttk.Treeview(tree_frame, show="tree headings")
        
        # Vertical scrollbar
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=v_scrollbar.set)
        
        # Horizontal scrollbar
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.data_tree.xview)
        self.data_tree.configure(xscrollcommand=h_scrollbar.set)
        
        # Pack scrollbars and treeview
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.data_tree.pack(fill=tk.BOTH, expand=True)
        
        # Initialize pagination
        self.current_page = 0
        self.total_pages = 0
    
    def _create_summary_tab(self) -> None:
        """Create the summary statistics tab."""
        self.summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_frame, text="Summary")
        
        # Create treeview for summary
        summary_tree_frame = ttk.Frame(self.summary_frame)
        summary_tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.summary_tree = ttk.Treeview(summary_tree_frame, columns=("Statistic", "Value"), show="headings")
        self.summary_tree.heading("Statistic", text="Statistic")
        self.summary_tree.heading("Value", text="Value")
        self.summary_tree.column("Statistic", width=200)
        self.summary_tree.column("Value", width=150)
        
        # Scrollbar for summary
        summary_scrollbar = ttk.Scrollbar(summary_tree_frame, orient=tk.VERTICAL, command=self.summary_tree.yview)
        self.summary_tree.configure(yscrollcommand=summary_scrollbar.set)
        
        summary_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.summary_tree.pack(fill=tk.BOTH, expand=True)
    
    def _create_missing_tab(self) -> None:
        """Create the missing data tab."""
        self.missing_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.missing_frame, text="Missing Data")
        
        # Create treeview for missing data info
        missing_tree_frame = ttk.Frame(self.missing_frame)
        missing_tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.missing_tree = ttk.Treeview(missing_tree_frame, 
                                        columns=("Column", "Missing Count", "Missing %", "Data Type"), 
                                        show="headings")
        self.missing_tree.heading("Column", text="Column")
        self.missing_tree.heading("Missing Count", text="Missing Count")
        self.missing_tree.heading("Missing %", text="Missing %")
        self.missing_tree.heading("Data Type", text="Data Type")
        
        self.missing_tree.column("Column", width=150)
        self.missing_tree.column("Missing Count", width=100)
        self.missing_tree.column("Missing %", width=100)
        self.missing_tree.column("Data Type", width=100)
        
        # Scrollbar for missing data
        missing_scrollbar = ttk.Scrollbar(missing_tree_frame, orient=tk.VERTICAL, command=self.missing_tree.yview)
        self.missing_tree.configure(yscrollcommand=missing_scrollbar.set)
        
        missing_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.missing_tree.pack(fill=tk.BOTH, expand=True)
    
    def _create_types_tab(self) -> None:
        """Create the data types tab."""
        self.types_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.types_frame, text="Data Types")
        
        # Create treeview for data types
        types_tree_frame = ttk.Frame(self.types_frame)
        types_tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.types_tree = ttk.Treeview(types_tree_frame, 
                                      columns=("Column", "Current Type", "Suggested Type", "Unique Values", "Sample Values"), 
                                      show="headings")
        self.types_tree.heading("Column", text="Column")
        self.types_tree.heading("Current Type", text="Current Type")
        self.types_tree.heading("Suggested Type", text="Suggested Type")
        self.types_tree.heading("Unique Values", text="Unique Values")
        self.types_tree.heading("Sample Values", text="Sample Values")
        
        self.types_tree.column("Column", width=120)
        self.types_tree.column("Current Type", width=100)
        self.types_tree.column("Suggested Type", width=100)
        self.types_tree.column("Unique Values", width=100)
        self.types_tree.column("Sample Values", width=200)
        
        # Scrollbar for data types
        types_scrollbar = ttk.Scrollbar(types_tree_frame, orient=tk.VERTICAL, command=self.types_tree.yview)
        self.types_tree.configure(yscrollcommand=types_scrollbar.set)
        
        types_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.types_tree.pack(fill=tk.BOTH, expand=True)
    
    def _create_controls(self) -> None:
        """Create control buttons."""
        control_frame = ttk.Frame(self.main_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Refresh button
        ttk.Button(control_frame, text="Refresh", command=self.refresh).pack(side=tk.LEFT, padx=5)
        
        # Export button
        ttk.Button(control_frame, text="Export Summary", command=self._export_summary).pack(side=tk.LEFT, padx=5)
        
        # Info label
        self.info_var = tk.StringVar(value="No data loaded")
        ttk.Label(control_frame, textvariable=self.info_var).pack(side=tk.RIGHT, padx=5)
    
    def set_data(self, data: pd.DataFrame) -> None:
        """Set the data to preview.
        
        Args:
            data: DataFrame to preview
        """
        self.data = data.copy() if data is not None else None
        self.current_page = 0
        self.refresh()
    
    def refresh(self) -> None:
        """Refresh all views."""
        if self.data is None:
            self._clear_all_views()
            self.info_var.set("No data loaded")
            return
        
        try:
            self._update_data_view()
            self._update_summary_view()
            self._update_missing_view()
            self._update_types_view()
            
            # Update info
            rows, cols = self.data.shape
            self.info_var.set(f"Data: {rows:,} rows × {cols} columns")
            
            self.logger.info(f"Data preview refreshed: {rows} rows, {cols} columns")
            
        except Exception as e:
            self.logger.error(f"Error refreshing data preview: {str(e)}")
            messagebox.showerror("Error", f"Failed to refresh data preview: {str(e)}")
    
    def _clear_all_views(self) -> None:
        """Clear all views."""
        # Clear data view
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        self.data_tree["columns"] = ()
        
        # Clear summary view
        for item in self.summary_tree.get_children():
            self.summary_tree.delete(item)
        
        # Clear missing view
        for item in self.missing_tree.get_children():
            self.missing_tree.delete(item)
        
        # Clear types view
        for item in self.types_tree.get_children():
            self.types_tree.delete(item)
        
        # Reset pagination
        self.current_page = 0
        self.total_pages = 0
        self.page_var.set("Page 1 of 1")
    
    def _update_data_view(self) -> None:
        """Update the data view."""
        if self.data is None:
            return
        
        # Clear existing data
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        # Set up columns
        columns = ["Index"] + list(self.data.columns)
        self.data_tree["columns"] = columns
        self.data_tree.column("#0", width=0, stretch=False)  # Hide tree column
        
        # Configure column headings and widths
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100, anchor=tk.W)
        
        # Calculate pagination
        rows_per_page = self.rows_per_page_var.get()
        total_rows = len(self.data)
        self.total_pages = max(1, (total_rows + rows_per_page - 1) // rows_per_page)
        
        # Ensure current page is valid
        if self.current_page >= self.total_pages:
            self.current_page = max(0, self.total_pages - 1)
        
        # Get data for current page
        start_idx = self.current_page * rows_per_page
        end_idx = min(start_idx + rows_per_page, total_rows)
        page_data = self.data.iloc[start_idx:end_idx]
        
        # Insert data
        for idx, (row_idx, row) in enumerate(page_data.iterrows()):
            values = [str(row_idx)] + [self._format_cell_value(val) for val in row]
            self.data_tree.insert("", tk.END, values=values)
        
        # Update page info
        self.page_var.set(f"Page {self.current_page + 1} of {self.total_pages}")
    
    def _update_summary_view(self) -> None:
        """Update the summary statistics view."""
        if self.data is None:
            return
        
        # Clear existing summary
        for item in self.summary_tree.get_children():
            self.summary_tree.delete(item)
        
        try:
            # Basic info
            rows, cols = self.data.shape
            self.summary_tree.insert("", tk.END, values=("Total Rows", f"{rows:,}"))
            self.summary_tree.insert("", tk.END, values=("Total Columns", f"{cols:,}"))
            
            # Memory usage
            memory_mb = self.data.memory_usage(deep=True).sum() / 1024 / 1024
            self.summary_tree.insert("", tk.END, values=("Memory Usage", f"{memory_mb:.2f} MB"))
            
            # Data types
            numeric_cols = len(self.data.select_dtypes(include=[np.number]).columns)
            categorical_cols = len(self.data.select_dtypes(include=['object', 'category']).columns)
            datetime_cols = len(self.data.select_dtypes(include=['datetime64']).columns)
            
            self.summary_tree.insert("", tk.END, values=("Numeric Columns", str(numeric_cols)))
            self.summary_tree.insert("", tk.END, values=("Categorical Columns", str(categorical_cols)))
            self.summary_tree.insert("", tk.END, values=("DateTime Columns", str(datetime_cols)))
            
            # Missing data
            total_missing = self.data.isnull().sum().sum()
            missing_percent = (total_missing / (rows * cols)) * 100
            self.summary_tree.insert("", tk.END, values=("Total Missing Values", f"{total_missing:,}"))
            self.summary_tree.insert("", tk.END, values=("Missing Data %", f"{missing_percent:.2f}%"))
            
            # Duplicates
            duplicates = self.data.duplicated().sum()
            self.summary_tree.insert("", tk.END, values=("Duplicate Rows", f"{duplicates:,}"))
            
            # Numeric summary for numeric columns
            numeric_data = self.data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                self.summary_tree.insert("", tk.END, values=("", ""))  # Separator
                self.summary_tree.insert("", tk.END, values=("=== Numeric Summary ===", ""))
                
                desc = numeric_data.describe()
                for stat in desc.index:
                    if stat == 'count':
                        continue  # Skip count as it's redundant
                    mean_val = desc.loc[stat].mean()
                    self.summary_tree.insert("", tk.END, values=(f"Mean {stat.title()}", format_number(mean_val)))
            
        except Exception as e:
            self.logger.error(f"Error updating summary view: {str(e)}")
            self.summary_tree.insert("", tk.END, values=("Error", "Failed to generate summary"))
    
    def _update_missing_view(self) -> None:
        """Update the missing data view."""
        if self.data is None:
            return
        
        # Clear existing missing data info
        for item in self.missing_tree.get_children():
            self.missing_tree.delete(item)
        
        try:
            total_rows = len(self.data)
            
            for col in self.data.columns:
                missing_count = self.data[col].isnull().sum()
                missing_percent = (missing_count / total_rows) * 100
                data_type = str(self.data[col].dtype)
                
                self.missing_tree.insert("", tk.END, values=(
                    col,
                    f"{missing_count:,}",
                    f"{missing_percent:.2f}%",
                    data_type
                ))
        
        except Exception as e:
            self.logger.error(f"Error updating missing data view: {str(e)}")
            self.missing_tree.insert("", tk.END, values=("Error", "Failed to analyze missing data", "", ""))
    
    def _update_types_view(self) -> None:
        """Update the data types view."""
        if self.data is None:
            return
        
        # Clear existing types info
        for item in self.types_tree.get_children():
            self.types_tree.delete(item)
        
        try:
            for col in self.data.columns:
                current_type = str(self.data[col].dtype)
                unique_count = self.data[col].nunique()
                
                # Suggest appropriate type
                suggested_type = self._suggest_data_type(self.data[col])
                
                # Get sample values
                sample_values = self._get_sample_values(self.data[col])
                
                self.types_tree.insert("", tk.END, values=(
                    col,
                    current_type,
                    suggested_type,
                    f"{unique_count:,}",
                    sample_values
                ))
        
        except Exception as e:
            self.logger.error(f"Error updating types view: {str(e)}")
            self.types_tree.insert("", tk.END, values=("Error", "Failed to analyze data types", "", "", ""))
    
    def _suggest_data_type(self, series: pd.Series) -> str:
        """Suggest appropriate data type for a series.
        
        Args:
            series: Pandas series to analyze
            
        Returns:
            Suggested data type as string
        """
        try:
            # Skip if all values are missing
            if series.isnull().all():
                return "unknown"
            
            # Current type
            current_dtype = series.dtype
            
            # If already numeric, check if it should be categorical
            if pd.api.types.is_numeric_dtype(series):
                unique_ratio = series.nunique() / len(series.dropna())
                if unique_ratio < 0.05 and series.nunique() < 20:
                    return "categorical"
                return "numeric"
            
            # If object type, try to infer
            if pd.api.types.is_object_dtype(series):
                # Try to convert to numeric
                try:
                    pd.to_numeric(series.dropna())
                    return "numeric"
                except (ValueError, TypeError):
                    pass
                
                # Try to convert to datetime
                try:
                    pd.to_datetime(series.dropna())
                    return "datetime"
                except (ValueError, TypeError):
                    pass
                
                # Check if categorical
                unique_ratio = series.nunique() / len(series.dropna())
                if unique_ratio < 0.1 or series.nunique() < 50:
                    return "categorical"
                
                return "text"
            
            # If datetime
            if pd.api.types.is_datetime64_any_dtype(series):
                return "datetime"
            
            # If categorical
            if pd.api.types.is_categorical_dtype(series):
                return "categorical"
            
            # If boolean
            if pd.api.types.is_bool_dtype(series):
                return "boolean"
            
            return str(current_dtype)
        
        except Exception:
            return "unknown"
    
    def _get_sample_values(self, series: pd.Series, n: int = 3) -> str:
        """Get sample values from a series.
        
        Args:
            series: Pandas series
            n: Number of sample values to get
            
        Returns:
            Comma-separated string of sample values
        """
        try:
            # Get non-null values
            non_null = series.dropna()
            if non_null.empty:
                return "All missing"
            
            # Get unique values
            unique_vals = non_null.unique()
            
            # Sample values
            sample_size = min(n, len(unique_vals))
            samples = unique_vals[:sample_size]
            
            # Format values
            formatted_samples = [self._format_cell_value(val) for val in samples]
            
            result = ", ".join(formatted_samples)
            if len(unique_vals) > sample_size:
                result += ", ..."
            
            return result
        
        except Exception:
            return "Error"
    
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
        if len(str_val) > 50:
            return str_val[:47] + "..."
        
        return str_val
    
    def _go_first(self) -> None:
        """Go to first page."""
        self.current_page = 0
        self._update_data_view()
    
    def _go_previous(self) -> None:
        """Go to previous page."""
        if self.current_page > 0:
            self.current_page -= 1
            self._update_data_view()
    
    def _go_next(self) -> None:
        """Go to next page."""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self._update_data_view()
    
    def _go_last(self) -> None:
        """Go to last page."""
        self.current_page = max(0, self.total_pages - 1)
        self._update_data_view()
    
    def _on_rows_per_page_changed(self, event=None) -> None:
        """Handle rows per page change."""
        self.current_page = 0  # Reset to first page
        self._update_data_view()
    
    def _export_summary(self) -> None:
        """Export summary to file."""
        if self.data is None:
            messagebox.showwarning("No Data", "No data to export.")
            return
        
        try:
            from tkinter import filedialog
            
            filename = filedialog.asksaveasfilename(
                title="Export Summary",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("PyADAP Data Summary\n")
                    f.write("=" * 50 + "\n\n")
                    
                    # Basic info
                    rows, cols = self.data.shape
                    f.write(f"Dataset Shape: {rows:,} rows × {cols} columns\n")
                    f.write(f"Memory Usage: {self.data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB\n\n")
                    
                    # Column info
                    f.write("Column Information:\n")
                    f.write("-" * 30 + "\n")
                    for col in self.data.columns:
                        dtype = str(self.data[col].dtype)
                        missing = self.data[col].isnull().sum()
                        missing_pct = (missing / len(self.data)) * 100
                        unique = self.data[col].nunique()
                        
                        f.write(f"{col}:\n")
                        f.write(f"  Type: {dtype}\n")
                        f.write(f"  Missing: {missing:,} ({missing_pct:.2f}%)\n")
                        f.write(f"  Unique: {unique:,}\n\n")
                    
                    # Numeric summary
                    numeric_data = self.data.select_dtypes(include=[np.number])
                    if not numeric_data.empty:
                        f.write("Numeric Summary:\n")
                        f.write("-" * 20 + "\n")
                        f.write(str(numeric_data.describe()))
                        f.write("\n\n")
                    
                    # Missing data summary
                    total_missing = self.data.isnull().sum().sum()
                    if total_missing > 0:
                        f.write("Missing Data Summary:\n")
                        f.write("-" * 25 + "\n")
                        missing_by_col = self.data.isnull().sum().sort_values(ascending=False)
                        for col, missing in missing_by_col.items():
                            if missing > 0:
                                pct = (missing / len(self.data)) * 100
                                f.write(f"{col}: {missing:,} ({pct:.2f}%)\n")
                
                messagebox.showinfo("Success", f"Summary exported to {filename}")
                self.logger.info(f"Data summary exported to {filename}")
        
        except Exception as e:
            self.logger.error(f"Error exporting summary: {str(e)}")
            messagebox.showerror("Error", f"Failed to export summary: {str(e)}")
    
    def load_data(self, data: pd.DataFrame) -> None:
        """Load data into the preview widget.
        
        Args:
            data: DataFrame to load
        """
        self.data = data
        self._refresh_all_tabs()
    
    def show(self) -> None:
        """Show the data preview window."""
        self.window.deiconify()
        self.window.lift()
        self.window.focus_force()
    
    def _refresh_all_tabs(self) -> None:
        """Refresh all tabs with current data."""
        if self.data is not None:
            self._update_data_view()
            self._update_summary_view()
            self._update_missing_view()
            self._update_types_view()