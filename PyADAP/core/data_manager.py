"""Advanced Data Management for PyADAP 3.0

This module provides intelligent data loading, preprocessing, and validation
with automatic type detection and quality assessment.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import boxcox, yeojohnson
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

from ..config import Config
from ..utils import Logger, Validator


class DataManager:
    """Advanced data management with intelligent preprocessing."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize DataManager.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.logger = Logger()
        self.validator = Validator()
        
        # Data storage
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}
        
        # Variable information
        self.subject_column: Optional[str] = None
        self.independent_vars: List[str] = []
        self.dependent_vars: List[str] = []
        self.covariate_vars: List[str] = []
        
        # Data quality information
        self.quality_report: Dict[str, Any] = {}
        self.transformations_applied: List[str] = []
        
    def load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load data from file with intelligent type detection.
        
        Args:
            file_path: Path to data file
            **kwargs: Additional parameters for pandas read functions
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext in ['.xlsx', '.xls']:
                self.raw_data = pd.read_excel(file_path, **kwargs)
            elif file_ext == '.csv':
                # Try to detect delimiter and encoding
                self.raw_data = self._smart_csv_read(file_path, **kwargs)
            elif file_ext in ['.tsv', '.txt']:
                self.raw_data = pd.read_csv(file_path, sep='\t', **kwargs)
            elif file_ext == '.json':
                self.raw_data = pd.read_json(file_path, **kwargs)
            elif file_ext in ['.pkl', '.pickle']:
                self.raw_data = pd.read_pickle(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            if self.raw_data.empty:
                raise ValueError("Loaded data is empty")
            
            # Store metadata
            self.metadata.update({
                'file_path': file_path,
                'file_size': os.path.getsize(file_path),
                'n_rows': len(self.raw_data),
                'n_columns': len(self.raw_data.columns),
                'load_time': pd.Timestamp.now()
            })
            
            # Auto-detect data types if enabled
            if self.config.data.auto_detect_types:
                self._auto_detect_types()
            
            # Generate initial quality report
            self._generate_quality_report()
            
            self.logger.info(f"Successfully loaded data: {self.metadata['n_rows']} rows, {self.metadata['n_columns']} columns")
            
            return self.raw_data.copy()
            
        except Exception as e:
            self.logger.error(f"Failed to load data from {file_path}: {str(e)}")
            raise
    
    def _smart_csv_read(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Intelligently read CSV files with automatic delimiter and encoding detection."""
        import chardet
        
        # Detect encoding
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10KB
            encoding_result = chardet.detect(raw_data)
            encoding = encoding_result['encoding'] or 'utf-8'
        
        # Try different delimiters
        delimiters = [',', ';', '\t', '|']
        best_delimiter = ','
        max_columns = 0
        
        for delimiter in delimiters:
            try:
                temp_df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding, nrows=5)
                if len(temp_df.columns) > max_columns:
                    max_columns = len(temp_df.columns)
                    best_delimiter = delimiter
            except Exception as e:
                self.logger.debug(f"Failed to read with delimiter '{delimiter}': {e}")
                continue
        
        # Read with best parameters
        return pd.read_csv(file_path, delimiter=best_delimiter, encoding=encoding, **kwargs)
    
    def _auto_detect_types(self) -> None:
        """Automatically detect and convert data types."""
        for column in self.raw_data.columns:
            # Try to convert to numeric
            if self.raw_data[column].dtype == 'object':
                # Check if it's numeric
                numeric_series = pd.to_numeric(self.raw_data[column], errors='coerce')
                if not numeric_series.isna().all():
                    # If most values can be converted to numeric
                    if numeric_series.notna().sum() / len(numeric_series) > 0.8:
                        self.raw_data[column] = numeric_series
                        continue
                
                # Check if it's datetime
                try:
                    datetime_series = pd.to_datetime(self.raw_data[column], errors='coerce')
                    if not datetime_series.isna().all():
                        if datetime_series.notna().sum() / len(datetime_series) > 0.8:
                            self.raw_data[column] = datetime_series
                            continue
                except Exception as e:
                    self.logger.debug(f"Failed to parse datetime for column {column}: {e}")
                    pass
                
                # Check if it's categorical
                unique_ratio = self.raw_data[column].nunique() / len(self.raw_data[column])
                if unique_ratio < 0.1:  # Less than 10% unique values
                    self.raw_data[column] = self.raw_data[column].astype('category')
    
    def set_variables(self,
                     independent_vars: List[str],
                     dependent_vars: List[str],
                     subject_column: Optional[str] = None,
                     covariate_vars: Optional[List[str]] = None) -> None:
        """Set variable roles for analysis.
        
        Args:
            independent_vars: List of independent variable names
            dependent_vars: List of dependent variable names
            subject_column: Name of subject identifier column
            covariate_vars: List of covariate variable names
        """
        # Check if data has been loaded
        if self.raw_data is None:
            raise ValueError("No data loaded. Please load data before setting variables.")
        
        # Validate variable names
        all_vars = independent_vars + dependent_vars + (covariate_vars or [])
        if subject_column:
            all_vars.append(subject_column)
        
        missing_vars = [var for var in all_vars if var not in self.raw_data.columns]
        if missing_vars:
            raise ValueError(f"Variables not found in data: {missing_vars}")
        
        self.independent_vars = independent_vars
        self.dependent_vars = dependent_vars
        self.subject_column = subject_column or self.raw_data.columns[0]
        self.covariate_vars = covariate_vars or []
        
        # Auto-detect subject column if not provided
        if not subject_column:
            self.subject_column = self._detect_subject_column()
        
        self.logger.info(f"Variables set - IV: {len(self.independent_vars)}, DV: {len(self.dependent_vars)}")
    
    def _detect_subject_column(self) -> str:
        """Automatically detect subject identifier column."""
        # Look for columns with names suggesting subject IDs
        subject_keywords = ['subject', 'participant', 'id', 'subj', 'part']
        
        for col in self.raw_data.columns:
            if any(keyword in col.lower() for keyword in subject_keywords):
                return col
        
        # If no obvious subject column, use first column
        return self.raw_data.columns[0]
    
    def preprocess_data(self) -> pd.DataFrame:
        """Comprehensive data preprocessing pipeline.
        
        Returns:
            Preprocessed DataFrame
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        self.processed_data = self.raw_data.copy()
        self.transformations_applied = []
        
        # Handle missing values
        self._handle_missing_values()
        
        # Detect and handle outliers
        self._handle_outliers()
        
        # Apply transformations
        self._apply_transformations()
        
        # Normalize/standardize if needed
        self._apply_scaling()
        
        self.logger.info(f"Data preprocessing completed. Transformations: {', '.join(self.transformations_applied)}")
        
        return self.processed_data.copy()
    
    def _handle_missing_values(self) -> None:
        """Handle missing values based on configuration."""
        missing_info = self.processed_data.isnull().sum()
        
        if missing_info.sum() == 0:
            return
        
        method = self.config.data.handle_missing
        threshold = self.config.data.missing_threshold
        
        # Drop columns with too many missing values
        high_missing_cols = missing_info[missing_info / len(self.processed_data) > threshold].index
        if len(high_missing_cols) > 0:
            self.processed_data = self.processed_data.drop(columns=high_missing_cols)
            self.transformations_applied.append(f"Dropped columns with >{threshold*100}% missing: {list(high_missing_cols)}")
        
        if method == 'drop':
            self.processed_data = self.processed_data.dropna()
            self.transformations_applied.append("Dropped rows with missing values")
        
        elif method == 'impute' or method == 'auto':
            # Separate numeric and categorical columns
            numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
            categorical_cols = self.processed_data.select_dtypes(include=['object', 'category']).columns
            
            # Impute numeric columns
            if len(numeric_cols) > 0:
                imputer = SimpleImputer(strategy='median')
                self.processed_data[numeric_cols] = imputer.fit_transform(self.processed_data[numeric_cols])
                self.transformations_applied.append("Imputed numeric missing values with median")
            
            # Impute categorical columns
            if len(categorical_cols) > 0:
                imputer = SimpleImputer(strategy='most_frequent')
                self.processed_data[categorical_cols] = imputer.fit_transform(self.processed_data[categorical_cols])
                self.transformations_applied.append("Imputed categorical missing values with mode")
    
    def _handle_outliers(self) -> None:
        """Detect and handle outliers."""
        method = self.config.statistical.outlier_method
        threshold = self.config.statistical.outlier_threshold
        
        numeric_cols = [col for col in self.dependent_vars if col in self.processed_data.select_dtypes(include=[np.number]).columns]
        
        outliers_detected = 0
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = self.processed_data[col].quantile(0.25)
                Q3 = self.processed_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = (self.processed_data[col] < lower_bound) | (self.processed_data[col] > upper_bound)
            
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(self.processed_data[col]))
                outliers = z_scores > threshold
            
            elif method == 'modified_zscore':
                median = np.median(self.processed_data[col])
                mad = np.median(np.abs(self.processed_data[col] - median))
                modified_z_scores = 0.6745 * (self.processed_data[col] - median) / mad
                outliers = np.abs(modified_z_scores) > threshold
            
            elif method == 'isolation_forest':
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outliers = iso_forest.fit_predict(self.processed_data[[col]].values.reshape(-1, 1)) == -1
            
            else:
                continue
            
            # Cap outliers instead of removing them
            if outliers.sum() > 0:
                outliers_detected += outliers.sum()
                # Cap at 5th and 95th percentiles
                lower_cap = self.processed_data[col].quantile(0.05)
                upper_cap = self.processed_data[col].quantile(0.95)
                self.processed_data.loc[outliers & (self.processed_data[col] < lower_cap), col] = lower_cap
                self.processed_data.loc[outliers & (self.processed_data[col] > upper_cap), col] = upper_cap
        
        if outliers_detected > 0:
            self.transformations_applied.append(f"Capped {outliers_detected} outliers using {method} method")
    
    def _apply_transformations(self) -> None:
        """Apply data transformations to improve normality."""
        method = self.config.data.transformation_method
        
        if method == 'none':
            return
        
        numeric_cols = [col for col in self.dependent_vars if col in self.processed_data.select_dtypes(include=[np.number]).columns]
        
        for col in numeric_cols:
            original_data = self.processed_data[col].copy()
            
            # Skip if data contains non-positive values for log/boxcox
            if method in ['log', 'boxcox'] and (original_data <= 0).any():
                continue
            
            try:
                if method == 'log':
                    self.processed_data[col] = np.log(original_data)
                    self.transformations_applied.append(f"Log transformation applied to {col}")
                
                elif method == 'sqrt':
                    if (original_data >= 0).all():
                        self.processed_data[col] = np.sqrt(original_data)
                        self.transformations_applied.append(f"Square root transformation applied to {col}")
                
                elif method == 'boxcox':
                    transformed_data, lambda_param = boxcox(original_data)
                    self.processed_data[col] = transformed_data
                    self.transformations_applied.append(f"Box-Cox transformation applied to {col} (λ={lambda_param:.3f})")
                
                elif method == 'yeo_johnson':
                    transformed_data, lambda_param = yeojohnson(original_data)
                    self.processed_data[col] = transformed_data
                    self.transformations_applied.append(f"Yeo-Johnson transformation applied to {col} (λ={lambda_param:.3f})")
                
                elif method == 'auto':
                    # Test normality before and after transformations
                    best_transformation = self._find_best_transformation(original_data, col)
                    if best_transformation:
                        self.transformations_applied.append(best_transformation)
            
            except Exception as e:
                self.logger.warning(f"Failed to apply {method} transformation to {col}: {str(e)}")
    
    def _find_best_transformation(self, data: pd.Series, col_name: str) -> Optional[str]:
        """Find the best transformation to improve normality."""
        from scipy.stats import shapiro
        
        # Test original data
        try:
            _, original_p = shapiro(data.dropna())
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            self.logger.warning(f"Shapiro-Wilk test failed: {str(e)}")
            return None
        
        transformations = []
        
        # Test different transformations
        if (data > 0).all():
            # Log transformation
            try:
                log_data = np.log(data)
                _, log_p = shapiro(log_data.dropna())
                transformations.append(('log', log_p, log_data))
            except Exception as e:
                self.logger.debug(f"Log transformation failed: {e}")
            
            # Box-Cox transformation
            try:
                boxcox_data, lambda_param = boxcox(data)
                _, boxcox_p = shapiro(boxcox_data)
                transformations.append(('boxcox', boxcox_p, boxcox_data, lambda_param))
            except Exception as e:
                self.logger.debug(f"Box-Cox transformation failed: {e}")
        
        if (data >= 0).all():
            # Square root transformation
            try:
                sqrt_data = np.sqrt(data)
                _, sqrt_p = shapiro(sqrt_data.dropna())
                transformations.append(('sqrt', sqrt_p, sqrt_data))
            except Exception as e:
                self.logger.debug(f"Square root transformation failed: {e}")
        
        # Yeo-Johnson (works with any data)
        try:
            yj_data, lambda_param = yeojohnson(data)
            _, yj_p = shapiro(yj_data)
            transformations.append(('yeo_johnson', yj_p, yj_data, lambda_param))
        except Exception as e:
                self.logger.debug(f"Yeo-Johnson transformation failed: {e}")
        
        # Find best transformation (highest p-value)
        if transformations:
            best_transform = max(transformations, key=lambda x: x[1])
            if best_transform[1] > original_p:
                self.processed_data[col_name] = best_transform[2]
                if best_transform[0] in ['boxcox', 'yeo_johnson']:
                    return f"{best_transform[0].replace('_', '-').title()} transformation applied to {col_name} (λ={best_transform[3]:.3f})"
                else:
                    return f"{best_transform[0].replace('_', ' ').title()} transformation applied to {col_name}"
        
        return None
    
    def _apply_scaling(self) -> None:
        """Apply scaling/normalization to data."""
        method = self.config.data.normalization_method
        
        if method == 'none':
            return
        
        numeric_cols = [col for col in self.dependent_vars if col in self.processed_data.select_dtypes(include=[np.number]).columns]
        
        if not numeric_cols:
            return
        
        if method == 'zscore':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'auto':
            # Choose based on data distribution
            scaler = RobustScaler()  # Generally more robust to outliers
        else:
            return
        
        self.processed_data[numeric_cols] = scaler.fit_transform(self.processed_data[numeric_cols])
        self.transformations_applied.append(f"{method.title()} scaling applied to dependent variables")
    
    def _generate_quality_report(self) -> None:
        """Generate comprehensive data quality report."""
        if self.raw_data is None:
            return
        
        self.quality_report = {
            'basic_info': {
                'n_rows': len(self.raw_data),
                'n_columns': len(self.raw_data.columns),
                'memory_usage': self.raw_data.memory_usage(deep=True).sum(),
                'dtypes': self.raw_data.dtypes.value_counts().to_dict()
            },
            'missing_values': {
                'total_missing': self.raw_data.isnull().sum().sum(),
                'missing_by_column': self.raw_data.isnull().sum().to_dict(),
                'missing_percentage': (self.raw_data.isnull().sum() / len(self.raw_data) * 100).to_dict()
            },
            'duplicates': {
                'duplicate_rows': self.raw_data.duplicated().sum(),
                'duplicate_percentage': self.raw_data.duplicated().sum() / len(self.raw_data) * 100
            },
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # Numeric columns summary
        numeric_cols = self.raw_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.quality_report['numeric_summary'][col] = {
                'mean': self.raw_data[col].mean(),
                'std': self.raw_data[col].std(),
                'min': self.raw_data[col].min(),
                'max': self.raw_data[col].max(),
                'skewness': self.raw_data[col].skew(),
                'kurtosis': self.raw_data[col].kurtosis(),
                'zeros': (self.raw_data[col] == 0).sum(),
                'negative': (self.raw_data[col] < 0).sum()
            }
        
        # Categorical columns summary
        categorical_cols = self.raw_data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            self.quality_report['categorical_summary'][col] = {
                'unique_values': self.raw_data[col].nunique(),
                'most_frequent': self.raw_data[col].mode().iloc[0] if not self.raw_data[col].mode().empty else None,
                'frequency_top': self.raw_data[col].value_counts().iloc[0] if not self.raw_data[col].empty else 0,
                'categories': self.raw_data[col].value_counts().head(10).to_dict()
            }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive data summary.
        
        Returns:
            Dictionary containing data summary information
        """
        summary = {
            'metadata': self.metadata,
            'variables': {
                'subject_column': self.subject_column,
                'independent_vars': self.independent_vars,
                'dependent_vars': self.dependent_vars,
                'covariate_vars': self.covariate_vars
            },
            'quality_report': self.quality_report,
            'transformations_applied': self.transformations_applied
        }
        
        return summary
    
    def export_processed_data(self, file_path: str, format: str = 'csv') -> None:
        """Export processed data to file.
        
        Args:
            file_path: Output file path
            format: Export format ('csv', 'excel', 'json')
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Call preprocess_data() first.")
        
        if format.lower() == 'csv':
            self.processed_data.to_csv(file_path, index=False)
        elif format.lower() == 'excel':
            self.processed_data.to_excel(file_path, index=False)
        elif format.lower() == 'json':
            self.processed_data.to_json(file_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Processed data exported to {file_path}")