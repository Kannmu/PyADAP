"""Automated Analysis Pipeline for PyADAP 3.0

This module provides an intelligent, automated analysis pipeline that integrates
data management, statistical analysis, and reporting with minimal user intervention.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd

from .data_manager import DataManager
from .statistical_analyzer import StatisticalAnalyzer
from ..config import Config
from ..utils import Logger, Validator
from ..visualization import PlotManager
from ..io import AppleStyleReportGenerator, DataExporter


class AnalysisPipeline:
    """Intelligent automated analysis pipeline."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the analysis pipeline.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.logger = Logger()
        self.validator = Validator()
        
        # Core components
        self.data_manager = DataManager(self.config)
        self.statistical_analyzer = StatisticalAnalyzer(self.config)
        self.plot_manager = PlotManager(self.config)
        self.report_generator = AppleStyleReportGenerator(self.config)
        self.data_exporter = DataExporter(self.config)
        
        # Pipeline state
        self.pipeline_id = self._generate_pipeline_id()
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.status = "initialized"
        
        # Results storage
        self.results: Dict[str, Any] = {
            'data_summary': {},
            'quality_report': {},
            'assumptions': {},
            'statistical_tests': {},
            'visualizations': {},
            'recommendations': []
        }
        
        # Output paths
        self.output_dir: Optional[Path] = None
        
    def set_data(self, data: pd.DataFrame, file_path: Optional[str] = None) -> None:
        """Set data for analysis.
        
        Args:
            data: DataFrame containing the data to analyze
            file_path: Optional path to the source data file
        """
        self.data_manager.raw_data = data
        self.data_manager.processed_data = None  # Reset processed data
        
        # Set file path in metadata if provided
        if file_path:
            self.data_manager.metadata['file_path'] = file_path
            # Update pipeline ID to reflect the new data file
            self.pipeline_id = self._generate_pipeline_id(file_path)
            
        # Generate quality report for the new data
        self.data_manager._generate_quality_report()
        
    def set_variables(self, 
                     dependent_vars: List[str],
                     independent_vars: List[str],
                     subject_var: Optional[str] = None,
                     covariate_vars: Optional[List[str]] = None) -> None:
        """Set variable roles for analysis.
        
        Args:
            dependent_vars: List of dependent variable names
            independent_vars: List of independent variable names
            subject_var: Name of subject identifier column
            covariate_vars: List of covariate variable names
        """
        # Delegate to data manager
        self.data_manager.set_variables(
            dependent_vars=dependent_vars,
            independent_vars=independent_vars,
            subject_column=subject_var,
            covariate_vars=covariate_vars
        )
        
    def _generate_pipeline_id(self, data_file: Optional[str] = None) -> str:
        """Generate unique pipeline ID based on data file name.
        
        Args:
            data_file: Path to the data file being analyzed
            
        Returns:
            Unique pipeline ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if data_file:
            # Extract filename without extension
            file_stem = Path(data_file).stem
            # Clean filename for use in directory name - more permissive cleaning
            clean_name = "".join(c for c in file_stem if c.isalnum() or c in ('-', '_', '.')).rstrip()
            # Ensure clean_name is not empty
            if not clean_name:
                clean_name = "data"
            return f"{clean_name}_{timestamp}"
        else:
            return f"PyADAP_Analysis_{timestamp}"
    
    def run_analysis(self, 
                    dependent_vars: List[str],
                    independent_vars: List[str],
                    subject_var: Optional[str] = None,
                    output_dir: Optional[str] = None,
                    **kwargs) -> Dict[str, Any]:
        """Run analysis on already loaded data.
        
        Args:
            dependent_vars: List of dependent variable names
            independent_vars: List of independent variable names
            subject_var: Name of subject identifier column
            output_dir: Directory for output files
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing all analysis results
        """
        self.start_time = datetime.now()
        self.status = "running"
        
        try:
            # Get data file from data_manager if available
            data_file = getattr(self.data_manager, 'metadata', {}).get('file_path', None)
            
            # Generate pipeline ID based on data file
            if data_file:
                self.pipeline_id = self._generate_pipeline_id(data_file)
            
            # Setup output directory with correct pipeline ID
            self._setup_output_directory(output_dir, data_file)
            
            # Set variables and validate (data should already be loaded)
            self.logger.info("Setting variables and validating...")
            self._set_variables_and_validate(dependent_vars, independent_vars, subject_var)
            
            # Data quality assessment
            self.logger.info("Assessing data quality...")
            self._assess_data_quality()
            
            # Check statistical assumptions
            self.logger.info("Checking statistical assumptions...")
            self._check_assumptions()
            
            # Run statistical analyses
            self.logger.info("Running statistical analyses...")
            self._run_statistical_analyses()
            
            # Generate visualizations
            self.logger.info("Generating visualizations...")
            self._generate_visualizations()
            
            # Generate recommendations
            self.logger.info("Generating recommendations...")
            self._generate_recommendations()
            
            # Create reports and export results
            self.logger.info("Creating reports and exporting results...")
            self._create_reports_and_export()
            
            self.status = "completed"
            self.end_time = datetime.now()
            
            self.logger.info(f"Analysis completed successfully in {self.end_time - self.start_time}")
            
            return self.results
            
        except Exception as e:
            self.status = "failed"
            self.end_time = datetime.now()
            self.logger.error(f"Analysis failed: {str(e)}")
            raise
    
    def run_full_pipeline(self, 
                         data_file: str,
                         dependent_vars: List[str],
                         independent_vars: List[str],
                         subject_column: Optional[str] = None,
                         output_dir: Optional[str] = None,
                         **kwargs) -> Dict[str, Any]:
        """Run the complete analysis pipeline.
        
        Args:
            data_file: Path to data file
            dependent_vars: List of dependent variable names
            independent_vars: List of independent variable names
            subject_column: Name of subject identifier column
            output_dir: Directory for output files
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing all analysis results
        """
        self.start_time = datetime.now()
        self.status = "running"
        
        try:
            # Generate pipeline ID first based on data file
            self.pipeline_id = self._generate_pipeline_id(data_file)
            
            # Setup output directory with the correct pipeline ID
            self._setup_output_directory(output_dir, data_file)
            
            # Step 1: Load and preprocess data
            self.logger.info("Step 1: Loading and preprocessing data...")
            self._load_and_preprocess_data(data_file, **kwargs)
            
            # Step 2: Set variables and validate
            self.logger.info("Step 2: Setting variables and validating...")
            self._set_variables_and_validate(dependent_vars, independent_vars, subject_column)
            
            # Step 3: Data quality assessment
            self.logger.info("Step 3: Assessing data quality...")
            self._assess_data_quality()
            
            # Step 4: Check statistical assumptions
            self.logger.info("Step 4: Checking statistical assumptions...")
            self._check_assumptions()
            
            # Step 5: Run statistical analyses
            self.logger.info("Step 5: Running statistical analyses...")
            self._run_statistical_analyses()
            
            # Step 6: Generate visualizations
            self.logger.info("Step 6: Generating visualizations...")
            self._generate_visualizations()
            
            # Step 7: Generate recommendations
            self.logger.info("Step 7: Generating recommendations...")
            self._generate_recommendations()
            
            # Step 8: Create reports and export results
            self.logger.info("Step 8: Creating reports and exporting results...")
            self._create_reports_and_export()
            
            self.status = "completed"
            self.end_time = datetime.now()
            
            self.logger.info(f"Pipeline completed successfully in {self.end_time - self.start_time}")
            
            return self.results
            
        except Exception as e:
            self.status = "failed"
            self.end_time = datetime.now()
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _setup_output_directory(self, output_dir: Optional[str], data_file: Optional[str] = None) -> None:
        """Setup output directory for results.
        
        Args:
            output_dir: Custom output directory path
            data_file: Path to the data file being analyzed
        """
        if output_dir:
            self.output_dir = Path(output_dir)
        elif data_file:
            # Create output directory in the same directory as the data file
            # Named using the pipeline_id (original filename + timestamp)
            data_path = Path(data_file)
            data_dir = data_path.parent
            # Use the pipeline_id which includes timestamp
            self.output_dir = data_dir / self.pipeline_id
        else:
            # Fallback to current working directory
            self.output_dir = Path.cwd() / "PyADAP_Results" / self.pipeline_id
        
        # Remove existing directory contents if it exists
        if self.output_dir.exists():
            import shutil
            shutil.rmtree(self.output_dir)
            self.logger.info(f"Removed existing output directory: {self.output_dir}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "exports").mkdir(exist_ok=True)
        
        self.logger.info(f"Output directory created: {self.output_dir}")
    
    def _load_and_preprocess_data(self, data_file: str, **kwargs) -> None:
        """Load and preprocess data."""
        
        # Load data
        raw_data = self.data_manager.load_data(data_file, **kwargs)
        
        # Store raw data summary
        self.results['data_summary']['raw'] = {
            'file_path': data_file,
            'n_rows': len(raw_data),
            'n_columns': len(raw_data.columns),
            'columns': list(raw_data.columns),
            'dtypes': raw_data.dtypes.to_dict(),
            'memory_usage': raw_data.memory_usage(deep=True).sum()
        }
        
        # Preprocess data if enabled
        if self.config.data.auto_preprocess:
            processed_data = self.data_manager.preprocess_data()
            
            # Store processed data summary
            self.results['data_summary']['processed'] = {
                'n_rows': len(processed_data),
                'n_columns': len(processed_data.columns),
                'transformations_applied': self.data_manager.transformations_applied
            }
        
        # Store data quality report
        self.results['quality_report'] = self.data_manager.quality_report
    
    def _set_variables_and_validate(self, 
                                   dependent_vars: List[str],
                                   independent_vars: List[str],
                                   subject_column: Optional[str]) -> None:
        """Set variables and validate them."""
        # Set variables in data manager
        self.data_manager.set_variables(
            independent_vars=independent_vars,
            dependent_vars=dependent_vars,
            subject_column=subject_column
        )
        
        # Get current data (processed if available, otherwise raw)
        current_data = self.data_manager.processed_data if self.data_manager.processed_data is not None else self.data_manager.raw_data
        
        # Set data in statistical analyzer
        self.statistical_analyzer.set_data(
            data=current_data,
            dependent_vars=dependent_vars,
            independent_vars=independent_vars,
            subject_column=self.data_manager.subject_column
        )
        
        # Validate variable types and distributions
        validation_results = self._validate_variables(current_data, dependent_vars, independent_vars)
        self.results['data_summary']['validation'] = validation_results
    
    def _validate_variables(self, data: pd.DataFrame, dependent_vars: List[str], independent_vars: List[str]) -> Dict[str, Any]:
        """Validate variables for analysis."""
        validation = {
            'dependent_variables': {},
            'independent_variables': {},
            'warnings': [],
            'recommendations': []
        }
        
        # Validate dependent variables
        for dv in dependent_vars:
            if dv in data.columns:
                dv_info = {
                    'type': str(data[dv].dtype),
                    'is_numeric': pd.api.types.is_numeric_dtype(data[dv]),
                    'unique_values': data[dv].nunique(),
                    'missing_values': data[dv].isnull().sum(),
                    'missing_percentage': data[dv].isnull().sum() / len(data) * 100
                }
                
                # Check if suitable for analysis
                if not dv_info['is_numeric']:
                    validation['warnings'].append(f"Dependent variable '{dv}' is not numeric")
                
                if dv_info['missing_percentage'] > 20:
                    validation['warnings'].append(f"Dependent variable '{dv}' has {dv_info['missing_percentage']:.1f}% missing values")
                
                if dv_info['unique_values'] < 3:
                    validation['warnings'].append(f"Dependent variable '{dv}' has very few unique values ({dv_info['unique_values']})")
                
                validation['dependent_variables'][dv] = dv_info
        
        # Validate independent variables
        for iv in independent_vars:
            if iv in data.columns:
                iv_info = {
                    'type': str(data[iv].dtype),
                    'is_numeric': pd.api.types.is_numeric_dtype(data[iv]),
                    'unique_values': data[iv].nunique(),
                    'missing_values': data[iv].isnull().sum(),
                    'missing_percentage': data[iv].isnull().sum() / len(data) * 100
                }
                
                # Check if suitable for analysis
                if iv_info['missing_percentage'] > 20:
                    validation['warnings'].append(f"Independent variable '{iv}' has {iv_info['missing_percentage']:.1f}% missing values")
                
                if iv_info['is_numeric'] and iv_info['unique_values'] > len(data) * 0.8:
                    validation['recommendations'].append(f"Consider treating '{iv}' as continuous rather than categorical")
                
                validation['independent_variables'][iv] = iv_info
        
        return validation
    
    def _assess_data_quality(self) -> None:
        """Assess overall data quality."""
        quality_assessment = {
            'overall_score': 0,
            'issues': [],
            'strengths': [],
            'recommendations': []
        }
        
        # Get quality report
        quality_report = self.data_manager.quality_report
        
        # Ensure quality report exists and has required keys
        if not quality_report or 'missing_values' not in quality_report or 'basic_info' not in quality_report:
            self.logger.warning("Quality report not available or incomplete, regenerating...")
            self.data_manager._generate_quality_report()
            quality_report = self.data_manager.quality_report
        
        # Calculate quality score (0-100)
        score_components = []
        
        # Missing data component (0-25 points)
        missing_percentage = quality_report['missing_values']['total_missing'] / (quality_report['basic_info']['n_rows'] * quality_report['basic_info']['n_columns']) * 100
        missing_score = max(0, 25 - missing_percentage)
        score_components.append(missing_score)
        
        if missing_percentage > 10:
            quality_assessment['issues'].append(f"High missing data rate: {missing_percentage:.1f}%")
        else:
            quality_assessment['strengths'].append(f"Low missing data rate: {missing_percentage:.1f}%")
        
        # Duplicate data component (0-25 points)
        duplicate_percentage = quality_report['duplicates']['duplicate_percentage']
        duplicate_score = max(0, 25 - duplicate_percentage * 5)
        score_components.append(duplicate_score)
        
        if duplicate_percentage > 5:
            quality_assessment['issues'].append(f"High duplicate rate: {duplicate_percentage:.1f}%")
        else:
            quality_assessment['strengths'].append(f"Low duplicate rate: {duplicate_percentage:.1f}%")
        
        # Sample size component (0-25 points)
        n_rows = quality_report['basic_info']['n_rows']
        if n_rows >= 100:
            sample_score = 25
            quality_assessment['strengths'].append(f"Adequate sample size: {n_rows}")
        elif n_rows >= 30:
            sample_score = 15
            quality_assessment['recommendations'].append("Consider increasing sample size for more robust results")
        else:
            sample_score = 5
            quality_assessment['issues'].append(f"Small sample size: {n_rows}")
        score_components.append(sample_score)
        
        # Data distribution component (0-25 points)
        distribution_score = 20  # Default good score
        
        # Check for extreme skewness in numeric variables
        extreme_skew_vars = []
        for var, stats in quality_report['numeric_summary'].items():
            if abs(stats['skewness']) > 2:
                extreme_skew_vars.append(var)
        
        if extreme_skew_vars:
            distribution_score -= len(extreme_skew_vars) * 3
            quality_assessment['issues'].append(f"Variables with extreme skewness: {', '.join(extreme_skew_vars)}")
        else:
            quality_assessment['strengths'].append("No extreme skewness detected")
        
        score_components.append(max(0, distribution_score))
        
        # Calculate overall score
        quality_assessment['overall_score'] = sum(score_components)
        
        # Generate recommendations based on score
        if quality_assessment['overall_score'] < 50:
            quality_assessment['recommendations'].append("Data quality is poor. Consider data cleaning and preprocessing")
        elif quality_assessment['overall_score'] < 75:
            quality_assessment['recommendations'].append("Data quality is moderate. Some preprocessing may improve results")
        else:
            quality_assessment['recommendations'].append("Data quality is good. Proceed with analysis")
        
        self.results['quality_report']['assessment'] = quality_assessment
    
    def _check_assumptions(self) -> None:
        """Check statistical assumptions."""
        assumptions = self.statistical_analyzer.check_assumptions()
        self.results['assumptions'] = assumptions
        
        # Generate assumption-based recommendations
        assumption_recommendations = []
        
        # Check normality assumptions
        if 'normality' in assumptions:
            non_normal_vars = [var for var, result in assumptions['normality'].items() 
                             if not result.get('overall_normal', True)]
            if non_normal_vars:
                assumption_recommendations.append(
                    f"Non-normal distributions detected in: {', '.join(non_normal_vars)}. "
                    "Consider non-parametric tests or data transformation."
                )
        
        # Check homogeneity assumptions
        if 'homogeneity' in assumptions:
            heterogeneous_comparisons = [comp for comp, result in assumptions['homogeneity'].items() 
                                       if not result.get('overall_homogeneous', True)]
            if heterogeneous_comparisons:
                assumption_recommendations.append(
                    f"Unequal variances detected in: {', '.join(heterogeneous_comparisons)}. "
                    "Consider Welch's t-test or non-parametric alternatives."
                )
        
        # Check independence assumptions
        if 'independence' in assumptions:
            if assumptions['independence'].get('repeated_measures_detected', False):
                assumption_recommendations.append(
                    "Repeated measures detected. Use appropriate repeated measures tests."
                )
        
        # Check multicollinearity
        if 'multicollinearity' in assumptions:
            if assumptions['multicollinearity'].get('multicollinearity_detected', False):
                assumption_recommendations.append(
                    "Multicollinearity detected among independent variables. "
                    "Consider removing highly correlated variables or using regularization."
                )
        
        self.results['recommendations'].extend(assumption_recommendations)
    
    def _run_statistical_analyses(self) -> None:
        """Run comprehensive statistical analyses."""
        # Run comprehensive analysis
        statistical_results = self.statistical_analyzer.run_comprehensive_analysis()
        
        # Convert results to serializable format
        serializable_results = {}
        for key, result in statistical_results.items():
            serializable_results[key] = {
                'test_name': result.test_name,
                'test_type': result.test_type.value,
                'statistic': result.statistic,
                'p_value': result.p_value,
                'effect_size': result.effect_size,
                'effect_size_interpretation': result.effect_size_interpretation,
                'confidence_interval': result.confidence_interval,
                'power': result.power,
                'sample_size': result.sample_size,
                'assumptions_met': result.assumptions_met,
                'interpretation': result.interpretation,
                'recommendations': result.recommendations,
                'additional_info': result.additional_info
            }
        
        self.results['statistical_tests'] = serializable_results
        
        # Generate statistical recommendations
        statistical_recommendations = []
        
        for key, result in statistical_results.items():
            if result.power is not None and result.power < 0.8:
                statistical_recommendations.append(
                    f"Low statistical power ({result.power:.3f}) for {key}. Consider increasing sample size."
                )
            
            if result.recommendations:
                statistical_recommendations.extend(result.recommendations)
        
        self.results['recommendations'].extend(statistical_recommendations)
    
    def _generate_visualizations(self) -> None:
        """Generate comprehensive visualizations."""
        if not self.config.visualization.generate_plots:
            self.logger.info("Plot generation is disabled in the configuration.")
            self.results['plots'] = {}
            self.results['visualizations'] = {}
            return
        
        visualization_results: Dict[str, Figure] = {}
        plot_figures: Dict[str, Figure] = {} # Explicitly for Figure objects
        
        # Get current data
        current_data = self.statistical_analyzer.data
        if current_data is None or current_data.empty:
            self.logger.warning("No data available for generating visualizations.")
            self.results['plots'] = {}
            self.results['visualizations'] = {}
            return

        # Ensure output directory for plots exists
        plots_output_dir = self.output_dir / "plots"
        plots_output_dir.mkdir(parents=True, exist_ok=True)

        # Generate plots for each DV-IV combination
        for dv in self.data_manager.dependent_vars:
            for iv in self.data_manager.independent_vars:
                plot_key_base = f"{dv}_by_{iv}"
                
                try:
                    iv_type = self._get_variable_type(current_data, iv)
                    dv_type = self._get_variable_type(current_data, dv)
                    
                    if dv_type == 'continuous' and iv_type == 'categorical':
                        # Box plot
                        fig_box = self.plot_manager.create_comparison_plot(
                            data=current_data,
                            dependent_var=dv,
                            independent_var=iv,
                            plot_type='box',
                            save_path=plots_output_dir / f"{plot_key_base}_boxplot.png"
                        )
                        if fig_box:
                            plot_figures[f"{plot_key_base}_boxplot"] = fig_box
                            visualization_results[f"{plot_key_base}_boxplot"] = str(plots_output_dir / f"{plot_key_base}_boxplot.png")
                        
                        # Distribution plot
                        # Assuming create_distribution_plot also returns a Figure and handles saving
                        fig_dist = self.plot_manager.create_distribution_plot(
                            data=current_data,
                            variable=dv,
                            group_by=iv,
                            save_path=plots_output_dir / f"{plot_key_base}_distribution.png"
                        )
                        if fig_dist:
                            plot_figures[f"{plot_key_base}_distribution"] = fig_dist
                            visualization_results[f"{plot_key_base}_distribution"] = str(plots_output_dir / f"{plot_key_base}_distribution.png")
                    
                    elif dv_type == 'continuous' and iv_type == 'continuous':
                        # Scatter plot
                        # Assuming create_scatter_plot also returns a Figure and handles saving
                        fig_scatter = self.plot_manager.create_scatter_plot(
                            data=current_data,
                            x_var=iv,
                            y_var=dv,
                            save_path=plots_output_dir / f"{plot_key_base}_scatter.png"
                        )
                        if fig_scatter:
                            plot_figures[f"{plot_key_base}_scatter"] = fig_scatter
                            visualization_results[f"{plot_key_base}_scatter"] = str(plots_output_dir / f"{plot_key_base}_scatter.png")
                    
                    elif dv_type == 'categorical' and iv_type == 'categorical':
                        # Contingency table heatmap
                        # Assuming create_contingency_plot also returns a Figure and handles saving
                        fig_contingency = self.plot_manager.create_contingency_plot(
                            data=current_data,
                            var1=dv,
                            var2=iv,
                            save_path=plots_output_dir / f"{plot_key_base}_contingency.png"
                        )
                        if fig_contingency:
                            plot_figures[f"{plot_key_base}_contingency"] = fig_contingency
                            visualization_results[f"{plot_key_base}_contingency"] = str(plots_output_dir / f"{plot_key_base}_contingency.png")
                
                except Exception as e:
                    self.logger.warning(f"Failed to create plot for {plot_key_base}: {str(e)}")
                    continue
        
        # Generate summary plots
        try:
            continuous_vars = [var for var in self.data_manager.dependent_vars + self.data_manager.independent_vars
                             if self._get_variable_type(current_data, var) == 'continuous']
            
            if len(continuous_vars) > 1:
                # Assuming create_correlation_matrix also returns a Figure and handles saving
                fig_corr = self.plot_manager.create_correlation_matrix(
                    data=current_data[continuous_vars],
                    save_path=plots_output_dir / "correlation_matrix.png"
                )
                if fig_corr:
                    plot_figures['correlation_matrix'] = fig_corr
                    visualization_results['correlation_matrix'] = str(plots_output_dir / "correlation_matrix.png")
        
        except Exception as e:
            self.logger.warning(f"Failed to create correlation matrix: {str(e)}")
        
        self.results['visualizations'] = visualization_results # Keep paths for reports
        self.results['plots'] = plot_figures  # Store Figure objects for GUI
    
    def _get_variable_type(self, data: pd.DataFrame, var_name: str) -> str:
        """Determine variable type."""
        if var_name not in data.columns:
            return 'unknown'
        
        if pd.api.types.is_numeric_dtype(data[var_name]):
            # Check if it's actually categorical
            unique_ratio = data[var_name].nunique() / len(data[var_name])
            if unique_ratio < 0.05 and data[var_name].nunique() < 10:
                return 'categorical'
            return 'continuous'
        else:
            return 'categorical'
    
    def _generate_recommendations(self) -> None:
        """Generate comprehensive recommendations."""
        # Recommendations are already generated in previous steps
        # Here we can add overall pipeline recommendations
        
        overall_recommendations = []
        
        # Check overall analysis quality
        quality_score = self.results['quality_report'].get('assessment', {}).get('overall_score', 0)
        
        if quality_score < 50:
            overall_recommendations.append(
                "Overall data quality is poor. Consider extensive data cleaning and preprocessing before analysis."
            )
        
        # Check if any statistical tests were significant
        significant_tests = []
        for test_name, result in self.results['statistical_tests'].items():
            if result['p_value'] < self.config.statistical.alpha_level:
                significant_tests.append(test_name)
        
        if significant_tests:
            overall_recommendations.append(
                f"Significant results found in: {', '.join(significant_tests)}. "
                "Consider effect sizes and practical significance."
            )
        else:
            overall_recommendations.append(
                "No statistically significant results found. "
                "Consider power analysis and sample size requirements."
            )
        
        # Add methodological recommendations
        overall_recommendations.extend([
            "Always interpret results in the context of your research question and domain knowledge.",
            "Consider replication and validation of findings with independent datasets.",
            "Report effect sizes and confidence intervals alongside p-values."
        ])
        
        self.results['recommendations'].extend(overall_recommendations)
    
    def _create_reports_and_export(self) -> None:
        """Create reports and export results."""
        # Generate statistical report
        statistical_report = self.statistical_analyzer.generate_report()
        
        # Save statistical report
        with open(self.output_dir / "reports" / "statistical_report.txt", 'w', encoding='utf-8') as f:
            f.write(statistical_report)
        
        # Generate comprehensive HTML report
        data_info = {
            'pipeline_id': self.pipeline_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'data_shape': self.data.shape if hasattr(self, 'data') and self.data is not None else None,
            'variables': {
                'dependent': getattr(self.data_manager, 'dependent_vars', []),
                'independent': getattr(self.data_manager, 'independent_vars', []),
                'subject': getattr(self.data_manager, 'subject_var', None)
            }
        }
        
        # Generate HTML report in the reports subdirectory
        html_filename = f"{self.pipeline_id}_report.html"
        html_report_path = self.report_generator.generate_comprehensive_report(
            results=self.results,
            data=self.data_manager.processed_data,
            data_info=data_info,
            filename=html_filename,
            output_dir=self.output_dir / "reports"
        )
        
        self.logger.info(f"HTML report saved to: {html_report_path}")
        
        # Export processed data
        if self.data_manager.processed_data is not None:
            self.data_exporter.export_data(
                data=self.data_manager.processed_data,
                file_path=self.output_dir / "exports" / "processed_data.xlsx",
                format='excel'
            )
        
        # Export results as JSON
        with open(self.output_dir / "exports" / "analysis_results.json", 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Export summary statistics
        summary_stats = self._create_summary_statistics()
        summary_stats.to_excel(self.output_dir / "exports" / "summary_statistics.xlsx", index=False)
        
        self.logger.info(f"Reports and exports saved to {self.output_dir}")
    
    def _create_summary_statistics(self) -> pd.DataFrame:
        """Create summary statistics table."""
        summary_data = []
        
        for test_name, result in self.results['statistical_tests'].items():
            summary_data.append({
                'Analysis': test_name,
                'Test': result['test_name'],
                'Statistic': result['statistic'],
                'p-value': result['p_value'],
                'Effect Size': result.get('effect_size', ''),
                'Effect Size Interpretation': result.get('effect_size_interpretation', ''),
                'Sample Size': result.get('sample_size', ''),
                'Interpretation': result.get('interpretation', '')
            })
        
        return pd.DataFrame(summary_data)
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get pipeline execution summary.
        
        Returns:
            Dictionary containing pipeline summary
        """
        duration = None
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
        
        return {
            'pipeline_id': self.pipeline_id,
            'status': self.status,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_seconds': duration,
            'output_directory': str(self.output_dir) if self.output_dir else None,
            'data_summary': self.results.get('data_summary', {}),
            'n_statistical_tests': len(self.results.get('statistical_tests', {})),
            'n_visualizations': len(self.results.get('visualizations', {})),
            'n_recommendations': len(self.results.get('recommendations', []))
        }
    
    def save_pipeline_state(self, file_path: Optional[str] = None) -> str:
        """Save pipeline state to file.
        
        Args:
            file_path: Path to save file (optional)
            
        Returns:
            Path to saved file
        """
        if file_path is None:
            file_path = self.output_dir / "pipeline_state.json" if self.output_dir else f"{self.pipeline_id}_state.json"
        
        pipeline_state = {
            'pipeline_summary': self.get_pipeline_summary(),
            'config': self.config.__dict__,
            'results': self.results
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(pipeline_state, f, indent=2, default=str)
        
        return str(file_path)
    
    @classmethod
    def load_pipeline_state(cls, file_path: str) -> 'AnalysisPipeline':
        """Load pipeline state from file.
        
        Args:
            file_path: Path to state file
            
        Returns:
            AnalysisPipeline instance with loaded state
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            pipeline_state = json.load(f)
        
        # Create new pipeline instance
        config = Config()
        config.__dict__.update(pipeline_state.get('config', {}))
        
        pipeline = cls(config)
        pipeline.results = pipeline_state.get('results', {})
        
        # Restore pipeline summary info
        summary = pipeline_state.get('pipeline_summary', {})
        pipeline.pipeline_id = summary.get('pipeline_id', pipeline.pipeline_id)
        pipeline.status = summary.get('status', 'loaded')
        
        if summary.get('start_time'):
            pipeline.start_time = datetime.fromisoformat(summary['start_time'])
        if summary.get('end_time'):
            pipeline.end_time = datetime.fromisoformat(summary['end_time'])
        if summary.get('output_directory'):
            pipeline.output_dir = Path(summary['output_directory'])
        
        return pipeline