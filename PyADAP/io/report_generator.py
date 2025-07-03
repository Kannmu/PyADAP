# -*- coding: utf-8 -*-
"""
Legacy Report Generator for PyADAP

This module provides backward compatibility for the old ReportGenerator interface
while using the new Apple-style generator as the backend.
"""

from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import pandas as pd
from ..config import Config
from ..utils import get_logger
from .apple_report_generator import AppleStyleReportGenerator


class ReportGenerator:
    """Legacy report generator that wraps the new Apple-style generator for backward compatibility."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the report generator.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.logger = get_logger("PyADAP.ReportGenerator")
        
        # Use the new Apple-style generator as backend
        self.apple_generator = AppleStyleReportGenerator(config)
        
        # Legacy compatibility
        self.output_dir = self.apple_generator.output_dir
        
        self.logger.info("ReportGenerator initialized (using Apple-style backend)")
    
    def generate_analysis_report(self, 
                               results: Dict[str, Any],
                               data_info: Dict[str, Any],
                               output_format: str = "html",
                               filename: Optional[str] = None,
                               output_dir: Optional[Path] = None) -> Path:
        """Generate a comprehensive analysis report.
        
        Args:
            results: Analysis results dictionary
            data_info: Information about the analyzed data
            output_format: Output format ('html', 'pdf')
            filename: Custom filename (optional)
            output_dir: Custom output directory (optional)
            
        Returns:
            Path to the generated report file
        """
        try:
            if output_format.lower() == "html":
                # Use the new Apple-style generator for HTML reports
                return self.apple_generator.generate_comprehensive_report(
                    results=results,
                    data=None,  # Legacy interface doesn't pass data
                    data_info=data_info,
                    filename=filename,
                    output_dir=output_dir
                )
            elif output_format.lower() == "pdf":
                return self._generate_pdf_report(results, data_info, filename, output_dir)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
                
        except Exception as e:
            self.logger.error(f"Failed to generate report: {str(e)}")
            raise
    
    def _generate_pdf_report(self, results: Dict[str, Any], 
                           data_info: Dict[str, Any], 
                           filename: Optional[str] = None,
                           output_dir: Optional[Path] = None) -> Path:
        """Generate PDF report (placeholder implementation)."""
        # First generate HTML using Apple-style generator
        html_filename = filename.replace('.pdf', '.html') if filename and filename.endswith('.pdf') else filename
        html_path = self.apple_generator.generate_comprehensive_report(
            results=results,
            data=None,
            data_info=data_info,
            filename=html_filename,
            output_dir=output_dir
        )
        
        # For now, return the HTML path as placeholder.
        self.logger.warning("PDF generation is not implemented due to missing dependencies (e.g., weasyprint, reportlab). Returning HTML report instead.")
        return html_path

    def export_data_summary(self, data: pd.DataFrame, 
                          filename: Optional[str] = None) -> Path:
        """Export data summary to file.
        
        Args:
            data: DataFrame to summarize
            filename: Output filename
            
        Returns:
            Path to the exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data_summary_{timestamp}.txt"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Data Summary Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Shape: {data.shape}\n\n")
            f.write("Column Information:\n")
            f.write(str(data.info()) + "\n\n")
            f.write("Descriptive Statistics:\n")
            f.write(str(data.describe()) + "\n\n")
            f.write("Missing Values:\n")
            f.write(str(data.isnull().sum()) + "\n")
        
        self.logger.info(f"Data summary exported: {output_path}")
        return output_path