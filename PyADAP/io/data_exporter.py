"""Data Exporter Module

This module provides functionality for exporting analysis results
and processed data in various formats.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pandas as pd
import json
from ..config import Config
from ..utils import get_logger


class DataExporter:
    """Exports data and analysis results in multiple formats."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the data exporter.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.logger = get_logger("PyADAP.DataExporter")
        
        # Export settings
        self.output_dir = Path(self.config.output_dir if hasattr(self.config, 'output_dir') else "./exports")
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("DataExporter initialized")
    
    def export_dataframe(self, 
                        data: pd.DataFrame,
                        filename: str,
                        format_type: str = "csv",
                        **kwargs) -> Path:
        """Export DataFrame to specified format.
        
        Args:
            data: DataFrame to export
            filename: Output filename (without extension)
            format_type: Export format ('csv', 'excel', 'json', 'parquet')
            **kwargs: Additional arguments for export functions
            
        Returns:
            Path to the exported file
        """
        try:
            # Add appropriate extension if not present
            if not any(filename.endswith(ext) for ext in ['.csv', '.xlsx', '.json', '.parquet']):
                filename = f"{filename}.{format_type}"
            
            output_path = self.output_dir / filename
            
            if format_type.lower() == "csv":
                data.to_csv(output_path, index=False, **kwargs)
            elif format_type.lower() == "excel":
                data.to_excel(output_path, index=False, **kwargs)
            elif format_type.lower() == "json":
                data.to_json(output_path, orient='records', indent=2, **kwargs)
            elif format_type.lower() == "parquet":
                data.to_parquet(output_path, **kwargs)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
            
            self.logger.info(f"Data exported to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to export data: {str(e)}")
            raise
    
    def export_results(self, 
                      results: Dict[str, Any],
                      filename: Optional[str] = None,
                      format_type: str = "json") -> Path:
        """Export analysis results to file.
        
        Args:
            results: Results dictionary to export
            filename: Output filename (optional)
            format_type: Export format ('json', 'csv', 'txt')
            
        Returns:
            Path to the exported file
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"analysis_results_{timestamp}.{format_type}"
            
            output_path = self.output_dir / filename
            
            if format_type.lower() == "json":
                self._export_results_json(results, output_path)
            elif format_type.lower() == "csv":
                self._export_results_csv(results, output_path)
            elif format_type.lower() == "txt":
                self._export_results_txt(results, output_path)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
            
            self.logger.info(f"Results exported to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to export results: {str(e)}")
            raise
    
    def _export_results_json(self, results: Dict[str, Any], output_path: Path) -> None:
        """Export results as JSON."""
        # Convert non-serializable objects to strings
        serializable_results = self._make_serializable(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    def _export_results_csv(self, results: Dict[str, Any], output_path: Path) -> None:
        """Export results as CSV."""
        # Flatten results for CSV export
        flattened = self._flatten_dict(results)
        df = pd.DataFrame([flattened])
        df.to_csv(output_path, index=False)
    
    def _export_results_txt(self, results: Dict[str, Any], output_path: Path) -> None:
        """Export results as formatted text."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Analysis Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            self._write_dict_to_file(results, f, indent=0)
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, str(v)))
        return dict(items)
    
    def _write_dict_to_file(self, d: Dict[str, Any], file, indent: int = 0) -> None:
        """Write dictionary to file with proper formatting."""
        for key, value in d.items():
            if isinstance(value, dict):
                file.write("  " * indent + f"{key}:\n")
                self._write_dict_to_file(value, file, indent + 1)
            else:
                file.write("  " * indent + f"{key}: {value}\n")
    
    def export_multiple_datasets(self, 
                               datasets: Dict[str, pd.DataFrame],
                               base_filename: str,
                               format_type: str = "csv") -> List[Path]:
        """Export multiple datasets with consistent naming.
        
        Args:
            datasets: Dictionary of dataset name -> DataFrame
            base_filename: Base filename for exports
            format_type: Export format
            
        Returns:
            List of paths to exported files
        """
        exported_files = []
        
        for name, data in datasets.items():
            filename = f"{base_filename}_{name}.{format_type}"
            output_path = self.export_dataframe(data, filename, format_type)
            exported_files.append(output_path)
        
        self.logger.info(f"Exported {len(exported_files)} datasets")
        return exported_files
    
    def create_export_manifest(self, 
                             exported_files: List[Path],
                             description: str = "") -> Path:
        """Create a manifest file listing all exported files.
        
        Args:
            exported_files: List of exported file paths
            description: Optional description of the export batch
            
        Returns:
            Path to the manifest file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest_path = self.output_dir / f"export_manifest_{timestamp}.json"
        
        manifest = {
            "export_timestamp": datetime.now().isoformat(),
            "description": description,
            "exported_files": [
                {
                    "filename": file.name,
                    "path": str(file),
                    "size_bytes": file.stat().st_size if file.exists() else 0
                }
                for file in exported_files
            ],
            "total_files": len(exported_files)
        }
        
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        
        self.logger.info(f"Export manifest created: {manifest_path}")
        return manifest_path