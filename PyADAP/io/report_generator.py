"""Report Generator Module

This module provides functionality for generating analysis reports
in various formats including HTML, PDF, and Word documents.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pandas as pd
from ..config import Config
from ..utils import get_logger


class ReportGenerator:
    """Generates comprehensive analysis reports in multiple formats."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the report generator.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.logger = get_logger("PyADAP.ReportGenerator")
        
        # Report templates and settings
        self.templates_dir = Path(__file__).parent / "templates"
        self.output_dir = Path(self.config.output_dir if hasattr(self.config, 'output_dir') else "./reports")
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("ReportGenerator initialized")
    
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
            output_format: Output format ('html', 'pdf', 'docx')
            filename: Custom filename (optional)
            output_dir: Custom output directory (optional)
            
        Returns:
            Path to the generated report file
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"analysis_report_{timestamp}.{output_format}"
            
            # Use custom output directory if provided
            target_dir = output_dir if output_dir else self.output_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            output_path = target_dir / filename
            
            if output_format.lower() == "html":
                return self._generate_html_report(results, data_info, output_path)
            elif output_format.lower() == "pdf":
                return self._generate_pdf_report(results, data_info, output_path)
            elif output_format.lower() == "docx":
                return self._generate_docx_report(results, data_info, output_path)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
                
        except Exception as e:
            self.logger.error(f"Failed to generate report: {str(e)}")
            raise
    
    def _generate_html_report(self, results: Dict[str, Any], 
                            data_info: Dict[str, Any], 
                            output_path: Path) -> Path:
        """Generate HTML report."""
        html_content = self._create_html_content(results, data_info)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report generated: {output_path}")
        return output_path
    
    def _generate_pdf_report(self, results: Dict[str, Any], 
                           data_info: Dict[str, Any], 
                           output_path: Path) -> Path:
        """Generate PDF report."""
        # Placeholder for PDF generation
        # In a real implementation, you would use libraries like reportlab or weasyprint
        html_content = self._create_html_content(results, data_info)
        
        # For now, save as HTML with .pdf extension as placeholder
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"PDF report generated: {output_path}")
        return output_path
    
    def _generate_docx_report(self, results: Dict[str, Any], 
                            data_info: Dict[str, Any], 
                            output_path: Path) -> Path:
        """Generate Word document report."""
        # Placeholder for DOCX generation
        # In a real implementation, you would use python-docx library
        html_content = self._create_html_content(results, data_info)
        
        # For now, save as HTML with .docx extension as placeholder
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"DOCX report generated: {output_path}")
        return output_path
    
    def _create_html_content(self, results: Dict[str, Any], 
                           data_info: Dict[str, Any]) -> str:
        """Create modern, interactive HTML content for the report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pipeline_id = data_info.get('pipeline_id', 'Unknown')
        
        # Extract data shape information
        data_shape = data_info.get('data_shape', (0, 0))
        rows, cols = data_shape if data_shape else (0, 0)
        
        # Extract variable information
        variables = data_info.get('variables', {})
        dependent_vars = variables.get('dependent', [])
        independent_vars = variables.get('independent', [])
        subject_var = variables.get('subject', None)
        
        # Format analysis duration
        start_time = data_info.get('start_time')
        end_time = data_info.get('end_time')
        duration = "N/A"
        if start_time and end_time:
            duration_seconds = (end_time - start_time).total_seconds()
            duration = f"{duration_seconds:.2f} seconds"
        
        html_template = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PyADAP 数据分析报告 - {pipeline_id}</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        }}
        
        .header h1 {{
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .header .subtitle {{
            color: #7f8c8d;
            font-size: 1.2em;
            margin-bottom: 20px;
        }}
        
        .timestamp {{
            background: #3498db;
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            display: inline-block;
            font-weight: 500;
        }}
        
        .dashboard {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .card {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
        }}
        
        .card-header {{
            display: flex;
            align-items: center;
            margin-bottom: 20px;
        }}
        
        .card-icon {{
            font-size: 2em;
            margin-right: 15px;
            padding: 15px;
            border-radius: 50%;
            color: white;
        }}
        
        .icon-data {{ background: #e74c3c; }}
        .icon-stats {{ background: #f39c12; }}
        .icon-quality {{ background: #27ae60; }}
        .icon-results {{ background: #9b59b6; }}
        
        .card h3 {{
            color: #2c3e50;
            font-size: 1.3em;
            margin-bottom: 15px;
        }}
        
        .metric {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        .metric:last-child {{
            border-bottom: none;
        }}
        
        .metric-label {{
            font-weight: 500;
            color: #7f8c8d;
        }}
        
        .metric-value {{
            font-weight: 700;
            color: #2c3e50;
            font-size: 1.1em;
        }}
        
        .variables-list {{
            margin: 10px 0;
        }}
        
        .variable-tag {{
            display: inline-block;
            background: #3498db;
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            margin: 3px;
            font-size: 0.9em;
        }}
        
        .variable-tag.dependent {{ background: #e74c3c; }}
        .variable-tag.independent {{ background: #27ae60; }}
        .variable-tag.subject {{ background: #f39c12; }}
        
        .results-section {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }}
        
        .results-tabs {{
            display: flex;
            margin-bottom: 20px;
            border-bottom: 2px solid #ecf0f1;
        }}
        
        .tab-button {{
            background: none;
            border: none;
            padding: 15px 25px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 500;
            color: #7f8c8d;
            border-bottom: 3px solid transparent;
            transition: all 0.3s ease;
        }}
        
        .tab-button.active {{
            color: #3498db;
            border-bottom-color: #3498db;
        }}
        
        .tab-content {{
            display: none;
            animation: fadeIn 0.3s ease;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .json-viewer {{
            background: #2c3e50;
            color: #ecf0f1;
            padding: 20px;
            border-radius: 10px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            overflow-x: auto;
            max-height: 400px;
            overflow-y: auto;
        }}
        
        .footer {{
            text-align: center;
            padding: 30px;
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.9em;
        }}
        
        .footer a {{
            color: rgba(255, 255, 255, 0.9);
            text-decoration: none;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 8px;
            background: #ecf0f1;
            border-radius: 4px;
            overflow: hidden;
            margin: 10px 0;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2ecc71);
            border-radius: 4px;
            animation: progressAnimation 2s ease-in-out;
        }}
        
        @keyframes progressAnimation {{
            from {{ width: 0%; }}
            to {{ width: 100%; }}
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
            text-transform: uppercase;
        }}
        
        .status-success {{
            background: #d5f4e6;
            color: #27ae60;
        }}
        
        .collapsible {{
            cursor: pointer;
            padding: 15px;
            background: #f8f9fa;
            border: none;
            text-align: left;
            outline: none;
            font-size: 1em;
            font-weight: 500;
            width: 100%;
            border-radius: 8px;
            margin: 5px 0;
            transition: background-color 0.3s ease;
        }}
        
        .collapsible:hover {{
            background: #e9ecef;
        }}
        
        .collapsible-content {{
            padding: 0 15px;
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
            background: #f8f9fa;
            border-radius: 0 0 8px 8px;
        }}
        
        .collapsible-content.active {{
            max-height: 500px;
            padding: 15px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-chart-line"></i> PyADAP 数据分析报告</h1>
            <p class="subtitle">{pipeline_id}</p>
            <div class="timestamp">
                <i class="fas fa-clock"></i> 生成时间: {timestamp}
            </div>
        </div>
        
        <div class="dashboard">
            <div class="card">
                <div class="card-header">
                    <div class="card-icon icon-data">
                        <i class="fas fa-database"></i>
                    </div>
                    <h3>数据概览</h3>
                </div>
                <div class="metric">
                    <span class="metric-label">数据行数</span>
                    <span class="metric-value">{rows:,}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">数据列数</span>
                    <span class="metric-value">{cols}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">分析耗时</span>
                    <span class="metric-value">{duration}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">分析状态</span>
                    <span class="status-badge status-success">完成</span>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <div class="card-icon icon-stats">
                        <i class="fas fa-chart-bar"></i>
                    </div>
                    <h3>变量配置</h3>
                </div>
                <div class="variables-list">
                    <p><strong>因变量:</strong></p>
                    {self._format_variables(dependent_vars, 'dependent')}
                    <p><strong>自变量:</strong></p>
                    {self._format_variables(independent_vars, 'independent')}
                    {f'<p><strong>主体变量:</strong></p><span class="variable-tag subject">{subject_var}</span>' if subject_var else ''}
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <div class="card-icon icon-quality">
                        <i class="fas fa-check-circle"></i>
                    </div>
                    <h3>数据质量</h3>
                </div>
                {self._format_quality_metrics(results)}
            </div>
            
            <div class="card">
                <div class="card-header">
                    <div class="card-icon icon-results">
                        <i class="fas fa-flask"></i>
                    </div>
                    <h3>分析结果</h3>
                </div>
                {self._format_analysis_summary(results)}
            </div>
        </div>
        
        <div class="results-section">
            <h2><i class="fas fa-microscope"></i> 详细分析结果</h2>
            <div class="results-tabs">
                <button class="tab-button active" onclick="showTab('summary')">概要</button>
                <button class="tab-button" onclick="showTab('quality')">数据质量</button>
                <button class="tab-button" onclick="showTab('statistics')">统计分析</button>
                <button class="tab-button" onclick="showTab('raw')">原始数据</button>
            </div>
            
            <div id="summary" class="tab-content active">
                {self._format_summary_tab(results, data_info)}
            </div>
            
            <div id="quality" class="tab-content">
                {self._format_quality_tab(results)}
            </div>
            
            <div id="statistics" class="tab-content">
                {self._format_statistics_tab(results)}
            </div>
            
            <div id="raw" class="tab-content">
                <h3>原始分析结果</h3>
                <div class="json-viewer">
                    <pre>{self._format_json_results(results)}</pre>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>由 <strong>PyADAP</strong> 自动生成 | Python 自动化数据分析管道</p>
            <p>© 2024 PyADAP Project</p>
        </div>
    </div>
    
    <script>
        function showTab(tabName) {{
            // Hide all tab contents
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.classList.remove('active'));
            
            // Remove active class from all buttons
            const buttons = document.querySelectorAll('.tab-button');
            buttons.forEach(button => button.classList.remove('active'));
            
            // Show selected tab and activate button
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }}
        
        // Initialize collapsible elements
        document.addEventListener('DOMContentLoaded', function() {{
            const collapsibles = document.querySelectorAll('.collapsible');
            collapsibles.forEach(collapsible => {{
                collapsible.addEventListener('click', function() {{
                    this.classList.toggle('active');
                    const content = this.nextElementSibling;
                    content.classList.toggle('active');
                }});
            }});
        }});
    </script>
</body>
</html>
        """
        
        return html_template
    
    def _format_results(self, results: Dict[str, Any]) -> str:
        """Format results for display in report."""
        formatted = ""
        for key, value in results.items():
            formatted += f"{key}: {value}\n"
        return formatted
    
    def _format_variables(self, variables: List[str], var_type: str) -> str:
        """Format variables as HTML tags."""
        if not variables:
            return '<span class="variable-tag">无</span>'
        
        tags = []
        for var in variables:
            tags.append(f'<span class="variable-tag {var_type}">{var}</span>')
        return '\n'.join(tags)
    
    def _format_quality_metrics(self, results: Dict[str, Any]) -> str:
        """Format data quality metrics."""
        quality_report = results.get('quality_report', {})
        
        if not quality_report:
            return '<div class="metric"><span class="metric-label">质量评估</span><span class="metric-value">未完成</span></div>'
        
        metrics = []
        
        # Missing values
        missing_info = quality_report.get('missing_values', {})
        if missing_info:
            total_missing = sum(missing_info.values()) if isinstance(missing_info, dict) else 0
            metrics.append(f'<div class="metric"><span class="metric-label">缺失值</span><span class="metric-value">{total_missing}</span></div>')
        
        # Duplicates
        duplicates = quality_report.get('duplicates', 0)
        metrics.append(f'<div class="metric"><span class="metric-label">重复行</span><span class="metric-value">{duplicates}</span></div>')
        
        # Data types
        data_types = quality_report.get('data_types', {})
        if data_types:
            type_count = len(data_types)
            metrics.append(f'<div class="metric"><span class="metric-label">数据类型</span><span class="metric-value">{type_count} 种</span></div>')
        
        return '\n'.join(metrics) if metrics else '<div class="metric"><span class="metric-label">质量评估</span><span class="metric-value">良好</span></div>'
    
    def _format_analysis_summary(self, results: Dict[str, Any]) -> str:
        """Format analysis summary."""
        summary_items = []
        
        # Statistical tests
        stats_tests = results.get('statistical_tests', {})
        if stats_tests:
            test_count = len(stats_tests)
            summary_items.append(f'<div class="metric"><span class="metric-label">统计检验</span><span class="metric-value">{test_count} 项</span></div>')
        
        # Visualizations
        visualizations = results.get('visualizations', {})
        if visualizations:
            viz_count = len(visualizations)
            summary_items.append(f'<div class="metric"><span class="metric-label">可视化图表</span><span class="metric-value">{viz_count} 个</span></div>')
        
        # Recommendations
        recommendations = results.get('recommendations', [])
        rec_count = len(recommendations) if isinstance(recommendations, list) else 0
        summary_items.append(f'<div class="metric"><span class="metric-label">建议</span><span class="metric-value">{rec_count} 条</span></div>')
        
        # Assumptions
        assumptions = results.get('assumptions', {})
        if assumptions:
            assumption_count = len(assumptions)
            summary_items.append(f'<div class="metric"><span class="metric-label">假设检验</span><span class="metric-value">{assumption_count} 项</span></div>')
        
        return '\n'.join(summary_items) if summary_items else '<div class="metric"><span class="metric-label">分析状态</span><span class="metric-value">已完成</span></div>'
    
    def _format_summary_tab(self, results: Dict[str, Any], data_info: Dict[str, Any]) -> str:
        """Format summary tab content."""
        content = '<h3>分析概要</h3>'
        
        # Pipeline information
        pipeline_id = data_info.get('pipeline_id', 'Unknown')
        start_time = data_info.get('start_time')
        end_time = data_info.get('end_time')
        
        content += f'''
        <div class="metric">
            <span class="metric-label">分析ID</span>
            <span class="metric-value">{pipeline_id}</span>
        </div>
        '''
        
        if start_time:
            content += f'''
            <div class="metric">
                <span class="metric-label">开始时间</span>
                <span class="metric-value">{start_time.strftime("%Y-%m-%d %H:%M:%S") if hasattr(start_time, 'strftime') else str(start_time)}</span>
            </div>
            '''
        
        if end_time:
            content += f'''
            <div class="metric">
                <span class="metric-label">结束时间</span>
                <span class="metric-value">{end_time.strftime("%Y-%m-%d %H:%M:%S") if hasattr(end_time, 'strftime') else str(end_time)}</span>
            </div>
            '''
        
        # Recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            content += '<h4>主要建议</h4><ul>'
            for rec in recommendations[:5]:  # Show top 5 recommendations
                content += f'<li>{rec}</li>'
            content += '</ul>'
        
        return content
    
    def _format_quality_tab(self, results: Dict[str, Any]) -> str:
        """Format quality tab content."""
        quality_report = results.get('quality_report', {})
        
        if not quality_report:
            return '<p>数据质量报告未生成。</p>'
        
        content = '<h3>数据质量详情</h3>'
        
        # Missing values detail
        missing_values = quality_report.get('missing_values', {})
        if missing_values and isinstance(missing_values, dict):
            content += '<h4>缺失值分布</h4>'
            content += '<div class="json-viewer"><pre>'
            for col, count in missing_values.items():
                content += f'{col}: {count}\n'
            content += '</pre></div>'
        
        # Data types
        data_types = quality_report.get('data_types', {})
        if data_types:
            content += '<h4>数据类型</h4>'
            content += '<div class="json-viewer"><pre>'
            for col, dtype in data_types.items():
                content += f'{col}: {dtype}\n'
            content += '</pre></div>'
        
        return content
    
    def _format_statistics_tab(self, results: Dict[str, Any]) -> str:
        """Format statistics tab content."""
        content = '<h3>统计分析结果</h3>'
        
        # Statistical tests
        stats_tests = results.get('statistical_tests', {})
        if stats_tests:
            content += '<h4>统计检验</h4>'
            for test_name, test_result in stats_tests.items():
                content += f'<button class="collapsible">{test_name}</button>'
                content += f'<div class="collapsible-content"><div class="json-viewer"><pre>{self._format_dict_content(test_result)}</pre></div></div>'
        
        # Assumptions
        assumptions = results.get('assumptions', {})
        if assumptions:
            content += '<h4>假设检验</h4>'
            for assumption_name, assumption_result in assumptions.items():
                content += f'<button class="collapsible">{assumption_name}</button>'
                content += f'<div class="collapsible-content"><div class="json-viewer"><pre>{self._format_dict_content(assumption_result)}</pre></div></div>'
        
        return content
    
    def _format_json_results(self, results: Dict[str, Any]) -> str:
        """Format results as JSON for display."""
        import json
        try:
            return json.dumps(results, indent=2, ensure_ascii=False, default=str)
        except Exception:
            return str(results)
    
    def _format_dict_content(self, content: Any) -> str:
        """Format dictionary content for display."""
        if isinstance(content, dict):
            formatted = ""
            for key, value in content.items():
                formatted += f"{key}: {value}\n"
            return formatted
        else:
            return str(content)
    
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