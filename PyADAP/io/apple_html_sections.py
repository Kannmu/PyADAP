#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apple-Style HTML Sections Generator for PyADAP Reports

This module contains methods for generating specific sections of the HTML report.
"""

from typing import Dict, List, Any


class AppleHTMLSections:
    """Generator for specific HTML sections."""
    
    def __init__(self):
        """Initialize the sections generator."""
        pass
    
    def create_overview_section(self, results: Dict[str, Any], data_info: Dict[str, Any], visualizations: Dict[str, str]) -> str:
        """Create the overview section with data summary and key metrics."""
        # Extract key metrics
        n_samples = data_info.get('n_samples', 0)
        n_features = data_info.get('n_features', 0)
        
        # Quality metrics
        quality_report = results.get('quality_report', {})
        missing_rate = quality_report.get('missing_rate', 0)
        duplicates = quality_report.get('duplicates', 0)
        
        # Statistical summary
        summary_stats = results.get('summary_statistics', {})
        
        return f"""
        <div class="overview-grid">
            <div class="metric-card">
                <div class="metric-icon">
                    <i class="fas fa-database"></i>
                </div>
                <div class="metric-content">
                    <div class="metric-value">{n_samples:,}</div>
                    <div class="metric-label">样本数量</div>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-icon">
                    <i class="fas fa-columns"></i>
                </div>
                <div class="metric-content">
                    <div class="metric-value">{n_features:,}</div>
                    <div class="metric-label">特征数量</div>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-icon">
                    <i class="fas fa-exclamation-triangle"></i>
                </div>
                <div class="metric-content">
                    <div class="metric-value">{missing_rate:.1f}%</div>
                    <div class="metric-label">缺失率</div>
                </div>
            </div>
            
            <div class="metric-card">
                <div class="metric-icon">
                    <i class="fas fa-copy"></i>
                </div>
                <div class="metric-content">
                    <div class="metric-value">{duplicates:,}</div>
                    <div class="metric-label">重复行</div>
                </div>
            </div>
        </div>
        
        {self._create_data_overview_visualization(visualizations)}
        """
    
    def create_quality_section(self, results: Dict[str, Any], visualizations: Dict[str, str]) -> str:
        """Create the data quality assessment section (public interface)."""
        return self._create_quality_section(results, visualizations)
    
    def create_analysis_section(self, results: Dict[str, Any], visualizations: Dict[str, str]) -> str:
        """Create the statistical analysis section (public interface)."""
        return self._create_analysis_section(results, visualizations)
    
    def create_visualizations_section(self, visualizations: Dict[str, str]) -> str:
        """Create the visualizations section (public interface)."""
        return self._create_visualizations_section(visualizations)
    
    def create_insights_section(self, results: Dict[str, Any]) -> str:
        """Create the insights and recommendations section."""
        # Extract key insights from results
        quality_report = results.get('quality_report', {})
        statistical_tests = results.get('statistical_tests', {})
        assumptions = results.get('assumptions', {})
        
        insights = []
        recommendations = []
        
        # Data quality insights
        missing_rate = quality_report.get('missing_rate', 0)
        if missing_rate > 10:
            insights.append({
                'type': 'warning',
                'title': '数据质量警告',
                'content': f'数据集存在{missing_rate:.1f}%的缺失值，可能影响分析结果的可靠性。'
            })
            recommendations.append({
                'priority': 'high',
                'action': '数据清洗',
                'description': '建议进行缺失值处理，可考虑删除、插值或使用模型预测填充。'
            })
        
        # Statistical test insights
        significant_tests = []
        for test_name, test_result in statistical_tests.items():
            if isinstance(test_result, dict):
                p_val = test_result.get('p_value', test_result.get('pvalue', 1))
                if hasattr(p_val, 'item'):
                    p_val = p_val.item()
                if isinstance(p_val, (int, float)) and float(p_val) < 0.05:
                    significant_tests.append(test_name)
        
        if significant_tests:
            insights.append({
                'type': 'success',
                'title': '显著性发现',
                'content': f'发现{len(significant_tests)}个具有统计显著性的检验结果。'
            })
        
        # Generate HTML
        insights_html = self._format_insights(insights)
        recommendations_html = self._format_recommendations(recommendations)
        
        return f"""
        <div class="insights-container">
            <div class="insights-section">
                <h3 class="insights-title">
                    <i class="fas fa-eye"></i>
                    关键洞察
                </h3>
                <div class="insights-content">
                    {insights_html}
                </div>
            </div>
            
            <div class="recommendations-section">
                <h3 class="recommendations-title">
                    <i class="fas fa-tasks"></i>
                    行动建议
                </h3>
                <div class="recommendations-content">
                    {recommendations_html}
                </div>
            </div>
        </div>
         """
     
    def _format_insights(self, insights: List[Dict[str, str]]) -> str:
        """Format insights into HTML."""
        if not insights:
            return '<p class="no-data">暂无关键洞察</p>'
        
        items = []
        for insight in insights:
            insight_type = insight.get('type', 'info')
            title = insight.get('title', '洞察')
            content = insight.get('content', '')
            
            icon_map = {
                'success': 'fas fa-check-circle',
                'warning': 'fas fa-exclamation-triangle', 
                'error': 'fas fa-times-circle',
                'info': 'fas fa-info-circle'
            }
            
            icon = icon_map.get(insight_type, 'fas fa-info-circle')
            
            items.append(f"""
            <div class="insight-item {insight_type}">
                <div class="insight-header">
                    <i class="{icon}"></i>
                    <span class="insight-title">{title}</span>
                </div>
                <div class="insight-content">{content}</div>
            </div>
            """)
        
        return ''.join(items)
    
    def _format_recommendations(self, recommendations: List[Dict[str, str]]) -> str:
        """Format recommendations into HTML."""
        if not recommendations:
            return '<p class="no-data">暂无行动建议</p>'
        
        items = []
        for rec in recommendations:
            priority = rec.get('priority', 'medium')
            action = rec.get('action', '建议')
            description = rec.get('description', '')
            
            priority_class = {
                'high': 'priority-high',
                'medium': 'priority-medium',
                'low': 'priority-low'
            }.get(priority, 'priority-medium')
            
            items.append(f"""
            <div class="recommendation-item {priority_class}">
                <div class="recommendation-header">
                    <span class="recommendation-action">{action}</span>
                    <span class="priority-badge {priority_class}">{priority.upper()}</span>
                </div>
                <div class="recommendation-description">{description}</div>
            </div>
            """)
        
        return ''.join(items)
     
    def _create_data_overview_visualization(self, visualizations: Dict[str, str]) -> str:
        """Create the data overview visualization section."""
        overview_viz = visualizations.get('data_overview', '')
        
        return f"""
        <div class="visualization-container">
            <h3 class="visualization-title">
                <i class="fas fa-chart-area"></i>
                数据概览仪表板
            </h3>
            <div class="visualization-content">
                {f'<img src="data:image/png;base64,{overview_viz}" alt="数据概览" class="visualization-image">' if overview_viz else '<p class="no-data">暂无数据概览图表</p>'}
            </div>
        </div>
        """
    
    def _create_quality_section(self, results: Dict[str, Any], visualizations: Dict[str, str]) -> str:
        """Create the data quality assessment section."""
        quality_report = results.get('quality_report', {})
        quality_viz = visualizations.get('data_quality', '')
        
        # Missing values analysis
        missing_values = quality_report.get('missing_values', {})
        missing_html = self._format_missing_values(missing_values)
        
        # Duplicates analysis
        duplicates = quality_report.get('duplicates', 0)
        
        # Data types analysis
        data_types = quality_report.get('data_types', {})
        types_html = self._format_data_types(data_types)
        
        return f"""
        <div class="content-grid">
            <div class="info-card">
                <div class="card-header">
                    <h3 class="card-title">
                        <i class="fas fa-exclamation-triangle"></i>
                        缺失值分析
                    </h3>
                </div>
                <div class="card-content">
                    {missing_html}
                </div>
            </div>
            
            <div class="info-card">
                <div class="card-header">
                    <h3 class="card-title">
                        <i class="fas fa-copy"></i>
                        重复值检测
                    </h3>
                </div>
                <div class="card-content">
                    <div class="info-grid">
                        <div class="info-item">
                            <span class="info-label">重复行数</span>
                            <span class="info-value">{duplicates:,}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">数据完整性</span>
                            <span class="status-badge {'success' if duplicates == 0 else 'warning'}">
                                {'优秀' if duplicates == 0 else '需要关注'}
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="visualization-container">
            <h3 class="visualization-title">
                <i class="fas fa-chart-doughnut"></i>
                数据质量可视化
            </h3>
            <div class="visualization-content">
                {f'<img src="data:image/png;base64,{quality_viz}" alt="数据质量" class="visualization-image">' if quality_viz else '<p class="no-data">暂无数据质量图表</p>'}
            </div>
        </div>
        
        <div class="info-card">
            <div class="card-header">
                <h3 class="card-title">
                    <i class="fas fa-list-alt"></i>
                    数据类型分布
                </h3>
            </div>
            <div class="card-content">
                {types_html}
            </div>
        </div>
        """
    
    def _create_analysis_section(self, results: Dict[str, Any], visualizations: Dict[str, str]) -> str:
        """Create the statistical analysis section."""
        stats_tests = results.get('statistical_tests', {})
        assumptions = results.get('assumptions', {})
        
        # Statistical tests results
        tests_html = self._format_statistical_tests(stats_tests)
        
        # Assumptions validation
        assumptions_html = self._format_assumptions(assumptions)
        
        # Visualizations
        stats_viz = visualizations.get('statistical_tests', '')
        correlation_viz = visualizations.get('correlation_heatmap', '')
        effect_sizes_viz = visualizations.get('effect_sizes', '')
        
        return f"""
        <div class="content-grid">
            <div class="info-card">
                <div class="card-header">
                    <h3 class="card-title">
                        <i class="fas fa-flask"></i>
                        统计检验结果
                    </h3>
                </div>
                <div class="card-content">
                    {tests_html}
                </div>
            </div>
            
            <div class="info-card">
                <div class="card-header">
                    <h3 class="card-title">
                        <i class="fas fa-check-circle"></i>
                        假设验证
                    </h3>
                </div>
                <div class="card-content">
                    {assumptions_html}
                </div>
            </div>
        </div>
        
        <div class="visualization-container">
            <h3 class="visualization-title">
                <i class="fas fa-chart-bar"></i>
                统计检验结果可视化
            </h3>
            <div class="visualization-content">
                {f'<img src="data:image/png;base64,{stats_viz}" alt="统计检验" class="visualization-image">' if stats_viz else '<p class="no-data">暂无统计检验图表</p>'}
            </div>
        </div>
        
        <div class="visualization-container">
            <h3 class="visualization-title">
                <i class="fas fa-th"></i>
                相关性热力图
            </h3>
            <div class="visualization-content">
                {f'<img src="data:image/png;base64,{correlation_viz}" alt="相关性分析" class="visualization-image">' if correlation_viz else '<p class="no-data">暂无相关性图表</p>'}
            </div>
        </div>
        
        <div class="visualization-container">
            <h3 class="visualization-title">
                <i class="fas fa-ruler"></i>
                效应量分析
            </h3>
            <div class="visualization-content">
                {f'<img src="data:image/png;base64,{effect_sizes_viz}" alt="效应量" class="visualization-image">' if effect_sizes_viz else '<p class="no-data">暂无效应量图表</p>'}
            </div>
        </div>
        """
    
    def _create_visualizations_section(self, visualizations: Dict[str, str]) -> str:
        """Create the visualizations section."""
        viz_sections = []
        
        viz_configs = [
            ('distribution_analysis', '分布分析', 'fas fa-chart-area', '变量分布和密度分析'),
            ('variable_importance', '变量重要性', 'fas fa-star', '特征重要性排序'),
            ('assumptions_validation', '假设验证', 'fas fa-check-double', '统计假设验证结果'),
            ('data_overview', '数据概览', 'fas fa-tachometer-alt', '综合数据仪表板')
        ]
        
        for viz_key, title, icon, description in viz_configs:
            viz_data = visualizations.get(viz_key, '')
            if viz_data:
                viz_sections.append(f"""
                <div class="visualization-container">
                    <h3 class="visualization-title">
                        <i class="{icon}"></i>
                        {title}
                    </h3>
                    <p class="visualization-description">{description}</p>
                    <div class="visualization-content">
                        <img src="data:image/png;base64,{viz_data}" alt="{title}" class="visualization-image">
                    </div>
                </div>
                """)
        
        if not viz_sections:
            viz_sections.append("""
            <div class="info-card">
                <div class="card-content">
                    <p class="no-data">暂无可视化图表生成</p>
                </div>
            </div>
            """)
        
        return '\n'.join(viz_sections)
    
    def _create_insights_section(self, results: Dict[str, Any]) -> str:
        """Create the insights and recommendations section."""
        # Extract insights from results
        stats_tests = results.get('statistical_tests', {})
        quality_report = results.get('quality_report', {})
        
        # Generate insights
        insights = self._generate_insights(stats_tests, quality_report)
        
        return f"""
        <div class="content-grid">
            <div class="info-card">
                <div class="card-header">
                    <h3 class="card-title">
                        <i class="fas fa-lightbulb"></i>
                        关键发现
                    </h3>
                </div>
                <div class="card-content">
                    <div class="insights-list">
                        {insights['key_findings']}
                    </div>
                </div>
            </div>
            
            <div class="info-card">
                <div class="card-header">
                    <h3 class="card-title">
                        <i class="fas fa-exclamation-circle"></i>
                        注意事项
                    </h3>
                </div>
                <div class="card-content">
                    <div class="insights-list">
                        {insights['warnings']}
                    </div>
                </div>
            </div>
        </div>
        
        <div class="info-card">
            <div class="card-header">
                <h3 class="card-title">
                    <i class="fas fa-tasks"></i>
                    行动建议
                </h3>
            </div>
            <div class="card-content">
                <div class="recommendations-list">
                    {insights['recommendations']}
                </div>
            </div>
        </div>
        
        <div class="info-card">
            <div class="card-header">
                <h3 class="card-title">
                    <i class="fas fa-chart-line"></i>
                    统计摘要
                </h3>
            </div>
            <div class="card-content">
                {insights['statistical_summary']}
            </div>
        </div>
        """
    
    def _format_missing_values(self, missing_values: Dict[str, Any]) -> str:
        """Format missing values information."""
        if not missing_values:
            return '<p class="no-data">无缺失值数据</p>'
        
        items = []
        for column, count in missing_values.items():
            if count > 0:
                items.append(f"""
                <div class="info-item">
                    <span class="info-label">{column}</span>
                    <span class="info-value">{count:,} 个缺失值</span>
                </div>
                """)
        
        if not items:
            return '<p class="no-data">所有变量均无缺失值</p>'
        
        return f'<div class="info-grid">{"".join(items)}</div>'
    
    def _format_data_types(self, data_types: Dict[str, Any]) -> str:
        """Format data types information."""
        if not data_types:
            return '<p class="no-data">无数据类型信息</p>'
        
        type_counts = {}
        for column, dtype in data_types.items():
            dtype_str = str(dtype)
            if 'int' in dtype_str or 'float' in dtype_str:
                category = '数值型'
            elif 'object' in dtype_str or 'string' in dtype_str:
                category = '文本型'
            elif 'bool' in dtype_str:
                category = '布尔型'
            elif 'datetime' in dtype_str:
                category = '日期型'
            else:
                category = '其他'
            
            type_counts[category] = type_counts.get(category, 0) + 1
        
        items = []
        for dtype, count in type_counts.items():
            items.append(f"""
            <div class="info-item">
                <span class="info-label">{dtype}</span>
                <span class="info-value">{count} 个变量</span>
            </div>
            """)
        
        return f'<div class="info-grid">{"".join(items)}</div>'
    
    def _format_statistical_tests(self, stats_tests: Dict[str, Any]) -> str:
        """Format statistical tests results."""
        if not stats_tests:
            return '<p class="no-data">无统计检验结果</p>'
        
        items = []
        for test_name, test_result in stats_tests.items():
            if isinstance(test_result, dict):
                p_val = test_result.get('p_value', test_result.get('pvalue', 'N/A'))
                if hasattr(p_val, 'item'):
                    p_val = p_val.item()
                
                # Format p-value
                if isinstance(p_val, (int, float)):
                    p_val_str = f"{float(p_val):.4f}"
                    significance = "显著" if float(p_val) < 0.05 else "不显著"
                    badge_class = "success" if float(p_val) < 0.05 else "warning"
                else:
                    p_val_str = str(p_val)
                    significance = "未知"
                    badge_class = "error"
                
                items.append(f"""
                <div class="test-result">
                    <div class="test-header">
                        <span class="test-name">{test_name}</span>
                        <span class="status-badge {badge_class}">{significance}</span>
                    </div>
                    <div class="test-details">
                        <span class="test-pvalue">p = {p_val_str}</span>
                    </div>
                </div>
                """)
        
        return f'<div class="tests-list">{"".join(items)}</div>'
    
    def _format_assumptions(self, assumptions: Dict[str, Any]) -> str:
        """Format assumptions validation results."""
        if not assumptions:
            return '<p class="no-data">无假设验证结果</p>'
        
        items = []
        for assumption_name, result in assumptions.items():
            if isinstance(result, dict):
                met = result.get('met', result.get('assumption_met', False))
                status = "满足" if met else "不满足"
                badge_class = "success" if met else "error"
                
                items.append(f"""
                <div class="assumption-result">
                    <div class="assumption-header">
                        <span class="assumption-name">{assumption_name}</span>
                        <span class="status-badge {badge_class}">{status}</span>
                    </div>
                </div>
                """)
        
        return f'<div class="assumptions-list">{"".join(items)}</div>'
    
    def _generate_insights(self, stats_tests: Dict[str, Any], quality_report: Dict[str, Any]) -> Dict[str, str]:
        """Generate insights and recommendations."""
        insights = {
            'key_findings': '',
            'warnings': '',
            'recommendations': '',
            'statistical_summary': ''
        }
        
        # Key findings
        findings = []
        significant_tests = 0
        total_tests = len(stats_tests)
        
        for test_result in stats_tests.values():
            if isinstance(test_result, dict):
                p_val = test_result.get('p_value', test_result.get('pvalue', 1.0))
                if hasattr(p_val, 'item'):
                    p_val = p_val.item()
                if isinstance(p_val, (int, float)) and float(p_val) < 0.05:
                    significant_tests += 1
        
        if total_tests > 0:
            significance_rate = (significant_tests / total_tests) * 100
            findings.append(f"<li>共进行了 {total_tests} 项统计检验，其中 {significant_tests} 项具有统计学意义 ({significance_rate:.1f}%)</li>")
        
        # Data quality findings
        missing_values = quality_report.get('missing_values', {})
        total_missing = sum(missing_values.values()) if isinstance(missing_values, dict) else 0
        if total_missing == 0:
            findings.append("<li>数据完整性良好，无缺失值</li>")
        else:
            findings.append(f"<li>检测到 {total_missing:,} 个缺失值，需要进行数据清理</li>")
        
        duplicates = quality_report.get('duplicates', 0)
        if duplicates == 0:
            findings.append("<li>数据唯一性良好，无重复记录</li>")
        else:
            findings.append(f"<li>检测到 {duplicates:,} 条重复记录</li>")
        
        insights['key_findings'] = f'<ul class="findings-list">{"".join(findings)}</ul>'
        
        # Warnings
        warnings = []
        if total_missing > 0:
            warnings.append("<li>存在缺失值，可能影响分析结果的准确性</li>")
        if duplicates > 0:
            warnings.append("<li>存在重复数据，建议进行去重处理</li>")
        if significant_tests == 0 and total_tests > 0:
            warnings.append("<li>所有统计检验均无显著性，可能需要调整分析策略</li>")
        
        if not warnings:
            warnings.append("<li>数据质量良好，无明显问题</li>")
        
        insights['warnings'] = f'<ul class="warnings-list">{"".join(warnings)}</ul>'
        
        # Recommendations
        recommendations = []
        if total_missing > 0:
            recommendations.append("<li>对缺失值进行适当的处理（删除、插值或标记）</li>")
        if duplicates > 0:
            recommendations.append("<li>清理重复数据以提高数据质量</li>")
        if significant_tests > 0:
            recommendations.append("<li>深入分析显著性结果，探索潜在的因果关系</li>")
        
        recommendations.append("<li>考虑进行更多的探索性数据分析</li>")
        recommendations.append("<li>验证分析结果的实际意义和业务价值</li>")
        
        insights['recommendations'] = f'<ul class="recommendations-list">{"".join(recommendations)}</ul>'
        
        # Statistical summary
        summary_items = []
        summary_items.append(f"<div class='summary-item'><span class='summary-label'>统计检验数量:</span><span class='summary-value'>{total_tests}</span></div>")
        summary_items.append(f"<div class='summary-item'><span class='summary-label'>显著结果数量:</span><span class='summary-value'>{significant_tests}</span></div>")
        summary_items.append(f"<div class='summary-item'><span class='summary-label'>缺失值总数:</span><span class='summary-value'>{total_missing:,}</span></div>")
        summary_items.append(f"<div class='summary-item'><span class='summary-label'>重复记录数:</span><span class='summary-value'>{duplicates:,}</span></div>")
        
        insights['statistical_summary'] = f'<div class="summary-grid">{"".join(summary_items)}</div>'
        
        return insights
    
    def _get_javascript_code(self, results: Dict[str, Any], visualizations: Dict[str, str]) -> str:
        """Generate JavaScript code for interactivity."""
        return """
        // Tab switching functionality
        document.addEventListener('DOMContentLoaded', function() {
            const tabButtons = document.querySelectorAll('.tab-button');
            const tabContents = document.querySelectorAll('.tab-content');
            
            tabButtons.forEach(button => {
                button.addEventListener('click', () => {
                    const targetTab = button.getAttribute('data-tab');
                    
                    // Remove active class from all buttons and contents
                    tabButtons.forEach(btn => btn.classList.remove('active'));
                    tabContents.forEach(content => content.classList.remove('active'));
                    
                    // Add active class to clicked button and corresponding content
                    button.classList.add('active');
                    document.getElementById(targetTab).classList.add('active');
                    
                    // Animate content
                    const activeContent = document.getElementById(targetTab);
                    activeContent.style.opacity = '0';
                    activeContent.style.transform = 'translateY(20px)';
                    
                    setTimeout(() => {
                        activeContent.style.transition = 'all 0.3s ease';
                        activeContent.style.opacity = '1';
                        activeContent.style.transform = 'translateY(0)';
                    }, 50);
                });
            });
            
            // Initialize hero chart
            initializeHeroChart();
            
            // Initialize animations
            initializeAnimations();
            
            // Initialize theme toggle
            initializeThemeToggle();
        });
        
        // Hero chart initialization
        function initializeHeroChart() {
            const ctx = document.getElementById('heroChart');
            if (!ctx) return;
            
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: ['数据加载', '质量检查', '统计分析', '可视化', '报告生成'],
                    datasets: [{
                        label: '分析进度',
                        data: [100, 100, 100, 100, 100],
                        borderColor: '#007AFF',
                        backgroundColor: 'rgba(0, 122, 255, 0.1)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        x: {
                            display: false
                        },
                        y: {
                            display: false,
                            min: 0,
                            max: 100
                        }
                    },
                    elements: {
                        point: {
                            radius: 0
                        }
                    }
                }
            });
        }
        
        // Initialize animations
        function initializeAnimations() {
            // Animate stat cards
            const statCards = document.querySelectorAll('.stat-card');
            statCards.forEach((card, index) => {
                card.style.opacity = '0';
                card.style.transform = 'translateY(30px)';
                
                setTimeout(() => {
                    card.style.transition = 'all 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94)';
                    card.style.opacity = '1';
                    card.style.transform = 'translateY(0)';
                }, index * 100);
            });
            
            // Animate stat values
            const statValues = document.querySelectorAll('.stat-value');
            statValues.forEach(value => {
                const finalValue = parseInt(value.textContent.replace(/,/g, ''));
                if (!isNaN(finalValue)) {
                    animateCounter(value, 0, finalValue, 2000);
                }
            });
        }
        
        // Counter animation
        function animateCounter(element, start, end, duration) {
            const startTime = performance.now();
            const isFloat = element.textContent.includes('.');
            
            function updateCounter(currentTime) {
                const elapsed = currentTime - startTime;
                const progress = Math.min(elapsed / duration, 1);
                
                const current = start + (end - start) * easeOutCubic(progress);
                
                if (isFloat) {
                    element.textContent = current.toFixed(1) + '%';
                } else {
                    element.textContent = Math.floor(current).toLocaleString();
                }
                
                if (progress < 1) {
                    requestAnimationFrame(updateCounter);
                }
            }
            
            requestAnimationFrame(updateCounter);
        }
        
        // Easing function
        function easeOutCubic(t) {
            return 1 - Math.pow(1 - t, 3);
        }
        
        // Theme toggle
        function initializeThemeToggle() {
            const themeToggle = document.querySelector('.nav-btn');
            if (!themeToggle) return;
            
            // Check for saved theme preference
            const savedTheme = localStorage.getItem('theme') || 'light';
            document.documentElement.setAttribute('data-theme', savedTheme);
            
            themeToggle.addEventListener('click', () => {
                const currentTheme = document.documentElement.getAttribute('data-theme');
                const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
                
                document.documentElement.setAttribute('data-theme', newTheme);
                localStorage.setItem('theme', newTheme);
                
                // Update icon
                const icon = themeToggle.querySelector('i');
                icon.className = newTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
            });
        }
        
        // Export functionality
        function exportReport() {
            window.print();
        }
        
        // Smooth scrolling for internal links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
        
        // Add loading states for images
        document.querySelectorAll('.visualization-image').forEach(img => {
            img.addEventListener('load', function() {
                this.style.opacity = '1';
            });
            
            img.addEventListener('error', function() {
                this.style.display = 'none';
                const container = this.closest('.visualization-content');
                if (container) {
                    container.innerHTML = '<p class="no-data">图表加载失败</p>';
                }
            });
        });
        """