#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apple-Style HTML Template Generator for PyADAP Reports

This module contains the HTML template generation methods for creating
modern, Apple-inspired analysis reports with interactive features.
"""

from typing import Dict, List, Any
from datetime import datetime
from .apple_html_sections import AppleHTMLSections


class AppleHTMLTemplate:
    """Apple-style HTML template generator."""
    
    def __init__(self):
        """Initialize the template generator."""
        self.apple_colors = {
            'blue': '#007AFF',
            'green': '#34C759', 
            'orange': '#FF9500',
            'red': '#FF3B30',
            'purple': '#AF52DE',
            'pink': '#FF2D92',
            'teal': '#5AC8FA',
            'indigo': '#5856D6',
            'gray': '#8E8E93',
            'gray_light': '#F2F2F7',
            'gray_dark': '#1C1C1E',
            'white': '#FFFFFF',
            'black': '#000000'
        }
    
    def create_apple_html_content(self, results: Dict[str, Any], 
                                data_info: Dict[str, Any], 
                                visualizations: Dict[str, str]) -> str:
        """Create the complete Apple-style HTML content.
        
        Args:
            results: Analysis results dictionary
            data_info: Information about the analyzed data
            visualizations: Dictionary of base64 encoded visualizations
            
        Returns:
            Complete HTML content string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pipeline_id = data_info.get('pipeline_id', 'PyADAP Analysis')
        
        # Extract basic information
        data_shape = data_info.get('data_shape', (0, 0))
        rows, cols = data_shape if data_shape else (0, 0)
        
        variables = data_info.get('variables', {})
        dependent_vars = variables.get('dependent', [])
        independent_vars = variables.get('independent', [])
        subject_var = variables.get('subject', None)
        
        # Calculate analysis duration
        start_time = data_info.get('start_time')
        end_time = data_info.get('end_time')
        duration = "N/A"
        if start_time and end_time:
            duration_seconds = (end_time - start_time).total_seconds()
            if duration_seconds < 60:
                duration = f"{duration_seconds:.1f} 秒"
            elif duration_seconds < 3600:
                duration = f"{duration_seconds/60:.1f} 分钟"
            else:
                duration = f"{duration_seconds/3600:.1f} 小时"
        
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PyADAP 数据分析报告 - {pipeline_id}</title>
    
    <!-- Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
    
    <!-- Animation Library -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/animejs/3.2.1/anime.min.js"></script>
    
    <!-- Prism.js for code highlighting -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-core.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/autoloader/prism-autoloader.min.js"></script>
    
    <style>
        {self._get_apple_css_styles()}
    </style>
</head>
<body>
    <!-- Navigation Bar -->
    <nav class="navbar">
        <div class="nav-container">
            <div class="nav-brand">
                <i class="fas fa-chart-line nav-icon"></i>
                <span class="nav-title">PyADAP</span>
            </div>
            <div class="nav-actions">
                <button class="nav-btn" onclick="exportReport()">
                    <i class="fas fa-download"></i>
                    <span>导出</span>
                </button>
                <button class="nav-btn" onclick="toggleTheme()">
                    <i class="fas fa-moon"></i>
                    <span>主题</span>
                </button>
            </div>
        </div>
    </nav>
    
    <!-- Main Container -->
    <div class="main-container">
        <!-- Hero Section -->
        <section class="hero-section">
            <div class="hero-content">
                <div class="hero-badge">
                    <i class="fas fa-sparkles"></i>
                    <span>AI驱动的数据分析</span>
                </div>
                <h1 class="hero-title">{pipeline_id}</h1>
                <p class="hero-subtitle">全面的统计分析报告与数据洞察</p>
                <div class="hero-meta">
                    <div class="meta-item">
                        <i class="fas fa-calendar-alt"></i>
                        <span>{timestamp}</span>
                    </div>
                    <div class="meta-item">
                        <i class="fas fa-database"></i>
                        <span>{rows:,} 行 × {cols} 列</span>
                    </div>
                    <div class="meta-item">
                        <i class="fas fa-clock"></i>
                        <span>{duration}</span>
                    </div>
                </div>
            </div>
            <div class="hero-visual">
                <div class="floating-card">
                    <div class="card-header">
                        <div class="card-dots">
                            <span class="dot red"></span>
                            <span class="dot yellow"></span>
                            <span class="dot green"></span>
                        </div>
                    </div>
                    <div class="card-content">
                        <div class="chart-preview">
                            <canvas id="heroChart" width="300" height="200"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        
        <!-- Quick Stats Dashboard -->
        <section class="stats-dashboard">
            <div class="dashboard-grid">
                {self._create_stats_cards(results, data_info)}
            </div>
        </section>
        
        <!-- Navigation Tabs -->
        <section class="content-navigation">
            <div class="tab-container">
                <div class="tab-list" role="tablist">
                    <button class="tab-button active" data-tab="overview" role="tab">
                        <i class="fas fa-chart-pie"></i>
                        <span>数据概览</span>
                    </button>
                    <button class="tab-button" data-tab="quality" role="tab">
                        <i class="fas fa-shield-alt"></i>
                        <span>数据质量</span>
                    </button>
                    <button class="tab-button" data-tab="analysis" role="tab">
                        <i class="fas fa-microscope"></i>
                        <span>统计分析</span>
                    </button>
                    <button class="tab-button" data-tab="visualizations" role="tab">
                        <i class="fas fa-chart-bar"></i>
                        <span>可视化</span>
                    </button>
                    <button class="tab-button" data-tab="insights" role="tab">
                        <i class="fas fa-lightbulb"></i>
                        <span>洞察建议</span>
                    </button>
                </div>
            </div>
        </section>
        
        <!-- Content Sections -->
        <main class="content-main">
            <!-- Overview Tab -->
            <section id="overview" class="tab-content">
                <div class="section-header">
                    <h2 class="section-title">
                        <i class="fas fa-chart-pie"></i>
                        数据概览
                    </h2>
                    <p class="section-subtitle">数据集的基本信息和变量配置</p>
                </div>
                
                <div class="content-grid">
                    <div class="info-card">
                        <div class="card-header">
                            <h3 class="card-title">
                                <i class="fas fa-info-circle"></i>
                                基本信息
                            </h3>
                        </div>
                        <div class="card-content">
                            <div class="info-grid">
                                <div class="info-item">
                                    <span class="info-label">数据行数</span>
                                    <span class="info-value">{rows:,}</span>
                                </div>
                                <div class="info-item">
                                    <span class="info-label">数据列数</span>
                                    <span class="info-value">{cols}</span>
                                </div>
                                <div class="info-item">
                                    <span class="info-label">分析耗时</span>
                                    <span class="info-value">{duration}</span>
                                </div>
                                <div class="info-item">
                                    <span class="info-label">分析状态</span>
                                    <span class="status-badge success">已完成</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="info-card">
                        <div class="card-header">
                            <h3 class="card-title">
                                <i class="fas fa-tags"></i>
                                变量配置
                            </h3>
                        </div>
                        <div class="card-content">
                            <div class="variables-section">
                                <div class="variable-group">
                                    <h4 class="variable-title">因变量</h4>
                                    <div class="variable-tags">
                                        {self._format_variable_tags(dependent_vars, 'dependent')}
                                    </div>
                                </div>
                                <div class="variable-group">
                                    <h4 class="variable-title">自变量</h4>
                                    <div class="variable-tags">
                                        {self._format_variable_tags(independent_vars, 'independent')}
                                    </div>
                                </div>
                                {f'<div class="variable-group"><h4 class="variable-title">主体变量</h4><div class="variable-tags"><span class="variable-tag subject">{subject_var}</span></div></div>' if subject_var else ''}
                            </div>
                        </div>
                    </div>
                </div>
                
                {AppleHTMLSections().create_overview_section(results, data_info, visualizations)}
            </section>
            
            <!-- Quality Tab -->
            <section id="quality" class="tab-content">
                <div class="section-header">
                    <h2 class="section-title">
                        <i class="fas fa-shield-alt"></i>
                        数据质量评估
                    </h2>
                    <p class="section-subtitle">数据完整性、一致性和准确性分析</p>
                </div>
                
                {AppleHTMLSections().create_quality_section(results, visualizations)}
            </section>
            
            <!-- Analysis Tab -->
            <section id="analysis" class="tab-content">
                <div class="section-header">
                    <h2 class="section-title">
                        <i class="fas fa-microscope"></i>
                        统计分析结果
                    </h2>
                    <p class="section-subtitle">详细的统计检验和假设验证结果</p>
                </div>
                
                {AppleHTMLSections().create_analysis_section(results, visualizations)}
            </section>
            
            <!-- Visualizations Tab -->
            <section id="visualizations" class="tab-content">
                <div class="section-header">
                    <h2 class="section-title">
                        <i class="fas fa-chart-bar"></i>
                        数据可视化
                    </h2>
                    <p class="section-subtitle">交互式图表和数据探索</p>
                </div>
                
                {AppleHTMLSections().create_visualizations_section(visualizations)}
            </section>
            
            <!-- Insights Tab -->
            <section id="insights" class="tab-content">
                <div class="section-header">
                    <h2 class="section-title">
                        <i class="fas fa-lightbulb"></i>
                        洞察与建议
                    </h2>
                    <p class="section-subtitle">基于分析结果的智能建议和行动指南</p>
                </div>
                
                {AppleHTMLSections().create_insights_section(results)}
            </section>
        </main>
    </div>
    
    <!-- Footer -->
    <footer class="footer">
        <div class="footer-content">
            <div class="footer-info">
                <div class="footer-brand">
                    <i class="fas fa-chart-line"></i>
                    <span>PyADAP</span>
                </div>
                <p class="footer-description">Python自动化数据分析管道 - 让数据分析更简单、更智能</p>
            </div>
            <div class="footer-meta">
                <p>© 2024 PyADAP Project. 由AI驱动的数据分析平台。</p>
                <p>生成时间: {timestamp}</p>
            </div>
        </div>
    </footer>
    
    <!-- Scripts -->
    <script>
        {self._get_javascript_code(results, visualizations)}
    </script>
</body>
</html>
        """
        
        return html_content
    
    def _get_apple_css_styles(self) -> str:
        """Get the complete Apple-style CSS."""
        return f"""
        /* Reset and Base Styles */
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        :root {{
            /* Apple Color Palette */
            --apple-blue: {self.apple_colors['blue']};
            --apple-green: {self.apple_colors['green']};
            --apple-orange: {self.apple_colors['orange']};
            --apple-red: {self.apple_colors['red']};
            --apple-purple: {self.apple_colors['purple']};
            --apple-pink: {self.apple_colors['pink']};
            --apple-teal: {self.apple_colors['teal']};
            --apple-indigo: {self.apple_colors['indigo']};
            --apple-gray: {self.apple_colors['gray']};
            --apple-gray-light: {self.apple_colors['gray_light']};
            --apple-gray-dark: {self.apple_colors['gray_dark']};
            --apple-white: {self.apple_colors['white']};
            --apple-black: {self.apple_colors['black']};
            
            /* Semantic Colors */
            --text-primary: #1D1D1F;
            --text-secondary: #86868B;
            --text-tertiary: #A1A1A6;
            --background-primary: #FBFBFD;
            --background-secondary: #F5F5F7;
            --background-tertiary: #FFFFFF;
            --border-color: #D2D2D7;
            --border-color-light: #E5E5E7;
            
            /* Shadows */
            --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.04);
            --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.06);
            --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.08);
            --shadow-xl: 0 16px 64px rgba(0, 0, 0, 0.12);
            
            /* Border Radius */
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
            --radius-xl: 24px;
            
            /* Spacing */
            --space-xs: 4px;
            --space-sm: 8px;
            --space-md: 16px;
            --space-lg: 24px;
            --space-xl: 32px;
            --space-2xl: 48px;
            --space-3xl: 64px;
            
            /* Typography */
            --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            --font-size-xs: 0.75rem;
            --font-size-sm: 0.875rem;
            --font-size-base: 1rem;
            --font-size-lg: 1.125rem;
            --font-size-xl: 1.25rem;
            --font-size-2xl: 1.5rem;
            --font-size-3xl: 1.875rem;
            --font-size-4xl: 2.25rem;
            --font-size-5xl: 3rem;
            
            /* Transitions */
            --transition-fast: 0.15s cubic-bezier(0.4, 0, 0.2, 1);
            --transition-normal: 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            --transition-slow: 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        /* Dark Theme */
        [data-theme="dark"] {{
            --text-primary: #F5F5F7;
            --text-secondary: #A1A1A6;
            --text-tertiary: #86868B;
            --background-primary: #000000;
            --background-secondary: #1C1C1E;
            --background-tertiary: #2C2C2E;
            --border-color: #38383A;
            --border-color-light: #48484A;
        }}
        
        /* Base Typography */
        html {{
            font-size: 16px;
            scroll-behavior: smooth;
        }}
        
        body {{
            font-family: var(--font-family);
            line-height: 1.6;
            color: var(--text-primary);
            background: var(--background-primary);
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            overflow-x: hidden;
        }}
        
        /* Navigation */
        .navbar {{
            position: sticky;
            top: 0;
            z-index: 1000;
            background: rgba(251, 251, 253, 0.8);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--border-color-light);
            padding: var(--space-md) 0;
        }}
        
        .nav-container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 var(--space-lg);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .nav-brand {{
            display: flex;
            align-items: center;
            gap: var(--space-sm);
        }}
        
        .nav-icon {{
            font-size: var(--font-size-xl);
            color: var(--apple-blue);
        }}
        
        .nav-title {{
            font-size: var(--font-size-xl);
            font-weight: 700;
            color: var(--text-primary);
        }}
        
        .nav-actions {{
            display: flex;
            gap: var(--space-sm);
        }}
        
        .nav-btn {{
            display: flex;
            align-items: center;
            gap: var(--space-xs);
            padding: var(--space-sm) var(--space-md);
            background: var(--background-tertiary);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            color: var(--text-primary);
            font-size: var(--font-size-sm);
            font-weight: 500;
            cursor: pointer;
            transition: all var(--transition-fast);
        }}
        
        .nav-btn:hover {{
            background: var(--background-secondary);
            transform: translateY(-1px);
            box-shadow: var(--shadow-md);
        }}
        
        /* Main Container */
        .main-container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 var(--space-lg);
        }}
        
        /* Hero Section */
        .hero-section {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: var(--space-3xl);
            align-items: center;
            padding: var(--space-3xl) 0;
            min-height: 60vh;
        }}
        
        .hero-content {{
            animation: slideInLeft 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        }}
        
        .hero-badge {{
            display: inline-flex;
            align-items: center;
            gap: var(--space-xs);
            padding: var(--space-xs) var(--space-md);
            background: linear-gradient(135deg, var(--apple-blue), var(--apple-purple));
            color: white;
            border-radius: var(--radius-xl);
            font-size: var(--font-size-sm);
            font-weight: 600;
            margin-bottom: var(--space-lg);
        }}
        
        .hero-title {{
            font-size: var(--font-size-5xl);
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: var(--space-md);
            background: linear-gradient(135deg, var(--text-primary), var(--apple-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .hero-subtitle {{
            font-size: var(--font-size-xl);
            color: var(--text-secondary);
            margin-bottom: var(--space-xl);
            line-height: 1.5;
        }}
        
        .hero-meta {{
            display: flex;
            flex-wrap: wrap;
            gap: var(--space-lg);
        }}
        
        .meta-item {{
            display: flex;
            align-items: center;
            gap: var(--space-sm);
            color: var(--text-secondary);
            font-size: var(--font-size-sm);
            font-weight: 500;
        }}
        
        .meta-item i {{
            color: var(--apple-blue);
        }}
        
        /* Hero Visual */
        .hero-visual {{
            display: flex;
            justify-content: center;
            align-items: center;
            animation: slideInRight 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        }}
        
        .floating-card {{
            background: var(--background-tertiary);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-xl);
            overflow: hidden;
            transform: perspective(1000px) rotateY(-5deg) rotateX(5deg);
            transition: transform var(--transition-slow);
        }}
        
        .floating-card:hover {{
            transform: perspective(1000px) rotateY(0deg) rotateX(0deg);
        }}
        
        .card-header {{
            padding: var(--space-md);
            border-bottom: 1px solid var(--border-color-light);
        }}
        
        .card-dots {{
            display: flex;
            gap: var(--space-xs);
        }}
        
        .dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}
        
        .dot.red {{ background: var(--apple-red); }}
        .dot.yellow {{ background: var(--apple-orange); }}
        .dot.green {{ background: var(--apple-green); }}
        
        .card-content {{
            padding: var(--space-lg);
        }}
        
        /* Stats Dashboard */
        .stats-dashboard {{
            margin: var(--space-3xl) 0;
        }}
        
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: var(--space-lg);
        }}
        
        .stat-card {{
            background: var(--background-tertiary);
            border-radius: var(--radius-lg);
            padding: var(--space-xl);
            box-shadow: var(--shadow-md);
            border: 1px solid var(--border-color-light);
            transition: all var(--transition-normal);
            position: relative;
            overflow: hidden;
        }}
        
        .stat-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--apple-blue), var(--apple-purple));
        }}
        
        .stat-card:hover {{
            transform: translateY(-4px);
            box-shadow: var(--shadow-lg);
        }}
        
        .stat-header {{
            display: flex;
            align-items: center;
            gap: var(--space-md);
            margin-bottom: var(--space-lg);
        }}
        
        .stat-icon {{
            width: 48px;
            height: 48px;
            border-radius: var(--radius-md);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: var(--font-size-xl);
            color: white;
        }}
        
        .stat-icon.blue {{ background: var(--apple-blue); }}
        .stat-icon.green {{ background: var(--apple-green); }}
        .stat-icon.orange {{ background: var(--apple-orange); }}
        .stat-icon.red {{ background: var(--apple-red); }}
        .stat-icon.purple {{ background: var(--apple-purple); }}
        
        .stat-title {{
            font-size: var(--font-size-lg);
            font-weight: 600;
            color: var(--text-primary);
        }}
        
        .stat-value {{
            font-size: var(--font-size-3xl);
            font-weight: 800;
            color: var(--text-primary);
            margin-bottom: var(--space-sm);
        }}
        
        .stat-description {{
            color: var(--text-secondary);
            font-size: var(--font-size-sm);
        }}
        
        /* Content Navigation */
        .content-navigation {{
            margin: var(--space-3xl) 0 var(--space-xl) 0;
        }}
        
        .tab-container {{
            background: var(--background-tertiary);
            border-radius: var(--radius-lg);
            padding: var(--space-sm);
            box-shadow: var(--shadow-md);
            border: 1px solid var(--border-color-light);
        }}
        
        .tab-list {{
            display: flex;
            gap: var(--space-xs);
            overflow-x: auto;
        }}
        
        .tab-button {{
            display: flex;
            align-items: center;
            gap: var(--space-sm);
            padding: var(--space-md) var(--space-lg);
            background: transparent;
            border: none;
            border-radius: var(--radius-md);
            color: var(--text-secondary);
            font-size: var(--font-size-sm);
            font-weight: 500;
            cursor: pointer;
            transition: all var(--transition-fast);
            white-space: nowrap;
        }}
        
        .tab-button:hover {{
            background: var(--background-secondary);
            color: var(--text-primary);
        }}
        
        .tab-button.active {{
            background: var(--apple-blue);
            color: white;
            box-shadow: var(--shadow-sm);
        }}
        
        /* Content Sections */
        .content-main {{
            margin-bottom: var(--space-3xl);
        }}
        
        .tab-content {{
            display: none;
            animation: fadeInUp 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        .section-header {{
            text-align: center;
            margin-bottom: var(--space-3xl);
        }}
        
        .section-title {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: var(--space-md);
            font-size: var(--font-size-3xl);
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: var(--space-md);
        }}
        
        .section-subtitle {{
            font-size: var(--font-size-lg);
            color: var(--text-secondary);
            max-width: 600px;
            margin: 0 auto;
        }}
        
        /* Content Grid */
        .content-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: var(--space-lg);
            margin-bottom: var(--space-xl);
        }}
        
        .info-card {{
            background: var(--background-tertiary);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-md);
            border: 1px solid var(--border-color-light);
            overflow: hidden;
            transition: all var(--transition-normal);
        }}
        
        .info-card:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }}
        
        .card-header {{
            padding: var(--space-lg);
            border-bottom: 1px solid var(--border-color-light);
            background: var(--background-secondary);
        }}
        
        .card-title {{
            display: flex;
            align-items: center;
            gap: var(--space-sm);
            font-size: var(--font-size-lg);
            font-weight: 600;
            color: var(--text-primary);
        }}
        
        .card-content {{
            padding: var(--space-lg);
        }}
        
        /* Info Grid */
        .info-grid {{
            display: grid;
            gap: var(--space-md);
        }}
        
        .info-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: var(--space-md) 0;
            border-bottom: 1px solid var(--border-color-light);
        }}
        
        .info-item:last-child {{
            border-bottom: none;
        }}
        
        .info-label {{
            color: var(--text-secondary);
            font-weight: 500;
        }}
        
        .info-value {{
            color: var(--text-primary);
            font-weight: 600;
        }}
        
        /* Status Badges */
        .status-badge {{
            padding: var(--space-xs) var(--space-md);
            border-radius: var(--radius-xl);
            font-size: var(--font-size-xs);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .status-badge.success {{
            background: rgba(52, 199, 89, 0.1);
            color: var(--apple-green);
        }}
        
        .status-badge.warning {{
            background: rgba(255, 149, 0, 0.1);
            color: var(--apple-orange);
        }}
        
        .status-badge.error {{
            background: rgba(255, 59, 48, 0.1);
            color: var(--apple-red);
        }}
        
        /* Variables */
        .variables-section {{
            display: flex;
            flex-direction: column;
            gap: var(--space-lg);
        }}
        
        .variable-group {{
            display: flex;
            flex-direction: column;
            gap: var(--space-sm);
        }}
        
        .variable-title {{
            font-size: var(--font-size-sm);
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .variable-tags {{
            display: flex;
            flex-wrap: wrap;
            gap: var(--space-sm);
        }}
        
        .variable-tag {{
            padding: var(--space-xs) var(--space-md);
            border-radius: var(--radius-xl);
            font-size: var(--font-size-sm);
            font-weight: 500;
            color: white;
        }}
        
        .variable-tag.dependent {{ background: var(--apple-red); }}
        .variable-tag.independent {{ background: var(--apple-green); }}
        .variable-tag.subject {{ background: var(--apple-orange); }}
        
        /* Visualization Containers */
        .visualization-container {{
            background: var(--background-tertiary);
            border-radius: var(--radius-lg);
            padding: var(--space-xl);
            margin: var(--space-lg) 0;
            box-shadow: var(--shadow-md);
            border: 1px solid var(--border-color-light);
        }}
        
        .visualization-title {{
            font-size: var(--font-size-xl);
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: var(--space-lg);
            text-align: center;
        }}
        
        .visualization-image {{
            width: 100%;
            height: auto;
            border-radius: var(--radius-md);
            box-shadow: var(--shadow-sm);
        }}
        
        /* Tables */
        .results-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            background: var(--background-tertiary);
            border-radius: var(--radius-lg);
            overflow: hidden;
            box-shadow: var(--shadow-md);
            border: 1px solid var(--border-color-light);
            margin: var(--space-lg) 0;
        }}
        
        .results-table th {{
            background: var(--apple-blue);
            color: white;
            padding: var(--space-md) var(--space-lg);
            text-align: left;
            font-weight: 600;
            font-size: var(--font-size-sm);
        }}
        
        .results-table td {{
            padding: var(--space-md) var(--space-lg);
            border-bottom: 1px solid var(--border-color-light);
            font-size: var(--font-size-sm);
        }}
        
        .results-table tr:hover {{
            background: var(--background-secondary);
        }}
        
        .results-table tr:last-child td {{
            border-bottom: none;
        }}
        
        /* Footer */
        .footer {{
            background: var(--background-secondary);
            border-top: 1px solid var(--border-color-light);
            padding: var(--space-3xl) 0;
            margin-top: var(--space-3xl);
        }}
        
        .footer-content {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 var(--space-lg);
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: var(--space-xl);
            align-items: center;
        }}
        
        .footer-brand {{
            display: flex;
            align-items: center;
            gap: var(--space-sm);
            font-size: var(--font-size-xl);
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: var(--space-md);
        }}
        
        .footer-brand i {{
            color: var(--apple-blue);
        }}
        
        .footer-description {{
            color: var(--text-secondary);
            line-height: 1.6;
        }}
        
        .footer-meta {{
            text-align: right;
            color: var(--text-secondary);
            font-size: var(--font-size-sm);
        }}
        
        /* Animations */
        @keyframes slideInLeft {{
            from {{
                opacity: 0;
                transform: translateX(-50px);
            }}
            to {{
                opacity: 1;
                transform: translateX(0);
            }}
        }}
        
        @keyframes slideInRight {{
            from {{
                opacity: 0;
                transform: translateX(50px);
            }}
            to {{
                opacity: 1;
                transform: translateX(0);
            }}
        }}
        
        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(30px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        /* Responsive Design */
        @media (max-width: 768px) {{
            .hero-section {{
                grid-template-columns: 1fr;
                text-align: center;
                gap: var(--space-xl);
            }}
            
            .hero-title {{
                font-size: var(--font-size-4xl);
            }}
            
            .dashboard-grid {{
                grid-template-columns: 1fr;
            }}
            
            .content-grid {{
                grid-template-columns: 1fr;
            }}
            
            .footer-content {{
                grid-template-columns: 1fr;
                text-align: center;
            }}
            
            .footer-meta {{
                text-align: center;
            }}
            
            .tab-list {{
                flex-direction: column;
            }}
            
            .nav-container {{
                flex-direction: column;
                gap: var(--space-md);
            }}
        }}
        
        /* Print Styles */
        @media print {{
            .navbar,
            .nav-actions,
            .tab-container {{
                display: none;
            }}
            
            .tab-content {{
                display: block !important;
            }}
            
            .main-container {{
                max-width: none;
                padding: 0;
            }}
        }}
        """
    
    def _create_stats_cards(self, results: Dict[str, Any], data_info: Dict[str, Any]) -> str:
        """Create the statistics dashboard cards."""
        # Extract basic statistics
        data_shape = data_info.get('data_shape', (0, 0))
        rows, cols = data_shape if data_shape else (0, 0)
        
        # Count statistical tests
        stats_tests = results.get('statistical_tests', {})
        test_count = len(stats_tests)
        
        # Count significant results
        significant_count = 0
        for test_result in stats_tests.values():
            if isinstance(test_result, dict):
                p_val = test_result.get('p_value', test_result.get('pvalue', 1.0))
                if hasattr(p_val, 'item'):
                    p_val = p_val.item()
                if float(p_val) < 0.05:
                    significant_count += 1
        
        # Data quality score
        quality_report = results.get('quality_report', {})
        missing_values = quality_report.get('missing_values', {})
        total_missing = sum(missing_values.values()) if isinstance(missing_values, dict) else 0
        duplicates = quality_report.get('duplicates', 0)
        
        # Calculate quality score
        if rows > 0 and cols > 0:
            completeness = max(0, (1 - total_missing / (rows * cols)) * 100)
            uniqueness = max(0, (1 - duplicates / rows) * 100)
            quality_score = (completeness + uniqueness) / 2
        else:
            quality_score = 100
        
        # Count visualizations
        visualizations = results.get('visualizations', {})
        viz_count = len(visualizations)
        
        return f"""
        <div class="stat-card">
            <div class="stat-header">
                <div class="stat-icon blue">
                    <i class="fas fa-database"></i>
                </div>
                <div class="stat-title">数据规模</div>
            </div>
            <div class="stat-value">{rows:,}</div>
            <div class="stat-description">数据行数，包含 {cols} 个变量</div>
        </div>
        
        <div class="stat-card">
            <div class="stat-header">
                <div class="stat-icon green">
                    <i class="fas fa-flask"></i>
                </div>
                <div class="stat-title">统计检验</div>
            </div>
            <div class="stat-value">{test_count}</div>
            <div class="stat-description">已完成的统计分析项目</div>
        </div>
        
        <div class="stat-card">
            <div class="stat-header">
                <div class="stat-icon orange">
                    <i class="fas fa-exclamation-triangle"></i>
                </div>
                <div class="stat-title">显著结果</div>
            </div>
            <div class="stat-value">{significant_count}</div>
            <div class="stat-description">具有统计学意义的结果 (p < 0.05)</div>
        </div>
        
        <div class="stat-card">
            <div class="stat-header">
                <div class="stat-icon purple">
                    <i class="fas fa-shield-alt"></i>
                </div>
                <div class="stat-title">数据质量</div>
            </div>
            <div class="stat-value">{quality_score:.1f}%</div>
            <div class="stat-description">综合数据质量评分</div>
        </div>
        
        <div class="stat-card">
            <div class="stat-header">
                <div class="stat-icon red">
                    <i class="fas fa-chart-bar"></i>
                </div>
                <div class="stat-title">可视化</div>
            </div>
            <div class="stat-value">{viz_count + 4}</div>
            <div class="stat-description">生成的图表和可视化</div>
        </div>
        """
    
    def _format_variable_tags(self, variables: List[str], var_type: str) -> str:
        """Format variables as HTML tags."""
        if not variables:
            return '<span class="variable-tag">无</span>'
        
        tags = []
        for var in variables:
            tags.append(f'<span class="variable-tag {var_type}">{var}</span>')
        return '\n'.join(tags)
    
    def _get_javascript_code(self, results: Dict[str, Any], visualizations: Dict[str, str]) -> str:
        """Get JavaScript code for interactive features."""
        return """
        // Export report functionality
        function exportReport() {
            window.print();
        }
        
        // Theme toggle functionality
        function toggleTheme() {
            document.body.classList.toggle('dark-theme');
            const isDark = document.body.classList.contains('dark-theme');
            localStorage.setItem('theme', isDark ? 'dark' : 'light');
            
            // Update icon
            const icon = event.target.closest('button').querySelector('i');
            if (icon) {
                icon.className = isDark ? 'fas fa-sun' : 'fas fa-moon';
            }
        }
        
        // Initialize page
        document.addEventListener('DOMContentLoaded', function() {
            // Tab switching functionality
            const tabButtons = document.querySelectorAll('.tab-button');
            const tabContents = document.querySelectorAll('.tab-content');
            
            tabButtons.forEach(button => {
                button.addEventListener('click', function(e) {
                    e.preventDefault();
                    const targetTab = this.getAttribute('data-tab');
                    
                    // Remove active class from all buttons and contents
                    tabButtons.forEach(btn => btn.classList.remove('active'));
                    tabContents.forEach(content => content.classList.remove('active'));
                    
                    // Add active class to clicked button and corresponding content
                    this.classList.add('active');
                    const targetContent = document.getElementById(targetTab);
                    if (targetContent) {
                        targetContent.classList.add('active');
                    }
                });
            });
            
            // Set first tab as active by default
            if (tabButtons.length > 0) {
                tabButtons[0].classList.add('active');
                const firstTabId = tabButtons[0].getAttribute('data-tab');
                const firstTabContent = document.getElementById(firstTabId);
                if (firstTabContent) {
                    firstTabContent.classList.add('active');
                }
            }
            
            // Add smooth scrolling
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
            
            // Theme toggle functionality - use class selector instead of specific class
            const themeButtons = document.querySelectorAll('[onclick="toggleTheme()"]');
            themeButtons.forEach(button => {
                button.addEventListener('click', function(e) {
                    e.preventDefault();
                    toggleTheme();
                });
            });
            
            // Export functionality - use class selector instead of specific class
            const exportButtons = document.querySelectorAll('[onclick="exportReport()"]');
            exportButtons.forEach(button => {
                button.addEventListener('click', function(e) {
                    e.preventDefault();
                    exportReport();
                });
            });
            
            // Load saved theme
            const savedTheme = localStorage.getItem('theme');
            if (savedTheme === 'dark') {
                document.body.classList.add('dark-theme');
            }
        });
        """