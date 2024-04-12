"""
PyADAP
=====

PyADAP: Python Automated Data Analysis Pipeline

Author: Kannmu
Date: 2024/3/31
License: MIT License
Repository: https://github.com/Kannmu/PyADAP

Framework:

1. 数据预处理
数据加载：使用Pandas加载数据。
数据清洗：处理缺失值、异常值、重复值等。
特征工程：根据需要进行特征选择、特征转换等。

2. 基本统计结果分析
描述性统计：计算均值、中位数、标准差、方差、最小/最大值、偏度和峰度等。
可视化分析：通过箱型图、直方图等进行数据分布的可视化。

3. 数据检验
正态性检验：使用Kolmogorov-Smirnov检验、Shapiro-Wilk检验等。
方差齐性检验：如Levene检验、Bartlett检验等。

4. 相关性分析
选择合适的相关性检验方法：
对于正态分布数据，使用Pearson相关系数。
对于非正态分布数据或等级数据，使用Spearman秩相关系数或Kendall的Tau相关系数。
计算相关性：计算变量之间的相关系数，并进行显著性检验。

5. 自动选择检验方法和相关性计算方法
根据数据检验的结果自动选择最适合的检验方法和相关性计算方法。

6. 报告生成
自动生成分析报告，包括统计图表、检验结果和相关性分析结果。

"""

import os

import pandas as pd

import PyADAP.Data as data
import PyADAP.File as file
import PyADAP.GUI as gui
import PyADAP.Plot as plt
import PyADAP.Statistic as statistic
import PyADAP.Utilities as utility


def Pipeline(
    Data: data.Data,
):
    """This function performs the entire process of analyzing the data and generates a report on the results.

    Args:
        Data (data.Data): The data to analyze.

    Returns:
        pd.DataFrame: The results of the analysis, in a Pandas DataFrame format.
    """
    StatisticsResults = statistic.statistics(Data)

    plt.BoxPlots(
        Data,
        os.path.dirname(Data.DataPath) + "\\Results",
        Split=True,
    )

    NormalTestResults = statistic.normality_test(Data)

    plt.QQPlots(
        Data,
        os.path.dirname(Data.DataPath) + "\\Results" + "\\QQ-Plots.png",
    )

    SphericityTestResults = statistic.sphericity_test(Data)

    ResultsFilePath = (
        os.path.dirname(Data.DataPath)
        + "\\Results"
        + "\\"
        + Data.DataFileName
        + "Results.xlsx"
    )

    file.SaveDataToExcel(
        ResultsFilePath, StatisticsResults, NormalTestResults, SphericityTestResults
    )
