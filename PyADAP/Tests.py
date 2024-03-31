"""
PyADAP
=====

PyADAP: Python Automated Data Analysis Pipeline

Tests

Author: Kannmu
Date: 2024/3/31
License: MIT License
Repository: https://github.com/Kannmu/PyADAP

"""

# Import necessary modules
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import kurtosis, skew
from prettytable import PrettyTable
from colorama import Fore, Back, Style, init
import matplotlib.pyplot as plt
from scipy import stats
import PyADAP.Utilities as U

init()

def statistics(data: pd.DataFrame):
    """
    Parameters:
    ----------
    data : pd.DataFrame
        The input DataFrame containing the data.

    Returns:
    ----------
    statistics : pd.DataFrame
        A DataFrame containing the calculated statistics for each column.
        The DataFrame includes the following columns:
        - Variable: The name of the variable (column).
        - Mean: The mean of the variable.
        - Standard Deviation: The standard deviation of the variable.
        - Kurtosis: The kurtosis of the variable.
        - Skewness: The skewness of the variable.
    """
    statistics_columns = ["Variable", "Mean", "Standard Deviation", "Kurtosis", "Skewness"]
    statistics = pd.DataFrame(columns=statistics_columns)
    StatisticsResultTable = PrettyTable()
    for column in data.columns:
        if np.issubdtype(data[column].dtype, np.number):
            mean = data[column].mean()
            std = data[column].std()
            kurt = kurtosis(data[column])
            skewness = skew(data[column])

            temp_result = pd.DataFrame(
                {
                    "Variable": column,
                    "Mean": mean,
                    "Standard Deviation": std,
                    "Kurtosis": kurt,
                    "Skewness": skewness,
                },
                index=[0],
            )

            statistics = pd.concat([statistics, temp_result], ignore_index=True)

    # Print Result in Table Form
    StatisticsResultTable.clear()
    StatisticsResultTable.field_names = statistics_columns

    for i, row in statistics.iterrows():
        StatisticsResultTable.add_row(row.apply(RoundFloat).values)

    print(Fore.YELLOW + "\nStatistics Results")
    print(Fore.BLUE)
    print(StatisticsResultTable)
    print(Fore.RESET)

    return statistics


def normality_test(data: pd.DataFrame):
    """
    Parameters:
    ----------
    data : pd.DataFrame
        The input DataFrame containing the data.

    Returns:
    ----------
    normality : pd.DataFrame
        A DataFrame containing the results of the normality test for each column.
        The DataFrame includes the following columns:
        - Variable: The name of the variable (column) being tested.
        - Result: Whether the variable follows a normal distribution (True/False).
        - Method: Normality test method.
        - Statistic: The test statistic value.
        - p-value: The p-value associated with the test.

    """
    ResultsColumns = ["Variable", "Result", "Method", "Statistic", "p-value"]
    NormalResultTable = PrettyTable()

    normality = pd.DataFrame(columns=ResultsColumns)

    normality = normality.astype({"Result": "bool"})

    for column in data.columns:
        if not np.issubdtype(data[column].values.dtype, np.number):
            continue
        
        n = len(data[column].values)

        if n < 50:
            Method = 'shapiro'
        elif n < 300:
            Method = 'normaltest'
        else:
            Method = 'kstest'

        test_result = pg.normality(data[column], method = Method)

        TempResult = pd.DataFrame(
            {
                "Variable": column,
                "Result": test_result["normal"],
                "Method": Method,
                "Statistic": test_result["W"],
                "p-value": test_result["pval"],
            }
        )

        # Concat Result
        normality = pd.concat(
            [
                normality,
                TempResult,
            ],
            ignore_index=True,
        )

    # Print Result in Table Form
    NormalResultTable.clear()
    NormalResultTable.field_names = ResultsColumns

    for i, row in normality.iterrows():
        NormalResultTable.add_row(row.apply(RoundFloat).values)

    print(Fore.YELLOW + "\nNormality Test Results")
    print(Fore.RED)
    print(NormalResultTable)
    print(Fore.RESET)
    
    return normality

def QQPlots(data):
    """
    绘制DataFrame中所有数字列的QQ-plot。

    参数:
    - data: pd.DataFrame, 包含需要绘制QQ-plot的数据。

    返回:
    - None, 但会显示一个包含所有数字列QQ-plot的图。
    """
    
    # 确定数据中的数字列
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    
    # 计算需要绘制的图的行数和列数，这里我们尝试让所有子图都尽可能地排成正方形
    n = len(numeric_cols)
    nrows = int(np.ceil(np.sqrt(n)))
    ncols = nrows if n > nrows * (nrows - 1) else nrows - 1
    
    # 创建一个足够大的figure来容纳所有的子图
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    
    # 如果只有一个子图，axes不是数组，为了统一处理，我们将其转换为数组
    if n == 1:
        axes = np.array([axes])
    
    # 对于每一个数字列，绘制QQ-plot
    for col, ax in zip(numeric_cols, axes.flatten()):
        stats.probplot(data[col].dropna(), dist="norm", plot=ax)
        ax.set_title(f'QQ-plot of {col}')
    
    # 如果子图的数量不等于数字列的数量，隐藏多余的子图
    for i in range(n, nrows * ncols):
        fig.delaxes(axes.flatten()[i])
    
    plt.tight_layout()
    plt.show()

def RoundFloat(X):
    if isinstance(X, float):
        return round(X, 6)
    else:
        return X
