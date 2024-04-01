"""
PyADAP
=====

PyADAP: Python Automated Data Analysis Pipeline

Plot

Author: Kannmu
Date: 2024/3/31
License: MIT License
Repository: https://github.com/Kannmu/PyADAP

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def BoxPlots(data: pd.DataFrame, SavePath: str = "", Split: bool = False):
    # Untested!!!!!!!!!!!!
    """
    Plots box plots for all the numerical columns in the provided DataFrame.
    Can plot all variables on one chart, or split into individual charts.

    Parameters:
    ----------
    - data: pd.DataFrame
        The data containing the columns for which box plots need to be plotted.
    - SavePath: string
        The path where to save the box plots. If Split is True, plots will be saved
        with variable names as filenames.
    - Split: bool
        If True, plot each variable on a separate chart and save them individually.

    Returns:
    ----------
    - None, but saves a plot or plots containing box plots for all the numerical columns.
    """

    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

    if Split:
        for col in numeric_cols:
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=data[col])
            plt.title(f'Box plot of {col}')
            plt.savefig(f"{SavePath}/{col}_boxplot.png", dpi=200)
            plt.close()  # Close the figure to avoid displaying it in the notebook
    else:
        plt.figure(figsize=(len(numeric_cols) * 5, 8))
        sns.boxplot(data=data[numeric_cols])
        plt.xticks(rotation=45)
        plt.title('Box plot of all variables')
        plt.tight_layout()
        plt.savefig(SavePath, dpi=200)

def QQPlots(data: pd.DataFrame, SavePath: str = ""):
    """
    Plots QQ-plots for all the numerical columns in the provided DataFrame.

    Parameters:
    ----------
    - data: pd.DataFrame
        The data containing the columns for which QQ-plots need to be plotted.
    - SavePath: string
        The path where to save the QQ plots.

    Returns:
    ----------
    - None, but saves a plot containing QQ-plots for all the numerical columns.
    """

    # Determine the numeric columns in the data
    numeric_cols = data.iloc[:,1:].select_dtypes(include=np.number).columns.tolist()

    # Calculate the number of rows and columns for the plots, trying to make them as square as possible
    n = len(numeric_cols)
    nrows = int(np.ceil(np.sqrt(n)))
    ncols = nrows if n > nrows * (nrows - 1) else nrows - 1

    # Create a figure with enough space for all the subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))

    # If there is only one subplot, axes is not an array, so we convert it to an array for consistent handling
    if n == 1:
        axes = np.array([axes])

    # For each numeric column, plot the QQ-plot
    for col, ax in zip(numeric_cols, axes.flatten()):
        stats.probplot(data[col].dropna(), dist="norm", plot=ax)
        ax.set_title(f"QQ-plot of {col}")

    # If the number of subplots is less than the number of numeric columns, hide the excess subplots
    for i in range(n, nrows * ncols):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
    plt.savefig(SavePath, dpi=200)
    # plt.show()