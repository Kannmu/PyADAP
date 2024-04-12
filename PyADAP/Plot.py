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

import PyADAP.Data as data


def ResidualPlots(DataInstance:data.Data):
    """
    This function calculates and plots residuals of input data

    Parameters:
    -----------
    - DataInstance: data.Data
        The input Data instance containing the data for which residuals are to be calculated.
    
    """



def BoxPlots(DataInstance:data.Data, SavePath: str = "", Split: bool = False):
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

    for dependentvar in DataInstance.DependentVars:
        for index, independentvar in enumerate(DataInstance.IndependentVars):
            IndependentLevels = DataInstance.IndependentLevelsList[index]
            
            plt.figure(figsize=(len(IndependentLevels) * 2, 8))
            
            Tempdf = DataInstance.RawData
            Tempdf['index_by_group'] = Tempdf.groupby(independentvar).cumcount()

            # 使用pivot_table来重塑DataFrame
            result_df = Tempdf.pivot_table(index='index_by_group', columns=independentvar, values=dependentvar, aggfunc='first').reset_index(drop=True)

            sns.boxplot(data=result_df)

            # plt.xticks(rotation=45)
            plt.title('Box plot of '+ independentvar)
            plt.tight_layout()
            plt.savefig(SavePath + "\\" +independentvar+  "Box-Plots.png", dpi=200)


def QQPlots(DataInstance: data.Data, SavePath: str = ""):
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
    # numeric_cols = data.iloc[:,1:].select_dtypes(include=np.number).columns.tolist()
    numeric_cols = DataInstance.RawData.columns.tolist()

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
        stats.probplot(DataInstance.RawData[col].dropna(), dist="norm", plot=ax)
        ax.set_title(f"QQ-plot of {col}")

    # If the number of subplots is less than the number of numeric columns, hide the excess subplots
    for i in range(n, nrows * ncols):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
    plt.savefig(SavePath, dpi=200)
    # plt.show()