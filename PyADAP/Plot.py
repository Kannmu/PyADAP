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
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats

import PyADAP.Data as data

plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 19
plt.rcParams["axes.titlesize"] = 24
plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


def ResidualPlots(DataInstance: data.Data):
    """
    This function calculates and plots residuals vs fitted values for each dependent variable using OLS model.

    Parameters:
    -----------
    - DataInstance: data.Data
        The input Data instance containing the data for which residuals are to be calculated.

    """
    # Residual vs Fitted plot for each dependent variable
    for dependent_var in DataInstance.DependentVars:
        # Create OLS model using statsmodels
        formula = f"{dependent_var} ~ {' + '.join(DataInstance.IndependentVars)}"
        model = smf.ols(formula, data=DataInstance.RawData).fit()
        # Get fitted values and residuals
        fitted_values = model.fittedvalues
        residuals = model.resid

        # Plotting residuals vs fitted values with different colors for each level of independent variables
        plt.figure(figsize=(8, 6))
        for independent_var in DataInstance.IndependentVars:
            levels = DataInstance.RawData[independent_var].unique()
            for level in levels:
                level_residuals = residuals[
                    DataInstance.RawData[independent_var] == level
                ]
                plt.scatter(
                    fitted_values[DataInstance.RawData[independent_var] == level],
                    level_residuals,
                    label=f"{independent_var} {level}",
                )
            plt.axhline(y=0, color="r", linestyle="--")
            plt.xlabel("Fitted Values")
            plt.ylabel("Residuals")
            plt.title(f"Residual vs. Fitted for {dependent_var}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                DataInstance.ImageFolderPath
                + f"\\ResidualVsFitted-{independent_var}--{dependent_var}.png",
                dpi=200,
            )
            plt.clf()
        plt.close()


def BoxPlots(DataInstance: data.Data):
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
            Tempdf["index_by_group"] = Tempdf.groupby([independentvar]).cumcount()

            # 使用pivot_table来重塑DataFrame
            result_df = Tempdf.pivot_table(
                index="index_by_group",
                columns=independentvar,
                values=dependentvar,
                aggfunc="first",
            ).reset_index(drop=True)

            sns.boxplot(data=result_df)

            # plt.xticks(rotation=45)
            plt.title("Box plot of " + independentvar + "-->" + dependentvar)
            plt.tight_layout()
            plt.savefig(
                DataInstance.ImageFolderPath
                + "\\"
                + independentvar
                + "--"
                + dependentvar
                + "_"
                + "Box-Plots.png",
                dpi=200,
            )


def QQPlots(DataInstance: data.Data):
    """
    Plots QQ-plots for each dependent variable and for each level of independent variables.
    Parameters:
    ----------
    - DataInstance: data.Data
        The input Data instance containing the data for which QQ plots are to be plotted.
    - SavePath: string
        The path where to save the QQ plots.
    """
    # QQ plot for each dependent variable
    for dependent_var in DataInstance.DependentVars:
        fig, ax = plt.subplots(figsize=(8, 8))
        stats.probplot(DataInstance.RawData[dependent_var], dist="norm", plot=ax)
        ax.set_title(f"QQ Plot of {dependent_var}")
        plt.tight_layout()
        plt.savefig(
            DataInstance.ImageFolderPath + f"\\QQPlot_{dependent_var}.png", dpi=200
        )
        plt.close()

    # QQ plot for each level of independent variables for all dependent variables
    for index, independent_var in enumerate(DataInstance.IndependentVars):
        IndependentLevels = DataInstance.IndependentLevelsList[index]
        for dependent_var in DataInstance.DependentVars:
            fig, axs = plt.subplots(
                1, len(IndependentLevels), figsize=(len(IndependentLevels) * 6, 6)
            )
            for i, level in enumerate(IndependentLevels):
                level_data = DataInstance.RawData[
                    DataInstance.RawData[independent_var] == level
                ]
                stats.probplot(level_data[dependent_var], dist="norm", plot=axs[i])
                axs[i].set_title(f"{independent_var}: {level} ({dependent_var})")
            plt.tight_layout()
            plt.savefig(
                DataInstance.ImageFolderPath
                + f"\\{independent_var}--{dependent_var}_QQPlot-levels.png",
                dpi=200,
            )
            plt.close()
