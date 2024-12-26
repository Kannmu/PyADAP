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
from statannotations.Annotator import Annotator
import PyADAP.Data as data
from itertools import combinations

plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 19
plt.rcParams["axes.titlesize"] = 24
plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

Colors = ["#FFA3B5", "#FFE8BD", "#A6B8FF","#92CED6","#BAE3FF"]

def SingleBoxPlot(dataIns: data.Data):
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

    for dependentVar in dataIns.DependentVarNames:
        for index, independentVar in enumerate(dataIns.IndependentVarNames):
            IndependentLevels = dataIns.IndependentVarLevels[independentVar]

            plt.figure(figsize=(len(IndependentLevels) * 2, 8))

            tempDataFrame = dataIns.RawData
            tempDataFrame["index_by_group"] = tempDataFrame.groupby([independentVar]).cumcount()

            # 使用pivot_table来重塑DataFrame
            result_df = tempDataFrame.pivot_table(
                index="index_by_group",
                columns=independentVar,
                values=dependentVar,
                aggfunc="first",
            ).reset_index(drop=True)

            melted_df = result_df.melt(var_name=independentVar, value_name=dependentVar)

            ax = sns.boxplot(data=melted_df,x=independentVar, y=dependentVar, palette=Colors)
            
            pairs = [(th1, th2) for i, th1 in enumerate(IndependentLevels) for th2 in IndependentLevels[i+1:]]
            
            # 初始化Annotator对象
            annotator = Annotator(ax, data=melted_df, x=independentVar, y=dependentVar,palette = Colors, pairs=pairs)

            # 配置显著性测试参数
            annotator.configure(alpha = dataIns.Alpha, test='t-test_ind', text_format='star', line_height=0.03, line_width=1,hide_non_significant = True)

            # 应用显著性标记
            annotator.apply_and_annotate()

            plt.tight_layout()
            plt.savefig(
                dataIns.ImageFolderPath
                + independentVar
                + "--"
                + dependentVar
                + "_"
                + "Box-Plots.png",
                dpi=200,
            )

def DoubleBoxPlot(dataIns: data.Data):
    if(len(dataIns.IndependentVarNames) != 2):
        return

    plt.figure(figsize=(len( dataIns.IndependentVarLevels[dataIns.IndependentVarNames[0]]) * 3, 10))
    ax = sns.boxenplot(data = dataIns.RawData, x = dataIns.IndependentVarNames[0], y = dataIns.DependentVarNames[0], hue=dataIns.IndependentVarNames[1],palette = Colors)
    
    # 初始化box_pairs列表
    box_pairs = []
    for Var1 in dataIns.IndependentVarLevels[dataIns.IndependentVarNames[0]]:
        if len(dataIns.IndependentVarLevels[dataIns.IndependentVarNames[1]]) > 1:
            th_mag_combinations = list(combinations(dataIns.IndependentVarLevels[dataIns.IndependentVarNames[1]], 2))  # 两两组合
            box_pairs.extend([(Var1, mag1), (Var1, mag2)] for (mag1, mag2) in th_mag_combinations)

    for Var2 in dataIns.IndependentVarLevels[dataIns.IndependentVarNames[1]]:
        if len(dataIns.IndependentVarLevels[dataIns.IndependentVarNames[0]]) > 1:
            th_combinations = list(combinations(dataIns.IndependentVarLevels[dataIns.IndependentVarNames[0]], 2))  # 两两组合
            box_pairs.extend([(th1, Var2), (th2, Var2)] for (th1, th2) in th_combinations)
    
    # 初始化Annotator对象
    annotator = Annotator(ax, data=dataIns.RawData, x = dataIns.IndependentVarNames[0], y = dataIns.DependentVarNames[0], hue=dataIns.IndependentVarNames[1],palette = Colors, pairs=box_pairs)

    # 配置显著性测试参数
    annotator.configure(alpha = dataIns.Alpha, test='t-test_ind', text_format='star', line_height=0.03, line_width=1,hide_non_significant = True)

    # 应用显著性标记
    annotator.apply_and_annotate()

    plt.tight_layout()
    plt.savefig(
        dataIns.ImageFolderPath
        + "_"
        + "DoubleBox-Plots.png",
        dpi=200,
    )

def SingleViolinPlot(dataIns: data.Data):
    for dependentvar in dataIns.DependentVarNames:
        for index, independentvar in enumerate(dataIns.IndependentVarNames):
            IndependentLevels = dataIns.IndependentVarLevels[independentvar]

            plt.figure(figsize=(len(IndependentLevels) * 3, 8))

            Tempdf = dataIns.RawData
            Tempdf["index_by_group"] = Tempdf.groupby([independentvar]).cumcount()

            # 使用pivot_table来重塑DataFrame
            result_df = Tempdf.pivot_table(
                index="index_by_group",
                columns=independentvar,
                values=dependentvar,
                aggfunc="first",
            ).reset_index(drop=True)

            melted_df = result_df.melt(var_name=independentvar, value_name=dependentvar)

            ax = sns.violinplot(data=melted_df,x=independentvar, y=dependentvar, palette=Colors)
            
            pairs = [(th1, th2) for i, th1 in enumerate(IndependentLevels) for th2 in IndependentLevels[i+1:]]
            
            # 初始化Annotator对象
            annotator = Annotator(ax, data=melted_df, x=independentvar, y=dependentvar,palette = Colors, pairs=pairs)

            # 配置显著性测试参数
            annotator.configure(alpha = dataIns.Alpha, test='t-test_ind', text_format='star', line_height=0.03, line_width=1,hide_non_significant = True)

            # 应用显著性标记
            annotator.apply_and_annotate()
            YMax = melted_df[dependentvar].max()
            YMin = melted_df[dependentvar].min()

            plt.ylim(YMin - 0.25*(YMax-YMin),1.5*YMax)

            # plt.xticks(rotation=45)
            # plt.title("Violin plot of " + independentvar + "-->" + dependentvar)
            plt.tight_layout()
            plt.savefig(
                dataIns.ImageFolderPath
                + independentvar
                + "--"
                + dependentvar
                + "_"
                + "Violin-Plots.png",
                dpi=200,
            )

def QQPlot(dataIns: data.Data):
    """
    Plots QQ-plots for each dependent variable and for each level of independent variables.
    Parameters:
    ----------
    - dataIns: data.Data
        The input Data instance containing the data for which QQ plots are to be plotted.
    - SavePath: string
        The path where to save the QQ plots.
    """
    # QQ plot for each dependent variable
    for dependent_var in dataIns.DependentVarNames:
        fig, ax = plt.subplots(figsize=(8, 8))
        stats.probplot(dataIns.RawData[dependent_var], dist="norm", plot=ax)
        ax.set_title(f"QQ Plot of {dependent_var}")
        plt.tight_layout()
        plt.savefig(
            dataIns.ImageFolderPath + f"QQPlot_{dependent_var}.png", dpi=200
        )
        plt.close()

    # QQ plot for each level of independent variables for all dependent variables
    for index, independent_var in enumerate(dataIns.IndependentVarNames):
        IndependentLevels = dataIns.IndependentVarLevels[independent_var]
        for dependent_var in dataIns.DependentVarNames:
            fig, axs = plt.subplots(
                1, len(IndependentLevels), figsize=(len(IndependentLevels) * 10, 6)
            )
            for i, level in enumerate(IndependentLevels):
                level_data = dataIns.RawData[
                    dataIns.RawData[independent_var] == level
                ]
                stats.probplot(level_data[dependent_var], dist="norm", plot=axs[i])
                axs[i].set_title(f"{independent_var}: {level} ({dependent_var})")
            plt.tight_layout()
            plt.savefig(
                dataIns.ImageFolderPath
                + f"{independent_var}--{dependent_var}_QQPlot-levels.png",
                dpi=200,
            )
            plt.close()

