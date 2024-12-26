"""
PyADAP
=====

PyADAP: Python Automated Data Analysis Pipeline

Author: Kannmu
Date: 2024/3/31
License: MIT License
Repository: https://github.com/Kannmu/PyADAP

"""

import os

import PyADAP.Data as data
import PyADAP.File as file
import PyADAP.GUI as gui
import PyADAP.Plot as plot
import PyADAP.Statistic as statistic
import PyADAP.Utilities as utility
import PyADAP.Writing as writing

def Pipeline(
    Data: data.Data,
):
    """This function performs the entire process of analyzing the data and generates a report on the results.

    Args:
        Data (data.Data): The data to analyze.

    Returns:
        pd.DataFrame: The results of the analysis, in a Pandas DataFrame format.
    """

    Data.Print2Log("Pipeline Started")
    print("Pipeline Started")

    Data.Print2Log("Performing Statistics Calculation")
    print("Performing Statistics Calculation")
    StatisticsResults = statistic.statistics(dataIns=Data)

    Data.Print2Log("Drawing Box Plots")
    print("Drawing Box Plots")
    if Data.IndependentVarNum == 1:
        plot.SingleBoxPlot(dataIns=Data)
    elif Data.IndependentVarNum == 2:
        plot.DoubleBoxPlot(dataIns=Data)
    else:
        raise Exception("Too many independent variables. PyADAP is only support one or two independent variables.")

    Data.Print2Log("Drawing Violinplot")
    print("Drawing Violinplot")
    plot.SingleViolinPlot(dataIns=Data)

    plot.DoubleBoxPlot(dataIns=Data)

    Data.Print2Log("Drawing QQ Plots")
    print("Drawing QQ Plots")
    plot.QQPlot(dataIns=Data)

    NormalTestResults = statistic.normality_test(dataIns=Data)

    Data.Print2Log("Performing Sphericity Test")
    print("Performing Sphericity Test")
    SphericityTestResults = statistic.sphericity_test(dataIns=Data)

    Data.Print2Log("Performing T-Test")
    print("Performing T-Test")
    TtestResults = statistic.ttest(dataIns=Data)

    Data.Print2Log("Performing One-Way ANOVA")
    print("Performing One-Way ANOVA")
    OneWayANOVAResults = statistic.OneWayANOVA(dataIns=Data)

    Data.Print2Log("Writing Results To Excel File")
    print("Writing Results Into Excel File")

    file.SaveDataToExcel(
        Data.ResultsFilePath,
        StatisticsResults,
        NormalTestResults,
        SphericityTestResults,
        TtestResults,
        OneWayANOVAResults,
    )

    