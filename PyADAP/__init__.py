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

import pandas as pd

import PyADAP.Data as data
import PyADAP.File as file
import PyADAP.Fit as fit
import PyADAP.GUI as gui
import PyADAP.Plot as plot
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

    Data.Print2Log("Pipeline Started")
    print("Pipeline Started")

    Data.Print2Log("Performing Statistics Calculation")
    print("Performing Statistics Calculation")
    StatisticsResults = statistic.statistics(DataInstance=Data)

    Data.Print2Log("Drawing Box Plots")
    print("Drawing Box Plots")
    plot.BoxPlots(
        DataInstance=Data,
    )

    Data.Print2Log("Drawing Residual Plots")
    print("Drawing Residual Plots")
    plot.ResidualPlots(DataInstance=Data)
    NormalTestResults = statistic.normality_test(DataInstance=Data)

    Data.Print2Log("Drawing QQ Plots")
    print("Drawing QQ Plots")
    plot.QQPlots(
        DataInstance=Data,
    )

    Data.Print2Log("Performing Sphericity Test")
    print("Performing Sphericity Test")
    SphericityTestResults = statistic.sphericity_test(DataInstance=Data)

    Data.Print2Log("Performing T-Test")
    print("Performing T-Test")
    TtestResults = statistic.ttest(DataInstance=Data)

    Data.Print2Log("Performing One-Way ANOVA")
    print("Performing One-Way ANOVA")
    OneWayANOVAResults = statistic.OneWayANOVA(DataInstance=Data)

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
    
    # Application Finished Log Message to Console and Log File.
    Data.Print2Log("Application Finished")
    print("Application Finished! Check the results in the following folder: " + Data.ResultsFolderPath)

    os.startfile(Data.ResultsFolderPath)
