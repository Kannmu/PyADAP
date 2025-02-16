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
    data: data.Data, interface: gui.Interface
):
    """This function performs the entire process of analyzing the data and generates a report on the results.

    Args:
        Data (data.Data): The data to analyze.

    Returns:
        pd.DataFrame: The results of the analysis, in a Pandas DataFrame format.
    """

    data.Print2Log("Pipeline Started")

    Statist = statistic.Statistics(dataIns=data)

    # Optional Data Cleaning Process
    if interface.isClean:
        data.DataCleaning()

    data.Print2Log("Performing Statistics Calculation")
    StatisticsResults = Statist.BasicStatistics()

    if interface.enableBoxCox:
        # data.BoxCoxConvert(data.Data)
        data.LogConvert(data.Data)

    data.Print2Log("Performing NormalTest Test")
    NormalTestResults = Statist.NormalityTest()

    data.Print2Log("Performing Sphericity Test")
    SphericityTestResults = Statist.SphericityTest()

    data.Print2Log("Performing T-Test")
    tTestResults = Statist.TTest()

    data.Print2Log("Performing One-Way ANOVA")
    OneWayANOVAResults = Statist.OneWayANOVA()

    if(data.IndependentVarNum > 1):
        # Performing Two-Way ANOVA
        data.Print2Log("Performing Two-Way ANOVA")
        TwoWayANOVAResults = Statist.TwoWayANOVA()
    else:
        TwoWayANOVAResults = None

    rmANOVAResults = Statist.RM_ANOVA()

    data.Print2Log("Drawing Box Plots")
    plot.SingleBoxPlot(dataIns=data)

    if data.IndependentVarNum == 2:
        plot.DoubleBoxPlot(dataIns=data)
    else:
        print("Too many independent variables. plots is only support one or two independent variables currently.")

    data.Print2Log("Drawing Violin Plot")
    plot.SingleViolinPlot(dataIns=data)

    plot.DoubleBoxPlot(dataIns=data)

    data.Print2Log("Drawing QQ Plots")
    plot.QQPlot(dataIns=data)


    data.Print2Log("Writing Results To Excel File")
    file.SaveDataToExcel(
        data,
        StatisticsResults,
        NormalTestResults,
        SphericityTestResults,
        tTestResults,
        OneWayANOVAResults,
        TwoWayANOVAResults,
        rmANOVAResults,
    )

