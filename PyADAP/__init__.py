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

import PyADAP.Analysis as analysis
import PyADAP.Clean as clean
import PyADAP.FileProcess as fp
import PyADAP.Tests as tests
import PyADAP.Utilities as utility


def Pipeline(
    data: pd.DataFrame,
    DataPath: str = "",
    Clean: bool = False,
    IndependentVars: list = [],
    DependentVars: list = [],
):

    DataFileName, _ = os.path.splitext(os.path.basename(DataPath))

    if Clean:
        data = clean.clean_numeric_columns(data)
        data = clean.clean_string_columns(data)

    StatisticsResults = tests.statistics(data,IndependentVars = IndependentVars, DependentVars=DependentVars)

    NormalTestResults = tests.normality_test(data,IndependentVars = IndependentVars, DependentVars=DependentVars)

    tests.QQPlots(data, os.path.dirname(DataPath) + "\\Results" + "\\QQ-Plots.png")

    SphericityTestResults = tests.sphericity_test(data, IndependentVars = IndependentVars, DependentVars=DependentVars)

    ResultsFilePath = (
        os.path.dirname(DataPath) + "\\Results" + "\\" + DataFileName + "Results.xlsx"
    )

    with pd.ExcelWriter(ResultsFilePath, engine="xlsxwriter") as writer:
        StatisticsResults.to_excel(writer, sheet_name="StatisticsResults", index=False)
        NormalTestResults.to_excel(writer, sheet_name='NormalTestResults', index=False)
        SphericityTestResults.to_excel(writer, sheet_name='SphericityTestResults', index=False)

    # ResultsDF = pd.concat([StatisticsResults, NormalTestResults.iloc[:,1:]], axis=1)

    fp.ExcelFormAdj(ResultsFilePath)
