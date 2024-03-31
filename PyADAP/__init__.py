"""
PyADAP
=====

PyADAP: Python Automated Data Analysis Pipeline

Author: Kannmu
Date: 2024/3/31
License: MIT License
Repository: https://github.com/Kannmu/PyADAP

"""


import PyADAP.Analysis as A
import PyADAP.Tests as T
import PyADAP.Utilities as U
import pandas as pd


def Pipeline(data:pd.DataFrame, DataPath:str = ""):
    StatisticsResults = T.statistics(data)
    NormalTestResults = T.normality_test(data)
    T.QQPlots(data)








