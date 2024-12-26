"""
PyADAP
=====

PyADAP: Python Automated Data Analysis Pipeline

Application

Author: Kannmu
Date: 2024/3/31
License: MIT License
Repository: https://github.com/Kannmu/PyADAP

"""

print("Now Loading, Please wait a few seconds")

# Import necessary libraries and modules
import os
import warnings

from colorama import Fore, init

import PyADAP as pap
import PyADAP.Writing as writing

warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.",
)

warnings.filterwarnings(
    "ignore",
    message="When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas.",
)
warnings.filterwarnings("ignore", message="SeriesGroupBy.grouper is deprecated")


# Select Data File in GUI
DataPath = pap.file.SelectDataFile()

# Construct Data Instance
Data = pap.data.Data(DataPath)

# Load Data From the DataPath
Data.LoadData()

# Select Parameters In GUI
Interface = pap.gui.Interface()
independentVars, dependentVars, isUsingDataClean, Alpha, apiKey = (
    Interface.ParametersSettingPage(Data.varsNames.tolist())
)

if(apiKey is ""):
    apiKey = "sk-caQnDXpwc00jFJu4LqO76ob5iPHeZzGwHuWdlZQZZFV9xMuv"

# Check if Independent Vars and Dependent Vars are not empty.
if not independentVars or not dependentVars:
    raise ValueError("IndependentVars or DependentVars cannot be empty.")

# Set Variables in Data Instance
Data.InitData(independentVars, dependentVars, Alpha)

# Optional Data Cleaning Process
if isUsingDataClean:
    Data.DataCleaning()

# Run Pipeline
pap.Pipeline(Data=Data)

# Writing Results Text
Writer = writing.Writer(dataIns=Data,apiKey = apiKey)

# Application Finished Log Message to Console and Log File.
Data.Print2Log("Application Finished")
print(
    "\nApplication Finished! Check the results in the following folder: "
    + Data.ResultsFolderPath
    + "\n"
)
os.startfile(Data.ResultsFolderPath)
os.system("Pause")
