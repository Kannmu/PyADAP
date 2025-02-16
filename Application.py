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
data = pap.data.Data(DataPath)

# Load Data From the DataPath
data.LoadData()

# Select Parameters In GUI
interface = pap.gui.Interface()

interface.ParametersSettingPage(data.varsNames.tolist())

if interface.apiKey == "":
    apiKey = "sk-1093dcab736946a39f4887dd226e07a8"

# Check if Independent Vars and Dependent Vars are not empty.
if not interface.independentVars or not interface.dependentVars:
    raise ValueError("IndependentVars or DependentVars cannot be empty.")

# Set Variables in Data Instance
data.InitData(interface.independentVars, interface.dependentVars, interface.alpha)

# Run Pipeline
pap.Pipeline(data=data,interface=interface)

if interface.enableWriting:
    # Writing Results Text
    Writer = writing.Writer(dataIns=data, apiKey=apiKey)

# Application Finished Log Message to Console and Log File.
data.Print2Log("Application Finished")
print(
    "\nApplication Finished! Check the results in the following folder: "
    + data.ResultsFolderPath
    + "\n"
)
os.startfile(data.ResultsFolderPath)
os.system("Pause")
