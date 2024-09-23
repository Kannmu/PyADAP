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
IndependentVars, DependentVars, IsClean, Alpha = Interface.ParametersSettingPage(
    Data.VarsNames.tolist()
)

# Check if Independent Vars and Dependent Vars are not empty.
if not IndependentVars or not DependentVars:
    raise ValueError("IndependentVars or DependentVars cannot be empty.")

# Set Variables in Data Instance
Data.SetVars(IndependentVars, DependentVars, Alpha)

# Optional Data Cleaning Process
if IsClean:
    Data.DataCleaning()

# Run Pipeline
pap.Pipeline(Data=Data)

os.system("Pause")
