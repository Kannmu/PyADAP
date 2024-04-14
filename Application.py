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
# Import necessary libraries and modules
import os
import sys

import pandas as pd

import PyADAP as pap

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
