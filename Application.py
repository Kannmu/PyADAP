import os
import sys

import pandas as pd

import PyADAP as pap

# sys.exit(0)

DataPath = pap.file.SelectDataFile()

# Construct Data Instance
Data = pap.data.Data(DataPath)

Data.LoadData()

# Select Parameters In GUI
Interface = pap.gui.Interface()
IndependentVars, DependentVars, IsClean = Interface.ParametersSettingPage(Data.VarsNames.tolist())

# 判断接收到的这两个变量是否为空
if not IndependentVars or not DependentVars:
    raise ValueError("IndependentVars or DependentVars cannot be empty.")

Data.SetVars(IndependentVars, DependentVars)

Data.DataCleaning(Clean = IsClean)

# Create Results Folder
pap.utility.CreateFolder(os.path.dirname(DataPath) + "\\Results")

pap.Pipeline(Data=Data)


