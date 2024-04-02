import os

import pandas as pd
import win32ui

import PyADAP as pap


def SelectDataFile():
    # Create a file selection dialog
    dlg = win32ui.CreateFileDialog(
        1, ".xlsx;.xls;.csv", None, 0, "Data (*.xlsx;*.xls;*.csv)|*.xlsx;*.xls;*.csv||"
    )

    # Display the file selection dialog
    dlg.DoModal()

    # Get the selected file path
    DataPath = dlg.GetPathName()

    # Return the selected file path
    return DataPath


DataPath = SelectDataFile()

# print("DataPath:", DataPath + "\n")

pap.utility.CreateSaveFolder(os.path.dirname(DataPath) + "\\Results")

# Construct Data Instance
Data = pap.data.Data(
    DataPath, IndependentVars=["BC", "BS"], DependentVars=["Time", "Speed"], Clean=False
)

pap.Pipeline(
    Data=Data,
)
