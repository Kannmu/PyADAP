import PyADAP as pap
import pandas as pd
import win32ui
import os

def SelectDataFile():
    # Create a file selection dialog
    dlg = win32ui.CreateFileDialog(1, ".xlsx;.xls;.csv", None, 0, "Data (*.xlsx;*.xls;*.csv)|*.xlsx;*.xls;*.csv||")
    
    # Display the file selection dialog
    dlg.DoModal()
    
    # Get the selected file path
    DataPath = dlg.GetPathName()
    
    # Return the selected file path
    return DataPath

def LoadData(DataPath:str):
    if DataPath.endswith(".xlsx") or DataPath.endswith(".xls"):
        # 读取Excel文件
        RawDataFrame = pd.read_excel(DataPath)
    elif DataPath.endswith(".csv"):
        # 读取CSV文件
        RawDataFrame = pd.read_csv(DataPath)
    else:
        # 文件类型不支持
        raise ValueError("Unsupported file format")
    
    return RawDataFrame


DataPath = SelectDataFile()

print("DataPath:", DataPath + "\n")

pap.U.CreateSaveFolder(os.path.dirname(DataPath) + "\\Results")

RawData = LoadData(DataPath)

# print(RawData)

pap.Pipeline(RawData,DataPath = DataPath)





