"""
PyADAP
=====

PyADAP: Python Automated Data Analysis Pipeline

Data Class

Author: Kannmu
Date: 2024/3/31
License: MIT License
Repository: https://github.com/Kannmu/PyADAP

"""

import os

import numpy as np
import pandas as pd

import PyADAP.Utilities as utility
from scipy.stats import boxcox


class Data:
    def __init__(
        self,
        DataPath: str,
    ) -> None:
        """
        Create Data Instance
        """

        self.DataPath = DataPath

        self.alpha = 0.05  # Default Value for Alpha

        self.DataFileName, self.DataFileType = os.path.splitext(
            os.path.basename(DataPath)
        )
        # Get Data File Name and Type
        self.ResultsFolderPath = (
            os.path.dirname(self.DataPath) + "\\" + self.DataFileName + " Results\\"
        )
        # Create Results Folder Path
        self.ResultsFilePath = (
            self.ResultsFolderPath + self.DataFileName + " " + "Results.xlsx"
        )
        # Create Image Folder Path
        self.ImageFolderPath = self.ResultsFolderPath + "Image\\"

        # Create Log File Path
        self.LogFilePath = self.ResultsFolderPath + self.DataFileName + ".log"

        # Create Results Text Path
        self.ResultsTextPath = self.ResultsFolderPath + "Results Text.txt"

        self.transformation = "None"

    def InitData(
        self,
        IndependentVarNames: list = [],
        DependentVarNames: list = [],
        alpha: float = 0.05,
    ):
        """
        Set Variables and Calculate Independent Variable Level Lists

        Parameters:
        -----------
        - IndependentVars: list[str]
            Independent Variables Name In DataFrame
        - DependentVars: list[str]
            Dependent Variables Name In DataFrame

        Returns:
        -----------
        None

        """
        self.IndependentVarNames = IndependentVarNames
        self.DependentVarNames = DependentVarNames
        self.alpha = alpha
        self.IndependentVarLevels = {}
        self.IndependentVarNum = len(self.IndependentVarNames)
        self.DependentVarNum = len(self.DependentVarNames)

        for Name in self.IndependentVarNames:
            TempIndependentLevels = self.Data[Name].unique().tolist()
            self.IndependentVarLevels[Name] = TempIndependentLevels

        self.CreateInitFolderAndFiles()

    def LoadData(self):
        """
        Load Data DataFrame
        """
        try:
            if self.DataPath.endswith((".xlsx", ".xls")):
                self.RawData = pd.read_excel(self.DataPath)
            elif self.DataPath.endswith(".csv"):
                self.RawData = pd.read_csv(self.DataPath)
            else:
                raise ValueError(f"Unsupported file format: {self.DataPath}")

            if self.RawData.empty:
                raise ValueError("Loaded data is empty")

            # Initialize a new DataFrame to store the transformed data
            self.Data = self.RawData.copy()

            self.subjectName = self.Data.columns[0]
            self.varsNames = self.Data.columns[1:]
            self.totalTrails = self.Data.shape[0]

        except Exception as e:
            raise RuntimeError(f"Failed to load data: {str(e)}")

    def Print2Log(self, S: str = "", enablePrint: bool = True):
        """
        Writes the provided string to the log file.

        Parameters:
        -----------
        - S: str
            The string to be written to the log file.

        Returns:
        -----------
        None

        """
        if enablePrint:
            print(S)
        with open(self.LogFilePath, "a", encoding="utf-8") as log_file:
            log_file.write(S + "\n")

    def ReadLogs(self):
        with open(self.LogFilePath, "r", encoding="utf-8") as file:
            content = file.read()
        return content

    def DataCleaning(self):
        """
        Data Cleaning
        """
        self.Data = self.clean_numeric_columns(self.Data)
        self.Data = self.clean_string_columns(self.Data)

    def CreateInitFolderAndFiles(self):

        # Create Folders
        # Main Results Folder
        utility.CreateFolder(self.ResultsFolderPath)
        # Create Image Folder
        utility.CreateFolder(self.ImageFolderPath)

        with open(self.LogFilePath, "w", encoding="utf8") as log_file:
            log_file.write("Log file for " + self.DataFileName + "\n")

    def clean_numeric_columns(self, data: pd.DataFrame):
        """
        Cleans the numeric columns in the provided DataFrame by replacing outliers and missing values with mean values.

        Parameters:
        ----------
        - data: pd.DataFrame
            The data containing the numeric columns to be cleaned.

        Returns:
        ----------
        - pd.DataFrame
            The cleaned DataFrame with replaced outliers and missing values.

        """
        for col in data.select_dtypes(include=[np.number]).columns:
            if col in [
                "subject",
                "subjects",
                "Subject" "Subjects",
                "participants",
                "Participants",
                "Participant",
            ]:
                continue
            mean_value = data[col].mean()
            std_value = data[col].std()

            # Apply the 3 sigma rule to find the indices of outliers
            outliers = data[
                (data[col] < (mean_value - 3 * std_value))
                | (data[col] > (mean_value + 3 * std_value))
            ].index

            # Replace outliers and missing values with the mean value
            data.loc[outliers, col] = np.nan
            data[col] = data[col].fillna(mean_value)

        return data

    def clean_string_columns(self, data: pd.DataFrame):
        """
        Cleans the string columns in the provided DataFrame by removing non-Unicode characters and rows with missing values.

        Parameters:
        ----------
        - data: pd.DataFrame
            The data containing the string columns to be cleaned.

        Returns:
        ----------
        - pd.DataFrame
            The cleaned DataFrame with removed non-Unicode characters and rows with missing values.

        """
        for col in data.select_dtypes(include=[object]).columns:
            # Remove non-Unicode characters
            data[col] = data[col].apply(
                lambda x: (
                    x.encode("ascii", "ignore").decode("ascii")
                    if isinstance(x, str)
                    else x
                )
            )
            # Drop rows with missing values
            data.dropna(subset=[col], inplace=True)
        return data

    def GetDataInfo(self)->pd.DataFrame:
        """
            Returns a pandas DataFrame containing information about the dataset, including the number of rows and columns, the names of the dependent variables, and the names of the independent variables.
        """
        
        info = {
            "Transformation": self.transformation,
            "Total Trials": self.totalTrails,
        }

        return pd.DataFrame(info)

    def BoxCoxConvert(self,originalData:pd.DataFrame):
        """
            Converts the dependent variables in the DataFrame to a normal distribution using the Box-Cox transformation.
            The transformed data is stored in a new DataFrame called self.TransformedData.
        """
        self.transformation = "Box-Cox"
        data = originalData.copy()
        # Apply Box-Cox transformation to each dependent variable
        for dep_var in self.DependentVarNames:
            # Ensure the data is positive (Box-Cox requires positive values)
            if (data[dep_var] <= 0).any():
                # Shift the data to make it positive
                min_value = data[dep_var].min()
                shift_value = abs(min_value) + 1 if min_value <= 0 else 0
                data[dep_var] += shift_value

            # Apply Box-Cox transformation
            transformed_data, lambda_value = boxcox(data[dep_var])
            self.Data[dep_var] = transformed_data

            # Log the lambda value used for the transformation
            self.Print2Log(
                f"Box-Cox transformation applied to {dep_var} with lambda = {lambda_value}"
            )

        # Log the completion of the transformation
        self.Print2Log("Box-Cox transformation completed for all dependent variables.")

    def LogConvert(self,originalData:pd.DataFrame):
        """
        Converts the dependent variables in the DataFrame to a normal distribution using the logarithmic transformation.
        The transformed data is stored in the same DataFrame.
        """
        self.transformation = "Log"
        data = originalData.copy()
        # Apply log transformation to each dependent variable
        for dep_var in self.DependentVarNames:
            # Ensure the data is positive (log transformation requires positive values)
            if (data[dep_var] <= 0).any():
                # Shift the data to make it positive
                min_value = data[dep_var].min()
                shift_value = abs(min_value) + 1 if min_value <= 0 else 0
                data[dep_var] += shift_value

            # Apply log transformation
            self.Data[dep_var] = np.log(data[dep_var])

            # Log the transformation
            self.Print2Log(f"Logarithmic transformation applied to {dep_var}.")

        # Log the completion of the transformation
        self.Print2Log("Logarithmic transformation completed for all dependent variables.")
