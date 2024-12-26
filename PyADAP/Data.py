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


class Data:
    def __init__(
        self,
        DataPath: str,
    ) -> None:
        """
        Create Data Instance
        """
        self.DataPath = DataPath

        self.Alpha = 0.05  # Default Value for Alpha

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

            self.subjectName = self.RawData.columns[0]
            self.varsNames = self.RawData.columns[1:]
            self.totalTrails = self.RawData.shape[0]

        except Exception as e:
            raise RuntimeError(f"Failed to load data: {str(e)}")

    def Print2Log(self, S: str = ""):
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
        with open(self.LogFilePath, "a") as log_file:
            log_file.write(S + "\n")

    def DataCleaning(self):
        """
        Data Cleaning
        """
        self.RawData = self.clean_numeric_columns(self.RawData)
        self.RawData = self.clean_string_columns(self.RawData)

    def InitData(
        self,
        IndependentVarNames: list = [],
        DependentVarNames: list = [],
        Alpha: float = 0.05,
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
        self.Alpha = Alpha
        self.IndependentVarLevels = {}
        
        self.IndependentVarNum = len(self.IndependentVarNames)
        self.DependentVarNum = len(self.DependentVarNames)

        for Name in self.IndependentVarNames:
            # IndependentLevels = list(set(self.RawData[independentvar]))
            TempIndependentLevels = self.RawData[Name].unique().tolist()
            # self.IndependentLevels.append(IndependentLevels)
            self.IndependentVarLevels[Name] = TempIndependentLevels

        self.CreateInitFolderAndFiles()

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
