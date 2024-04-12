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


class Data:
    def __init__(
        self,
        DataPath: str,
    ) -> None:
        """
        Create Data Instance
        """
        self.DataPath = DataPath

        self.DataFileName, self.DataFileType = os.path.splitext(
            os.path.basename(DataPath)
        )
        

    def LoadData(self):
        """
        Load Data DataFrame
        """
        if self.DataPath.endswith(".xlsx") or self.DataPath.endswith(".xls"):
            # Read the Excel file
            self.RawData = pd.read_excel(self.DataPath)
        elif self.DataPath.endswith(".csv"):
            # Read the CSV file
            self.RawData = pd.read_csv(self.DataPath)
        else:
            # File format not supported
            raise ValueError("Unsupported file format")
        
        self.VarsNames = self.RawData.columns[1:]

        self.Trails = self.RawData.shape[0]

    
    def DataCleaning(self, Clean: bool = False):
        """
        Data Cleaning

        Parameters:
        -----------
        - Clean: bool
            Toggle Data Cleaning
        """
        if not hasattr(self, 'RawData'):
            raise Exception("Error: Data need to be loaded first")
        if Clean:
            self.RawData = self.clean_numeric_columns(self.RawData)
            self.RawData = self.clean_string_columns(self.RawData)

    def SetVars(self, IndependentVars: list = [], DependentVars: list = []):
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
        self.IndependentVars = IndependentVars
        self.DependentVars = DependentVars

        self.IndependentLevelsList = []

        for independentvar in self.IndependentVars:
            IndependentLevels = list(set(self.RawData[independentvar]))
            self.IndependentLevelsList.append(IndependentLevels)

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
            data[col].fillna(mean_value, inplace=True)

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
