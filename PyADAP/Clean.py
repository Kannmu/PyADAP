"""
PyADAP
=====

PyADAP: Python Automated Data Analysis Pipeline

Data Clearing

Author: Kannmu
Date: 2024/3/31
License: MIT License
Repository: https://github.com/Kannmu/PyADAP

"""

import numpy as np
import pandas as pd


def clean_numeric_columns(data: pd.DataFrame):
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
        if col in  ["subject","subjects","Subject""Subjects","participants","Participants","Participant"]:
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


def clean_string_columns(data: pd.DataFrame):
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
                x.encode("ascii", "ignore").decode("ascii") if isinstance(x, str) else x
            )
        )

        # Drop rows with missing values
        data.dropna(subset=[col], inplace=True)

    return data
