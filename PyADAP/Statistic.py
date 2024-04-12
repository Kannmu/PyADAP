"""
PyADAP
=====

PyADAP: Python Automated Data Analysis Pipeline

Statistic

Author: Kannmu
Date: 2024/3/31
License: MIT License
Repository: https://github.com/Kannmu/PyADAP

"""

# Import necessary modules


import itertools

import numpy as np
import pandas as pd
import pingouin as pg
from colorama import Back, Fore, Style, init
from prettytable import PrettyTable
from scipy.stats import kurtosis, skew

import PyADAP.Data as data
import PyADAP.Utilities as utility

init()


def statistics(DataInstance: data.Data):
    """
    This function calculates and returns a DataFrame containing various statistical measures for each numerical column in the input DataFrame.

    Parameters:
    ----------
    - DataInstance : data.Data
        The input Data instance containing the data for which statistics are to be calculated.

    Returns:
    ----------
    - statistics : pd.DataFrame
        A DataFrame containing the calculated statistics for each numerical column.
        The DataFrame includes the following columns:
        - "Variable": The name of the variable (column).
        - "Mean": The mean (average) value of the variable.
        - "Standard Deviation": The standard deviation of the variable, a measure of dispersion or variability.
        - "Kurtosis": The kurtosis of the variable, a measure of the "tailedness" of the probability distribution.
        - "Skewness": The skewness of the variable, a measure of the asymmetry of the probability distribution.

    Example Usage:
    --------------
    >>> input_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> data_instance = data.Data(input_data, ['A'], ['B'])
    >>> result = statistics(data_instance)
    >>> print(result)
        Variable  Mean  Standard Deviation  Kurtosis  Skewness
    0        A   2.0      1.414214    -1.200000   0.000000
    1        B   5.0      1.414214     0.000000   0.000000

    """

    # Define the list of column names for the statistics DataFrame.
    statistics_columns = [
        "Variable",
        "Mean",
        "Standard Deviation",
        "Kurtosis",
        "Skewness",
    ]

    # Initialize an empty DataFrame to store the statistics.
    statistics = pd.DataFrame(columns=statistics_columns)

    # Initialize a PrettyTable object for printing the results in a formatted table.
    StatisticsResultTable = PrettyTable()

    # Loop through each column in the input DataFrame.
    for dependentvar in DataInstance.DependentVars:
        Tempdata = DataInstance.RawData[dependentvar]
        statistics = SingleStatistic(Tempdata, dependentvar, statistics)

        # Single independent variable different levels
        for index, independentvar in enumerate(DataInstance.IndependentVars):

            IndependentLevels = DataInstance.IndependentLevelsList[index]

            for indelevel in IndependentLevels:
                # Calculate the mean, standard deviation, kurtosis, and skewness for the column.
                Tempdata = DataInstance.RawData[
                    DataInstance.RawData[independentvar] == indelevel
                ][dependentvar]
                statistics = SingleStatistic(
                    Tempdata,
                    independentvar + "->" + str(indelevel) + "-->" + dependentvar,
                    statistics,
                )

        # Interaction Between independent variables different levels
        for IndependentvarCombination in itertools.product(
            *DataInstance.IndependentLevelsList
        ):

            conditions = (
                DataInstance.RawData[var] == value
                for var, value in zip(
                    DataInstance.IndependentVars, IndependentvarCombination
                )
            )
            combined_condition = pd.Series(True, index=DataInstance.RawData.index)
            for condition in conditions:
                combined_condition &= condition
            Tempdata = DataInstance.RawData[combined_condition][dependentvar]
            # Calculate the mean, standard deviation, kurtosis, and skewness for the column.

            VariableStrList = [
                "; " + invar + "->" + str(level)
                for invar, level in zip(
                    DataInstance.IndependentVars, IndependentvarCombination
                )
            ]

            VariableStr = (
                ("".join(VariableStrList) + "-->" + dependentvar)[1:]
            ).lstrip()

            statistics = SingleStatistic(
                Tempdata,
                VariableStr,
                statistics,
            )

    # Prepare the PrettyTable object for printing the statistics.
    StatisticsResultTable.clear()
    StatisticsResultTable.field_names = statistics_columns
    # Prepare the PrettyTable object for printing the statistics.
    StatisticsResultTable.clear()
    StatisticsResultTable.field_names = statistics_columns

    # Iterate over the rows of the statistics DataFrame and format the floating-point numbers.
    for i, row in statistics.iterrows():
        row_formatted = row.apply(utility.RoundFloat)
        StatisticsResultTable.add_row(row_formatted.values)

    # Print the statistics in a formatted table.
    print(Fore.YELLOW + "\nStatistics Results:")
    print(Fore.BLUE)
    print(StatisticsResultTable)
    print(Fore.RESET)

    # Return the DataFrame containing the calculated statistics.
    return statistics


def SingleStatistic(Tempdata: pd.DataFrame, VariableStr: str, statistics):
    mean = Tempdata.mean()
    std = Tempdata.std()
    kurt = kurtosis(Tempdata)
    skewness = skew(Tempdata)
    # Create a temporary DataFrame with the calculated statistics for the current column.
    temp_result = pd.DataFrame(
        {
            "Variable": [VariableStr],
            "Mean": [mean],
            "Standard Deviation": [std],
            "Kurtosis": [kurt],
            "Skewness": [skewness],
        },
        index=[0],
    )
    # Concatenate the temporary DataFrame with the main statistics DataFrame.
    statistics = pd.concat([statistics, temp_result], ignore_index=True, axis=0)

    return statistics


def normality_test(DataInstance: data.Data):
    """
    This function performs a normality test on each numeric column of the provided DataFrame to determine if the data follows a normal distribution.

    Parameters:
    ----------
    - data : pd.DataFrame
        The input DataFrame containing the data to be tested. Each column represents a variable that will be checked for normality.

    Returns:
    ----------
    - normality : pd.DataFrame
        A DataFrame containing the results of the normality test for each numeric column.
        The DataFrame includes the following columns:
        - 'Variable': The name of the variable (column) being tested.
        - 'Normal': A boolean indicating whether the variable follows a normal distribution (True if the null hypothesis of normality is not rejected, False otherwise).
        - 'Method': The method used for the normality test, which can be 'shapiro', 'normaltest', or 'kstest' based on the sample size.
        - 'Statistic': The test statistic value obtained from the normality test.
        - 'p-value': The p-value associated with the test, which helps to determine if the null hypothesis can be rejected.

    Notes:
    -----
    The function uses different normality tests depending on the sample size of the data:
    - 'shapiro': Suitable for small sample sizes (less than 50 observations).
    - 'normaltest': An omnibus normality test suitable for samples larger than 50 but less than 300.
    - 'kstest': The Kolmogorov-Smirnov test, suitable for larger sample sizes (300 or more).

    The 'normality' DataFrame is initialized with the 'Normal' column as a boolean type to store the True/False results of the normality tests.

    Example Usage:
    --------------
    >>> input_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> result = normality_test(input_data)
    >>> print(result)
      Variable  Normal   Method  Statistic  p-value
    0        A    False  shapiro       -1.53      0.13
    1        B     True  normaltest       NaN      1.00

    The function prints the results in a table format for easy readability and returns the DataFrame for further analysis.
    """

    # Define the column names for the results DataFrame
    ResultsColumns = ["Variable", "Normal", "Method", "Statistic", "p-value"]

    # Initialize a PrettyTable object for displaying the results in a table format
    NormalResultTable = PrettyTable()

    # Create an empty DataFrame to store the results
    normality = pd.DataFrame(columns=ResultsColumns)

    # Ensure the 'Normal' column is of boolean type
    normality = normality.astype({"Normal": "bool"})

    for dependentvar in DataInstance.DependentVars:
        Tempdata = DataInstance.RawData[dependentvar]
        normality = SingleNormalTest(Tempdata, dependentvar, normality)

        # Single independent variable different levels
        for index, independentvar in enumerate(DataInstance.IndependentVars):
            IndependentLevels = DataInstance.IndependentLevelsList[index]
            for indelevel in IndependentLevels:

                Tempdata = DataInstance.RawData[
                    DataInstance.RawData[independentvar] == indelevel
                ][dependentvar]
                normality = SingleNormalTest(
                    Tempdata,
                    independentvar + "->" + str(indelevel) + "-->" + dependentvar,
                    normality,
                )

        # Interaction Between independent variables different levels
        for IndependentvarCombination in itertools.product(
            *DataInstance.IndependentLevelsList
        ):
            # print(IndependentvarCombination)
            conditions = (
                DataInstance.RawData[var] == value
                for var, value in zip(
                    DataInstance.IndependentVars, IndependentvarCombination
                )
            )
            combined_condition = pd.Series(True, index=DataInstance.RawData.index)
            for condition in conditions:
                combined_condition &= condition
            Tempdata = DataInstance.RawData[combined_condition][dependentvar]
            VariableStrList = [
                "; " + invar + "->" + str(level)
                for invar, level in zip(
                    DataInstance.IndependentVars, IndependentvarCombination
                )
            ]

            VariableStr = ("".join(VariableStrList) + "-->" + dependentvar)[1:]
            VariableStr = VariableStr.lstrip()

            normality = SingleNormalTest(Tempdata, VariableStr, normality)

    # Display the results in table format using PrettyTable
    NormalResultTable.clear()
    NormalResultTable.field_names = ResultsColumns

    for i, row in normality.iterrows():
        # Add each row of results to the PrettyTable
        NormalResultTable.add_row(row.apply(utility.RoundFloat).values)

    # Print the table with the results
    print(Fore.YELLOW + "\nNormality Test Results:")
    print(Fore.RED)
    print(NormalResultTable)
    print(Fore.RESET)

    # Return the results DataFrame
    return normality


def SingleNormalTest(Tempdata: pd.DataFrame, VariableStr: str, normality: pd.DataFrame):
    # Get the number of observations in the column
    n = len(Tempdata.values)

    # Select the appropriate normality test based on the number of observations
    if n < 50:
        Method = "shapiro"
    elif n < 300:
        Method = "normaltest"
    else:
        Method = "jarque_bera"
    # print(Method)
    
    # Conduct the normality test using the selected method
    test_result = pg.normality(Tempdata, method = Method)

    # Prepare the result for output
    TempResult = pd.DataFrame(
        {
            "Variable": [VariableStr],
            "Normal": test_result["normal"],
            "Method": ["omnibus" if Method == "normaltest" else Method][0],
            "Statistic": test_result["W"],
            "p-value": test_result["pval"],
        }
    )
    # Concatenate the current results with the overall results DataFrame
    normality = pd.concat([normality, TempResult], ignore_index=True)

    return normality


def sphericity_test(DataInstance: data.Data):
    """
    Parameters:
    ----------
    - data : pd.DataFrame
        The input DataFrame containing the data.

    Returns:
    ----------
    - sphericity : pd.DataFrame
        A DataFrame containing the result of the sphericity test.
        The DataFrame includes the following columns:
        - Variable: The name of the variable (column) being tested.
        - Sphericity: Whether the data meet the sphericity assumption (True/False).
        - Statistic: The test statistic value.
        - p-value: The p-value associated with the test.
        - EPS: The epsilon value used to correct degrees of freedom if sphericity is violated.
    """

    ResultsColumns = ["Variable", "Sphericity", "Statistic", "p-value", "EPS"]
    SphericityResultTable = PrettyTable()

    # Prepare the DataFrame for results
    sphericity = pd.DataFrame(columns=ResultsColumns)
    sphericity = sphericity.astype({"Sphericity": "bool"})

    for dependentvar in DataInstance.DependentVars:

        for independentvar in DataInstance.IndependentVars:
            # Conduct the sphericity test
            test_result = pg.sphericity(
                DataInstance.RawData,
                dv=dependentvar,
                within=independentvar,
                subject="Subject",
            )

            # Prepare the result for output
            TempResult = pd.DataFrame(
                {
                    "Variable": [(independentvar + "-->" + dependentvar)],
                    "Sphericity": test_result[0],
                    "Statistic": test_result[1],
                    "p-value": test_result[2],
                    "EPS": test_result[3],
                }
            )
            # Concatenate the result
            sphericity = pd.concat([sphericity, TempResult], ignore_index=True)

    # Print Result in Table Form
    SphericityResultTable.clear()
    SphericityResultTable.field_names = ResultsColumns

    for i, row in sphericity.iterrows():
        SphericityResultTable.add_row(row.apply(utility.RoundFloat).values)

    print(Fore.YELLOW + "\nSphericity Test Results:")
    print(Fore.GREEN)
    print(SphericityResultTable)
    print(Fore.RESET)

    return sphericity
