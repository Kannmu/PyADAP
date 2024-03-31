"""
PyADAP
=====

PyADAP: Python Automated Data Analysis Pipeline

Tests

Author: Kannmu
Date: 2024/3/31
License: MIT License
Repository: https://github.com/Kannmu/PyADAP

"""

# Import necessary modules

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from colorama import Back, Fore, Style, init
from prettytable import PrettyTable
from scipy import stats
from scipy.stats import kurtosis, skew

import PyADAP.Utilities as utility

init()


def statistics(data: pd.DataFrame, IndependentVars: list = [], DependentVars:list = []):
    """
    This function calculates and returns a DataFrame containing various statistical measures for each numerical column in the input DataFrame.

    Parameters:
    ----------
    - data : pd.DataFrame
        The input DataFrame containing the data for which statistics are to be calculated.

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
    >>> result = statistics(input_data)
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
    for dependentvar in DependentVars:
        for independentvar in IndependentVars:
            
            IndependentLevels = list(set(data[independentvar]))
            
            for indelevel in IndependentLevels:
                

                # Calculate the mean, standard deviation, kurtosis, and skewness for the column.
                Tempdata = data[data[independentvar] == indelevel][dependentvar]
                mean = Tempdata.mean()
                std = Tempdata.std()
                kurt = kurtosis(Tempdata)
                skewness = skew(Tempdata)

                # Create a temporary DataFrame with the calculated statistics for the current column.
                temp_result = pd.DataFrame(
                    {
                        "Variable": [independentvar+"->" + str(indelevel)+"-->"+ dependentvar],
                        "Mean": [mean],
                        "Standard Deviation": [std],
                        "Kurtosis": [kurt],
                        "Skewness": [skewness],
                    },
                    index=[0],
                )

                # Concatenate the temporary DataFrame with the main statistics DataFrame.
                statistics = pd.concat([statistics, temp_result], ignore_index=True,axis = 0)
                
    # print(statistics)
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


def normality_test(data: pd.DataFrame, IndependentVars: list = [], DependentVars:list = []):
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

    # Iterate over each column in the input DataFrame
    for column in data.columns:
        if column in  ["subject","subjects","Subject""Subjects","participants","Participants","Participant"]:
            continue
        # Skip non-numeric columns
        if not np.issubdtype(data[column].values.dtype, np.number):
            continue

        # Get the number of observations in the column
        n = len(data[column].values)

        # Select the appropriate normality test based on the number of observations
        if n < 50:
            Method = "shapiro"
        elif n < 300:
            Method = "normaltest"
        else:
            Method = "kstest"

        # Conduct the normality test using the selected method
        test_result = pg.normality(data[column], method=Method)

        # Prepare the results for the current column
        TempResult = pd.DataFrame(
            {
                "Variable": column,
                "Normal": test_result["normal"],
                "Method": ["omnibus" if Method == "normaltest" else Method][0],
                "Statistic": test_result["W"],
                "p-value": test_result["pval"],
            }
        )

        # Concatenate the current results with the overall results DataFrame
        normality = pd.concat([normality, TempResult], ignore_index=True)

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


def QQPlots(data: pd.DataFrame, SavePath: str = ""):
    """
    Plots QQ-plots for all the numerical columns in the provided DataFrame.

    Parameters:
    ----------
    - data: pd.DataFrame
        The data containing the columns for which QQ-plots need to be plotted.
    - SavePath: string
        The path where to save the QQ plots.

    Returns:
    ----------
    - None, but saves a plot containing QQ-plots for all the numerical columns.
    """

    # Determine the numeric columns in the data
    numeric_cols = data.iloc[:,1:].select_dtypes(include=np.number).columns.tolist()

    # Calculate the number of rows and columns for the plots, trying to make them as square as possible
    n = len(numeric_cols)
    nrows = int(np.ceil(np.sqrt(n)))
    ncols = nrows if n > nrows * (nrows - 1) else nrows - 1

    # Create a figure with enough space for all the subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))

    # If there is only one subplot, axes is not an array, so we convert it to an array for consistent handling
    if n == 1:
        axes = np.array([axes])

    # For each numeric column, plot the QQ-plot
    for col, ax in zip(numeric_cols, axes.flatten()):
        stats.probplot(data[col].dropna(), dist="norm", plot=ax)
        ax.set_title(f"QQ-plot of {col}")

    # If the number of subplots is less than the number of numeric columns, hide the excess subplots
    for i in range(n, nrows * ncols):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
    plt.savefig(SavePath, dpi=200)
    # plt.show()


def sphericity_test(data: pd.DataFrame, IndependentVars: list = [], DependentVars:list = []):
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
    # if len(Within) <= 2:
    #     print(Fore.YELLOW + "\n Sphericity assumptions must meet between two variables")
    #     return pd.DataFrame(
    #         {
    #             "Variable": ["Overall"],
    #             "Sphericity": "True",
    #             "Statistic": "N/A",
    #             "p-value": "N/A",
    #             "EPS": "N/A",
    #         }
    #     )

    ResultsColumns = ["Variable", "Sphericity", "Statistic", "p-value", "EPS"]
    SphericityResultTable = PrettyTable()

    # Prepare the DataFrame for results
    sphericity = pd.DataFrame(columns=ResultsColumns)
    sphericity = sphericity.astype({"Sphericity":"bool"})

    for dependentvar in DependentVars:
        
        for independentvar in IndependentVars:
            # Conduct the sphericity test
            test_result = pg.sphericity(data, dv = dependentvar, within = independentvar, subject = "Subject")

            # Prepare the result for output
            TempResult = pd.DataFrame(
                {
                    "Variable": [(independentvar+"-->"+dependentvar)],
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
