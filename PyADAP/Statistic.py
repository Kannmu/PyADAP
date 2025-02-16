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
from colorama import Fore, init
from prettytable import PrettyTable
from scipy.stats import kurtosis, skew

import PyADAP.Data as data
import PyADAP.Utilities as utility
import pandas as pd
from scipy.stats import shapiro
from prettytable import PrettyTable
from colorama import Fore
import statsmodels.formula.api as smf
init()

class Statistics:
    def __init__(self, dataIns:data.Data):
        self.dataIns = dataIns

    def BasicStatistics(self):
        """
        This function calculates and returns a DataFrame containing various statistical measures for each numerical column in the input DataFrame.

        Parameters:
        ----------
        - dataIns : data.Data
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
        for dependentVar in self.dataIns.DependentVarNames:
            tempData = self.dataIns.Data[dependentVar]
            statistics = self.SingleStatistic(tempData, dependentVar, statistics)

            # Single independent variable different levels
            for index, independentVar in enumerate(self.dataIns.IndependentVarNames):

                IndependentLevels = self.dataIns.IndependentVarLevels[independentVar]

                for Level in IndependentLevels:
                    # Calculate the mean, standard deviation, kurtosis, and skewness for the column.
                    tempData = self.dataIns.Data[self.dataIns.Data[independentVar] == Level][
                        dependentVar
                    ]
                    statistics = self.SingleStatistic(
                        tempData,
                        independentVar + "->" + str(Level) + "-->" + dependentVar,
                        statistics,
                    )

            # Interaction Between independent variables different levels
            for independentVarCombination in itertools.product(
                *self.dataIns.IndependentVarLevels.values()
            ):

                conditions = (
                    self.dataIns.Data[var] == value
                    for var, value in zip(
                        self.dataIns.IndependentVarNames, independentVarCombination
                    )
                )
                combined_condition = pd.Series(True, index=self.dataIns.Data.index)
                for condition in conditions:
                    combined_condition &= condition
                tempData = self.dataIns.Data[combined_condition][dependentVar]
                # Calculate the mean, standard deviation, kurtosis, and skewness for the column.

                VariableStrList = [
                    "; " + invar + "->" + str(level)
                    for invar, level in zip(
                        self.dataIns.IndependentVarNames, independentVarCombination
                    )
                ]

                VariableStr = (
                    ("".join(VariableStrList) + "-->" + dependentVar)[1:]
                ).lstrip()

                statistics = self.SingleStatistic(
                    tempData,
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

        # Print To Log File
        self.dataIns.Print2Log("\nStatistics Results:", False)
        self.dataIns.Print2Log(str(StatisticsResultTable), False)

        # Return the DataFrame containing the calculated statistics.
        return statistics

    def SingleStatistic(self, tempData: pd.DataFrame, VariableStr: str, statistics):
        mean = tempData.mean()
        std = tempData.std()
        kurt = kurtosis(tempData)
        skewness = skew(tempData)
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

    def NormalityTest(self):
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

        for dependentVar in self.dataIns.DependentVarNames:
            tempData = self.dataIns.Data[dependentVar]
            normality = self.SingleNormalTest(
                tempData, dependentVar, normality, alpha=self.dataIns.alpha
            )
            # Single independent variable different levels
            for index, independentVar in enumerate(self.dataIns.IndependentVarNames):
                IndependentLevels = self.dataIns.IndependentVarLevels[independentVar]
                for Level in IndependentLevels:
                    tempData = self.dataIns.Data[self.dataIns.Data[independentVar] == Level][
                        dependentVar
                    ]
                    normality = self.SingleNormalTest(
                        tempData,
                        independentVar + "->" + str(Level) + "-->" + dependentVar,
                        normality,
                        alpha=self.dataIns.alpha,
                    )

            # Interaction Between independent variables different levels
            for independentVarCombination in itertools.product(
                *self.dataIns.IndependentVarLevels.values()
            ):
                # print(self.dataIns.IndependentVarLevels)
                # print(independentVarCombination)
                conditions = [
                    self.dataIns.Data[var] == value
                    for var, value in zip(
                        self.dataIns.IndependentVarNames, independentVarCombination
                    )
                ]
                combined_condition = pd.Series(True, index=self.dataIns.Data.index)
                for condition in conditions:
                    combined_condition &= condition
                tempData = self.dataIns.Data[combined_condition][dependentVar]
                VariableStrList = [
                    "; " + invar + "->" + str(level)
                    for invar, level in zip(
                        self.dataIns.IndependentVarNames, independentVarCombination
                    )
                ]

                VariableStr = ("".join(VariableStrList) + "-->" + dependentVar)[1:]
                VariableStr = VariableStr.lstrip()

                normality = self.SingleNormalTest(
                    tempData, VariableStr, normality, alpha=self.dataIns.alpha
                )

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

        # Print To Log File
        self.dataIns.Print2Log("\nNormality Test Results:", False)
        self.dataIns.Print2Log(str(NormalResultTable), False)

        # Return the results DataFrame
        return normality

    def SingleNormalTest(
        self,
        tempData: pd.DataFrame,
        variableStr: str,
        normality: pd.DataFrame,
        alpha: float = 0.05,
    ):
        # Get the number of observations in the column
        n = len(tempData.values)

        # Select the appropriate normality test based on the number of observations
        if n < 50:
            method = "shapiro"
        elif n < 300:
            method = "normaltest"
        else:
            method = "jarque_bera"
        # print(Method)
        # Conduct the normality test using the selected method
        test_result = pg.normality(tempData, method=method, alpha=alpha)
        # Prepare the result for output
        tempResult = pd.DataFrame(
            {
                "Variable": [variableStr],
                "Normal": test_result["normal"],
                "Method": ["omnibus" if method == "normaltest" else method][0],
                "Statistic": test_result["W"],
                "p-value": test_result["pval"],
            }
        )
        # Concatenate the current results with the overall results DataFrame
        normality = pd.concat([normality, tempResult], ignore_index=True)

        return normality

    def ResidualNormalityTest(self):
        """
        Performs normality test on the residuals of the model.

        Parameters:
        ----------
        dataIns : data.Data
            An instance of the Data class containing the data to be analyzed.

        Returns:
        ----------
        normality_results : pd.DataFrame
            A DataFrame containing the results of the normality test.
            The DataFrame includes the following columns:
            - Variable: The name of the dependent variable being tested.
            - Statistic: The test statistic value.
            - p-value: The p-value associated with the test.
            - Normality: Whether the residuals are normally distributed (True/False).
        """
        # Define the column names for the results DataFrame
        ResultsColumns = ["Variable", "Normal", "Method", "Statistic", "p-value"]

        # Initialize a PrettyTable object for displaying the results in a table format
        NormalResultTable = PrettyTable()

        # Create an empty DataFrame to store the results
        normality = pd.DataFrame(columns=ResultsColumns)

        # Ensure the 'Normal' column is of boolean type
        normality = normality.astype({"Normal": "bool"})

        model = smf.mixedlm(f"{self.dataIns.DependentVarNames[0]} ~ {self.dataIns.IndependentVarNames[0]} + {self.dataIns.IndependentVarNames[1]}", data=self.dataIns.Data, groups=self.dataIns.subjectName)
        fitted_model = model.fit()
        
        # Obtain residuals
        self.dataIns.Data["Residuals"] = fitted_model.resid
        
        # Loop through dependent variables and levels of independent variables
        for dependentVar in self.dataIns.DependentVarNames:
            # Single independent variable different levels
            for independentVar in self.dataIns.IndependentVarNames:
                for Level in self.dataIns.IndependentVarLevels[independentVar]:
                    subset = self.dataIns.Data[
                        (self.dataIns.Data[independentVar] == Level)
                    ]
                    tempResiduals = subset["Residuals"]
                    variableStr = f"{independentVar}->{Level}->{dependentVar}"
                    normality = self.SingleResidualNormalityTest(
                        tempResiduals, variableStr, normality, alpha=self.dataIns.alpha
                    )
            
            # Interaction between independent variables different levels
            for independentVarCombination in itertools.product(
                *self.dataIns.IndependentVarLevels.values()
            ):
                conditions = [
                    self.dataIns.Data[var] == value
                    for var, value in zip(
                        self.dataIns.IndependentVarNames, independentVarCombination
                    )
                ]
                combined_condition = pd.Series(True, index=self.dataIns.Data.index)
                for condition in conditions:
                    combined_condition &= condition
                subset = self.dataIns.Data[combined_condition]
                tempResiduals = subset["Residuals"]
                variableStr = f"{'->'.join([f'{var}->{level}' for var, level in zip(self.dataIns.IndependentVarNames, independentVarCombination)])}->{dependentVar}"
                normality = self.SingleResidualNormalityTest(
                    tempResiduals, variableStr, normality, alpha=self.dataIns.alpha
                )

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

        # Print To Log File
        self.dataIns.Print2Log("\nNormality Test Results:", False)
        self.dataIns.Print2Log(str(NormalResultTable), False)

        # Return the results DataFrame
        return normality

    def SingleResidualNormalityTest(self,
        tempData: pd.DataFrame,
        variableStr: str,
        normality: pd.DataFrame,
        alpha: float = 0.05,
    ):
        # Get the number of observations in the column
        n = len(tempData.values)

        # Select the appropriate normality test based on the number of observations
        if n < 50:
            method = "shapiro"
        elif n < 300:
            method = "normaltest"
        else:
            method = "jarque_bera"
        # print(Method)
        # Conduct the normality test using the selected method
        test_result = pg.normality(tempData - tempData.mean(), method=method, alpha=alpha)
        # Prepare the result for output
        tempResult = pd.DataFrame(
            {
                "Variable": [variableStr],
                "Normal": test_result["normal"],
                "Method": ["omnibus" if method == "normaltest" else method][0],
                "Statistic": test_result["W"],
                "p-value": test_result["pval"],
            }
        )
        # Concatenate the current results with the overall results DataFrame
        normality = pd.concat([normality, tempResult], ignore_index=True)

        return normality



    def SphericityTest(self):
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

        for dependentVar in self.dataIns.DependentVarNames:

            for independentVar in self.dataIns.IndependentVarNames:
                # Conduct the sphericity test
                test_result = pg.sphericity(
                    self.dataIns.Data,
                    dv=dependentVar,
                    within=independentVar,
                    subject=self.dataIns.subjectName,
                    alpha=self.dataIns.alpha,
                )

                # Prepare the result for output
                TempResult = pd.DataFrame(
                    {
                        "Variable": [(independentVar + "-->" + dependentVar)],
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

        # Print To Log File
        self.dataIns.Print2Log("\nSphericity Test Results:", False)
        self.dataIns.Print2Log(str(SphericityResultTable), False)

        return sphericity

    def TTest(self):
        """
        Performs t-tests on the data based on the dependent and independent variables.

        Parameters:
        ----------
        dataIns : data.Data
            An instance of the Data class containing the data to be analyzed.

        Returns:
        ----------
        t_test_results : pd.DataFrame
            A DataFrame containing the results of the t-tests.
            The DataFrame includes the following columns:
            - Variable: The name of the variable combination being tested.
            - Test: The type of t-test performed (e.g., "One-Sample", "Independent", "Paired").
            - Statistic: The test statistic value.
            - p-value: The p-value associated with the test.
        """

        ResultsColumns = ["Variable", "Significance", "Statistic", "p-value"]
        tTestResultTable = PrettyTable()

        # Prepare the DataFrame for results
        t_test_results = pd.DataFrame(columns=ResultsColumns)
        t_test_results = t_test_results.astype({"Significance": "bool"})

        # Paired t-test for repeated measures
        for dependentVar in self.dataIns.DependentVarNames:
            for independentVar in self.dataIns.IndependentVarNames:
                IndependentLevels = self.dataIns.IndependentVarLevels[independentVar]
                for level1, level2 in itertools.combinations(IndependentLevels, 2):
                    tempData1 = self.dataIns.Data[self.dataIns.Data[independentVar] == level1][
                        dependentVar
                    ]
                    tempData2 = self.dataIns.Data[self.dataIns.Data[independentVar] == level2][
                        dependentVar
                    ]
                    test_result = pg.ttest(
                        tempData1, tempData2, confidence=1 - self.dataIns.alpha,paired=True,correction="b"
                    )

                    # print("\n",test_result,"\n",test_result["p-val"].values[-1],type(test_result["p-val"].values[-1]))

                    TempResult = pd.DataFrame(
                        {
                            "Variable": [
                                f"{independentVar}: {level1} vs {level2} --> {dependentVar}"
                            ],
                            "Significance": [
                                (
                                    True
                                    if test_result["p-val"].values[-1] < self.dataIns.alpha
                                    else False
                                )
                            ],
                            "Statistic": test_result["T"].values[-1],
                            "p-value": test_result["p-val"].values[-1],
                        }
                    )

                    t_test_results = pd.concat(
                        [t_test_results, TempResult], ignore_index=True
                    )

        # Display the results in table format using PrettyTable
        tTestResultTable.clear()
        tTestResultTable.field_names = ResultsColumns

        for i, row in t_test_results.iterrows():
            tTestResultTable.add_row(row.apply(utility.RoundFloat).values)

        print(Fore.YELLOW + "\nT-test Results:")
        print(Fore.CYAN)
        print(tTestResultTable)
        print(Fore.RESET)

        # Print To Log File
        self.dataIns.Print2Log("\nT-test Results:", False)
        self.dataIns.Print2Log(str(tTestResultTable), False)

        return t_test_results

    def OneWayANOVA(self):
        """
        Performs one-way ANOVA on the data based on the independent and dependent variables.

        Parameters:
        ----------
        dataIns : data.Data
            An instance of the Data class containing the data to be analyzed.

        Returns:
        ----------
        anova_results : pd.DataFrame
            A DataFrame containing the results of the one-way ANOVA.
            The DataFrame includes the following columns:
            - Variable: The name of the variable combination being tested.
            - df: Degrees of freedom.
            - F: The F-statistic value.
            - p-value: The p-value associated with the test.
            - np2: Partial eta squared as a measure of effect size.
        """

        ResultsColumns = ["Variable", "Significance", "df", "F", "p-value", "np2"]
        anovaResultTable = PrettyTable()

        # Prepare the DataFrame for results
        anovaResults = pd.DataFrame(columns=ResultsColumns)
        anovaResults = anovaResults.astype({"Significance": "bool"})

        # One-way ANOVA for each combination of independent and dependent variables
        for dependentVar in self.dataIns.DependentVarNames:
            for independentVar in self.dataIns.IndependentVarNames:
                test_result = pg.anova(
                    data=self.dataIns.Data,
                    dv=dependentVar,
                    between=independentVar,
                    detailed=True,
                )

                # Prepare the result for output
                TempResult = pd.DataFrame(
                    {
                        "Variable": [f"{independentVar} --> {dependentVar}"],
                        "Significance": [
                            True if test_result["p-unc"][0] < self.dataIns.alpha else False
                        ],
                        "df": [test_result["DF"][0]],
                        "F": [test_result["F"][0]],
                        "p-value": [test_result["p-unc"][0]],
                        "np2": [test_result["np2"][0]],
                    }
                )

                anovaResults = pd.concat([anovaResults, TempResult], ignore_index=True)

        # Display the results in table format using PrettyTable
        anovaResultTable.clear()
        anovaResultTable.field_names = ResultsColumns

        for i, row in anovaResults.iterrows():
            anovaResultTable.add_row(row.apply(utility.RoundFloat).values)

        print(Fore.YELLOW + "\nOne-Way ANOVA Results:")
        print(Fore.MAGENTA)
        print(anovaResultTable)
        print(Fore.RESET)

        # Print To Log File
        self.dataIns.Print2Log("\nOne-Way ANOVA Results:", False)
        self.dataIns.Print2Log(str(anovaResultTable), False)

        return anovaResults

    def TwoWayANOVA(self):
        """
        Performs two-way ANOVA on the data based on the independent and dependent variables.

        Parameters:
        ----------
        dataIns : data.Data
            An instance of the Data class containing the data to be analyzed.

        Returns:
        ----------
        anova_results : pd.DataFrame
            A DataFrame containing the results of the two-way ANOVA.
            The DataFrame includes the following columns:
            - Variable: The combination of independent variables and dependent variable being tested.
            - Significance: Whether the interaction is statistically significant (True/False).
            - df: Degrees of freedom.
            - F: The F-statistic value.
            - p-value: The p-value associated with the test.
            - np2: Partial eta squared as a measure of effect size.
        """

        # Ensure there are at least two independent variables
        if len(self.dataIns.IndependentVarNames) < 2:
            print(
                Fore.RED
                + "Two-way ANOVA requires at least two independent variables."
                + Fore.RESET
            )
            self.dataIns.Print2Log(
                "Two-way ANOVA requires at least two independent variables."
            )
            return pd.DataFrame()

        ResultsColumns = ["Variable", "Significance", "df", "F", "p-value", "np2"]
        AnovaResultTable = PrettyTable()

        # Prepare the DataFrame for results
        anova_results = pd.DataFrame(columns=ResultsColumns)
        anova_results = anova_results.astype({"Significance": "bool"})

        # Two-way ANOVA for each combination of two independent variables and each dependent variable
        for dependentVar in self.dataIns.DependentVarNames:
            # Get all combinations of two independent variables
            independent_var_pairs = itertools.combinations(
                self.dataIns.IndependentVarNames, 2
            )
            for pair in independent_var_pairs:
                independentVar1, independentVar2 = pair
                # Perform two-way ANOVA
                try:
                    test_result = pg.anova(
                        data=self.dataIns.Data,
                        dv=dependentVar,
                        between=[independentVar1, independentVar2],
                        detailed=True,
                    )
                    # Check if there is an interaction term
                    # if 'Interaction' in test_result['Source']:
                    test_result = test_result.iloc[0:-1]

                    for index, row in test_result.iterrows():
                        TempResult = pd.DataFrame(
                            {
                                "Variable": [row["Source"]],
                                "Significance": [row["p-unc"] < self.dataIns.alpha],
                                "df": [row["DF"]],
                                "F": [row["F"]],
                                "p-value": [row["p-unc"]],
                                "np2": [row["np2"]],
                            }
                        )
                        anova_results = pd.concat(
                            [anova_results, TempResult], ignore_index=True
                        )
                except Exception as e:
                    print(
                        Fore.RED
                        + f"Error performing two-way ANOVA for {dependentVar}: {str(e)}"
                        + Fore.RESET
                    )
                    self.dataIns.Print2Log(
                        f"Error performing two-way ANOVA for {dependentVar}: {str(e)}"
                    )
                    continue

        # Display the results in table format using PrettyTable
        AnovaResultTable.clear()
        AnovaResultTable.field_names = ResultsColumns

        for i, row in anova_results.iterrows():
            AnovaResultTable.add_row(row.apply(utility.RoundFloat).values)

        print(Fore.YELLOW + "\nTwo-Way ANOVA Results:")
        print(Fore.MAGENTA)
        print(AnovaResultTable)
        print(Fore.RESET)

        # Print To Log File
        self.dataIns.Print2Log("\nTwo-Way ANOVA Results:", False)
        self.dataIns.Print2Log(str(AnovaResultTable), False)

        return anova_results
    
    def RM_ANOVA(self):
        """
        Perform Repeated Measures ANOVA for each dependent variable.
        """
        ResultsColumns = ["Variable", "Significance", "df", "F", "p-value","p-GG-corr", "ng2"]
        rm_anova_results = pd.DataFrame(columns=ResultsColumns)
        rm_anova_results = rm_anova_results.astype({"Significance": "bool"})

        # Create a PrettyTable for displaying results
        RMAnovaResultTable = PrettyTable()
        RMAnovaResultTable.field_names = ResultsColumns

        # Perform Repeated Measures ANOVA for each dependent variable
        for dependentVar in self.dataIns.DependentVarNames:
            try:
                # Perform Repeated Measures ANOVA
                test_result = pg.rm_anova(
                    data=self.dataIns.Data,
                    dv=dependentVar,
                    within=self.dataIns.IndependentVarNames,
                    detailed=True,
                    subject=self.dataIns.subjectName,
                )
                print(test_result)

                # Iterate over the results and store them in the DataFrame
                for index, row in test_result.iterrows():
                    TempResult = pd.DataFrame(
                        {
                            "Variable": [row["Source"]],
                            "Significance": [row["p-unc"] < self.dataIns.alpha],
                            "df": [row["ddof1"]],
                            "F": [row["F"]],
                            "p-value": [row["p-unc"]],
                            "p-GG-corr":[row["p-GG-corr"]],
                            "ng2": [row["ng2"]],
                        }
                    )
                    rm_anova_results = pd.concat(
                        [rm_anova_results, TempResult], ignore_index=True
                    )

            except Exception as e:
                print(
                    Fore.RED
                    + f"Error performing Repeated Measures ANOVA for {dependentVar}: {str(e)}"
                    + Fore.RESET
                )
                self.dataIns.Print2Log(
                    f"Error performing Repeated Measures ANOVA for {dependentVar}: {str(e)}"
                )
                continue

        # Display the results in table format using PrettyTable
        print(Fore.YELLOW + "\nRepeated Measures ANOVA Results:")
        print(Fore.MAGENTA)
        for i, row in rm_anova_results.iterrows():
            RMAnovaResultTable.add_row(row.apply(utility.RoundFloat).values)
        print(RMAnovaResultTable)
        print(Fore.RESET)

        # Print To Log File
        self.dataIns.Print2Log("\nRepeated Measures ANOVA Results:", False)
        self.dataIns.Print2Log(str(RMAnovaResultTable), False)

        return rm_anova_results
