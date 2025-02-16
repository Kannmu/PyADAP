"""
PyADAP
=====

PyADAP: Python Automated Data Analysis Pipeline

Utilities

Author: Kannmu
Date: 2024/3/31
License: MIT License
Repository: https://github.com/Kannmu/PyADAP

"""
# Import necessary modules

import os

import numpy as np
from colorama import Back, Fore, Style, init
import PyADAP.Data as data
def norm(X):
    """
    Data normalization, map data into range of [0,1]
    
    Parameters
    ----------
    X: array-like
        Input data
    
    Returns
    ----------
    X: numpy array
        normalized data
    """
    X = np.asarray(X)
    if np.max(X) == np.min(X):
        return X
    else:
        _range = np.max(X) - np.min(X)
        return (X - np.min(X)) / _range

def stand(X):
    """
    Data standardization, map data into distribution of mean = 0, std = 1
    
    Parameters
    ----------
    X: array-like
        Input data
    
    Returns
    ----------
    X: numpy array
        Standardized data
    """
    X = np.asarray(X)
    mu = np.mean(X)
    sigma = np.std(X)
    return (X - mu) / (sigma + 1e-20)

def CreateFolder(FolderPath):
    """
    Checks if a directory exists at the specified file path. 
    If it exists, deletes all subdirectories and files within it. 
    Then creates a new empty directory at the same file path.

    Parameters:
    ----------
    - FolderPath : str
        The path of the directory to be created.

    Returns:
    ----------
    - None

    """
    # Check if the directory exists
    if os.path.exists(FolderPath):
        # Delete all subdirectories and files within the directory
        for root, dirs, files in os.walk(FolderPath, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        print(Fore.RED +f"\nAll subdirectories and files in {FolderPath} have been deleted.\n")
    # Create the directory
    os.makedirs(FolderPath, exist_ok=True)
    print(Fore.RED +f"Directory {FolderPath} created successfully.")

def RoundFloat(X,Digits:int = 6):
    """
    Round a  float number to 6 decimal places.
    
    Parameters
    ----------
    X: float
        Input float number
    
    Returns
    ----------
    X: float
        Rounded float number
    
    """
    if isinstance(X, float):
        return round(X, Digits)
    else:
        return X
    
