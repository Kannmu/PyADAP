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
import scipy


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
    Checks if a directory exists at the specified file path and creates it if it doesn't exist.

    Parameters:
    ----------
    - file_path : str
        The path of the directory to be created.

    Returns:
    ----------
    - None

    """
    # Check if the directory exists
    if not os.path.exists(FolderPath):
        # Create the directory
        os.makedirs(FolderPath)
        print(f"Directory {FolderPath} created successfully.")
    else:
        print(f"Directory {FolderPath} already exists.")


def RoundFloat(X):
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
        return round(X, 6)
    else:
        return X