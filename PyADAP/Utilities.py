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
import numpy as np
import scipy
import os

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

def CreateSaveFolder(file_path):
    # 检查目录是否存在
    if not os.path.exists(file_path):
        # 创建目录
        os.makedirs(file_path)
        print(f"目录 {file_path} 创建成功")
    else:
        print(f"目录 {file_path} 已存在")