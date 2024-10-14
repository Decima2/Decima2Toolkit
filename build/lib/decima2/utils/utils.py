import pandas as pd
import numpy as np


"""
Module: utils

This module provides utility functions for processing and handling feature importance data in machine learning models. 
It includes functions to extract feature names, calculate their importances, and sort them in descending order of 
importance.

Dependencies:
-------------
- pandas: Used for handling DataFrame structures and sorting feature importances.
- numpy: Used for numerical computations, such as calculating the mean and absolute values of feature importances.

Functions:
----------

1. feature_names(X, attributions):
    This function generates a DataFrame of features along with their respective importances. The importances are averaged 
    over multiple repetitions, and the result is sorted in descending order of importance.

    Parameters:
    -----------
    - X: DataFrame of features.
    - attributions: A list or array-like object of feature importances, where each entry corresponds to a feature in X.

    Returns:
    --------
    - DataFrame: A pandas DataFrame containing two columns: 'Feature' and 'Importance', sorted by importance in descending order.
    
    Example Usage:
    --------------
    ```python
    feature_importances = feature_names(X, attributions)
    print(feature_importances)
    ```

2. sort_features_by_importance(importances, feature_names):
    This function takes a list of feature importances and their corresponding feature names and returns both sorted 
    in descending order of importance.

    Parameters:
    -----------
    - importances: List of feature importances.
    - feature_names: List of feature names.

    Returns:
    --------
    - Tuple: A tuple containing two elements:
        1. Sorted list of importances.
        2. Sorted list of feature names corresponding to the sorted importances.

    Example Usage:
    --------------
    ```python
    sorted_importances, sorted_feature_names = sort_features_by_importance(importances, feature_names)
    print(sorted_importances)
    print(sorted_feature_names)
    ```

General Notes:
--------------
- The `feature_names` function is useful for transforming the output of feature importance calculations into a readable 
  DataFrame format for further analysis or visualization.
- The `sort_features_by_importance` function helps in ranking features based on their significance to the model's 
  predictions.

Warnings:
---------
- Ensure that the length of the `attributions` matches the number of columns in the feature DataFrame `X`.
"""


def feature_names(X,attributions):
       
    
    feature_names = X.columns

    # Average the importances over all repeats
    feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': [round(np.abs(np.mean(attributions[i])),5) for i in range(0,len(feature_names))]
    })

    return feature_importances.sort_values(by='Importance', ascending=False)


def sort_features_by_importance(importances, feature_names):
    """
    Sort feature importances and corresponding feature names in descending order of importance.

    :param importances: List of feature importances.
    :param feature_names: List of feature names.
    :return: Tuple of sorted importances and corresponding feature names.
    """
    # Pair the importances and feature names
    paired = list(zip(importances, feature_names))
    
    # Sort by importance in descending order
    sorted_paired = sorted(paired, key=lambda x: x[0], reverse=True)
    
    # Unzip the sorted pairs into separate lists
    sorted_importances, sorted_feature_names = zip(*sorted_paired)
    
    return sorted_importances, sorted_feature_names


