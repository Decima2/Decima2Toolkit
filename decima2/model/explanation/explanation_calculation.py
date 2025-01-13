
import pandas as pd
import numpy as np
import requests
import random

# Replace this URL with the Google Cloud Function URL you get after deployment
CLOUD_FUNCTION_URL_MODEL_EXPLANATION = "https://europe-west2-decima2.cloudfunctions.net/call_private_model_explanation"
CLOUD_FUNCTION_URL_GROUPED_EXPLANATION = "https://europe-west2-decima2.cloudfunctions.net/call_private_grouped_explanation"


#CLOUD_FUNCTION_URL_MODEL_EXPLANATION = "https://model-explanation-api-gateway-1y81cisz.nw.gateway.dev/call_private_model_explanation"
#CLOUD_FUNCTION_URL_GROUPED_EXPLANATION = "https://grouped-explanation-api-gateway-1y81cisz.nw.gateway.dev/call_private_grouped_explanation"


"""
Module: model_explanation_calculation

This module computes the explanations machine learning model using different techniques including feature importance and grouped feature importance
It includes functions for calculating feature importance of a model and of groups of test data. 

Dependencies:
-------------
- pandas: For data manipulation and handling DataFrames.
- numpy: For numerical computations and array manipulations.

Functions:
----------
1. model_feature_importance(model_evaluator, X, X_d, y):
   Computes the permutation feature importance for a given model and dataset.

   Args:
   - model_evaluator: An instance of ModelEvaluator that provides a method to evaluate the model.
   - X (pd.DataFrame): The feature dataset used for evaluation.
   - X_d (pd.DataFrame): A version of the feature dataset used for computing similarities.
   - y (pd.Series or np.ndarray): The target variable.

   Returns:
   - dict: A dictionary containing feature names as keys and their computed importances as values.

2. grouped_feature_importance(model_evaluator, X, X_d, y):
   Computes the permutation feature importance for a given model and dataset.

    Args:
    - model: A fitted model (e.g., from scikit-learn).
    - X (pd.DataFrame): Original Feature dataset.
    - y (pd.Series or np.ndarray): Original target vector
    - X_d (pd.DataFrame): Encoded Feature dataset
    - X_d_selected (pd.DataFrame): Encoded group of features 
    - X_selected (pd.DataFrame): group of features
    - y_selected (pd.Series or np.ndarray) group of targets

    Returns:
    - feature_importances (dict): Dictionary containing feature names and their importances.

3. public_model_feature_importance
    Public function that interacts with the Google Cloud Function, invoked by model_feature_imporance, our novel
    similarity algorithm is hosted in Google Cloud
    - X_d(pd.DataFrame): Dataset used as representation of data manifold
    - X_i (pd.DataFrame): Intervened dataset used to find reference points on manifold
    - i (int) index of column under investigation
    Returns: most_similar_datapoints (list) list of indexes representing datapoints on the data manifold

4. public_grouped_feature_importance
    Public function that interacts with the Google Cloud Function, invoked by grouped_feature_importance our novel
    similarity algorithm is hosted in Google Cloud
    - X_d(pd.DataFrame): Dataset used as representation of data manifold
    - X_i (pd.DataFrame): Intervened dataset used to find reference points on manifold
    - i (int) index of column under investigation
    Returns: most_similar_datapoints (list) list of indexes representing datapoints on the data manifold
"""


def model_feature_importance(model_evaluator, X, X_d, y):
    """
    Computes the permutation feature importance for a given model and dataset.

    Args:
    - model: A fitted model (e.g., from scikit-learn).
    - X (pd.DataFrame): Feature dataset.
    - y (pd.Series or np.ndarray): Target variable.

    Returns:
    - feature_importances (dict): Dictionary containing feature names and their importances.
    """
    # Ensure X and y are in the right format
    #feature_names = X_d.columns if isinstance(X, pd.DataFrame) else [f'Feature {i}' for i in range(X_d.shape[1])]
    
   

    feature_names = X_d.columns

    # Calculate the baseline score
    baseline_score = model_evaluator.evaluate(X, y)

    # Initialize a dictionary to hold the importance scores
    importances = {name: 0 for name in feature_names}

    # Permute each feature and compute the new score
    for i, col in enumerate(feature_names):
        #print(f"Processing {col}...")

        # Copy the original data and permute the feature
        X_intervened = X_d.copy()

        number_uniques = X_d[col].nunique()
        
        if number_uniques < 3: 
            X_intervened[col] = (X_intervened[col] + 1) % number_uniques
        else:
            random_array = [random.randint(1, number_uniques-1) for _ in range(X_intervened.shape[0])]
            X_intervened[col] = (X_intervened[col] + random_array) % number_uniques

        # Get most similar data points
        most_similar_datapoints = public_model_feature_importance(X_d.values, X_intervened.values, i)

        most_similar_datapoints = np.asarray(most_similar_datapoints)

        # Get realistic samples
        X_realistic = X.iloc[most_similar_datapoints]
        
        # Compute the new score
        new_score = model_evaluator.evaluate(X_realistic, y)

        # Calculate importance as the difference in scores
        importances[col] += (baseline_score - new_score)

    return importances


def grouped_feature_importance(model_evaluator, X, y, X_d, X_d_selected, X_selected, y_selected):

    """
    Computes the permutation feature importance for a given model and dataset.

    Args:
    - model: A fitted model (e.g., from scikit-learn).
    - X (pd.DataFrame): Original Feature dataset.
    - y (pd.Series or np.ndarray): Original target vector
    - X_d (pd.DataFrame): Encoded Feature dataset
    - X_d_selected (pd.DataFrame): Encoded group of features 
    - X_selected (pd.DataFrame): group of features
    - y_selected (pd.Series or np.ndarray) group of targets

    Returns:
    - feature_importances (dict): Dictionary containing feature names and their importances.
    """
   
    feature_names = X_d.columns


    # Calculate the baseline score

    baseline_score = model_evaluator.evaluate(X_selected, y_selected)

    # Initialize a dictionary to hold the importance scores
    importances = {name: 0 for name in feature_names}

    # Permute each feature and compute the new score
    for i, col in enumerate(feature_names):
        #print(f"Processing {col}...")

        # Copy the original data and permute the feature
        X_intervened = X_d_selected.copy()
        number_uniques = X_d[col].nunique()
        if number_uniques < 3: 
            X_intervened[col] = (X_intervened[col] + 1) % number_uniques
        else:
            random_array = [random.randint(1, number_uniques-1) for _ in range(X_intervened.shape[0])]
            X_intervened[col] = (X_intervened[col] + random_array) % number_uniques
        most_similar_datapoints = public_grouped_feature_importance(X_d.values, X_intervened.values, i)
        most_similar_datapoints = np.asarray(most_similar_datapoints)

        # Get realistic samples
        X_realistic = X.iloc[most_similar_datapoints]

        y_new = [y[i] for i in most_similar_datapoints]

        
        # Compute the new score
        new_score = model_evaluator.evaluate(X_realistic, y_selected)


        # Calculate importance as the difference in scores
        importances[col] += (baseline_score - new_score)

    return importances


def public_model_feature_importance(X_d=0,X_i=0,index=-1):
    # Public function that interacts with the Google Cloud Function
    try:
        # Send a POST request to the Google Cloud Function
        response = requests.post(CLOUD_FUNCTION_URL_MODEL_EXPLANATION, json={"X_d" : X_d.tolist(), "X_i" : X_i.tolist(), "index": index})
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            return data['message']
        else:
            data = response.json()
            return data['error']
    except Exception as e:
        return f"Error: {e}"


def public_grouped_feature_importance(X_d=0,X_i=0,index=-1):
    # Public function that interacts with the Google Cloud Function
    try:
        # Send a POST request to the Google Cloud Function
        response = requests.post(CLOUD_FUNCTION_URL_GROUPED_EXPLANATION, json={"X_d" : X_d.tolist(), "X_i" : X_i.tolist(), "index": index})
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            return data['message']
        else:
            data = response.json()
            return data['error']
    except Exception as e:
        return f"Error: {e}"
