
import pandas as pd
import numpy as np
import requests

# Replace this URL with the Google Cloud Function URL you get after deployment
CLOUD_FUNCTION_URL_MODEL_EXPLANATION = "https://europe-west2-decima2.cloudfunctions.net/call_private_model_explanation"

"""
Module: feature_importance

This module computes the feature importance of a machine learning model using permutation importance 
techniques, specifically tailored for categorical data. It includes functions for calculating 
feature importance based on model evaluation scores and measuring similarity using Jaccard metrics.

Dependencies:
-------------
- pandas: For data manipulation and handling DataFrames.
- numpy: For numerical computations and array manipulations.

Functions:
----------
1. feature_importance(model_evaluator, X, X_d, y):
   Computes the permutation feature importance for a given model and dataset.

   Args:
   - model_evaluator: An instance of ModelEvaluator that provides a method to evaluate the model.
   - X (pd.DataFrame): The feature dataset used for evaluation.
   - X_d (pd.DataFrame): A version of the feature dataset used for computing similarities.
   - y (pd.Series or np.ndarray): The target variable.

   Returns:
   - dict: A dictionary containing feature names as keys and their computed importances as values.

2. categorical_jaccard_similarity(test_data, reference_points, priority_index, priority_weight=1000000):
   Finds the most similar data points to each reference point using Jaccard similarity for binary data, 
   with the ability to prioritize a specific feature.

   Args:
   - test_data (numpy.ndarray): The test dataset (shape: [n_samples, n_features]).
   - reference_points (numpy.ndarray): The reference data points (shape: [n_refs, n_features]).
   - priority_index (int): The index of the feature to prioritize in similarity calculations.
   - priority_weight (float): A weighting factor for the prioritized index.

   Returns:
   - numpy.ndarray: Indices of the most similar test data points for each reference point.
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

        X_intervened[col] = (X_intervened[col] + 1) % len(feature_names)

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

#def feature_importance(model_evaluator, X, X_d, y):
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
    """feature_names = X_d.columns

    # Calculate the baseline score
    baseline_score = model_evaluator.evaluate(X, y)

    # Initialize a dictionary to hold the importance scores
    importances = {name: 0 for name in feature_names}

    # Permute each feature and compute the new score
    for i, col in enumerate(feature_names):
        #print(f"Processing {col}...")

        # Copy the original data and permute the feature
        X_intervened = X_d.copy()
        X_intervened[col] = (X_intervened[col] + 1) % len(feature_names)

        # Get most similar data points
        most_similar_datapoints = categorical_jaccard_similarity(X_d.values, X_intervened.values, i)

        # Get realistic samples
        X_realistic = X.iloc[most_similar_datapoints]

        # Compute the new score
        new_score = model_evaluator.evaluate(X_realistic, y)

        # Calculate importance as the difference in scores
        importances[col] += (baseline_score - new_score)

    return importances"""


#def categorical_jaccard_similarity(test_data, reference_points, priority_index, priority_weight=1000000):
    """
    Find the most similar data points to each reference point using Jaccard similarity 
    for binary data with priority weighting.

    Args:
    - test_data (numpy.ndarray): The test dataset (shape: [n_samples, n_features]).
    - reference_points (numpy.ndarray): The reference data points (shape: [n_refs, n_features]).
    - priority_index (int): The index to prioritize for all reference points.
    - priority_weight (float): The weight to apply to the priority index.

    Returns:
    - most_similar_indices (numpy.ndarray): Indices of the most similar test data points for each reference point.
    """
    # Calculate intersection and union using matrix operations
    """intersection = np.dot(reference_points, test_data.T)  # Shape: (n_refs, n_samples)
    reference_counts = np.sum(reference_points, axis=1, keepdims=True)  # Shape: (n_refs, 1)
    test_counts = np.sum(test_data, axis=1, keepdims=True)  # Shape: (n_samples, 1)

    union = reference_counts + test_counts.T - intersection  # Shape: (n_refs, n_samples)

    # Jaccard similarity
    jaccard_similarities = intersection / (union + 1e-6)  # Avoid division by zero

    # Priority adjustment
    priority_adjustment = reference_points[:, priority_index] * priority_weight
    jaccard_similarities += priority_adjustment[:, None]  # Broadcasting

    # Find the most similar indices
    most_similar_indices = np.argmax(jaccard_similarities, axis=1)

    return most_similar_indices"""