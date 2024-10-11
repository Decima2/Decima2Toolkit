import pandas as pd
import numpy as np
import warnings

"""
Module: data_processing

This module provides functions for validating, processing, and evaluating datasets for machine learning models. 
It includes methods to handle target variables, assess DataFrame structures, and categorize features based on their types.

Dependencies:
-------------
- pandas: For handling DataFrame structures and data manipulation.
- numpy: For numerical operations and array manipulations.
- warnings: To issue warnings about potential issues in dataset processing.

Functions:
----------

1. target_handler(y):
    Validates the target variable `y` and determines its type (classification or regression).
    
    Parameters:
    ----------
    - y: The target variable to validate (1D NumPy array or Pandas Series).
    
    Returns:
    --------
    - target_type: A string indicating whether the target is for 'classification' or 'regression'.

    Raises:
    -------
    - ValueError: If `y` is not a valid 1D array.

2. assert_size(df, y):
    Checks the size of the DataFrame and target variable `y`. Issues warnings if the size may lead to long computation times 
    and adjusts their sizes if necessary.

    Parameters:
    ----------
    - df: The DataFrame to check.
    - y: The target variable to check.

    Returns:
    --------
    - df, y: Adjusted DataFrame and target variable if applicable.

3. data_discretiser(df, y):
    Validates the input DataFrame and target variable. It discretizes continuous features into categories and returns a 
    processed DataFrame.

    Parameters:
    ----------
    - df: The input DataFrame.
    - y: The target variable.

    Returns:
    --------
    - discretised_data_frame: A DataFrame with discretized features.
    - df: The original DataFrame.
    - y: The target variable.

4. validate_target(y):
    Validates whether the target variable is a 1D NumPy array or Pandas Series.

    Parameters:
    ----------
    - y: The target variable to validate.

    Returns:
    --------
    - True if valid, False otherwise.

5. determine_target_type(y, threshold=10):
    Determines if the target variable is for classification or regression based on the number of unique values.

    Parameters:
    ----------
    - y: A 1D NumPy array representing the target variable.
    - threshold: The threshold for classification (default is 10).

    Returns:
    --------
    - 'classification' or 'regression'.

6. validate_dataframe_target(df, y):
    Checks if the lengths of the DataFrame and target variable match.

    Parameters:
    ----------
    - df: The input DataFrame.
    - y: The target variable.

    Raises:
    -------
    - ValueError: If the lengths do not match.

7. validate_dataframe(df, check_empty=True, check_column_types=True, check_duplicates=False):
    Validates a pandas DataFrame by checking for NaN, infinite values, and optional conditions such as emptiness 
    and duplicates.

    Parameters:
    ----------
    - df: The DataFrame to validate.
    - check_empty: Whether to check if the DataFrame is empty.
    - check_column_types: Whether to check for valid column types.
    - check_duplicates: Whether to check for duplicate rows.

    Returns:
    --------
    - (bool, dict): A tuple indicating validity and a dictionary with validation details.

8. is_categorical(column):
    Checks if a column is of categorical type.

    Parameters:
    ----------
    - column: A DataFrame column to check.

    Returns:
    --------
    - True if categorical, False otherwise.

9. is_continuous(column):
    Checks if a column is of numeric type (continuous feature).

    Parameters:
    ----------
    - column: A DataFrame column to check.

    Returns:
    --------
    - True if continuous, False otherwise.

10. ModelEvaluator:
    A class for evaluating machine learning models.
    
    Methods:
    --------
    - __init__(model): Initializes the evaluator with a model.
    - evaluate(X_test, y_test): Evaluates the model on the given test set and returns a score or metric.

11. determine_data_types(df):
    Determines which columns in a DataFrame are continuous or discrete.

    Parameters:
    ----------
    - df: The input DataFrame.

    Returns:
    --------
    - (continuous_columns, discrete_columns): A tuple containing lists of column names.

12. discretise_data(data, n_categories):
    Discretizes numerical data into specified categories (bins) and returns the discretized data along with bin bounds.

    Parameters:
    ----------
    - data: Numerical data to be discretized.
    - n_categories: Number of categories (bins) to create.

    Returns:
    --------
    - discretized_data: List of discretized category labels.
    - bin_bounds: List of tuples containing (lower_bound, upper_bound) for each bin.

General Notes:
--------------
- The functions provided in this module facilitate the preprocessing of data for machine learning, ensuring that 
  datasets are properly formatted and validated before model training and evaluation.
- Warnings are issued for potential issues such as large dataset sizes or invalid feature types.

"""



def determine_target_type_valid(y):


    is_valid = validate_target(y)
    if not is_valid:
        raise ValueError("Target is not a 1-d numpy array")

    target_type = determine_target_type(y)
    return target_type

def assert_size(df,y):


    if int(df.shape[0]) * int(df.shape[1]) > 10000:
        warnings.warn("The dataset passed may require a long computation time. Automatically adjusting size of reference set", UserWarning)

        new_row_dimension_1 = int(10000/df.shape[1])
        new_row_dimension_2 = int(df.shape[1]*50)
        new_row_dimension = min(new_row_dimension_1,new_row_dimension_2)
        df = df[:new_row_dimension]
        y = y[:new_row_dimension]
        if df.shape[0]/df.shape[1] < 20:
            warnings.warn("Reccommend using a model with less features to obtain meaningful explanations", UserWarning)

    elif df.shape[0]/df.shape[1] < 20:
        warnings.warn("Increase the number of test instances to increase the reliability of feature importances ", UserWarning)

    return df, y


def data_discretiser(df,y):
    """"
    function first validates DataFrame by calling validate_dataframe. If dataframe is not valid, raise
    an error. 



    """

    df, y = assert_size(df,y)

    is_valid, details = validate_dataframe(df)
    if not is_valid:
        raise ValueError(details)


    validate_dataframe_target(df,y)

    discretised_data_frame = pd.DataFrame()
    continuous_columns, discrete_columns = determine_data_types(df)
    #set up dictionaries for feature category names to be used with grouped feature importance
    discretised_category_names = {}

    for column in discrete_columns:
        discretised_data_frame[column] = df[column]

    for column in continuous_columns:
        discretised_values, category_names = discretise_data(df[column].values, 2)
        discretised_category_names[column] = category_names
        discretised_data_frame[column] = discretised_values

    # not yet returning discretised_category_names, or one_hot_encoded_names leave this for grouped feature importance
    return discretised_data_frame, df, y 


def validate_target(y):
    """
    Check if the input y is a valid 1-dimensional NumPy array or 1D Pandas Series
    
    :param y: The input to check.
    :return: True if y is a valid 1D NumPy array, False otherwise.
    """
    if isinstance(y, np.ndarray) and y.ndim == 1:
        return True
    elif  isinstance(y, pd.Series) and y.ndim == 1:
        return True
    
    else:
        return False


import numpy as np

def determine_target_type(y, threshold=10):
    """
    Determine whether the target variable is for classification or regression.
    
    :param y: A 1-dimensional NumPy array representing the target variable.
    :param threshold: The number of unique values below which the array is considered for classification.
                      Default is 10 (for cases like binary/multi-class classification).
    :return: 'classification' if the target is discrete, 'regression' if continuous.
    """
    unique_values = np.unique(y)
    
    # Check if all unique values are integers (common in classification tasks)
    all_integers = np.all(np.mod(unique_values, 1) == 0)
    
    # Classification: if there are relatively few unique values and they are integers
    if len(unique_values) <= threshold and all_integers:
        return 'classification'
    
    # Otherwise, it's a regression problem
    return 'regression'

def validate_dataframe_target(df,y):
    if len(df) != len(y):
        raise ValueError(f"The number of rows in X ({len(X)}) must match the length of y ({len(y)}).")



def validate_dataframe(df, check_empty=True, check_column_types=True, check_duplicates=False):
    """
    Validates a pandas DataFrame by checking for NaN values, infinite values,
    and optionally checks if it's empty or contains duplicate rows.

    Parameters:
    df (pd.DataFrame): The DataFrame to validate.
    check_empty (bool): If True, will check if the DataFrame is empty.
    check_duplicates (bool): If True, will check for duplicate rows.

    Returns:
    bool: True if the DataFrame is valid, False otherwise.
    dict: A dictionary containing details about validation failures.
    """

    dimension_flag = 0 
    validation_result = {
        "valid": True,
        "errors": []
    }



    if not isinstance(df, pd.DataFrame):
        validation_result["valid"] = False
        validation_result["errors"].append("Input X must be a Pandas DataFrame")

    # Check if DataFrame is empty
    elif check_empty and df.empty:
        validation_result["valid"] = False
        validation_result["errors"].append("The DataFrame is empty.")

    # Check for NaN values
    elif df.isnull().values.any():
        validation_result["valid"] = False
        validation_result["errors"].append("The DataFrame contains NaN values.")

    # Check for infinite values
    elif np.isinf(df.values).any():
        validation_result["valid"] = False
        validation_result["errors"].append("The DataFrame contains infinite values.")

    # Check for duplicate rows
    elif check_duplicates and df.duplicated().any():
        validation_result["valid"] = False
        validation_result["errors"].append("The DataFrame contains duplicate rows.")

    # check that columns are same shape as dataframe: 
    elif df.columns.shape[0] != df.shape[1]:
        validation_result["valid"] = False
        validation_result["errors"].append("The DataFrame columns do not match the data")



    # Check if columns are numerical
    elif check_column_types:
        invalid_columns = []
        for col in df.columns:
            if not (is_numeric(df[col])):
                invalid_columns.append(col)

        if invalid_columns:
            validation_result["valid"] = False
            validation_result["errors"].append(f"The following columns are not numerical {invalid_columns}")

    

    return validation_result["valid"], validation_result


def is_numeric(column):
    """Check if a column is of numeric type."""

    return pd.api.types.is_numeric_dtype(column)






class ModelEvaluator:
    def __init__(self, model):
        """
        Initialize the evaluator with the model.
        :param model: A pre-trained machine learning model (e.g., scikit-learn, XGBoost, etc.)
        """
        self.model = model

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the given test set (X_test, y_test).
        :param X_test: Features of the test set.
        :param y_test: Labels/targets of the test set.
        :return: A score or metric depending on the model.
        """
        if hasattr(self.model, 'score'):
            # For scikit-learn models or others that have a 'score' method
            return self.model.score(X_test, y_test)
        elif hasattr(self.model, 'evaluate'):
            # For deep learning models like Keras with an 'evaluate' method
            return self.model.evaluate(X_test, y_test, verbose=0)
        else:
            raise TypeError("The provided model doesn't have a 'score' or 'evaluate' method")


def determine_data_types(df):
    """
    Determines which columns in a DataFrame are continuous or discrete.

    Args:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - data_types (dict): A dictionary with 'continuous' and 'discrete' as keys, 
                         containing lists of column names.
    """
    continuous_columns = []
    discrete_columns = []

    for col in df.columns:
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            # If the column has float values, we assume it to be continuous
            if pd.api.types.is_float_dtype(df[col]):
                continuous_columns.append(col)
            # If it's integer and has fewer unique values, it's discrete (e.g., categories)
            elif pd.api.types.is_integer_dtype(df[col]):
                unique_values = df[col].nunique()
                if unique_values <= 10:  # Arbitrary threshold for categorical data
                    discrete_columns.append(col)
                else:
                    continuous_columns.append(col)
        # For non-numeric types, assume it's discrete (e.g., strings, categories)
        else:
            discrete_columns.append(col)

    return (continuous_columns, discrete_columns)

def discretise_data(data, n_categories):
    """
    Discretizes numerical data into a specified number of categories (bins)
    with roughly equal amounts of data in each bin, and returns the discretized data
    along with the lower and upper bounds of each bin.

    Args:
    - data (list or pd.Series): Numerical data to be discretized.
    - n_categories (int): Number of categories (bins) to create.

    Returns:
    - discretized_data (list): List of discretized category labels.
    - bin_bounds (list of tuples): List of tuples containing (lower_bound, upper_bound) for each bin.
    """
    # Convert to a pandas Series if the input is a list
    if isinstance(data, list):
        data = pd.Series(data)
    
    # Ensure n_categories is greater than 1
    if n_categories < 1:
        raise ValueError("n_categories must be at least 1")

    # Discretize the data into quantile bins and get the bin edges
    discretized_data, bin_edges = pd.qcut(data, q=n_categories, retbins=True, labels=False, duplicates='drop')

    # Create list of bin bounds as tuples (lower_bound, upper_bound)
    bin_bounds = [(bin_edges[i], bin_edges[i + 1]) for i in range(len(bin_edges) - 1)]

    return discretized_data.tolist(), bin_bounds



