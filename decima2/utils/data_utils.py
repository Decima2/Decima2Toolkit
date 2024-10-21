import pandas as pd
import numpy as np
import warnings

"""
Module for data preprocessing and model evaluation.

This module includes functions for validating inputs, determining target types,
discretizing numerical data, and evaluating machine learning models. It is designed
to facilitate the preparation and evaluation of datasets for machine learning tasks.
"""

def determine_target_type_valid(y):

    """
    Determine the target type after validating the input.

    Validates the target input and determines if it is for classification or regression.

    :param y: Target variable, can be a 1D NumPy array or Pandas Series.
    :return: Type of the target variable, either 'classification' or 'regression'.
    :raises ValueError: If the input is not a valid 1D array or Series.
    """

    is_valid = validate_target(y)
    if not is_valid:
        raise ValueError("Target is not a 1-d numpy array")

    target_type = determine_target_type(y)
    return target_type

def determine_target_type(y, threshold=10):
    """
    Determine if the target variable is for classification or regression.

    Checks the number of unique values in the target variable and whether they are integers.

    :param y: A 1-dimensional NumPy array representing the target variable.
    :param threshold: Number of unique values below which the variable is considered for classification.
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

def assert_size(df,y):

    """Assert the size of the DataFrame and target variable.

    Checks if the dataset is larger than 50,000 elements and warns the user if so.
    Automatically reduces the size of the DataFrame and target variable if necessary.

    :param df: Input DataFrame.
    :param y: Target variable, can be a NumPy array or Pandas Series.
    :return: Resized DataFrame and target variable.
    :raises UserWarning: If the dataset size exceeds 50,000 elements or has too few rows.
    """

    if int(df.shape[0]) * int(df.shape[1]) > 50000:
        warnings.warn("The dataset passed may require a long computation time. Automatically adjusting size of reference set", UserWarning)

        new_row_dimension_1 = int(50000/df.shape[1])
        new_row_dimension_2 = int(df.shape[1]*50)
        new_row_dimension = min(new_row_dimension_1,new_row_dimension_2)
        if new_row_dimension/df.shape[1] < 50:
            new_row_dimension_1 = int(100000/df.shape[1])
            new_row_dimension_2 = int(df.shape[1]*50)
            new_row_dimension = min(new_row_dimension_1,new_row_dimension_2)
        df = df[:new_row_dimension]
        y = y[:new_row_dimension]
        if df.shape[0]/df.shape[1] < 20:
            warnings.warn("Recommend using a model with less features to obtain meaningful explanations", UserWarning)

    elif df.shape[0]/df.shape[1] < 20:
        warnings.warn("Increase the number of test instances to increase the reliability of feature importances ", UserWarning)

    return df, y



def validate_inputs(df,y):

    """
    Validate the input DataFrame and target variable.

    Performs size checks and resets the DataFrame index.

    :param df: Input DataFrame.
    :param y: Target variable, can be a NumPy array or Pandas Series.
    :return: Validated DataFrame and target variable.
    :raises ValueError: If the DataFrame or target variable is invalid.
    """

    df, y = assert_size(df,y)

    #reset index on df
    df = df.reset_index(drop=True)



    is_valid, details = validate_dataframe(df)
    if not is_valid:
        raise ValueError(details)


    validate_dataframe_target(df,y)

    return df, y



def data_discretiser(df):
    """
    Discretizes continuous columns in a DataFrame.

    Returns a new DataFrame with discretized continuous columns and the original DataFrame.

    :param df: Input DataFrame to be discretized.
    :return: Tuple containing the discretized DataFrame, original DataFrame, and target variable.
    """


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
    return discretised_data_frame


def validate_target(y):
    """
    Validate if the input target is a 1-dimensional NumPy array or Pandas Series.

    :param y: Input target to validate.
    :return: True if valid, otherwise False.
    """

    if isinstance(y, np.ndarray) and y.ndim == 1:
        return True
    elif  isinstance(y, pd.Series) and y.ndim == 1:
        return True
    
    else:
        return False


def validate_dataframe_target(df,y):
    """
    Validate that the number of rows in the DataFrame matches the length of the target variable.

    :param df: Input DataFrame.
    :param y: Target variable.
    :raises ValueError: If the number of rows in DataFrame does not match the length of target variable.
    """
    if len(df) != len(y):
        raise ValueError(f"The number of rows in X ({len(X)}) must match the length of y ({len(y)}).")



def validate_dataframe(df, check_empty=True, check_column_types=True, check_duplicates=False):
    """
    Validate a Pandas DataFrame for common data quality issues.

    Checks for NaN values, infinite values, duplicates, and the integrity of the DataFrame structure.

    :param df: The DataFrame to validate.
    :param check_empty: If True, checks if the DataFrame is empty.
    :param check_column_types: If True, checks if all columns are numeric.
    :param check_duplicates: If True, checks for duplicate rows.
    :return: Tuple of (bool, dict) indicating validity and details of validation errors.
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




def determine_data_types(df):
    """
    Determine which columns in a DataFrame are continuous or discrete.

    Analyzes each column and classifies it as either continuous or discrete based on its data type.

    :param df: Input DataFrame.
    :return: Tuple of lists with 'continuous' and 'discrete' column names.
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
    Discretize numerical data into a specified number of categories (bins).

    Returns discretized data along with the bounds of each bin.

    :param data: Numerical data to be discretized, can be a list or Pandas Series.
    :param n_categories: Number of categories (bins) to create.
    :return: Tuple containing the discretized data and bin bounds.
    :raises ValueError: If n_categories is less than 1.
    """

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



def validate_selected_feature(X,feature_name, number_of_categories):
    """
    Validate the selected feature and determine the appropriate number of categories for discretization.

    :param X: Input DataFrame containing features.
    :param feature_name: Name of the feature to validate.
    :param number_of_categories: Number of categories to be used for discretization.
    :return: Adjusted number of categories.
    :raises ValueError: If the feature_name is not in DataFrame columns.
    :raises TypeError: If number_of_categories is not an int when not set to 'auto'.
    """
    if feature_name not in X.columns:
        raise ValueError(f"{feature_name} is not one of the dataframe columns")

    if number_of_categories != 'auto':
        if type(number_of_categories) != int:
            raise TypeError("number_of_categories must be an int if not set to auto")
        k = int(np.ceil(np.log2(X.shape[0]) + 1))
        if number_of_categories > k:
            warnings.warn("Number of categories may be too high to return meaningful results, automatically adjusting number of categories ", UserWarning)
            return 'auto'
        else:
            return number_of_categories
    else:
        return number_of_categories

def format_feature_categories(feature_categories, feature_name):
    """
    Format the category names for a feature based on its bins.

    :param feature_categories: List of bin categories.
    :param feature_name: Name of the feature being categorized.
    :return: List of formatted category names.
    """
    formatted_feature_categories = []
    for i in feature_categories:
        new_i = [str(round(x,3)) for x in i]
        new_category_name = new_i [0] + " >= " + feature_name + " < " + new_i[1]
        formatted_feature_categories.append(new_category_name)
    return formatted_feature_categories



def discretise_selected_feature(feature_values,number_of_categories,feature_name):
    """
    Discretize a specific feature based on the number of desired categories.

    Automatically adjusts the number of categories if set to 'auto'.

    :param feature_values: Values of the feature to be discretized.
    :param number_of_categories: Number of categories for discretization.
    :param feature_name: Name of the feature.
    :return: Tuple containing discretized values and formatted feature categories.
    """
    if number_of_categories == 'auto':
        unique_values = feature_values.nunique()
        if unique_values <= 10:  # Arbitrary threshold for categorical data
            #discretised_values, feature_categories = data_utils_temp.discretise_data(feature_values, unique_values)

            discretised_values = feature_values.values
            unique_values = np.unique(feature_values)
            formatted_feature_categories = [feature_name + ' = ' +str(i) for i in unique_values]
        else:
            discretised_values, feature_categories = discretise_data(feature_values, 5)

            formatted_feature_categories = format_feature_categories(feature_categories, feature_name)

    else:
        if number_of_categories == feature_values.nunique():
            discretised_values = feature_values.values
            unique_values = np.unique(feature_values)
            formatted_feature_categories = [feature_name + ' = ' +str(i) for i in unique_values]

        else:
            discretised_values, feature_categories = discretise_data(feature_values, number_of_categories)
            formatted_feature_categories = format_feature_categories(feature_categories, feature_name)

    return discretised_values, formatted_feature_categories 


