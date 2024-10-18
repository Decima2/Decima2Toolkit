import pytest
import pandas as pd
import numpy as np
import warnings

from decima2.utils.data_utils  import (
    assert_size,
    data_discretiser,
    validate_target,
    determine_target_type,
    validate_dataframe_target,
    validate_dataframe,
    is_numeric,
    determine_data_types,
    discretise_data,
    validate_selected_feature,
    format_feature_categories,
    discretise_selected_feature
)


# Sample data for testing
@pytest.fixture
def sample_data():
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [1.1, 2.2, 3.3, 4.4, 5.5],
        'C': ['a', 'b', 'c', 'd', 'e']
    })
    y_classification = pd.Series([0, 1, 0, 1, 0])  # For classification
    y_regression = pd.Series([1.2, 2.4, 3.2, 4.0, 5.0])  # For regression
    return df, y_classification, y_regression

def test_validate_target_valid(sample_data):
    _, y_classification, _ = sample_data
    assert validate_target(y_classification) is True

def test_validate_target_invalid():
    assert validate_target([1, 2, 3]) is False  # List is not valid

def test_determine_target_type_classification(sample_data):
    _, y_classification, _ = sample_data
    assert determine_target_type(y_classification) == 'classification'

def test_determine_target_type_regression(sample_data):
    _, _, y_regression = sample_data
    assert determine_target_type(y_regression) == 'regression'

def test_assert_size_large_dataframe():
    # Create a large DataFrame (for example, 2000 rows and 30 columns)
    n_rows = 2000
    n_cols = 30
    large_df = pd.DataFrame(np.random.rand(n_rows, n_cols))
    
    # Create a target variable with the same number of rows
    y = np.random.rand(n_rows)

    # Use a context manager to catch warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # Capture all warnings
        resized_df, resized_y = assert_size(large_df, y)

        # Check if any warnings were raised
        assert len(w) > 0  # Ensure that at least one warning was captured
        assert "The dataset passed may require a long computation time." in str(w[-1].message)

    # Ensure the DataFrame size is reduced as expected
    assert resized_df.shape[0] <= 50000 / n_cols
    assert len(resized_y) <= 50000 / n_cols  # Ensure the target variable size is reduced accordingly

    # Further checks can be added based on expected dimensions after resizing
    assert resized_df.shape[0] == resized_y.shape[0]  # Ensure the sizes of df and y match

    # Check if the dimensions of resized_df is still less than original
    assert resized_df.shape[0] < n_rows


def test_data_discretiser1():
    df = pd.DataFrame({
        'A': [1, 2, 3, 1],
        'B': [1.1, 2.2, 3.3, 4.4],
    })
    y_classification = pd.Series([0, 1, 0, 1])
    discretised_df = data_discretiser(df)
    assert 'B' in discretised_df.columns  # Check if column 'A' is still present
    assert discretised_df['B'].nunique() == 2  # Check if 'A' was discretized into 2 categories

def test_validate_dataframe_valid():
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [1.1, 2.2, 3.3, 4.4, 5.5],
    })
    is_valid, details = validate_dataframe(df)
    assert is_valid is True

def test_validate_dataframe_invalid():
    df_invalid = pd.DataFrame({
        'A': [1, 2, 3, np.nan],
        'B': [1.1, 2.2, 3.3, 4.4]
    })
    is_valid, details = validate_dataframe(df_invalid)
    assert is_valid is False
    assert "The DataFrame contains NaN values." in details["errors"]

def test_is_numeric():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
    assert is_numeric(df['A']) is True
    assert is_numeric(df['B']) is False

def test_determine_data_types(sample_data):
    df, _, _ = sample_data
    continuous_columns, discrete_columns = determine_data_types(df)
    assert 'A' in discrete_columns
    assert 'B' in continuous_columns

def test_discretise_data():
    data = [1, 2, 3, 4, 5]
    discretized_data, bin_bounds = discretise_data(data, 2)
    assert len(discretized_data) == len(data)
    assert len(bin_bounds) == 2  # Should create 2 bins



# Sample DataFrame for testing
@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'Feature_1': np.random.choice([0, 1, 2, 3, 4], size=100),
        'Feature_2': np.random.randint(10, 100, size=100),
        'Feature_3': np.random.randint(100, 1000, size=100)
    })


import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch


@patch('decima2.utils.data_utils.discretise_data')
def test_discretise_selected_feature_auto(mock_discretise_data):
    # Test case where number_of_categories is 'auto' and unique_values <= 10
    feature_values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9])
    feature_name = "Feature_1"
    number_of_categories = 'auto'
    
    discretised_values, formatted_feature_categories = discretise_selected_feature(feature_values, number_of_categories, feature_name)

    # Expect all unique values to be returned in the formatted_feature_categories
    expected_categories = [f"Feature_1 = {i}" for i in range(1, 10)]
    
    assert np.array_equal(discretised_values, feature_values.values)
    assert formatted_feature_categories == expected_categories


@patch('decima2.utils.data_utils.discretise_data')
def test_discretise_selected_feature_auto_high_unique_values(mock_discretise_data):
    # Test case where number_of_categories is 'auto' and unique_values > 10
    feature_values = pd.Series(np.random.randint(1, 100, 50))  # More than 10 unique values
    feature_name = "Feature_1"
    number_of_categories = 'auto'
    
    # Mocking discretise_data to return dummy values
    mock_discretise_data.return_value = (np.arange(50), [[0, 20], [20, 40], [40, 60], [60, 80], [80, 100]])

    discretised_values, formatted_feature_categories = discretise_selected_feature(feature_values, number_of_categories, feature_name)

    # Expect 5 categories as defined in the mock
    expected_categories = [
        "0 >= Feature_1 < 20", 
        "20 >= Feature_1 < 40", 
        "40 >= Feature_1 < 60", 
        "60 >= Feature_1 < 80", 
        "80 >= Feature_1 < 100"
    ]

    assert len(discretised_values) == 50
    assert formatted_feature_categories == expected_categories


@patch('decima2.utils.data_utils.discretise_data')
def test_discretise_selected_feature_with_exact_categories(mock_discretise_data):
    # Test case where number_of_categories equals the number of unique values
    feature_values = pd.Series([1, 2, 3, 4])
    feature_name = "Feature_2"
    number_of_categories = 4

    discretised_values, formatted_feature_categories = discretise_selected_feature(feature_values, number_of_categories, feature_name)

    # Since number_of_categories equals number of unique values, expect each unique value to be a category
    expected_categories = [f"Feature_2 = {i}" for i in feature_values.unique()]

    assert np.array_equal(discretised_values, feature_values.values)
    assert formatted_feature_categories == expected_categories


@patch('decima2.utils.data_utils.discretise_data')
def test_discretise_selected_feature_with_less_categories(mock_discretise_data):
    # Test case where number_of_categories is less than the number of unique values
    feature_values = pd.Series(np.random.randint(1, 100, 50))
    feature_name = "Feature_2"
    number_of_categories = 3

    # Mocking discretise_data to return dummy values
    mock_discretise_data.return_value = (np.arange(50), [[0, 30], [30, 60], [60, 100]])

    discretised_values, formatted_feature_categories = discretise_selected_feature(feature_values, number_of_categories, feature_name)

    expected_categories = [
        "0 >= Feature_2 < 30", 
        "30 >= Feature_2 < 60", 
        "60 >= Feature_2 < 100"
    ]

    assert len(discretised_values) == 50
    assert formatted_feature_categories == expected_categories


@patch('decima2.utils.data_utils.discretise_data')
def test_discretise_selected_feature_with_auto_for_high_unique(mock_discretise_data):
    # Test case where unique values > 10 and auto sets number_of_categories to 5
    feature_values = pd.Series(np.random.randint(1, 100, 100))  # Large unique values
    feature_name = "Feature_3"
    number_of_categories = 'auto'

    # Mock discretization
    mock_discretise_data.return_value = (np.arange(100), [[0, 20], [20, 40], [40, 60], [60, 80], [80, 100]])

    discretised_values, formatted_feature_categories = discretise_selected_feature(feature_values, number_of_categories, feature_name)

    expected_categories = [
        "0 >= Feature_3 < 20", 
        "20 >= Feature_3 < 40", 
        "40 >= Feature_3 < 60", 
        "60 >= Feature_3 < 80", 
        "80 >= Feature_3 < 100"
    ]

    assert len(discretised_values) == 100
    assert formatted_feature_categories == expected_categories


def test_valid_feature(sample_dataframe):
    """Test with a valid feature name."""
    result = validate_selected_feature(sample_dataframe, 'Feature_1', 'auto')
    assert result == 'auto', "Expected 'auto' for valid feature name when number_of_categories is 'auto'"

def test_invalid_feature(sample_dataframe):
    """Test with an invalid feature name."""
    with pytest.raises(ValueError, match="Feature_4 is not one of the dataframe columns"):
        validate_selected_feature(sample_dataframe, 'Feature_4', 'auto')

def test_number_of_categories_auto(sample_dataframe):
    """Test with number_of_categories set to 'auto'."""
    result = validate_selected_feature(sample_dataframe, 'Feature_1', 'auto')
    assert result == 'auto', "Expected 'auto' when number_of_categories is set to 'auto'"

def test_warning_high_number_of_categories(sample_dataframe):
    """Test that a warning is raised when number_of_categories exceeds the limit."""
    k = int(np.ceil(np.log2(sample_dataframe.shape[0]) + 1))  # Calculate k based on sample size
    high_value = k + 1  # A value greater than k to trigger the warning

    with pytest.warns(UserWarning, match="Number of categories may be too high to return meaningful results"):
        result = validate_selected_feature(sample_dataframe, 'Feature_1', high_value)
        assert result == 'auto', "Expected 'auto' to be returned after warning is raised"

def test_valid_number_of_categories(sample_dataframe):
    """Test with a valid number of categories."""
    k = int(np.ceil(np.log2(sample_dataframe.shape[0]) + 1))  # Calculate k based on sample size
    valid_value = k  # Set a valid number of categories

    result = validate_selected_feature(sample_dataframe, 'Feature_1', valid_value)
    assert result == valid_value, "Expected valid number of categories to be returned"



def test_format_feature_categories_normal():
    """Test with typical feature categories."""
    feature_categories = [(1.0, 5.0), (5.0, 10.0)]
    feature_name = 'Feature_A'
    expected_output = ['1.0 >= Feature_A < 5.0', '5.0 >= Feature_A < 10.0']
    
    result = format_feature_categories(feature_categories, feature_name)
    assert result == expected_output, "Expected formatted feature categories do not match"

def test_format_feature_categories_single_range():
    """Test with a single feature category range."""
    feature_categories = [(0.0, 1.0)]
    feature_name = 'Feature_B'
    expected_output = ['0.0 >= Feature_B < 1.0']
    
    result = format_feature_categories(feature_categories, feature_name)
    assert result == expected_output, "Expected formatted output for a single range does not match"

def test_format_feature_categories_zero_values():
    """Test with feature categories including zero values."""
    feature_categories = [(0.0, 0.0), (0.0, 1.0)]
    feature_name = 'Feature_C'
    expected_output = ['0.0 >= Feature_C < 0.0', '0.0 >= Feature_C < 1.0']
    
    result = format_feature_categories(feature_categories, feature_name)
    assert result == expected_output, "Expected formatted output for zero values does not match"

def test_format_feature_categories_negative_values():
    """Test with negative feature category values."""
    feature_categories = [(-5.0, -1.0), (-1.0, 0.0)]
    feature_name = 'Feature_D'
    expected_output = ['-5.0 >= Feature_D < -1.0', '-1.0 >= Feature_D < 0.0']
    
    result = format_feature_categories(feature_categories, feature_name)
    assert result == expected_output, "Expected formatted output for negative values does not match"

def test_format_feature_categories_floats():
    """Test with floating point values in feature categories."""
    feature_categories = [(1.12345, 2.12345)]
    feature_name = 'Feature_E'
    expected_output = ['1.123 >= Feature_E < 2.123']
    
    result = format_feature_categories(feature_categories, feature_name)
    assert result == expected_output, "Expected formatted output for floating point values does not match"

def test_format_feature_categories_empty():
    """Test with an empty feature category list."""
    feature_categories = []
    feature_name = 'Feature_F'
    expected_output = []
    
    result = format_feature_categories(feature_categories, feature_name)
    assert result == expected_output, "Expected empty output for empty feature categories list"