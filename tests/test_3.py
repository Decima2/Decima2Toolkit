# test_data_processing.py
import pytest
import pandas as pd
import numpy as np
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

def test_assert_size_large_dataframe(sample_data):
    df, _, _ = sample_data
    df_large = pd.concat([df]*3000)  # Make a large DataFrame
    df_resized, _ = assert_size(df_large, df_large['A'])
    assert df_resized.shape[0] <= 200  # Check if resizing worked

def test_data_discretiser1():
    df = pd.DataFrame({
        'A': [1, 2, 3, 1],
        'B': [1.1, 2.2, 3.3, 4.4],
    })
    y_classification = pd.Series([0, 1, 0, 1])
    discretised_df, _, _ = data_discretiser(df, y_classification)
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

