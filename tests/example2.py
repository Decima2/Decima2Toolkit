import pytest
import pandas as pd
import numpy as np
import warnings

# Import the functions and classes from the data_processing module
# Assuming the module is named data_processing.py and located in the same directory
from decima.utils.data_utils import (
    determine_target_type_valid,
    assert_size,
    data_discretiser,
    validate_target,
    determine_target_type,
    validate_dataframe_target,
    validate_dataframe,
    is_numeric,
    determine_data_types,
    discretise_data
)

def test_determine_target_type_valid_classification():
    y = np.array([0, 1, 1, 0])
    assert determine_target_type_valid(y) == 'classification'

def test_determine_target_type_valid_regression():
    y = np.array([1.0, 2.5, 3.0, 4.0])
    assert determine_target_type_valid(y) == 'regression'

def test_determine_target_type_invalid():
    y = np.array([[1, 2], [3, 4]])  # Invalid shape
    with pytest.raises(ValueError):
        determine_target_type_valid(y)

def test_assert_size_warning():
    df_large = pd.DataFrame(np.random.rand(200, 100))  # 200 rows, 100 columns
    y_large = np.random.rand(200)
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df_adjusted, y_adjusted = assert_size(df_large, y_large)
        assert len(w) > 0  # Check if a warning was issued
        assert df_adjusted.shape[0] <= 100  # Ensure size is adjusted

def test_data_discretiser():
    df = pd.DataFrame({'feature_1': np.random.rand(10), 'feature_2': np.arange(10)})
    y = np.random.randint(0, 2, size=10)
    
    discretised_df, original_df, original_y = data_discretiser(df, y)
    
    # Check if discretised_df has the same columns as original df
    assert list(discretised_df.columns) == list(df.columns)
    assert len(discretised_df) == len(original_df) == len(original_y)

def test_validate_target():
    assert validate_target(np.array([1, 2, 3])) is True
    assert validate_target(pd.Series([1, 2, 3])) is True
    assert validate_target([1, 2, 3]) is False  # List is invalid

def test_validate_dataframe_target():
    df = pd.DataFrame({'feature_1': [1, 2, 3]})
    y = np.array([1, 2, 3])
    assert validate_dataframe_target(df, y) is None  # Should pass without exception

    y_invalid = np.array([1, 2])  # Mismatch in length
    with pytest.raises(ValueError):
        validate_dataframe_target(df, y_invalid)

def test_validate_dataframe():
    df_valid = pd.DataFrame({'feature_1': [1, 2, 3], 'feature_2': [4, 5, 6]})
    assert validate_dataframe(df_valid) == (True, {'valid': True, 'errors': []})

    df_invalid = pd.DataFrame({'feature_1': [1, 2, None], 'feature_2': [4, 5, 6]})
    assert validate_dataframe(df_invalid)[0] is False  # Should fail due to NaN

    df_empty = pd.DataFrame()
    assert validate_dataframe(df_empty) == (False, {'valid': False, 'errors': ['The DataFrame is empty.']})

def test_is_numeric():
    df_numeric = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
    assert is_numeric(df_numeric['col1']) is True
    assert is_numeric(df_numeric['col2']) is False

def test_determine_data_types():
    df = pd.DataFrame({
        'continuous': [1.0, 2.5, 3.0],
        'discrete': [1, 2, 3],
        'categorical': ['a', 'b', 'c']
    })
    
    continuous, discrete = determine_data_types(df)
    assert 'continuous' in continuous
    assert 'discrete' in discrete
    assert 'categorical' in discrete  # Categorical columns are treated as discrete

def test_discretise_data():
    data = [1, 2, 3, 4, 5]
    discretized_data, bin_bounds = discretise_data(data, n_categories=2)

    assert len(discretized_data) == len(data)
    assert len(bin_bounds) == 2  # Two bins created

# Example class for ModelEvaluator tests
class DummyModel:
    def score(self, X, y):
        return 0.95  # Dummy score

def test_model_evaluator():
    model = DummyModel()
    evaluator = ModelEvaluator(model)

    df_test = pd.DataFrame({'feature_1': [1, 2], 'feature_2': [3, 4]})
    y_test = [1, 0]
    
    score = evaluator.evaluate(df_test, y_test)
    assert score == 0.95  # Check if the dummy score returned is correct

if __name__ == "__main__":
    pytest.main([__file__])
