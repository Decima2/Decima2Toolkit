import pandas as pd
import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_regression

# Import the function from test_1.py
from decima2 import grouped_feature_importance
from decima2.utils.data_utils import validate_inputs
from decima2.utils.data_utils import discretise_selected_feature 

# Function to create a custom DataFrame with predefined characteristics
def create_custom_dataframe():
    num_rows = 10000
    data = {
        'Feature_0': np.random.choice([0, 1, 2, 3, 4], size=num_rows),  # Categorical feature with 5 unique values
        'Feature_1': np.random.randint(10, 100, size=num_rows),         # Continuous feature
        'Feature_2': np.random.randint(100, 1000, size=num_rows)        # Continuous feature
    }
    
    X = pd.DataFrame(data)
    y = np.random.choice([0, 1], size=num_rows)  # Binary target
    return X, y

# Function to train a RandomForest classifier
def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Test for grouped feature importance with a categorical feature column
def test_categorical_data_categorical_column():
    X, y = create_custom_dataframe()
    model, X_test, y_test = train_random_forest(X, y)
    
    # Run the grouped feature importance for regression
    X_adjusted,y_adjusted = validate_inputs(X_test,y_test)
    _, formatted_feature_categories = discretise_selected_feature(X_adjusted['Feature_0'],'auto', 'Feature_0')
    attributions = grouped_feature_importance(X_test, y_test, model, 'Feature_0', output='text')
    
    # Check if the app initializes without errors

    assert len(attributions) == len(formatted_feature_categories)

# Test for grouped feature importance with a continuous feature column
def test_categorical_data_categorical_column():
    X, y = create_custom_dataframe()
    model, X_test, y_test = train_random_forest(X, y)
    
    # Run the grouped feature importance for regression
    X_adjusted,y_adjusted = validate_inputs(X_test,y_test)
    _, formatted_feature_categories = discretise_selected_feature(X_adjusted['Feature_1'],'auto', 'Feature_1')
    attributions = grouped_feature_importance(X_test, y_test, model, 'Feature_1', output='text')
    
    # Check if the app initializes without errors

    assert len(attributions) == len(formatted_feature_categories)

# Generate regression data with many samples
def generate_regression_data(samples, features):
    X_regression, y_regression = make_regression(n_samples=samples, n_features=features, n_targets=1, noise=0.1, random_state=42)
    X_regression_df = pd.DataFrame(X_regression)
    return X_regression_df, y_regression

# Test with a large number of regression samples
def test_regression():
    X_regression, y_regression = generate_regression_data(samples=1000, features=10)
    X_regression.columns = ['Feature_' + str(i) for i in range(10)]
    X_train, X_test, y_train, y_test = train_test_split(X_regression, y_regression, test_size=0.4, random_state=42)
    
    # Train RandomForest for regression
    model = RandomForestRegressor(max_depth=100, random_state=42)
    model.fit(X_train, y_train)
    
    X_adjusted,y_adjusted = validate_inputs(X_test,y_test)
    _, formatted_feature_categories = discretise_selected_feature(X_adjusted['Feature_2'],'auto', 'Feature_2')
    attributions = grouped_feature_importance(X_test, y_test, model, 'Feature_2', output='text')
    
    # Check if the app initializes without errors

    assert len(attributions) == len(formatted_feature_categories)


# Test with user defined catgeories regression features
def test_regression_defined_categories():
    X_regression, y_regression = generate_regression_data(samples=100, features=4)
    X_regression.columns = ['Feature_' + str(i) for i in range(4)]
    X_train, X_test, y_train, y_test = train_test_split(X_regression, y_regression, test_size=0.4, random_state=42)
    
    # Train RandomForest for regression
    model = RandomForestRegressor(max_depth=100, random_state=42)
    model.fit(X_train, y_train)

    X_adjusted,y_adjusted = validate_inputs(X_test,y_test)
    _, formatted_feature_categories = discretise_selected_feature(X_adjusted['Feature_2'],3,'Feature_2')
    attributions = grouped_feature_importance(X_test, y_test, model, 'Feature_2', output='text', number_of_categories=3)
    
    # Check if the app initializes without errors



    assert len(attributions) == len(formatted_feature_categories)


# Test with a large number of regression features
def test_regression_features_big():
    X_regression, y_regression = generate_regression_data(samples=10000, features=50)
    X_regression.columns = ['Feature_' + str(i) for i in range(50)]
    X_train, X_test, y_train, y_test = train_test_split(X_regression, y_regression, test_size=0.4, random_state=42)
    
    # Train RandomForest for regression
    model = RandomForestRegressor(max_depth=100, random_state=42)
    model.fit(X_train, y_train)

    X_adjusted,y_adjusted = validate_inputs(X_test,y_test)
    _, formatted_feature_categories = discretise_selected_feature(X_adjusted['Feature_2'],2,'Feature_2')

    attributions = grouped_feature_importance(X_test, y_test, model, 'Feature_2', output='text', number_of_categories=2)
    
    # Check if the app initializes without errors
    print(len(attributions))
    print(len(formatted_feature_categories))

    assert len(attributions) == len(formatted_feature_categories)

    
