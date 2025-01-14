import pytest 
from decima2 import individual_feature_importance
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
import pandas as pd



def test_RFC_scikit():
    X_adult, y_adult = shap.datasets.adult()
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_adult, y_adult, test_size=0.20, random_state=42)
    model1 = RandomForestClassifier(max_depth=100, random_state=42)
    model1.fit(X_train1, y_train1)
    explanation = individual_feature_importance(model1,X_test1,y_test1,2)


def test_RFR_scikit():
    def generate_regression_data():
        X_regression, y_regression = make_regression(
            n_samples=500, n_features=10, n_targets=1, noise=0.1, random_state=42
        )
        X_regression_df = pd.DataFrame(X_regression, columns=[f"feature_{i}" for i in range(10)])
        return X_regression_df, y_regression

    # Example usage
    X_regression, y_regression = generate_regression_data()

    X_train3, X_test3, y_train3, y_test3 = train_test_split(X_regression, y_regression, test_size=0.20, random_state=42)
    model3 = RandomForestRegressor(max_depth=100, random_state=42)
    model3.fit(X_train3, y_train3)

    explanation = explanation = individual_feature_importance(model3,X_test3,y_test3,2)
    