import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import accuracy_score, r2_score
from sklearn.datasets import make_classification, make_regression
import torch
import torch.nn as nn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


from decima2.utils.model_utils import ModelEvaluator  # Adjust the import based on your structure


# Fixture for scikit-learn classification model
@pytest.fixture
def sklearn_classification_model():
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
    model = LogisticRegression()
    model.fit(X, y)
    return model, X, y


# Fixture for scikit-learn regression model
@pytest.fixture
def sklearn_regression_model():
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
    model = RandomForestRegressor()
    model.fit(X, y)
    return model, X, y


# Fixture for PyTorch classification model
@pytest.fixture
def pytorch_classification_model():
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc = nn.Linear(20, 2)

        def forward(self, x):
            return self.fc(x)

    model = SimpleNN()
    model.train()  # Set model to training mode

    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Training the model for 10 epochs
    for _ in range(10):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])


    return model, X, y


# Fixture for PyTorch regression model
@pytest.fixture
def pytorch_regression_model():
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc = nn.Linear(20, 1)

        def forward(self, x):
            return self.fc(x)

    model = SimpleNN()
    model.train()  # Set model to training mode

    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Training the model for 10 epochs
    for _ in range(10):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor.view(-1, 1))
        loss.backward()
        optimizer.step()


    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])

    return model, X, y


# Fixture for Keras classification model
@pytest.fixture
def keras_classification_model():
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(20,)))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
    model.fit(X, y, epochs=10, verbose=0)

    return model, X, y


# Fixture for Keras regression model
@pytest.fixture
def keras_regression_model():
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(20,)))
    model.add(Dense(1))  # Single output for regression
    model.compile(optimizer='adam', loss='mean_squared_error')

    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    model.fit(X, y, epochs=10, verbose=0)

    return model, X, y


def test_evaluate_sklearn_classification(sklearn_classification_model):
    model, X, y = sklearn_classification_model
    evaluator = ModelEvaluator(model, problem_type='classification')
    score = evaluator.evaluate(X, y)
    assert score >= 0.0 and score <= 1.0  # Accuracy score should be between 0 and 1


def test_evaluate_sklearn_regression(sklearn_regression_model):
    model, X, y = sklearn_regression_model
    evaluator = ModelEvaluator(model, problem_type='regression')
    score = evaluator.evaluate(X, y)
    assert score >= -1.0 and score <= 1.0  # R^2 score should be between -1 and 1


def test_evaluate_pytorch_classification(pytorch_classification_model):
    model, X, y = pytorch_classification_model
    evaluator = ModelEvaluator(model, problem_type='classification', is_pytorch_model=True)
    score = evaluator.evaluate(X, y)
    assert score >= 0.0 and score <= 1.0  # Accuracy score should be between 0 and 1


def test_evaluate_pytorch_regression(pytorch_regression_model):
    model, X, y = pytorch_regression_model
    evaluator = ModelEvaluator(model, problem_type='regression', is_pytorch_model=True)
    score = evaluator.evaluate(X, y)
    assert score >= -1.0 and score <= 1.0  # R^2 score should be between -1 and 1


def test_evaluate_keras_classification(keras_classification_model):
    model, X, y = keras_classification_model
    evaluator = ModelEvaluator(model, problem_type='classification', is_keras_model=True)
    score = evaluator.evaluate(X, y)
    assert score >= 0.0 and score <= 1.0  # Accuracy score should be between 0 and 1


def test_evaluate_keras_regression(keras_regression_model):
    model, X, y = keras_regression_model
    evaluator = ModelEvaluator(model, problem_type='regression', is_keras_model=True)
    score = evaluator.evaluate(X, y)
    assert score >= -1.0 and score <= 1.0  # R^2 score should be between -1 and 1
