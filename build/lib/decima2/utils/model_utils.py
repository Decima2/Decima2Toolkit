import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

from torch.nn import Module
from tensorflow.keras import Model
from tensorflow.keras import Sequential

"""
Module: model_evaluator

This module provides a unified interface for evaluating machine learning models across different libraries, including 
scikit-learn, PyTorch, Keras. It supports both classification and regression problems and can compute 
relevant metrics such as accuracy or R^2 score.

Dependencies:
- torch: Used for evaluating PyTorch models.
- numpy: For handling arrays and numerical computations.
- pandas: For managing data in DataFrame format.
- sklearn.metrics: Provides metrics like accuracy_score and r2_score for classification and regression.

Classes:
--------
ModelEvaluator:
    A class to handle the evaluation of various types of models (scikit-learn, PyTorch, Keras, TensorFlow). It determines 
    the appropriate method for evaluating models based on the type of model and problem (classification or regression).

    Methods:
    --------
    __init__(model, problem_type='classification', is_pytorch_model=False, is_keras_model=False):
        Initializes the evaluator with a model and problem type, and detects whether the model is from PyTorch or Keras.

        Parameters:
        - model: A pre-trained machine learning model (e.g., scikit-learn, PyTorch, Keras, TensorFlow).
        - problem_type: Type of problem - 'classification' or 'regression' (default is 'classification').
        - is_pytorch_model: Boolean to indicate if the model is a PyTorch model.
        - is_keras_model: Boolean to indicate if the model is a Keras model.
    
    evaluate(X_test, y_test):
        Evaluates the given model on the provided test data (X_test, y_test), using the appropriate method based on the 
        model type and problem type (classification or regression).
        
        Parameters:
        - X_test: Features of the test set (DataFrame, numpy array, or tensor).
        - y_test: Labels/targets of the test set (DataFrame, numpy array, or tensor).

        Returns:
        - A score or metric depending on the problem type (accuracy for classification, R^2 score for regression).

    _evaluate_pytorch_model(X_test, y_test):
        A helper method for evaluating PyTorch models. It converts data to PyTorch tensors and handles both classification 
        and regression problems.

        Parameters:
        - X_test: Features of the test set (torch.Tensor or DataFrame).
        - y_test: Labels/targets of the test set (torch.Tensor, numpy array, or pandas Series).

        Returns:
        - A score or metric depending on the problem type (accuracy for classification, R^2 score for regression).

    _evaluate_keras_model(X_test, y_test):
        A helper method for evaluating Keras models. It converts data from DataFrame to numpy array if needed and handles 
        both classification and regression problems.

        Parameters:
        - X_test: Features of the test set (DataFrame or numpy array).
        - y_test: Labels/targets of the test set (numpy array).

        Returns:
        - A score or metric depending on the problem type (accuracy for classification, R^2 score for regression).

    Notes:
    ------
    - For scikit-learn models, the `score()` method is used.
    - For Keras models, the `evaluate()` method is used if available; otherwise, the `predict()` method is used.
    - For PyTorch models, data is converted to tensors and evaluated directly in PyTorch.
    - If a model doesn’t have `score`, `evaluate`, or `predict` methods, a TypeError is raised.

Examples:
---------
```python
# Example usage for a scikit-learn model:
evaluator = ModelEvaluator(model, problem_type='classification')
score = evaluator.evaluate(X_test, y_test)
print("Accuracy:", score)

# Example usage for a PyTorch model:
evaluator = ModelEvaluator(pytorch_model, problem_type='regression', is_pytorch_model=True)
r2 = evaluator.evaluate(X_test, y_test)
print("R^2 score:", r2)

# Example usage for a Keras model:
evaluator = ModelEvaluator(keras_model, problem_type='classification', is_keras_model=True)
accuracy = evaluator.evaluate(X_test, y_test)
print("Accuracy:", accuracy)
"""


def validate_model_dataframe(model_evaluator,X,y):
    if model_evaluator.evaluate(X, y) is None:
        raise ValueError("data does not match expected type of the model")



class ModelEvaluator:
    def __init__(self, model, problem_type='classification', is_pytorch_model=False, is_keras_model=False):
        """
        Initialize the evaluator with the model and problem type.

        :param model: A pre-trained machine learning model (e.g., scikit-learn, XGBoost, Keras, PyTorch, TensorFlow).
        :param problem_type: The type of problem - 'classification' or 'regression'. Default is 'classification'.
        :param is_pytorch_model: Boolean indicating whether the model is a PyTorch model.
        :param is_keras_model: Boolean indicating whether the model is a Keras model.
        :param is_tensorflow_model: Boolean indicating whether the model is a TensorFlow model.
        """
        self.model = model
        self.problem_type = problem_type
        self.is_pytorch_model = isinstance(model, Module)
        self.is_keras_model = isinstance(model, (Model, Sequential))
        self.metric = 'Accuracy'

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the given test set (X_test, y_test).

        :param X_test: Features of the test set (DataFrame or tensor).
        :param y_test: Labels/targets of the test set.
        :return: A score or metric depending on the model and problem type.
        """
        if self.is_pytorch_model:
            return self._evaluate_pytorch_model(X_test, y_test)
        
        elif self.is_keras_model:
            return self._evaluate_keras_model(X_test, y_test)
        
        
        elif hasattr(self.model, 'score'):
            # For scikit-learn models or others that have a 'score' method
            if self.problem_type == 'regression':
                self.metric = 'R Squared Score'

            return self.model.score(X_test, y_test)

        
        elif hasattr(self.model, 'evaluate'):
            # For deep learning models like Keras with an 'evaluate' method

            metric = self.model.evaluate(X_test, y_test, verbose=0)
            if isinstance(metric, list):
                return metric[1]
            else:
                self.metric = 'Loss'
                return metric

        
        elif hasattr(self.model, 'predict'):
            # For models that use the 'predict' method (e.g., scikit-learn or Keras)
            predictions = self.model.predict(X_test)
            
            # Handle different types of problems (classification vs regression)
            if self.problem_type == 'classification':
                # Convert probabilities (if any) to class labels
                if predictions.ndim > 1 and predictions.shape[1] > 1:
                    predictions = predictions.argmax(axis=1)
                return accuracy_score(y_test, predictions)
            
            elif self.problem_type == 'regression':
                # For regression problems, compute the R^2 or other relevant metrics
                self.metric = 'R Squared Score'
                return r2_score(y_test, predictions)
        
        else:
            raise TypeError("The provided model doesn't have 'score', 'evaluate', or 'predict' method.")
    
    def _evaluate_pytorch_model(self, X_test, y_test):
        """
        Evaluate a PyTorch model on the given test set (X_test, y_test).

        :param X_test: Features of the test set (torch.Tensor).
        :param y_test: Labels/targets of the test set (torch.Tensor or numpy array).
        :return: A score or metric depending on the problem type.
        """
        self.model.eval()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)

        if isinstance(y_test, np.ndarray):
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32 if self.problem_type == 'regression' else torch.long).to(device)
        elif isinstance(y_test, pd.Series):
            y_test = np.array(y_test)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32 if self.problem_type == 'regression' else torch.long).to(device)
        else:
            y_test_tensor = y_test.to(device)

        with torch.no_grad():
            y_pred_tensor = self.model(X_test_tensor)

            if self.problem_type == 'classification':
                if y_pred_tensor.ndim > 1 and y_pred_tensor.shape[1] > 1:
                    y_pred = y_pred_tensor.argmax(dim=1).cpu().numpy()
                else:
                    y_pred = (y_pred_tensor >= 0.5).long().cpu().numpy()  # Binary classification
                
                return accuracy_score(y_test_tensor.cpu().numpy(), y_pred)
            
            elif self.problem_type == 'regression':
                self.metric = 'R Squared Score'
                y_pred = y_pred_tensor.cpu().numpy()
                return r2_score(y_test_tensor.cpu().numpy(), y_pred)

    def _evaluate_keras_model(self, X_test, y_test):
        """
        Evaluate a Keras model on the given test set (X_test, y_test).

        :param X_test: Features of the test set (numpy array or DataFrame).
        :param y_test: Labels/targets of the test set (numpy array).
        :return: A score or metric depending on the problem type.
        """
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values  # Convert DataFrame to numpy array

        predictions = self.model.predict(X_test)

        if self.problem_type == 'classification':
            if predictions.ndim > 1 and predictions.shape[1] > 1:
                predictions = predictions.argmax(axis=1)
            else:
                predictions = (predictions >= 0.5).astype(int)  # Binary classification
            
            return accuracy_score(y_test, predictions)
        
        elif self.problem_type == 'regression':
            self.metric = 'R Squared Score'
            return r2_score(y_test, predictions)
