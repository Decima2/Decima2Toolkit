
from decima2.utils import data_utils
from decima2.model.explanation import explanation_calculation
from decima2.visualisation.model import visualisation_explanation
from decima2.utils import utils
from decima2.utils import model_utils 
import warnings

"""
Module: model_explanations

This module provides functionality to generate model summary explanations based on feature importance. 
It supports generating textual, static, or interactive visualizations of the model's feature attributions. 
The key functions within this module perform data validation, feature importance calculations, and visualization.

Dependencies:
- data_handler: Handles data preprocessing, including binarizing and transforming data for interpretable analysis.
- feature_importance: Calculates the feature importance scores for the provided model.
- visualisation: Provides functionality to generate both static and interactive visualizations of feature importances.
- utils: Contains utility functions, such as sorting features by importance and extracting feature names.
- models: Handles model evaluation based on the problem type (classification or regression).

Functions:
-----------
model_explanations(X, y, model, output='dynamic'):
    Generates explanations for the provided model, which can be returned as text, static image, or interactive app.
    
    Parameters:
    -----------
    - X: pandas DataFrame
        The feature matrix (input data) which was used to test the model. 
    - y: pandas Series or Numpy array
        The target variable corresponding to X.
    - model: scikit-learn, Keras, Pytorch compatible model
        The machine learning model for which the feature importance and explanations are generated.
    - output: str, default='dynamic'
        The output type for explanations. Options include:
        - 'text': Returns a textual summary of feature importances.
        - 'static': Generates a static image visualizing feature importances.
        - 'dynamic': Returns an interactive dashboard (using Dash) for visualizing feature importances.

    Returns:
    --------
    - Depending on the `output` argument, the function returns either:
      - A textual summary of feature names with their importances,
      - A static Plotly figure visualizing feature importances, or
      - An interactive Dash app for exploring feature importances dynamically.

    Raises:
    -------
    - ValueError: Raised if the number of feature importances does not match the number of features.
    - UserWarning: Warns if feature importance is all zeros or suggests using a model with fewer features.

    Notes:
    ------
    - The function begins by validating and preprocessing the input data using `data_handler`.
    - It computes feature importance scores using the provided model and returns results in the specified format.
    - In case of all-zero feature importances, it warns the user about potential issues with the model's explanation capacity.

Example usage:
--------------
```python
# Example usage for generating interactive explanations:
app = model_explanations(X, y, model, output='dynamic')
app.run_server()

# Example usage for static image output:
model_explanations(X, y, model, output='static')

# Example usage for textual output:
text_summary = model_explanations(X, y, model, output='text')
print(text_summary)

"""


def model_feature_importance(X,y,model,output='dynamic'):
	# this validates dataframe and returns a binarised interpretable dataframe upon which we perform our interventions
	X_d, X_adjusted,y_adjusted = data_utils.data_discretiser(X,y)
	problem_type = data_utils.determine_target_type_valid(y_adjusted)
	model_evaluator = model_utils.ModelEvaluator(model, problem_type=problem_type)
	model_utils.validate_model_dataframe(model_evaluator,X_adjusted,y_adjusted)
	#this calculates feature importance
	importances = explanation_calculation.model_feature_importance(model_evaluator, X_adjusted, X_d, y_adjusted)
	list_importances = list(importances.values())

	if all(x == 0 for x in list_importances):
		if X_adjusted.shape[0] != X.shape[0]:

			warnings.warn("Reccommend using a model with less features to obtain meaningful explanations", UserWarning)

		else:
			warnings.warn("Reccommend using a model with less features to obtain meaningful explanations", UserWarning)
	if len(list_importances) == 0:
		raise ValueError("Number of attributions is 0, algorithm did not perform as expected")
	if len(list_importances) != len(X_adjusted.columns):
		raise ValueError("Number of attributions does not match number of features")


	if output=='text':
		return utils.feature_names(X_adjusted,list_importances)
	elif output=='static':
		features = list(X_adjusted.columns)
		importances = [round(x,3) for x in list_importances]
		original_accuracy = model_evaluator.evaluate(X_adjusted,y_adjusted)
		sorted_importances, sorted_features = utils.sort_features_by_importance(importances,features)
		fig = visualisation_explanation.create_model_explanation_plot(sorted_features, sorted_importances, original_accuracy, model_evaluator.metric, sorted_features)
		fig.show() 

	else:

		features = list(X_adjusted.columns)
		importances = [round(x,3) for x in list_importances]
		original_accuracy = model_evaluator.evaluate(X_adjusted,y_adjusted)
		sorted_importances, sorted_features = utils.sort_features_by_importance(importances,features)
		app = visualisation_explanation.create_model_explanation_app(sorted_features,sorted_importances,original_accuracy,model_evaluator.metric)

		return app

