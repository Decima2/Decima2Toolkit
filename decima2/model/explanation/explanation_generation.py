
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
1. model_feature_importance(X, y, model, output='dynamic'):
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

2.  grouped_feature_importance(X,y,model,feature_name, output='dynamic', number_of_categories='auto')
	
	Calculate and visualize the feature importance of a machine learning model grouped by specific categories or ranges 
    of a selected feature. This function provides insights into how feature importance varies across different segments 
    of the dataset based on the provided feature.

    The function can return either a textual summary of grouped feature importances or an interactive visualization 
    using a Dash app.

    Args:
    -----
    - X (pandas.DataFrame): The feature matrix (input data) used for model evaluation.
    - y (pandas.Series or numpy.ndarray): The target variable corresponding to X.
    - model (scikit-learn, Keras, Pytorch compatible model): The machine learning model for which feature importances 
      will be calculated.
    - feature_name (str): The name of the feature used to group the data into different categories or ranges for 
      analysis.
    - output (str, default='dynamic'): Specifies the output type. Options include:
        - 'text': Returns a textual summary of the grouped feature importances.
        - 'dynamic': Returns an interactive Dash app to visualize and explore feature importances for the different 
          feature groupings.
    - number_of_categories (str or int, default='auto'): If not 'auto' input must be an int Defines the number of categories or ranges into which 
      `feature_name` should be discretized. If set to 'auto', the function will automatically determine the number 
      of categories based on the uniqueness of the feature values. Must be less than log2(size of X)

    Returns:
    --------
    - If `output == 'text'`: Returns a dictionary of grouped feature importances where the keys are the categories 
      (or ranges) of the `feature_name`, and the values are dictionaries of feature importances for each group.
    - If `output == 'dynamic'`: Returns an interactive Dash app to visualize the feature importances grouped by 
      `feature_name` and allows users to explore the feature importance dynamically.

    Raises:
    -------
    - ValueError: If the calculated feature importances are inconsistent with the number of features in the dataset.
    - UserWarning: Raised if all feature importances are zeros, suggesting potential issues with the model or feature 
      selection.

    Notes:
    ------
    - This function discretizes the selected `feature_name` into a specified or automatically determined number of 
      categories or ranges. The feature importance of each range is then calculated and presented either as text or 
      through a visual interface.
    - In the case of an interactive output, users can interactively explore the importance of features for each 
      category of the selected feature via an interactive app.
    - The function leverages several utility functions to preprocess the input data, evaluate the model, and calculate 
      grouped feature importances.

    Example usage:
    --------------
    ```python
    # Example for textual output of grouped feature importance:
    grouped_importances = grouped_feature_importance(X, y, model, feature_name='age', output='text')
    print(grouped_importances)

    # Example for dynamic visualization of grouped feature importance:
    app = grouped_feature_importance(X, y, model, feature_name='age', output='dynamic')
    app.run_server()
    ```

"""

def model_feature_importance(X,y,model,output='dynamic'):
    X_adjusted,y_adjusted = data_utils.validate_inputs(X,y)
    X_d = data_utils.data_discretiser(X_adjusted)
    problem_type = data_utils.determine_target_type_valid(y_adjusted)
    model_evaluator = model_utils.ModelEvaluator(model, problem_type=problem_type)
    model_utils.validate_model_dataframe(model_evaluator,X_adjusted,y_adjusted)
    #this calculates feature importance
    importances = explanation_calculation.model_feature_importance(model_evaluator, X_adjusted, X_d, y_adjusted)
    list_importances = list(importances.values())

    if all(x == 0 for x in list_importances):
    	if X_adjusted.shape[0] != X.shape[0]:

    		warnings.warn("Recommend using a model with less features to obtain meaningful explanations", UserWarning)

    	else:
    		warnings.warn("Recommend using a model with less features to obtain meaningful explanations", UserWarning)
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


def grouped_feature_importance(X,y,model,feature_name, output='dynamic', number_of_categories='auto'): 
    X_adjusted,y_adjusted = data_utils.validate_inputs(X,y)
    number_of_categories = data_utils.validate_selected_feature(X_adjusted, feature_name, number_of_categories)
    X_d = data_utils.data_discretiser(X_adjusted.drop(feature_name,axis=1))

    problem_type = data_utils.determine_target_type_valid(y_adjusted)
    model_evaluator = model_utils.ModelEvaluator(model, problem_type=problem_type)
    model_utils.validate_model_dataframe(model_evaluator,X_adjusted,y_adjusted)

    
    X_d[feature_name], formatted_feature_categories = data_utils.discretise_selected_feature(X_adjusted[feature_name],number_of_categories, feature_name)
        
    grouped_feature_importance = {}
    original_accuracies = {}
    most_important_features = {}
    for i in range(len(formatted_feature_categories)):#change this back after testing

        X_d_selected = X_d[X_d[feature_name]==i]
        y_selected = y_adjusted[X_d_selected.index]

        X_adjusted_safe = X_adjusted.copy()
        X_adjusted_safe['category'] = [1 if j == i else 0 for j in X_d[feature_name]]
        
        
         # Use the positional index
        X_selected = X_adjusted_safe[X_adjusted_safe['category']==1]
        X_selected = X_selected.drop('category',axis=1)
        X_adjusted_safe = X_adjusted_safe.drop('category',axis=1)

 
        original_accuracies[formatted_feature_categories[i]] = model_evaluator.evaluate(X_selected,y_selected)
        category_feature_importances = explanation_calculation.grouped_feature_importance(model_evaluator,X_adjusted_safe,y_adjusted,X_d,X_d_selected,X_selected,y_selected )
        most_important_feature = max(category_feature_importances, key=category_feature_importances.get)
        sorted_category_feature_importances = dict(sorted(category_feature_importances.items(), reverse=True, key=lambda item: item[1]))


        grouped_feature_importance[formatted_feature_categories[i]] = sorted_category_feature_importances
        most_important_features[formatted_feature_categories[i]] = most_important_feature


    if output =='text':
        return grouped_feature_importance

    if output == 'dynamic':
        app = visualisation_explanation.create_grouped_explanation_app(grouped_feature_importance, original_accuracies, model_evaluator.metric, most_important_features, feature_name)
        return app



