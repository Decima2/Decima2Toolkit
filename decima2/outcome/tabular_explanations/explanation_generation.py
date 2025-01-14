
from decima2.utils import data_utils

from decima2.utils import model_utils
from decima2.visualisation.outcome import visualisation

import pandas as pd
import numpy as np

import requests


CLOUD_FUNCTION_URL_INDIVIDUAL_EXPLANATION = "https://europe-west2-decima2.cloudfunctions.net/call_private_individual_explanation"


def public_individual_feature_importance(target_row=0,dataset=0):
    # Public function that interacts with the Google Cloud Function
    try:
        # Send a POST request to the Google Cloud Function
        response = requests.post(CLOUD_FUNCTION_URL_INDIVIDUAL_EXPLANATION, json={"target_row" : target_row.tolist(), "dataset" : dataset.tolist()})
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            return data['message']
        else:
            data = response.json()
            return data['error']
    except Exception as e:
        return f"Error: {e}"


def individual_feature_importance(model,dataset,y,selected_index):
    
    # Machine learning model 
    # dataset of interest including instance to be explainaed
    # machine learning predictions - should be of shape (n_samples)
    # selected index of dataset we want to be explained

	X_adjusted,y_adjusted = data_utils.validate_inputs(dataset,y)
	problem_type = data_utils.determine_target_type_valid(y_adjusted)
	
	target_axis = 0
	if problem_type == 'classification':
		target_axis = int(y[selected_index])


	model_evaluator = model_utils.ModelEvaluator(model, problem_type=problem_type)


	selected_instance = X_adjusted.iloc[selected_index].values.reshape(1,-1)
	selected_instance_index = selected_index
	selected_prediction = model_evaluator.predict(selected_instance, target_axis)

	full_dataset = X_adjusted.copy()
	full_dataset['predictions'] = model_evaluator.predict(X_adjusted, target_axis)

	increase_set_indices = full_dataset[full_dataset['predictions'] > selected_prediction[0]].index
	decrease_set_indices = full_dataset[full_dataset['predictions'] < selected_prediction[0]].index


	discretized_dataset = data_utils.data_discretiser(full_dataset)
	discretized_selected_instance = discretized_dataset.iloc[selected_instance_index]
	selected_prediction = selected_prediction.tolist()  # Ensure predictions are lists



	if len(increase_set_indices) == 0:
		increase_prediction = 0 

	else:
		discretized_increasers = discretized_dataset.iloc[increase_set_indices]

		trial = (discretized_selected_instance.values).tolist()

		nearest_neighbour_upper_index = public_individual_feature_importance(discretized_selected_instance.values, discretized_increasers.values)
		
		
		increase_index = list(increase_set_indices)[nearest_neighbour_upper_index]

		increase_instance = X_adjusted.loc[increase_index]
		increase_prediction = model_evaluator.predict(increase_instance.values.reshape(1,-1), target_axis)



		increaser_actions = increase_instance.values.reshape(1,-1) - selected_instance
		increaser_actions = list(increaser_actions)[0]  # Ensure actions are lists

		increase_dict = {}
		for i in range(len(dataset.columns)):
		    increase_dict[dataset.columns[i]] = round(increaser_actions[i], 2)

		increase_dict = {key: value for key, value in increase_dict.items() if value != 0}


	if len(decrease_set_indices) == 0: 
		decrease_dict = {}
		decrease_prediction = 0

	else:
		discretized_decreasers= discretized_dataset.iloc[decrease_set_indices]
		nearest_neighbour_lower_index = public_individual_feature_importance(discretized_selected_instance.values, discretized_decreasers.values)
		decrease_index = list(decrease_set_indices)[nearest_neighbour_lower_index]

		decrease_instance = X_adjusted.loc[decrease_index]
		decrease_prediction = model_evaluator.predict(decrease_instance.values.reshape(1,-1), target_axis)

		decreaser_actions = decrease_instance.values.reshape(1,-1) - selected_instance
		decreaser_actions = list(decreaser_actions)[0]  # Ensure actions are lists

		#decrease_dict = {key: value for key, value in decrease_dict.items()}

		decrease_dict = {}
		for i in range(len(dataset.columns)):
		    decrease_dict[dataset.columns[i]] = round(decreaser_actions[i], 2)

		decrease_dict = {key: value for key, value in decrease_dict.items() if value != 0}

	
	# Convert NumPy arrays to lists to avoid the TypeError during serialization

	# Return the response as JSON
	selected_instance_dict = {}
	for i in range(len(dataset.columns)):
		    selected_instance_dict[dataset.columns[i]] = selected_instance[0][i]


	if problem_type == 'classification':
		app = visualisation.create_individual_explanation_classification_app(selected_instance_dict, increase_dict, decrease_dict, target_axis, selected_prediction[0], increase_prediction[0], decrease_prediction[0])
	else:
		app = visualisation.create_individual_explanation_regression_app(selected_instance_dict, increase_dict, decrease_dict, selected_prediction[0], increase_prediction[0], decrease_prediction[0])


	return app 
