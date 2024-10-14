

import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output



"""
Module: visualisation

This module provides functions to create visual representations of model explanations, particularly focused on feature 
importance. It leverages Plotly and Dash to create interactive bar charts that display the impact of each feature on 
the model's performance metric.

Dependencies:
-------------
- plotly.graph_objects: For creating interactive visualizations such as bar charts.
- dash: For building web applications with interactive user interfaces.
- dash.dependencies: To manage the dependencies between different components in the Dash app.

Functions:
----------

1. create_model_explanation_plot(features, importances, original_accuracy, metric, selected_features):
    Generates a Plotly bar chart displaying the importance of selected features. The heights of the bars represent 
    the decrease in a specified performance metric when the corresponding feature is removed from the model. 
    The chart is customizable with various layout settings.

    Parameters:
    -----------
    - features: List of feature names.
    - importances: List of feature importances corresponding to the features.
    - original_accuracy: The original accuracy or performance metric of the model before any feature removal.
    - metric: A string representing the performance metric (e.g., 'accuracy', 'R² score').
    - selected_features: List of features currently selected for visualization.

    Returns:
    --------
    - fig: A Plotly Figure object representing the bar chart.

    Example Usage:
    --------------
    ```python
    fig = create_model_explanation_plot(features, importances, original_accuracy, metric, selected_features)
    fig.show()
    ```

2. create_model_explanation_app(features, importances, original_accuracy, metric):
    Initializes and returns a Dash application that allows users to interactively visualize feature importance. 
    The app includes a bar chart for feature importances and a dropdown menu for selecting specific features to focus on. 
    Additionally, it provides an interpretation panel that explains the impact of feature removal on the model's performance.

    Parameters:
    -----------
    - features: List of feature names.
    - importances: List of feature importances corresponding to the features.
    - original_accuracy: The original accuracy or performance metric of the model before any feature removal.
    - metric: A string representing the performance metric (e.g., 'accuracy', 'R² score').

    Returns:
    --------
    - app: A Dash app object that can be run in a web server.

    Example Usage:
    --------------
    ```python
    app = create_model_explanation_app(features, importances, original_accuracy, metric)
    app.run_server(debug=True)
    ```

General Notes:
--------------
- The visualizations provided by this module are intended to help users interpret the results of machine learning 
  models by highlighting the importance of different features.
- The Dash app allows for interactive exploration, where users can select features and see how their removal would 
  affect the model's performance.

Customization:
--------------
- The appearance of the plots and the layout of the Dash app can be customized through the parameters in the 
  respective functions, allowing for flexibility in visualization styles and usability.

Warnings:
---------
- Ensure that the input lists (`features`, `importances`, `selected_features`) are of the same length to avoid 
  indexing errors.
"""


def create_model_explanation_plot(features, importances, original_accuracy, metric, selected_features):
    # Convert features to a list if it is a pandas Index
    features_list = list(features)

    # Filter the data based on selected features
    filtered_features = [f for f in selected_features if f in features_list]
    filtered_importances = [importances[features_list.index(f)] for f in filtered_features]

    # Create bar chart
    fig = go.Figure(
        data=[go.Bar(
            x=filtered_features,
            y=filtered_importances,
            text=[f"{imp:.3f}" for imp in filtered_importances],  # Display values with 3 decimal places
            textposition='auto',
            marker_color='rgba(128, 203, 196, 1)',  # Set the color for the bars
            hoverinfo = 'none',
            textfont=dict(color="white")  # Set text color to white
        )]
    )

    # Update layout with automatic y-axis scaling
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Features",
        yaxis_title=f"{metric} Decrease If Feature Removed",
        plot_bgcolor='#2F4A6D',  # Background color
        paper_bgcolor='#2F4A6D',
        font=dict(color="white", family='Arial'),  # Set axis and title text color and font
        height=500,  # Fixed height
        yaxis=dict(autorange=True),  # Automatically scale y-axis
    )

    return fig



def create_model_explanation_app(features,importances,original_accuracy,metric):
    # Create Dash app
    app = Dash(__name__)
    
    # Set a fixed height for the components
    fixed_height = 500  # Adjust this value as needed
    
    
    app.layout = html.Div(style={'display': 'flex', 'flexDirection': 'row'}, children=[
        dcc.Graph(
            id='bar-chart',
            config={'displayModeBar': False},  # Disable Plotly toolbar
            style={'height': f'{fixed_height}px', 'width': '70%'}  # Fixed height for the graph
        ),
        html.Div(
            style={
                'width': '30%',  # Adjust width as needed
                'backgroundColor': '#3a5b8c',  # Lighter shade of blue
                'color': 'white',
                'height': f'{fixed_height}px',  # Same fixed height as the graph
                'overflow': 'hidden'  # Prevent overflow
            },
            children=[
                html.P("Feature Selection", style={'fontFamily': 'Arial', 'padding-left': '5%', 'padding-right': '5%'}),
                dcc.Dropdown(
                    id='feature-dropdown',
                    options=[{'label': feature, 'value': feature} for feature in features],
                    value=features[:5],  # Default to the top 5 features
                    multi=True,
                    style={
                        'color': '#3a5b8c',  # Text color
                        'fontFamily': 'Arial',
                        'margin-top': '5%',
                        'padding-left': '5%',
                        'padding-right': '5%',
                    },
                    placeholder="Select features..."  # Optional placeholder text
                ),
                html.P("Interpretation Panel", style={'fontFamily': 'Arial', 'padding-left': '5%', 'padding-right': '5%'}),
                html.P(id='interpretation-text', 
                       style={'fontFamily': 'Arial', 'padding-left': '5%', 'padding-right': '5%', 'margin-top': '5%'},
                       children=f"Each feature importance corresponds to the decrease in {metric} which would happen if that feature was removed from the dataset. The greater the change in {metric}, the more important the feature is.")
                    
            ]
        )
    ])
    
    # Initialize variable to store the index of the clicked bar
    clicked_feature = None
    
    @app.callback(
    Output('bar-chart', 'figure'),
    Output('interpretation-text', 'children'),
    Output('feature-dropdown', 'value'),
    Input('bar-chart', 'clickData'),
    Input('feature-dropdown', 'value')
)
    def update_chart(clickData, selected_features):
        nonlocal clicked_feature
        interpretation_text = f"Each feature importance corresponds to the decrease in {metric} which would happen if that feature was removed from the dataset. The greater the change in {metric}, the more important the feature is."
        
        # Convert features to a list if it is a pandas Index
        features_list = list(features)

        # Create a list to hold the opacity values
        opacity = [1] * len(selected_features)  # Start with full opacity

        if clickData:
            clicked_feature = clickData['points'][0]['x']
            
            # Check if the clicked feature is in selected features
            if clicked_feature in selected_features:
                clicked_idx = selected_features.index(clicked_feature)

                # If the same bar is clicked again, reset
                if clicked_feature == clicked_feature:
                    opacity[clicked_idx] = 0.5  # Set lower opacity for the clicked bar
                else:
                    clicked_feature = None  # Reset clicked feature
                    opacity = [1] * len(selected_features)  # Reset opacity for all bars

                interpretation_text = f"If the feature, {clicked_feature} was removed from the model, the {metric} would go from {round(original_accuracy,3)} to {round(original_accuracy - importances[features_list.index(clicked_feature)], 3)}." 
                
        # Filter the data based on selected features
        filtered_features = [f for f in selected_features if f in features_list]
        filtered_importances = [importances[features_list.index(f)] for f in filtered_features]

        # Create bar chart
        fig = go.Figure(
            data=[go.Bar(
                x=filtered_features,
                y=filtered_importances,
                text=[f"{imp:.3f}" for imp in filtered_importances],  # Display values with 3 decimal places
                textposition='auto',
                marker_color=[f'rgba(128, 203, 196, {op})' for op in opacity],  # Use RGBA for colors with opacity
                hoverinfo='none',  # Disable hover information
                textfont=dict(color="white")  # Set text color to white
            )]
        )

        # Update layout with automatic y-axis scaling
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Features",
            yaxis_title=f"{metric} Decrease If Feature Removed",
            plot_bgcolor='#2F4A6D',  # Background color
            paper_bgcolor='#2F4A6D',
            font=dict(color="white", family='Arial'),  # Set axis and title text color and font
            height=fixed_height,  # Match the fixed height
            yaxis=dict(autorange=True),  # Automatically scale y-axis
        )

        # Prevent deselection of clicked feature in the dropdown
        if clicked_feature is not None and clicked_feature not in selected_features:
            # If trying to deselect a clicked feature, reset its transparency
            clicked_feature = None
            opacity = [1] * len(selected_features)

        return fig, interpretation_text, selected_features

    return app