from dash import Dash, dcc, html, Input, Output
import dash_table
from dash import callback_context  # Import callback_context


def create_individual_explanation_regression_app(selected_instance, increaser_dict, decreaser_dict, original_score, increased_score, decreased_score):
    # Create Dash app
    app = Dash(__name__)

    # Set a fixed height for the components
    fixed_height = 100  # Adjust this value as needed

    # Extract features (keys) and importances (values) from the dictionaries
    increaser_features_list = list(increaser_dict.keys())
    increaser_values = list(increaser_dict.values())

    decreaser_features_list = list(decreaser_dict.keys())
    decreaser_values = list(decreaser_dict.values())

    # Layout with headings above and below the table
    app.layout = html.Div(
        style={'backgroundColor': 'white', 'padding': '10px', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'},  # White background and flex layout
        children=[
            # Heading: "Selected Instance To Explain"
            html.H3(
                'Selected instance to explain', 
                style={
                    #'color': 'black',  # Change text color to black for visibility on white background
                    'color' : '#2F4A6D',
                    'fontFamily': 'Arial', 
                    'fontWeight': 'bold',
                    'marginBottom': '5px'  # Reduced margin below the heading
                }
            ),
            
            # Table displaying selected instance (horizontal table)
            dash_table.DataTable(
                id='selected-instance-table',
                columns=[
                    {"name": feature, "id": feature} for feature in selected_instance.keys()
                ],
                data=[{
                    feature: value for feature, value in selected_instance.items()
                }],
                style_table={'height': f'{fixed_height}px', 'overflowY': 'auto', 'width': '100%'},  # Table width set to 100%
                style_cell={
                    'textAlign': 'center',
                    'padding': '8px',  # Reduced padding
                    'backgroundColor': 'white',  # Match background color
                    'color' : '#2F4A6D',  # Text color for the table
                    'fontFamily': 'Arial',
                    'whiteSpace': 'normal',  # Allow the text to wrap within cells
                    'overflow': 'hidden'  # Prevent text from overflowing
                },
                style_header={
                    'backgroundColor': 'white',  # Same background for the header
                    'fontWeight': 'bold',
                    'color' : '#2F4A6D',  # Change header text color to black
                }
            ),

            # Heading: "Model Prediction: 1"
            html.H3(
                f"Current prediction: {round(original_score,2)}", 
                style={
                    'color' : '#2F4A6D',  # Change text color to black
                    'fontFamily': 'Arial', 
                    'fontWeight': 'bold',
                    'marginTop': '5px'  # Very small margin above this heading
                }
            ),
            
            # Heading: "What change do you want to make to the outcome?"
            html.H4(
                'What change do you want to make to the outcome?', 
                style={
                    'color' : '#2F4A6D',  # Change text color to black
                    'fontFamily': 'Arial', 
                    'fontWeight': 'bold',
                    'marginTop': '5px',  # Reduced margin top
                    'marginBottom': '5px'  # Reduced margin bottom
                }
            ),

            # Buttons: "Increase Outcome" and "Decrease Outcome"
            html.Div(
                style={
                    'display': 'flex',
                    'gap': '20px',  # Reduced gap between buttons
                    'justifyContent': 'center',  # Center the buttons horizontally
                },
                children=[
                    # Increase Outcome Button
                    html.Button(
                        'Increase Outcome', 
                        id='increase-outcome-button',
                        style={
                            'backgroundColor': '#28a745',  # Green color for increase
                            'color': 'white',
                            'padding': '8px 16px',  # Reduced padding
                            'fontSize': '14px',  # Smaller font size
                            'border': 'none',
                            'borderRadius': '5px',
                            'cursor': 'pointer',
                        }
                    ),
                    # Decrease Outcome Button
                    html.Button(
                        'Decrease Outcome', 
                        id='decrease-outcome-button',
                        style={
                            'backgroundColor': '#dc3545',  # Red color for decrease
                            'color': 'white',
                            'padding': '8px 16px',  # Reduced padding
                            'fontSize': '14px',  # Smaller font size
                            'border': 'none',
                            'borderRadius': '5px',
                            'cursor': 'pointer',
                        }
                    ),
                ]
            ),

            # Placeholder for the result table
            html.Div(id='output-table', style={'marginTop': '10px'})
        ]
    )

    # Callback to update the table based on button click
    @app.callback(
        Output('output-table', 'children'),
        [Input('increase-outcome-button', 'n_clicks'),
         Input('decrease-outcome-button', 'n_clicks')]
    )
    def display_table(increase_clicks, decrease_clicks):
        # Initialize variables to avoid errors if no button is clicked
        ctx = callback_context  # Use callback_context instead of dash.callback_context

        if not ctx.triggered:
            return None  # No action yet

        # Check which button was clicked
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # Check if increaser_dict is empty when the "Increase Outcome" button is clicked
        if button_id == 'increase-outcome-button':
            if not increaser_dict:
                # If increaser_dict is empty, show the message
                return html.Div(
                    "There is no way to increase the outcome",
                    style={
                        'color' : '#2F4A6D',  # Change text color to black for visibility on white background
                        'fontFamily': 'Arial',
                        'fontSize': '18px',
                        'textAlign': 'center'
                    }
                )
            else:
                # Display the increaser dict in a table with green text
                result_table = dash_table.DataTable(
                    columns=[
                        {"name": 'Feature', "id": 'Feature'},
                        {"name": 'Recommended Change', "id": 'Increase Effect'}
                    ],
                    data=[{
                        'Feature': feature,
                        'Increase Effect': round(value, 3)
                    } for feature, value in zip(increaser_features_list, increaser_values)],
                    style_table={'width': '100%'},
                    style_cell={
                        'textAlign': 'center',
                        'padding': '8px',  # Reduced padding
                        'backgroundColor': 'white',
                        'color': 'green',  # Green text for increase
                        'fontFamily': 'Arial',
                        'whiteSpace': 'normal',
                        'overflow': 'hidden'
                    },
                    style_header={
                        'backgroundColor': 'white',
                        'fontWeight': 'bold',
                        'color' : '#2F4A6D',
                    }
                )

                # Add the increased probability message below the table
                increased_probability_message = html.Div(
                    f"The resulting prediction is {round(increased_score,2)}",
                    style={
                        'color': 'green',  # Green text for increase
                        'fontFamily': 'Arial',
                        'fontSize': '18px',
                        'textAlign': 'center',
                        'marginTop': '10px'
                    }
                )

                # Return both the table and the increased probability message
                return html.Div([result_table, increased_probability_message])

        elif button_id == 'decrease-outcome-button':
            # Check if decreaser_dict is empty when the "Decrease Outcome" button is clicked
            if not decreaser_dict:
                # If decreaser_dict is empty, show the message
                return html.Div(
                    "There is no way to decrease the outcome",
                    style={
                        'color' : '#2F4A6D',  # Change text color to black for visibility on white background
                        'fontFamily': 'Arial',
                        'fontSize': '18px',
                        'textAlign': 'center'
                    }
                )
            else:
                # Display the decreaser dict in a table with red text
                result_table = dash_table.DataTable(
                    columns=[
                        {"name": 'Feature', "id": 'Feature'},
                        {"name": 'Recommended Change', "id": 'Decrease Effect'}
                    ],
                    data=[{
                        'Feature': feature,
                        'Decrease Effect': round(value, 3)
                    } for feature, value in zip(decreaser_features_list, decreaser_values)],
                    style_table={'width': '100%'},
                    style_cell={
                        'textAlign': 'center',
                        'padding': '8px',  # Reduced padding
                        'backgroundColor': 'white',
                        'color': 'red',  # Red text for decrease
                        'fontFamily': 'Arial',
                        'whiteSpace': 'normal',
                        'overflow': 'hidden'
                    },
                    style_header={
                        'backgroundColor': 'white',
                        'fontWeight': 'bold',
                        'color' : '#2F4A6D',
                    }
                )

                # Add the decreased probability message below the table
                decreased_probability_message = html.Div(
                    f"The resulting prediction is {round(decreased_score,2)}",
                    style={
                        'color': 'red',  # Red text for decrease
                        'fontFamily': 'Arial',
                        'fontSize': '18px',
                        'textAlign': 'center',
                        'marginTop': '10px'
                    }
                )

                # Return both the table and the decreased probability message
                return html.Div([result_table, decreased_probability_message])

    return app

from dash import Dash, dcc, html, Input, Output
import dash_table
from dash import callback_context  # Import callback_context

def create_individual_explanation_classification_app(selected_instance, increaser_dict, decreaser_dict, axis, original_score, increased_score, decreased_score):
    # Create Dash app
    app = Dash(__name__)

    # Set a fixed height for the components
    fixed_height = 100  # Adjust this value as needed

    # Extract features (keys) and importances (values) from the dictionaries
    increaser_features_list = list(increaser_dict.keys())
    increaser_values = list(increaser_dict.values())

    decreaser_features_list = list(decreaser_dict.keys())
    decreaser_values = list(decreaser_dict.values())

    # Layout with headings above and below the table
    app.layout = html.Div(
        style={'backgroundColor': 'white', 'padding': '10px', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'},  # White background and flex layout
        children=[
            # Heading: "Selected Instance To Explain"
            html.H3(
                'Selected instance to explain', 
                style={
                    #'color': 'black',  # Change text color to black for visibility on white background
                    'color' : '#2F4A6D',
                    'fontFamily': 'Arial', 
                    'fontWeight': 'bold',
                    'marginBottom': '5px'  # Reduced margin below the heading
                }
            ),
            
            # Table displaying selected instance (horizontal table)
            dash_table.DataTable(
                id='selected-instance-table',
                columns=[
                    {"name": feature, "id": feature} for feature in selected_instance.keys()
                ],
                data=[{
                    feature: value for feature, value in selected_instance.items()
                }],
                style_table={'height': f'{fixed_height}px', 'overflowY': 'auto', 'width': '100%'},  # Table width set to 100%
                style_cell={
                    'textAlign': 'center',
                    'padding': '8px',  # Reduced padding
                    'backgroundColor': 'white',  # Match background color
                    'color' : '#2F4A6D',  # Text color for the table
                    'fontFamily': 'Arial',
                    'whiteSpace': 'normal',  # Allow the text to wrap within cells
                    'overflow': 'hidden'  # Prevent text from overflowing
                },
                style_header={
                    'backgroundColor': 'white',  # Same background for the header
                    'fontWeight': 'bold',
                    'color' : '#2F4A6D',  # Change header text color to black
                }
            ),

            # Heading: "Model Prediction: 1"
            html.H3(
                f"Current class prediction: {axis} with probability: {round(original_score,2)}", 
                style={
                    'color' : '#2F4A6D',  # Change text color to black
                    'fontFamily': 'Arial', 
                    'fontWeight': 'bold',
                    'marginTop': '5px'  # Very small margin above this heading
                }
            ),
            
            # Heading: "What change do you want to make to the outcome?"
            html.H4(
                'What change do you want to make to the outcome?', 
                style={
                    'color' : '#2F4A6D',  # Change text color to black
                    'fontFamily': 'Arial', 
                    'fontWeight': 'bold',
                    'marginTop': '5px',  # Reduced margin top
                    'marginBottom': '5px'  # Reduced margin bottom
                }
            ),

            # Buttons: "Increase Outcome" and "Decrease Outcome"
            html.Div(
                style={
                    'display': 'flex',
                    'gap': '20px',  # Reduced gap between buttons
                    'justifyContent': 'center',  # Center the buttons horizontally
                },
                children=[
                    # Increase Outcome Button
                    html.Button(
                        'Increase Outcome', 
                        id='increase-outcome-button',
                        style={
                            'backgroundColor': '#28a745',  # Green color for increase
                            'color': 'white',
                            'padding': '8px 16px',  # Reduced padding
                            'fontSize': '14px',  # Smaller font size
                            'border': 'none',
                            'borderRadius': '5px',
                            'cursor': 'pointer',
                        }
                    ),
                    # Decrease Outcome Button
                    html.Button(
                        'Decrease Outcome', 
                        id='decrease-outcome-button',
                        style={
                            'backgroundColor': '#dc3545',  # Red color for decrease
                            'color': 'white',
                            'padding': '8px 16px',  # Reduced padding
                            'fontSize': '14px',  # Smaller font size
                            'border': 'none',
                            'borderRadius': '5px',
                            'cursor': 'pointer',
                        }
                    ),
                ]
            ),

            # Placeholder for the result table
            html.Div(id='output-table', style={'marginTop': '10px'})
        ]
    )

    # Callback to update the table based on button click
    @app.callback(
        Output('output-table', 'children'),
        [Input('increase-outcome-button', 'n_clicks'),
         Input('decrease-outcome-button', 'n_clicks')]
    )
    def display_table(increase_clicks, decrease_clicks):
        # Initialize variables to avoid errors if no button is clicked
        ctx = callback_context  # Use callback_context instead of dash.callback_context

        if not ctx.triggered:
            return None  # No action yet

        # Check which button was clicked
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # Check if increaser_dict is empty when the "Increase Outcome" button is clicked
        if button_id == 'increase-outcome-button':
            if not increaser_dict:
                # If increaser_dict is empty, show the message
                return html.Div(
                    "There is no way to increase the outcome",
                    style={
                        'color' : '#2F4A6D',  # Change text color to black for visibility on white background
                        'fontFamily': 'Arial',
                        'fontSize': '18px',
                        'textAlign': 'center'
                    }
                )
            else:
                # Display the increaser dict in a table with green text
                result_table = dash_table.DataTable(
                    columns=[
                        {"name": 'Feature', "id": 'Feature'},
                        {"name": 'Recommended Change', "id": 'Increase Effect'}
                    ],
                    data=[{
                        'Feature': feature,
                        'Increase Effect': round(value, 3)
                    } for feature, value in zip(increaser_features_list, increaser_values)],
                    style_table={'width': '100%'},
                    style_cell={
                        'textAlign': 'center',
                        'padding': '8px',  # Reduced padding
                        'backgroundColor': 'white',
                        'color': 'green',  # Green text for increase
                        'fontFamily': 'Arial',
                        'whiteSpace': 'normal',
                        'overflow': 'hidden'
                    },
                    style_header={
                        'backgroundColor': 'white',
                        'fontWeight': 'bold',
                        'color' : '#2F4A6D',
                    }
                )

                # Add the increased probability message below the table
                increased_probability_message = html.Div(
                    f"The resulting predicted probability of class {axis} is {round(increased_score,2)}",
                    style={
                        'color': 'green',  # Green text for increase
                        'fontFamily': 'Arial',
                        'fontSize': '18px',
                        'textAlign': 'center',
                        'marginTop': '10px'
                    }
                )

                # Return both the table and the increased probability message
                return html.Div([result_table, increased_probability_message])

        elif button_id == 'decrease-outcome-button':
            # Check if decreaser_dict is empty when the "Decrease Outcome" button is clicked
            if not decreaser_dict:
                # If decreaser_dict is empty, show the message
                return html.Div(
                    "There is no way to decrease the outcome",
                    style={
                        'color' : '#2F4A6D',  # Change text color to black for visibility on white background
                        'fontFamily': 'Arial',
                        'fontSize': '18px',
                        'textAlign': 'center'
                    }
                )
            else:
                # Display the decreaser dict in a table with red text
                result_table = dash_table.DataTable(
                    columns=[
                        {"name": 'Feature', "id": 'Feature'},
                        {"name": 'Recommended Change', "id": 'Decrease Effect'}
                    ],
                    data=[{
                        'Feature': feature,
                        'Decrease Effect': round(value, 3)
                    } for feature, value in zip(decreaser_features_list, decreaser_values)],
                    style_table={'width': '100%'},
                    style_cell={
                        'textAlign': 'center',
                        'padding': '8px',  # Reduced padding
                        'backgroundColor': 'white',
                        'color': 'red',  # Red text for decrease
                        'fontFamily': 'Arial',
                        'whiteSpace': 'normal',
                        'overflow': 'hidden'
                    },
                    style_header={
                        'backgroundColor': 'white',
                        'fontWeight': 'bold',
                        'color' : '#2F4A6D',
                    }
                )

                # Add the decreased probability message below the table
                decreased_probability_message = html.Div(
                    f"The resulting predicted probability of class {axis} is {round(decreased_score,2)}",
                    style={
                        'color': 'red',  # Red text for decrease
                        'fontFamily': 'Arial',
                        'fontSize': '18px',
                        'textAlign': 'center',
                        'marginTop': '10px'
                    }
                )

                # Return both the table and the decreased probability message
                return html.Div([result_table, decreased_probability_message])

    return app


