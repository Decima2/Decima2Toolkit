import dash
from dash import dcc, html, Input, Output

def create_nlp_app(input_text_1, input_text_2, original_similarity, top_importance_subset, bottom_importance_subset):
    """
    Creates a Dash web application for visualizing NLP text similarity and the impact of key term pairs on similarity scores.
    The application displays the similarity score between two input texts and allows users to view and interact with term pairs 
    that increase or decrease similarity.

    Parameters
    ----------
    input_text_1 : str
        The first input text for similarity analysis.
    input_text_2 : str
        The second input text for similarity analysis.
    original_similarity : float
        The initial similarity score between the two texts.
    top_importance_subset : list of tuples
        A list of tuples representing the top similarity increasers. Each tuple should contain:
            - term_1 (str): A term from the first text.
            - term_2 (str): A term from the second text.
            - impact (float): The impact on similarity if these terms are removed.
    bottom_importance_subset : list of tuples
        A list of tuples representing the top similarity decreasers. Each tuple should contain:
            - term_1 (str): A term from the first text.
            - term_2 (str): A term from the second text.
            - impact (float): The impact on similarity if these terms are removed.

    Returns
    -------
    app : dash.Dash
        A Dash web application instance for interactive NLP similarity analysis.

    Application Interface
    ---------------------
    - Left Panel: Displays the top similarity increasers (terms seen as similar by the model).
      Clicking on a term pair shows the impact on the similarity score.
    - Middle Panel: Displays the input texts and the initial similarity score between them.
    - Right Panel: Displays the top similarity decreasers (terms seen as dissimilar by the model).
      Clicking on a term pair shows the impact on the similarity score.

    Example
    -------
    >>> app = create_nlp_app(
            input_text_1="Natural language processing allows computers to understand human language.",
            input_text_2="Machine learning algorithms help machines interpret human language.",
            original_similarity=0.85,
            top_importance_subset=[("language", "language", 0.02), ("processing", "learning", 0.015)],
            bottom_importance_subset=[("computers", "machines", -0.03), ("understand", "interpret", -0.02)]
        )
    >>> app.run_server(debug=True)

    Notes
    -----
    - The application is designed to be run in a Dash-compatible environment.
    - Uses Google Fonts to load the "Inconsolata" font for styling.
    """

    app = dash.Dash(__name__)

    # Load Inconsolata font
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            <link href="https://fonts.googleapis.com/css2?family=Inconsolata:wght@400;700&display=swap" rel="stylesheet">
            {%metas%}
            <title>Text Similarity App</title>
            {%css%}
        </head>
        <body style="font-family: 'Inconsolata', monospace;">
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''

    # Middle panel
    middle_panel = html.Div(
        style={
            'flex': '1 1 0',
            'backgroundColor': '#3a5b8c',
            'display': 'flex',
            'flexDirection': 'column',
            'alignItems': 'center',
            'overflowY': 'auto',  # Enable scrolling if content is long
            'height': 'auto',     # Allow height to adjust with content
            'textAlign': 'center',
            'padding': '5%',      # Ensure some padding around content
            'marginBottom': '5%'
        },
        children=[
            html.Div(f"The Similarity Between Both Texts Is {round(float(original_similarity),2)}", style={
                'color': 'white',
                'fontSize': 18,
                'fontWeight': 'bold',
                'marginTop': '5%',
            }),
            html.Div("Text 1", style={
                'color': 'white',
                'fontSize': 16,
                'marginTop': '5%',
                'textDecoration': 'none'
            }),
            html.Div(f"{input_text_1}", style={
                'color': 'white',
                'fontSize': 16,
                'fontWeight': 'bold',
                'marginTop': '5%',
                'padding': '5%',
            }),
            html.Div("Text 2", style={
                'color': 'white',
                'fontSize': 16,
                'marginTop': '15%',
                'fontWeight': 'bold',
            }),
            html.Div(f"{input_text_2}", style={
                'color': 'white',
                'fontSize': 16,
                'fontWeight': 'bold',
                'marginTop': '5%',
                'padding': '5%',
            }),
        ]
    )

    # Left panel
    left_panel = html.Div(
        id='left_panel',
        style={
            'flex': '1 1 0',
            'backgroundColor': 'white',
            'display': 'flex',
            'flexDirection': 'column',
            'alignItems': 'center',
            'justifyContent': 'start',
            'height': '100%',
            'overflowY': 'auto',
            'textAlign': 'center',
            'border': '4px solid #3a5b8c',
        },
        children=[
            html.Div("Top Similarity Increasers", style={
                'fontSize': 18, 
                'fontWeight': 'bold', 
                'marginBottom': '10px', 
                'color': '#3a5b8c'
            }),
            html.Div("These pairs of terms are seen by the model as similar", style={
                'fontSize': 18, 
                'fontWeight': 'bold', 
                'marginBottom': '10px', 
                'color': '#3a5b8c'
            }),
            html.Ul(style={'listStyleType': 'none', 'padding': '0', 'textAlign': 'center'}, children=[
                html.Li(style={'color': 'green', 'cursor': 'pointer', 'margin': '10%'}, children=[
                    html.Span(f"{top_importance_subset[i][0]} <--> {top_importance_subset[i][1]}"),
                    dcc.Store(id={'type': 'top-item-store', 'index': i}, data=top_importance_subset[i][2])
                ], id={'type': 'top-item', 'index': i})
                for i in range(len(top_importance_subset))
            ]),
            html.Div(id='similarity-score-left', style={
                'color': 'green',
                'fontSize': 16,
                'marginTop': '10px'
            })
        ]
    )

    # Right panel
    right_panel = html.Div(
        id='right_panel',
        style={
            'flex': '1 1 0',
            'backgroundColor': 'white',
            'display': 'flex',
            'flexDirection': 'column',
            'alignItems': 'center',
            'justifyContent': 'start',
            'height': '100%',
            'overflowY': 'auto',
            'textAlign': 'center',
            'border': '4px solid #3a5b8c',
        },
        children=[
            html.Div("Top Similarity Decreasers", style={
                'fontSize': 18, 
                'fontWeight': 'bold', 
                'marginBottom': '10px', 
                'color': '#3a5b8c'
            }),
            html.Div("These pairs of terms are seen by the model as dissimilar", style={
                'fontSize': 18, 
                'fontWeight': 'bold', 
                'marginBottom': '10px', 
                'color': '#3a5b8c'
            }),
            html.Ul(style={'listStyleType': 'none', 'padding': '0', 'textAlign': 'center'}, children=[
                html.Li(style={'color': 'red', 'cursor': 'pointer', 'margin': '10%'}, children=[
                    html.Span(f"{bottom_importance_subset[i][0]} <--> {bottom_importance_subset[i][1]}"),
                    dcc.Store(id={'type': 'bottom-item-store', 'index': i}, data=bottom_importance_subset[i][2])
                ], id={'type': 'bottom-item', 'index': i})
                for i in range(len(bottom_importance_subset))
            ]),
            html.Div(id='similarity-score-right', style={
                'color': 'red',
                'fontSize': 16,
                'marginTop': '10px'
            }),
        ]
    )

    # Combine the left, middle, and right panels into a complete layout
    app.layout = html.Div(style={'display': 'flex', 'height': '100vh', 'overflowY': 'auto'}, children=[left_panel, middle_panel, right_panel])

    # Callback function for updating similarity scores (unchanged)
    @app.callback(
        [Output('similarity-score-left', 'children'),
         Output('similarity-score-right', 'children')],
        [Input({'type': 'top-item', 'index': dash.dependencies.ALL}, 'n_clicks_timestamp'),
         Input({'type': 'bottom-item', 'index': dash.dependencies.ALL}, 'n_clicks_timestamp')],
        prevent_initial_call=True
    )
    def update_similarity_score(top_clicks, bottom_clicks):
        left_score = dash.no_update
        right_score = dash.no_update

        # Find the most recently clicked item in the left panel
        if any(top_clicks):
            latest_top_click = max((click for click in top_clicks if click), default=None)
            if latest_top_click is not None:
                latest_top_index = top_clicks.index(latest_top_click)
                similarity_score = round(top_importance_subset[latest_top_index][2], 2)
                left_score = f"The similiarity between these terms is : {similarity_score}"

        # Find the most recently clicked item in the right panel
        if any(bottom_clicks):
            latest_bottom_click = max((click for click in bottom_clicks if click), default=None)
            if latest_bottom_click is not None:
                latest_bottom_index = bottom_clicks.index(latest_bottom_click)
                similarity_score = round(bottom_importance_subset[latest_bottom_index][2], 2)
                right_score = f"The similarity between these terms is {similarity_score}"

        return left_score, right_score

    return app
