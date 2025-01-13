
import pandas as pd
import numpy as np
import requests
import random

#CLOUD_FUNCTION_URL_INDIVIDUAL_NLP_EXPLANATION = "https://individual-nlp-explanation-api-gateway-1y81cisz.nw.gateway.dev/call_private_individual_nlp_explanation"
CLOUD_FUNCTION_URL_INDIVIDUAL_NLP_EXPLANATION = "https://europe-west2-decima2.cloudfunctions.net/call_private_individual_nlp_explanation"


def public_individual_nlp_explanation(text=0,start_index=-1,finish_index=-1):
    """
    Calls an external Google Cloud Function to retrieve NLP-based explanations for a specified segment of a text. 
    Sends a POST request to the specified API endpoint with the provided text and index range, receiving a message 
    with extracted bigrams and keys or an error message if the request fails.

    Parameters
    ----------
    text : str
        The input text for analysis.
    start_index : int
        The starting index for extracting bigrams from the text.
    finish_index : int
        The ending index for extracting bigrams from the text.

    Returns
    -------
    list or str
        - If the request is successful, returns a list containing:
            - bigrams : list of str
                Extracted bigram phrases from the specified text segment.
            - keys : list of str
                Key phrases or terms derived from the specified text segment.
        - If the request fails, returns an error message indicating the issue.

    Example
    -------
    To retrieve bigrams and keys from a text segment using specific start and finish indices:

    >>> text = "Natural language processing allows computers to understand human language."
    >>> start_index = 2
    >>> finish_index = 5
    >>> result = public_individual_nlp_explanation(text, start_index, finish_index)
    >>> print(result)
    # Output (example): [['natural language', 'language processing'], ['natural', 'processing']]

    Notes
    -----
    - `CLOUD_FUNCTION_URL_INDIVIDUAL_NLP_EXPLANATION` is the endpoint URL for the external API.
    - Returns `data['message']` if the status code is 200, otherwise returns `data['error']`.
    - Captures any exceptions during the API request, returning an error message if a network or other 
      unexpected error occurs.
    """
    try:
        # Send a POST request to the Google Cloud Function
        response = requests.post(CLOUD_FUNCTION_URL_INDIVIDUAL_NLP_EXPLANATION, json={"text" : text, "start_index" : start_index, "finish_index": finish_index})
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            return data['message']
        else:
            data = response.json()
            return data['error']
    except Exception as e:
        return f"Error: {e}"