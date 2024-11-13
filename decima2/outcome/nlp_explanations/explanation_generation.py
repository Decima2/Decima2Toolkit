import numpy as np
from scipy.spatial.distance import cosine

import sys

from decima2.visualisation.outcome.visualisation_nlp import create_nlp_app
from decima2.utils.nlp_utils import EmbeddingModel,measure_embedding_similarity

from decima2.outcome.nlp_explanations.nlp_private_function import public_individual_nlp_explanation

def individual_nlp_explanation(text1, text2, model_name, output='dynamic') :
    """
    Analyzes the similarities and differences between two input texts by extracting key bigrams, generating 
    embeddings, and comparing them to identify impactful pairs. Returns pairs that increase and decrease similarity 
    between the two texts, or generates a dynamic visualization app.

    Parameters
    ----------
    text1 : str
        The first text input to analyze.
    text2 : str
        The second text input to analyze.
    model_name : str
        The name of the embedding model to use for encoding text bigrams.
    output : str, optional, default='dynamic'
        Specifies the output type:
        - 'text': returns lists of impactful bigram pairs with similarity effects.
        - 'dynamic': returns an interactive app for visualizing similarity effects.

    Returns
    -------
    tuple or app
        If `output` is 'text', returns a tuple of two lists:
        - similarity_increasers : list of tuples
            Pairs of bigrams that increase the similarity between `text1` and `text2`.
        - similarity_decreasers : list of tuples
            Pairs of bigrams that decrease the similarity between `text1` and `text2`.
        
        If `output` is 'dynamic', returns a Flask app for interactive visualization of 
        similarity increasers and decreasers.

    Examples
    --------

    Example 1: Dynamic Visualization Output
    ---------------------------------------
    To generate a dynamic visualization app for a side-by-side comparison of similarity-increasing 
    and similarity-decreasing bigrams:

    >>> text1 = "Data science and machine learning are growing fields."
    >>> text2 = "Machine learning is an important aspect of data-driven research."
    >>> model_name = "distilbert-base-uncased"
    >>> app = individual_nlp_explanation(text1, text2, model_name, output='dynamic')
    >>> app.run()  # This will start a web server for visualizing similarity comparisons.

    Example 2: Text Output
    ----------------------
    To get the list of bigram pairs that increase and decrease similarity between two texts:

    >>> text1 = "Natural language processing is a fascinating field of AI."
    >>> text2 = "The field of artificial intelligence includes NLP and machine learning."
    >>> model_name = "bert-base-uncased"
    >>> similarity_increasers, similarity_decreasers = individual_nlp_explanation(text1, text2, model_name, outcome='')
    >>> print("Pairs that increase similarity:", similarity_increasers)
    >>> print("Pairs that decrease similarity:", similarity_decreasers)

    Notes
    -----
    - In the second example, `similarity_increasers` and `similarity_decreasers` contain tuples 
      of (bigram_from_text1, bigram_from_text2, similarity_effect) for bigram pairs.
    - In the first example, running `app.run()` launches an interactive Flask app for viewing 
      and exploring the similarity effects in a web interface.
    """

    # Step 1: Extract concepts

    top_k = 10

    if len(text1) > 300:
        text1 = text1[0:300]
    if len(text2) > 300:
        text2 = text2[0:300]

    word_list1 = text1.split(" ")
    word_list2 = text2.split(" ")

    length_list_1 = len(word_list1)
    length_list_2 = len(word_list2)

    
    # start and finish index for text1 
    start_index_1 = 1
    if length_list_1 < 3:
        start_index_1 = 1
    #finish_index_1 = max(2,length_list_1 - 1)
    #if finish_index_1 > 11:
    #    finish_index_1 = 10
    finish_index_1 = 4

    # start and finish index for text1 
    start_index_2 = 1
    if length_list_2 < 3:
        start_index_2 = 1
    #finish_index_2 = max(2,length_list_2 - 1)
    #if finish_index_2 > 11:
    #    finish_index_2 = 10
    finish_index_2 = 4

    response = public_individual_nlp_explanation(text1, start_index_1, finish_index_1)
    [bigrams1, keys1] = response
    response = public_individual_nlp_explanation(text2, start_index_2, finish_index_2)
    [bigrams2, keys2] = response

    # Step 2: Generate embeddings
    model = EmbeddingModel(model_name)
    embeddings_bi_1 = model.encode(bigrams1)
    embeddings_bi_2 = model.encode(bigrams2)

    # Step 3: Calculate cosine similarities
    similarities = []
    for emb1 in embeddings_bi_1:
        similarities.append([1 - cosine(emb1.numpy(), emb2.numpy()) for emb2 in embeddings_bi_2])
    
    similarities = np.array(similarities)

    # Step 4: Prepare impactful pairs with their similarities

    original_similarity = measure_embedding_similarity(text1,text2,model_name)

    impactful_pairs = []
    for i in range(len(bigrams1)):
        for j in range(len(bigrams2)):
            impactful_pairs.append((keys1[i], keys2[j], similarities[i][j]))

    # Step 5: Sort and get top and bottom results
    impactful_pairs = sorted(impactful_pairs, key=lambda x: x[2])
    
    top_pairs = impactful_pairs[-top_k:] # Get top_k pairs (highest similarity)
    top_pairs.reverse()
    bottom_pairs = impactful_pairs[:top_k]  # Get bottom_k pairs (lowest similarity)
    similarity_increasers = []
    
    for pair in top_pairs:
        tolerance = 1e-2  # Define a small tolerance
        if pair[2] > original_similarity:
            similarity_increasers.append(pair)
   
    similarity_decreasers = []
    for pair in bottom_pairs:
        tolerance = 1e-2  # Define a small tolerance
        if (pair[2]) < original_similarity:
            similarity_decreasers.append(pair)

    if output == 'text':
        return similarity_increasers, similarity_decreasers

    if output == 'dynamic':
        app = create_nlp_app(text1, text2, original_similarity, similarity_increasers, similarity_decreasers)
        return app



