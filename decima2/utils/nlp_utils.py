"""
Module for embedding-based NLP similarity analysis using pre-trained transformer models.

This module provides tools to:
- Generate sentence embeddings using a specified transformer model.
- Measure similarity between embeddings using cosine similarity.

Dependencies
------------
- sentence_transformers
- transformers
- torch
- scipy

Classes
-------
EmbeddingModel
    Loads a specified transformer model, tokenizes input text, and generates sentence embeddings.

Functions
---------
measure_embedding_similarity(text1, text2, model_name)
    Measures cosine similarity between embeddings of two input texts using the specified model.

Examples
--------
To create an `EmbeddingModel` instance and encode a list of sentences:

>>> model = EmbeddingModel("bert-base-uncased")
>>> embeddings = model.encode(["This is a test sentence.", "Here is another sentence."])
>>> print(embeddings.shape)
# Output: torch.Size([2, hidden_size])

To calculate the similarity between two sentences:

>>> similarity = measure_embedding_similarity("I love programming.", "Coding is my passion.", "bert-base-uncased")
>>> print(f"Cosine Similarity: {similarity:.4f}")
# Output: Cosine Similarity: 0.7683 (example value)

"""

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine

class EmbeddingModel:
    """
    A class to generate embeddings for text using a specified pre-trained transformer model.

    Parameters
    ----------
    model_name : str
        The name of the pre-trained transformer model to load. This should match a model 
        available in the Hugging Face model hub.

    Methods
    -------
    encode(sentences)
        Generates embeddings for a list of sentences, returning the embeddings of the [CLS] token.
    """

    def __init__(self, model_name):
        """
        Initializes the EmbeddingModel with a specified transformer model by loading its 
        tokenizer and model.

        Parameters
        ----------
        model_name : str
            The name of the transformer model (e.g., "bert-base-uncased") to load from 
            Hugging Face’s model hub.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set the model to evaluation mode to disable dropout and gradient updates

    def encode(self, sentences):
        """
        Encodes a list of sentences into embeddings.

        Parameters
        ----------
        sentences : list of str
            A list of sentences or text snippets to encode.

        Returns
        -------
        torch.Tensor
            A tensor of shape (batch_size, hidden_size) containing embeddings for each sentence, 
            extracted from the [CLS] token's representation.
        """
        inputs = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():  # Disable gradient calculation for efficiency
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # Return embeddings for the [CLS] token


def measure_embedding_similarity(text1, text2, model_name):
    """
    Measures the cosine similarity between the embeddings of two input texts.

    Parameters
    ----------
    text1 : str
        The first text input to compare.
    text2 : str
        The second text input to compare.
    model_name : str
        The name of the transformer model to use for generating embeddings.

    Returns
    -------
    float
        The cosine similarity score between the embeddings of `text1` and `text2`, ranging 
        from -1 (opposite) to 1 (identical).

    Example
    -------
    >>> similarity = measure_embedding_similarity("Hello world", "Hi there", "bert-base-uncased")
    >>> print(f"Similarity score: {similarity:.4f}")
    """
    model = EmbeddingModel(model_name)
    emb1 = model.encode([text1])  # Get embedding for text1 as a batch of 1
    emb2 = model.encode([text2])  # Get embedding for text2 as a batch of 1

    emb1_flat = emb1.squeeze().numpy()  # Convert to a 1D numpy array
    emb2_flat = emb2.squeeze().numpy()  # Convert to a 1D numpy array

    return 1 - cosine(emb1_flat, emb2_flat)
