import pytest
import numpy as np
from decima2.outcome.nlp_explanations.explanation_generation import individual_nlp_explanation  # Replace with actual module path
from decima2.outcome.nlp_explanations.nlp_private_function import public_individual_nlp_explanation

def test_cloud_function():
    text1 = "espresso machine with milk thingy"
    response = public_individual_nlp_explanation(text1, 1, 3)
    print(response)
    assert isinstance(response,list)



def test_individual_nlp_explanation_bert():

    text1 = "espresso machine with milk thingy"
    #text2 = "this is a description of a coffeee machine, I am making it nice and long so we can test out our functions this coffee machine has lots of capabilities like a milk frother and a milk steamer."
    text2 = "torty sivill loves espresso thingy"

    model_name = 'bert-base-uncased'  # You can change this to any model name from Hugging Face

    similarity_increasers, similarity_decreasers = individual_nlp_explanation(text1, text2, model_name,output='text')
    
    assert isinstance(similarity_increasers, list)
    assert isinstance(similarity_decreasers, list)


def test_individual_nlp_explanation_distilbert():

    text1 = "espresso"
    #text2 = "this is a description of a coffeee machine, I am making it nice and long so we can test out our functions this coffee machine has lots of capabilities like a milk frother and a milk steamer."
    text2 = "machine"

    model_name = 'distilbert-base-uncased'  # You can change this to any model name from Hugging Face

    similarity_increasers, similarity_decreasers = individual_nlp_explanation(text1, text2, model_name,output='text')
    
    assert isinstance(similarity_increasers, list)
    assert isinstance(similarity_decreasers, list)


def test_individual_nlp_explanation_roberta():

    text1  = "torty sivill loves espresso thingy torty sivill loves espresso thingy torty sivill loves espresso thingy torty sivill loves espresso thingy torty sivill loves espresso thingy espresso thingy torty sivill loves espresso thingy espresso thingy torty sivill loves espresso thingy espresso thingy torty sivill loves espresso thingy espresso thingy torty sivill loves espresso thingy espresso thingy torty sivill loves espresso thingy espresso thingy torty sivill loves espresso thingy espresso thingy torty sivill loves espresso thingy espresso thingy torty sivill loves espresso thingy"

    #text2 = "this is a description of a coffeee machine, I am making it nice and long so we can test out our functions this coffee machine has lots of capabilities like a milk frother and a milk steamer."
    text2 = "katie sivill loves espresso thingy katie sivill loves espresso thingy torty sivill loves espresso thingy torty sivill loves espresso thingy torty sivill loves espresso thingy espresso thingy torty sivill loves espresso thingy espresso thingy torty sivill loves espresso thingy espresso thingy torty sivill loves espresso thingy espresso thingy torty sivill loves espresso thingy espresso thingy torty sivill loves espresso thingy espresso thingy torty sivill loves espresso thingy espresso thingy torty sivill loves espresso thingy espresso thingy torty sivill loves espresso thingy"


    model_name = 'roberta-base'  # You can change this to any model name from Hugging Face

    similarity_increasers, similarity_decreasers = individual_nlp_explanation(text1, text2, model_name,output='text')
    
    assert isinstance(similarity_increasers, list)
    assert isinstance(similarity_decreasers, list)

