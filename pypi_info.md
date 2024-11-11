
# Decima2 AI Evaluation Toolkit

## Introduction
Welcome to the Decima2 AI Evaluation Toolkit — a comprehensive suite of tools designed to empower developers with the insights needed to effectively evaluate and enhance machine learning models. Our toolkit focuses on making complex concepts in machine learning intuitive and accessible, allowing users to derive meaningful insights without the steep learning curve often associated with advanced analytics.



## Table of Contents
1. [Installation](#installation)
2. [Model Tools](#model-tools)
   - [Model Feature Importance](#model-feature-importance)
   - [Grouped Feature Importance](#grouped-feature-importance)
3. [Data Tools](#data-tools)
4. [Outcome Tools](#outcome-tools)
	- [Individual NLP Explanations](#individual-nlp-explanations)
5. [License](#license)
6. [Contributing](#contributing)
7. [Contact](#contact)


## Installation

You can install the package using pip:

<pre>
pip install decima2
</pre>


## Model Tools

Gain insights into how your models make predictions with clear, interpretable visualizations and explanations, making it easier to communicate results. 


### Model Feature Importance
For Tabular Data 

This tool allows users to examine which features were most important for their model's perfomance. Given a numerical dataset and pre-trained model, the model_feature_importance module returns either a textual or graphical representation of which features were most important. 

#### Instructions
For detailed usage instructions and to explore how the module works check out our [Developer Docs](https://docs.decima2.co.uk/docs/explanation/model-feature-importance) 
#### Tutorial 
To explore tutorials on model feature importance including motivations and comparisons with SHAP check out our [Jupyter Notebooks](https://github.com/Decima2/Decima2Toolkit/tree/main/examples/model_insights/model_explanations)  



### Grouped Feature Importance
For Tabular Data 


<pre>
from decima2 import grouped_feature_importance
</pre>

This tool builds on model_feature_importance to give users an insight into which features were most influential for their model over a **select group of data**. For example, a user may want to compare the most important feature across men and women in their data, or people with an income over 65k and under 65k. 

#### Instructions
For detailed usage instructions and to explore how the module works check out our [Developer Docs](https://docs.decima2.co.uk/docs/explanation/grouped-feature-importance) 
#### Tutorial 
To explore tutorials on grouped feature importance including motivation and use-cases check out our [Jupyter Notebooks](https://github.com/Decima2/Decima2Toolkit/tree/main/examples/model_insights/model_explanations)  


## Data Tools
### Coming Soon
These tools help you evaluate your data 

## Outcome Tools
These tools help you to evaluate the outcomes of your model

### Individual NLP Explanations 
Understand which terms were most impactful in driving the similarity between two embedded texts

<pre>
from decima2 import individual_nlp_explanation
</pre>

This tool allows users to explore which terms were most influential in driving similarity score between the two texts in embedded space as determined by the user specified model.  

#### Instructions
For detailed usage instructions and to explore how the module works check out our [Developer Docs](https://docs.decima2.co.uk/docs/nlp/individual-nlp-explanation) 
#### Tutorial 
To explore tutorials on individual nlp explanation and use-cases check out our [Jupyter Notebooks](https://github.com/Decima2/Decima2Toolkit/tree/main/examples/outcome_insights/individual_nlp_explanation)  



## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing
Contributions are welcome! Please create a pull request or open an issue for any improvements, bugs, or feature requests.

## Contact
For inquiries, please reach out to torty.sivill@decima2.co.uk
