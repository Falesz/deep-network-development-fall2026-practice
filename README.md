# Deep Network Development - Fall 2026 - Practice
In this repo I aim to create various small, self-contained machine learning / neural network projects, neatly organised into separate folders.  
My AI policy:  
- I use AI heavily for researching topics, programming language syntax, summarizing and fetching documentation and generic semantics. Example prompts: _"Show me a generic way to write a forward function in a model containing an input layer, a single hidden layer and an output layer, with the input layer and the hidden layer utilizing a ReLU activation function."_ or _"What is the purpose of an activation function?"_
- All code explicitly appearing in the repository is 100% organic human handwritten.
- All ideas are my originals, with their flaws and learning potentials (or the lack of them) included.

## Exhibit 1 - Linear regression

### Linear data generator

`linear_data_generator.py` is a simple script that generates 10000 data points roughly around the `y = 3x + 10` line. The aim here is to create a dataset of points onto which a single neuron can learn to fit a straight line.  
You may run this script with the command:  
`py linear_data_generator.py`