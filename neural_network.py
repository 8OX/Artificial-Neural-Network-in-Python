import numpy as np
from nnfs.datasets import spiral_data
import nnfs
import matplotlib.pyplot as plt
import matplotlib


# Our dense layer will begin with two methods
class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        # initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:

    def forward(self, inputs):
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)

# softmax activatation


class Activation_Softmax:

    # forward pass
    def forward(self, inputs):

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities
