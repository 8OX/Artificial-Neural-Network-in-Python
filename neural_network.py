import numpy as np
from nnfs.datasets import spiral_data
import nnfs
import matplotlib.pyplot as plt
import matplotlib



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

class Activation_Softmax:

    # forward pass
    def forward(self, inputs):

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities
 
# Common loss
class Loss:

    # Calculates the dataand regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):

        # Calculate samples losses
        # forward called when inherited
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Return loss
        return data_loss

# inherits from Loss class
class CategoricalCrossEntropy(Loss):

    # forward pass
    def forward(self, y_pred, y_true):

        # Number of samples in batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not introduce a bias in any direction
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values - only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # Mask values - only for one-hot encoded labels
        elif len(class_targets.shape) == 2:
        correct_confidences = np.sum(
            y_pred_clipped * y_true,
            axis=1
        )

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods