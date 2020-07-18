import numpy as np
import math
import pandas as pd
import random
from array import *
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from random import seed
from random import randrange
from csv import reader


# region NeuralNetwork
class NeuralNetwork:

    def __init__(self, learning_rate, threshold):
        self.learning_rate = learning_rate
        self.threshold = threshold
        np.random.seed(1)
        self.weights = 2 * np.random.random((2, 1)) - 1

    def step(self, x):
        if x > float(self.threshold):
            return 1
        else:
            return 0

    def train(self, training_inputs, training_outputs, training_iterations):
        for i in range(training_iterations):
            output_activation_fn = self.think(training_inputs)
            cal_error = training_outputs - output_activation_fn
            adj = np.dot(training_inputs.T, cal_error * self.learning_rate)
            self.weights = self.weights + adj

    def think(self, inputs):
        inputs = inputs.astype(float)
        output_in = np.sum(np.dot(inputs, self.weights))
        aggregate_sum = self.step(output_in)
        return aggregate_sum


# endregion NeuralNetwork


# endregion

# region Neural_Network_Main_Fn
def NN_Main():
    learning_rate = 0.1
    threshold = -0.2
    neural_network = NeuralNetwork(learning_rate, threshold)

    print("Beginning Randomly Generated Weights: ")
    print(neural_network.weights)

    training_inputs = np.array([[0, 0],
                                [0, 1],
                                [1, 0],
                                [1, 1]])

    training_outputs = np.array([[0, 0, 0, 1]]).T

    neural_network.train(training_inputs, training_outputs, 100)

    print("Ending Weights After Training: ")
    print(neural_network.weights)

    inputTestCase = [1, 1]

    print("Considering New Situation: ", inputTestCase[0], inputTestCase[1], end=" ")
    print("New Output data: ", end=" ")
    print(neural_network.think(np.array(inputTestCase)))
    print("Wow, we did it!")


# endregion

######################## MAIN ###########################33
if __name__ == '__main__':
    NN_Main()
