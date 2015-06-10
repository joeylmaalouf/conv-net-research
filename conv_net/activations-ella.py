import convnet
import numpy as np
# import ELLA
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":
	activations = convnet.ConvolutionalNeuralNetwork().load_data("saved/activations.txt", (10000, 625))
	print(activations.shape)
