import convnet
# import ELLA
import load
import numpy as np
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":
	load_activations = convnet.ConvolutionalNeuralNetwork().load_data
	trA1 = load_activations("saved/trA-1.txt", (30000, 625))
	trA2 = load_activations("saved/trA-2.txt", (30000, 625))
	trA = np.concatenate((trA1, trA2))
	teA = load_activations("saved/teA.txt", (10000, 625))
	print("trA.shape: {0}".format(trA.shape))
	trX, teX, trY, teY = load.mnist(onehot = True)
	classes = np.argmax(trY, axis = 1)
	print("classes.shape: {0}".format(classes.shape))
	lr = LogisticRegression()
	lr.fit(trA, classes)
	print("teA.shape: {0}".format(teA.shape))
	predictions = lr.predict(teA)
	print("predictions.shape: {0}".format(predictions.shape))
	accuracy = np.mean(predictions == classes)
	print("accuracy: {0}".format(accuracy))
