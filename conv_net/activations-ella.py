import convnet
# import ELLA
import load
import numpy as np
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":
	load_activations = convnet.ConvolutionalNeuralNetwork().load_data
	trA = load_activations("saved/trA.txt", (60000, 625))
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
