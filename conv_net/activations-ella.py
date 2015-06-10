import convnet
# import ELLA
import load
import numpy as np
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":
	activations = convnet.ConvolutionalNeuralNetwork().load_data("saved/activations.txt", (10000, 625)) # (10000 individual 25x25 images? weren't they 28x28?)
	trX, teX, trY, teY = load.mnist(onehot = True)
	classes = np.argmax(teY, 1)
	print("activations.shape: {0}".format(activations.shape))
	print("classes.shape: {0}".format(classes.shape))
	lr = LogisticRegression()
	lr.fit(activations, classes)
	predictions = lr.predict(activations)
	accuracy = np.mean([int(i) for i in (predictions == classes)])
	print("accuracy: {0}".format(accuracy))
	# resave activations of trX and teX, train on trX and trY then predict on teX and compare to teY?
