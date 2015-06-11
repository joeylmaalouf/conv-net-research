import convnet
# import ELLA
import load
import numpy as np
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":
	print("\nLoading Data:")
	load_activations = convnet.ConvolutionalNeuralNetwork().load_data
	num_chunks = 20
	trAs = [load_activations("saved/trA{0:02d}.txt".format(i), (60000/num_chunks, 625)) for i in range(num_chunks)]
	trA = np.concatenate(trAs)
	print("trA.shape: {0}".format(trA.shape))
	teA = load_activations("saved/teA.txt", (10000, 625))
	print("teA.shape: {0}".format(teA.shape))
	trX, teX, trY, teY = load.mnist(onehot = True)

	print("\nCreating Regression Model:")
	classes = np.argmax(trY, axis = 1)
	print("classes.shape: {0}".format(classes.shape))
	lr = LogisticRegression()
	lr.fit(trA, classes)

	print("\nAnalyzing Training Data:")
	predictions = lr.predict(trA)
	print("predictions.shape: {0}".format(predictions.shape))
	accuracy = np.mean(predictions == classes)
	print("accuracy: {0}".format(accuracy))

	print("\nAnalyzing Testing Data:")
	predictions = lr.predict(teA)
	print("predictions.shape: {0}".format(predictions.shape))
	accuracy = np.mean(predictions == classes)
	print("accuracy: {0}".format(accuracy))

	print("\n")
