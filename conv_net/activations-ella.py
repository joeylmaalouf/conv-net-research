import numpy as np
from sklearn.linear_model import LogisticRegression
import load
import convnet
import ELLA


def binarize(classifications, task_id):
	return np.asarray([int(elem == task_id) for elem in classifications])


if __name__ == "__main__":
	print("\nLoading Data...")
	load_activations = convnet.ConvolutionalNeuralNetwork().load_data
	num_chunks = 20
	trAs = [load_activations("saved/trA{0:02d}.txt".format(i), (60000 / num_chunks, 625)) for i in range(num_chunks)]
	trA = np.concatenate(trAs)
	print("trA.shape: {0}".format(trA.shape))
	teA = load_activations("saved/teA.txt", (10000, 625))
	print("teA.shape: {0}".format(teA.shape))
	trX, teX, trY, teY = load.mnist(onehot = True)
	trC = np.argmax(trY, axis = 1)
	print("trC.shape: {0}".format(trC.shape))
	teC = np.argmax(teY, axis = 1)
	print("teC.shape: {0}".format(teC.shape))
	print("Done.")

	print("\nCreating ELLA Model...")
	num_params = 625
	num_latent = 20
	ella = ELLA.ELLA(num_params, num_latent, LogisticRegression, mu = 10 ** -3)
	for task in range(10):
		result_vector = binarize(trC, task)
		ella.fit(trA, result_vector, task)
		print("Trained task {0}".format(task))
	print("Sparsity coefficients: {0}".format(ella.S))
	print("Done.")

	print("\nAnalyzing Training Data...")
	predictions = np.argmax(np.asarray([ella.predict_logprobs(trA, i) for i in range(ella.T)]), axis = 0)
	print("predictions.shape: {0}".format(predictions.shape))
	accuracy = np.mean(predictions == trC)
	print("accuracy: {0:0.04f}".format(accuracy))
	print("per-task binary accuracy:")
	for task_id in range(ella.T):
		print("  {0} - {1:0.04f}".format(task_id, np.mean(binarize(predictions, task_id) == binarize(trC, task_id))))
	print("Done.")

	print("\nAnalyzing Testing Data...")
	predictions = np.argmax(np.asarray([ella.predict_logprobs(teA, i) for i in range(ella.T)]), axis = 0)
	print("predictions.shape: {0}".format(predictions.shape))
	accuracy = np.mean(predictions == teC)
	print("accuracy: {0:0.04f}".format(accuracy))
	print("per-task binary accuracy:")
	for task_id in range(ella.T):
		print("  {0} - {1:0.04f}".format(task_id, np.mean(binarize(predictions, task_id) == binarize(teC, task_id))))
	print("Done.")

	print("\nExecution complete.\nProgram terminated successfully.\n")
