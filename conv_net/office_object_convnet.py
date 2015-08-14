import numpy as np
from convnet import ConvolutionalNeuralNetwork


if __name__ == "__main__":
	data_dir = "/data1/user_data/office_objects/"
	trX = np.load(data_dir + "trX.npy")
	trY = np.load(data_dir + "trY.npy")
	trY = np.concatenate((np.logical_not(trY).astype(np.int64), trY), axis = 1)
	teX = np.load(data_dir + "teX.npy")
	teY = np.load(data_dir + "teY.npy")
	teY = np.concatenate((np.logical_not(teY).astype(np.int64), teY), axis = 1)
	shape_dict = {
		"trX": (-1, 1, 96, 128),
		"teX": (-1, 1, 96, 128),
		"w1": (32, 1, 3, 3),
		"w2": (64, 32, 3, 3),
		"w3": (128, 64, 3, 3),
		"w4": (128 * 15 * 11, 11625),
		"wo": (11625, 2)
	}

	cnn = ConvolutionalNeuralNetwork().initialize_dataset(trX, trY, teX, teY, shape_dict) \
	.create_model_functions().train_net(epochs = 5, batch_size = 50, verbose = True)
	# method chaining is cool, guys
