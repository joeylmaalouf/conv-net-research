import numpy as np
import sys
sys.path.append("..")
from convnet import ConvolutionalNeuralNetwork
from datasets.Load import mnist, cifar10


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: {filename} <dataset>".format(filename = sys.argv[0].split("/")[-1]))
		sys.exit(1)

	dataset = sys.argv[1].lower()

	if dataset == "mnist":
		trX, trY, teX, teY = mnist(onehot = True)
		shape_dict = {
			"w1": (32, 1, 3, 3),
			"w2": (64, 32, 3, 3),
			"w3": (128, 64, 3, 3),
			"w4": (128 * 3 * 3, 625),
			"wo": (625, 10)
		}

	elif dataset == "cifar":
		trX, trY, teX, teY = cifar10(onehot = True)
		shape_dict = {
			"w1": (32, 1, 3, 3),
			"w2": (64, 32, 3, 3),
			"w3": (128, 64, 3, 3),
			"w4": (128 * 3 * 3, 841),
			"wo": (841, 10)
		}

	elif dataset == "office":
		data_dir = "/data1/user_data/office_objects/"
		trX = np.load(data_dir + "trX.npy")
		trX = trX[:, np.newaxis, :, :]
		trY = np.load(data_dir + "trY.npy")
		trY = np.concatenate((np.logical_not(trY).astype(np.int64), trY), axis = 1)
		teX = np.load(data_dir + "teX.npy")
		teX = teX[:, np.newaxis, :, :]
		teY = np.load(data_dir + "teY.npy")
		teY = np.concatenate((np.logical_not(teY).astype(np.int64), teY), axis = 1)
		shape_dict = {
			"w1": (32, 1, 3, 3),
			"w2": (64, 32, 3, 3),
			"w3": (128, 64, 3, 3),
			"w4": (128 * 15 * 11, 11625),
			"wo": (11625, 2)
		}

	else:
		print("Dataset must be mnist, cifar, or office.")
		sys.exit(1)

	cnn = ConvolutionalNeuralNetwork().initialize_dataset(trX, trY, teX, teY, shape_dict)
	cnn.create_model_functions().train_net(epochs = 10, batch_size = 50, verbose = False)
	print("Accuracy on the {0} dataset: {1:0.02f}%".format(dataset, cnn.calc_accuracy(teX, teY)*100))
