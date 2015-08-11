from convnet import ConvolutionalNeuralNetwork

if __name__ == "__main__":
	cnn = ConvolutionalNeuralNetwork().initialize_office().create_model_functions()
	print cnn.trX.shape
	print cnn.trY.shape
	print cnn.teX.shape
	print cnn.teY.shape
	accuracies = cnn.train_model(verbose = True, epochs = 10, batch_size = 50)
	print(accuracies)
