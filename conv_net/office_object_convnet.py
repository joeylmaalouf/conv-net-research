from convnet import ConvolutionalNeuralNetwork


if __name__ == "__main__":
	cnn = ConvolutionalNeuralNetwork().initialize_office().create_model_functions().train_net(epochs = 10, batch_size = 50, verbose = True)
	# method chaining is cool, guys
