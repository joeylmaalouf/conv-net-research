import numpy as np
import convnet_chair
import itertools
import processing_data_gvs

class gvs(object):
	def __init__(self):
		super(gvs, self).__init__()
	def range_data(self):
		print "[10, 0.001, 0.2, 0.5]"
		# self.scale_min = input('image data input scale multiple number min: ')
		# self.scale_max = input('image data input scale multiple number max: ')
		# self.s_between = input('points between: ')

		# self.lr_min = input('learning rate min: ')
		# self.lr_max = input('learning rate max: ')
		# self.l_between = input('points between: ')

		# self.dropout_conv_min = input('dropout conv rate min: ')
		# self.dropout_conv_max = input('dropout conv rate max: ')
		# self.d_conv_between = input('points between: ')

		# self.dropout_hidden_min = input('dropout hidden rate min: ')
		# self.dropout_hidden_max = input('dropout hidden rate max: ')
		# self.d_hidden_between = input('points between: ')

		self.scale_min = 8
		self.scale_max = 10
		self.s_between = 4

		self.lr_min = 0.0008
		self.lr_max = 0.0012
		self.l_between = 4

		self.dropout_conv_min = 0.2
		self.dropout_conv_max = 0.2
		self.d_conv_between = 1

		self.dropout_hidden_min = 0.5
		self.dropout_hidden_max = 0.5
		self.d_hidden_between = 1


		# scale = np.arange(self.scale_min, self.scale_max).astype(int)
		# lr = np.arange(self.lr_min, self.lr_max)
		# drop_out_conv = np.arange(self.dropout_conv_min, self.dropout_conv_max)
		# drop_out_hidden = np.arange(self.dropout_hidden_min, self.dropout_hidden_max)

		scale = np.linspace(self.scale_min, self.scale_max, self.s_between).astype(int)
		lr = np.linspace(self.lr_min, self.lr_max, self.l_between)
		drop_out_conv = np.linspace(self.dropout_conv_min, self.dropout_conv_max, self.d_conv_between)
		drop_out_hidden = np.linspace(self.dropout_hidden_min, self.dropout_hidden_max, self.d_hidden_between)

		print scale
		print lr
		print drop_out_conv
		print drop_out_hidden
		self.combination = [c for c in itertools.product((scale, lr, drop_out_conv, drop_out_hidden))]
		# self.combination = cartesian((scale, lr, drop_out_conv, drop_out_hidden))
		print self.combination[0][0]
	def testing_model(self):
		accuracies = []
		for i in range (len(self.combination)):
			scale = self.combination[i][0][0]
			print "scale", scale
			size = processing_data_gvs.get_data(scale=scale)
			print size
			print self.combination[i][0]
			self.cnn = convnet_chair.ConvolutionalNeuralNetwork()
			self.cnn.get_data(data = self.combination[i][0], size = size)
			accuracy = self.cnn.chair_example(verbose = True, save = False)

			accuracies. append(accuracy)
		index = np.argmax(accuracises)
		np.save("paramater_combination.npy", combination)
		np.save("accuracy.npy", accuracises)

	def run(self):
		self.range_data()
		self.testing_model()

if __name__ == '__main__':
	gvs = gvs()
	gvs.run()