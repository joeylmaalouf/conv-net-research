import numpy as np
from convnet import ConvolutionalNeuralNetwork

class multi_net(object):

	def __init__(self, arg):
		supermulti_nete, self).__init__()
		self.arg = arg
		self.data_list=[]
		self.net_list = []
		self.task_id=1
		self.epochs=10
		self.batch_size=40
	# def binarize(self, classifications, task_id):
	# 	return np.asarray(np.asarray(classifications) == task_id, dtype = np.uint8)

	def create_cnn(self):	
		self.cnn = ConvolutionalNeuralNetwork()	
		self.cnn.create_model_functions()
		if self.task_id-1 == 0:
			self.cnn.initialize_mnist()
		else:
			cnn.load_all_weights(self.task_id-1)	
		return self.cnn

	def process_dat(self):
		self.trX08, self.trY08, self.trX9, self.trY9 = self.split_data (self.cnn.trX, self.cnn.trY, 9)
		self.trX07, self.trY07, self.trX8, self.trY8 = self.split_data (self.cnn.trX, self.cnn.trY, 8)
		self.trX06, self.trY06, self.trX7, self.trY7 = self.split_data (self.cnn.trX, self.cnn.trY, 7)
		self.trX05, self.trY05, self.trX6, self.trY6 = self.split_data (self.cnn.trX, self.cnn.trY, 6)
		self.trX04, self.trY04, self.trX5, self.trY5 = self.split_data (self.cnn.trX, self.cnn.trY, 5)
		self.trX03, self.trY03, self.trX4, self.trY4 = self.split_data (self.cnn.trX, self.cnn.trY, 4)
		self.trX02, self.trY02, self.trX3, self.trY3 = self.split_data (self.cnn.trX, self.cnn.trY, 3)
		self.trX01, self.trY01, self.trX2, self.trY2 = self.split_data (self.cnn.trX, self.cnn.trY, 2)
		self.trX00, self.trY00, self.trX1, self.trY1 = self.split_data (self.cnn.trX, self.cnn.trY, 1)

		self.data_list.append([self.trX00, self.trY00, self.trX1, self.trY1])
		self.data_list.append([self.trX01, self.trY01, self.trX2, self.trY2])
		self.data_list.append([self.trX02, self.trY02, self.trX3, self.trY3])
		self.data_list.append([self.trX03, self.trY03, self.trX4, self.trY4])
		self.data_list.append([self.trX04, self.trY04, self.trX5, self.trY5])
		self.data_list.append([self.trX05, self.trY05, self.trX6, self.trY6])
		self.data_list.append([self.trX06, self.trY06, self.trX7, self.trY7])
		self.data_list.append([self.trX07, self.trY07, self.trX8, self.trY8])										
		self.data_list.append([self.trX08, self.trY08, self.trX9, self.trY9])
		

	def split_data(self, dataset, label, task_id):
		matching = np.nonzero(label == task_id)
		nonmatching = np.nonzero(label != task_id)
		return dataset[nonmatching], label[nonmatching], dataset[matching], dataset[nonmatching] 

	def run(self):

		for self.task_id in range(1,10):
			if len(net_list) == 0:
				self.cnn=self.create_cnn()
				self.train(self.cnn, data_list[self.task_id-1])
			else:
				if len(net_list) != self.task_id:
					self.cnn=self.create_cnn()
					trX, trY, teX, teY=self.check_pre()
					self.train(self.cnn, trX, trY, teX, teY)
			self.cnn.save_all_weights(task_id)
		self.test()

	def check_pre(self):
		trX, trY, teX, teY = net_list[self.task_id-1]
		if self.task_id-1 >= 0:
			trX_1, trY_1, teX_1, teY_1 = net_list[self.task_id-2]
			trX = self.crossing_share(trX, trX_1, .5)
		if self.task_id-2 >= 0:
			trX_2, trY_2, teX_2, teY_2 = net_list[self.task_id-3]
			trX = self.crossing_share(trX, trX_2, .3)
		if self.task_id-3 >= 0:
			trX_3, trY_3, teX_3, teY_3 = net_list[self.task_id-4]
			trX = self.crossing_share(trX, trX_3, .1)
		return trX, trY, teX, teY

	def crossing_share(self, a, b, rate):
		b = b[rate*len(b), :, :]
		a = np.concatenate((a, d), axis=0)
		return a


	def train(self, cnn, trX, trY, teX, teY):
		accuracies = []
		for i in range(self.epochs):
			for start, end in zip(range(0, len(trX), self.batch_size), range(self.batch_size, len(trX), self.batch_size)):
				cnn.cost = cnn.train(trX[start:end], trY[start:end])
			accuracy = np.mean(np.argmax(teY, axis = 1) == cnn.predict(teX))
			accuracies.append(accuracy)
			if verbose:
				print("for task_id {0} accuracy is {1}".format(self.task_id, accuracy))


	def test(self):




if "__name__" == __main__:
	mn=multi_net()
	mn.run()
	print "processing complete"