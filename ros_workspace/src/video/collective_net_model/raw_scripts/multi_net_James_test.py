import numpy as np
from convnet_James_Test import ConvolutionalNeuralNetwork
import random
from theano import tensor as T
# from load import mnist

class multi_net(object):

	def __init__(self):
		self.epochs=5
		self.batch_size=40
		self.signal = True


	def create_cnn(self):
		self.cnn = ConvolutionalNeuralNetwork()
		self.cnn.initialize_mnist()	
		self.cnn.create_model_functions()
		return self.cnn

	def get_sub_test_data(self):
		teX = self.cnn.teX
		teY = self.cnn.teY
		X = []
		Y = []
		for i in range(len(teX)):
			index = np.argmax(teY[i])
			if self.signal:
				if index <= self.task_id:
					X.append(teX[i,:,:,:])
					Y.append(teY[i])
			else:
				if index > self.task_id:
					X.append(teX[i,:,:,:])
					Y.append(teY[i])
		return X, Y

	def get_train_data(self):
		teX = self.cnn.teX
		teY = self.cnn.teY
		X = []
		Y = []
		for i in range(len(teX)):
			index = np.argmax(teY[i])
			if index <= self.task_id:
				X.append(teX[i,:,:,:])
				Y.append(teY[i])
		return X, Y


	def run(self):
		for self.task_id in range(1,10):
			self.cnn=self.create_cnn()
			trX, trY = self.get_train_data() 
			teX, teY = self.get_sub_test_data()
			self.train(self.cnn, trX, trY, teX, teY)
			# self.cnn.save_all_weights(self.task_id)


	def train(self, cnn, trX, trY, teX, teY):
		signals=[]

		for i in range(self.epochs):
			for start, end in zip(range(0, len(trX), self.batch_size), range(self.batch_size, len(trX), self.batch_size)):
				self.cnn.cost = self.cnn.train(trX[start:end], trY[start:end])

		for start, end in zip(range(0, len(teX), self.batch_size), range(self.batch_size, len(teX), self.batch_size)):
			signal = self.cnn.predict(teX[start:end])
			signals.extend(signal)

		if self.signal:
			# print accuracies
			print "{0}_in_range".format(self.task_id)
			np.save("range1/{0}_in_range.npy".format(self.task_id),signals)
		else:
			print "{0}_out_range".format(self.task_id)
			np.save("range1/{0}_out_range.npy".format(self.task_id),signals)
			# print("{0} for task_id {1} accuracy is {2}".format("cnn", self.task_id, np.mean(accuracies)))
		
		if self.signal:
			# print accuracies
			print "{0}_in_range".format(self.task_id)
			np.save("range2/s{0}_in_range.npy".format(self.task_id),ss)
		else:
			print "{0}_out_range".format(self.task_id)
			np.save("range2/s{0}_out_range.npy".format(self.task_id),ss)

if __name__ == "__main__":
	mn=multi_net()
	mn.run()
	print "processing complete"