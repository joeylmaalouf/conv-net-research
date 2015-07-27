import numpy as np
from convnet_James_Test import ConvolutionalNeuralNetwork
import random
from theano import tensor as T
import collections
import operator

# from load import mnist

class multi_net(object):

	def __init__(self):
		self.data_list=[]
		self.net_list = []
		self.task_id=1
		self.epochs=10
		self.batch_size=10
		self.data_set={}
		self.cnn = ConvolutionalNeuralNetwork()
		self.cnn.initialize_mnist()	
		self.trX_spf, self.trY_spf = self.split_data (self.cnn.trX, self.cnn.trY, 0)
		self.data_set[0] = [self.trX_spf, self.trY_spf]


	def create_cnn(self):
		self.cnn = ConvolutionalNeuralNetwork()
		self.cnn.initialize_mnist()	
		if self.task_id != 1:	
			self.cnn.load_all_weights(self.task_id-1)	
		self.cnn.create_model_functions()
		return self.cnn

	def shuffle_data(self, x, y):
		c=zip(x,y)
		random.shuffle(c)
		x, y=zip(*c)
		return x, y 	

	def get_sub_test_data(self):
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


	def process_data(self):
		self.trX_spf, self.trY_spf = self.split_data (self.cnn.trX, self.cnn.trY, self.task_id)
		self.data_set[self.task_id] = [self.trX_spf, self.trY_spf]
		trX, trY, teX, teY = self. check_pre()
		trX, trY = self.shuffle_data(trX, trY)
		teX, teY = self.shuffle_data(teX, teY)
		return trX, trY, teX, teY

	def split_data(self, dataset, label, task_id):
		matching = np.nonzero(np.argmax(label, axis = 1) == task_id)[0]
		return dataset[matching], label[matching] 

	def run(self):
		for self.task_id in range(1,10):
			# if self.task_id == 0:
			# 	self.cnn = self.create_cnn()
			# 	trX, trY, teX, teY=self.process_data()
			# 	self.train(self.cnn, trX, trY, teX, teY)
			# 	self.cnn.save_all_weights(self.task_id)
			if len(self.net_list)-1 != self.task_id:
				self.cnn=self.create_cnn()
				trX, trY, teX, teY=self.process_data()
				self.train(self.cnn, trX, trY, teX, teY)
				self.cnn.save_all_weights(self.task_id)
			if len(self.net_list)-1 == self.task_id:
				pass
			self.net_list.append(self.cnn)
		# self.test()


	def check_pre(self):
		trX, trY, = self.data_set[self.task_id]
		# task_number=[self.task_id]
		if self.task_id >= 1:
			trX_1, trY_1 = self.data_set[self.task_id-1]
			trX = self.crossing_share_X(trX, trX_1, .8, trX)
			trY = self.crossing_share_Y(trY, trY_1, .8, trY)
			# task_number.append(self.task_id-1)

		if self.task_id >= 3:
			trX_2, trY_2 = self.data_set[self.task_id-3]
			trX = self.crossing_share_X(trX, trX_2, .6, trX)
			trY = self.crossing_share_Y(trY, trY_2, .6, trY)
			# task_number.append(self.task_id-2)

		if self.task_id >= 5:
			trX_3, trY_3 = self.data_set[self.task_id-5]
			trX = self.crossing_share_X(trX, trX_3, .4, trX)
			trY = self.crossing_share_Y(trY, trY_3, .4, trY)

		if self.task_id >= 7:
			trX_4, trY_4 = self.data_set[self.task_id-7]
			trX = self.crossing_share_X(trX, trX_4, .4, trX)
			trY = self.crossing_share_Y(trY, trY_4, .4, trY)

		if self.task_id >= 9:
			trX_4, trY_4 = self.data_set[self.task_id-9]
			trX = self.crossing_share_X(trX, trX_4, .2, trX)
			trY = self.crossing_share_Y(trY, trY_4, .2, trY)
			# task_number.append(self.task_id-3)
		# trX_neg, trY_neg = self.get_neg_train_data(task_number)
		# trX = self.crossing_share_X(trX, trX_neg, .99, trX_neg)
		# trY = self.crossing_share_Y(trY, trY_neg, .99, trY_neg)
		teX, teY = self.get_sub_test_data()
		# self.data_set[self.task_id-1]=[trX, trY, teX, teY]
		return trX, trY, teX, teY

	def crossing_share_X(self, a, b, rate, c):
		index = rate*len(c)
		b = b[:index,:,:,:]
		a = np.concatenate((a, b), axis=0)
		return a

	def crossing_share_Y(self, a, b, rate, c):
		index = rate*len(c)
		b = b[:index,:]
		a = np.concatenate((a, b), axis=0)
		return a

	def train(self, cnn, trX, trY, teX, teY):
		for i in range(self.epochs):
			for start, end in zip(range(0, len(trX), self.batch_size), range(self.batch_size, len(trX), self.batch_size)):
				cnn.cost = cnn.train(trX[start:end], trY[start:end])

			accuracies = []
			accuracies1 = []
		for start, end in zip(range(0, len(teX), self.batch_size), range(self.batch_size, len(teX), self.batch_size)):
			if self.task_id >= 6:
				flag = "cnn_6"
			elif self.task_id >= 3:
				flag = "cnn_3"
			else:
				flag = "cnn"
			accuracy = self.calculate_accuracy(flag, teX[start:end], teY[start:end])
			accuracies.append(accuracy)
		print("{0} for task_id {1} accuracy is {2}".format(flag, self.task_id, np.mean(accuracies)))

			
	def calculate_accuracy(self, flag, teX, teY):
		
		last_sec = self.cnn.predict(teX)
		predict = last_sec
		if flag == "cnn_3":
			cnn_3 = self.net_list[self.task_id-3]
			predict += cnn_3.predict(teX) 

		elif flag == "cnn_6":
			cnn_3 = self.net_list[self.task_id-3]
			predict += cnn_3.predict(teX) 
			cnn_6 = self.net_list[self.task_id-6]
			predict += cnn_6.predict(teX)

		predict = self.cnn.softmax(predict)
		predict = np.argmax(predict, axis = 1)
		accuracy = np.mean(np.argmax(teY, axis = 1) == predict)
		 
		return accuracy



	def test(self):
		pass

if __name__ == "__main__":
	mn=multi_net()
	mn.run()
	print "processing complete"