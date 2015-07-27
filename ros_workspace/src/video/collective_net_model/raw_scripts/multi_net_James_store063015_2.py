import numpy as np
from convnet_James import ConvolutionalNeuralNetwork
import random
from theano import tensor as T
# from load import mnist

class multi_net(object):

	def __init__(self):
		self.data_list=[]
		self.net_list = []
		self.task_id=1
		self.epochs=10
		self.batch_size=40
		self.data_set={}
	# def binarize(self, classifications, task_id):
	# 	return np.asarray(np.asarray(classifications) == task_id, dtype = np.uint8)



	def create_cnn(self):
		self.cnn = ConvolutionalNeuralNetwork()
		# self.cnn.trX, self.cnn.teX, self.cnn.trY, self.cnn.teY = mnist(onehot = True)	
		self.cnn.initialize_mnist()	
		if self.task_id != 0:	
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
		teX = teX[:1000,:,:,:]
		teY = teY[:1000,:]
		X = []
		Y = []
		for i in range(1000):
			index = np.nonzero(teY[i])[0][0]
			if index <= self.task_id:
				X.append(teX[i,:,:,:])
				Y.append(teY[i])
		return X, Y

	# def get_neg_train_data(self, task_number):	
	# 	trX = self.cnn.trX
	# 	trY = self.cnn.trY
	# 	trX = trX[:1000,:,:,:]
	# 	trY = trY[:1000, :]
	# 	for i in range(1000):
	# 		index = np.nonzero(trY[i])[0][0]
	# 		if  index not in list(task_number):
	# 			trY[i][index] = 0
	# 	return trX, trY

	def process_data(self):
		self.trX_spf, self.trY_spf = self.split_data (self.cnn.trX, self.cnn.trY, self.task_id)
		self.data_set[self.task_id] = [self.trX_spf, self.trY_spf]
		trX, trY, teX, teY = self. check_pre()
		trX, trY = self.shuffle_data(trX, trY)
		teX, teY = self.shuffle_data(teX, teY)
		return trX, trY, teX, teY

	def split_data(self, dataset, label, task_id):
		matching = np.nonzero(np.argmax(label, axis = 1) == task_id)[0]
		# nonmatching = np.nonzero(label != task_id)[0]
		return dataset[matching], label[matching] 

	def run(self):
		for self.task_id in range(0,10):
			if self.task_id == 0:
				self.cnn = self.create_cnn()
				trX, trY, teX, teY=self.process_data()
				self.train(self.cnn, trX, trY, teX, teY)
				self.cnn.save_all_weights(self.task_id)
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
		if self.task_id-1 >= 0:
			trX_1, trY_1 = self.data_set[self.task_id-1]
			trX = self.crossing_share_X(trX, trX_1, .5, trX)
			trY = self.crossing_share_Y(trY, trY_1, .5, trY)
			# task_number.append(self.task_id-1)

		if self.task_id-2 >= 0:
			trX_2, trY_2 = self.data_set[self.task_id-2]
			trX = self.crossing_share_X(trX, trX_2, .3, trX)
			trY = self.crossing_share_Y(trY, trY_2, .3, trY)
			# task_number.append(self.task_id-2)

		if self.task_id-3 >= 0:
			trX_3, trY_3 = self.data_set[self.task_id-3]
			trX = self.crossing_share_X(trX, trX_3, .1, trX)
			trY = self.crossing_share_Y(trY, trY_3, .1, trY)
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
			if self.task_id-6 >= 0:
				self.calculate_accuracy("cnn_6", teX, teY)
			elif self.task_id-3 >= 0:
				self.calculate_accuracy("cnn_3", teX, teY)
			else:
				self.calculate_accuracy("cnn", teX, teY)
	
				
	def calculate_accuracy(self, flag, teX, teY):
		accuracies = []
		last_sec = self.cnn.predict(teX)
		predict = last_sec
		if flag == "cnn_3":
			last_sec = last_sec[:len(last_sec), -3:]
			cnn_3 = self.net_list[self.task_id-3]
			mid_sec = cnn_3.predict(teX) 
			mid_sec = mid_sec[:len(mid_sec), :-3]
			predict = np.concatenate((mid_sec, last_sec), axis = 1)

		elif flag == "cnn_6":
			last_sec = last_sec[:len(last_sec), -3:]

			cnn_3 = self.net_list[self.task_id-3]
			mid_sec = cnn_3.predict(teX) 
			mid_sec = mid_sec[:len(mid_sec), :-3]
			predict = np.concatenate((mid_sec, last_sec), axis = 1)

			cnn_6 = self.net_list[self.task_id-6]
			first_sec = cnn_6.predict(teX)
			first_sec = first_sec[:len(first_sec), :-6]
			predict = predict[:len(predict), -6:]
			predict = np.concatenate((first_sec, predict), axis = 1)
		
		last_sec = np.argmax(last_sec, axis = 1)
		predict = np.argmax(predict, axis = 1)
		accuracy1 = np.mean(np.argmax(teY, axis = 1) == last_sec)
		accuracy = np.mean(np.argmax(teY, axis = 1) == predict)
		accuracies.append(accuracy)
		print "last_sec", accuracy1
		print("{0} for task_id {1} accuracy is {2}".format(flag, self.task_id, accuracy))
		print predict[0]

	def test(self):
		pass

if __name__ == "__main__":
	mn=multi_net()
	mn.run()
	print "processing complete"