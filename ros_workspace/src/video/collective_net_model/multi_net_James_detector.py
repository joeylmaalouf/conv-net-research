import numpy as np
from convnet_James_Test import ConvolutionalNeuralNetwork
import random
from theano import tensor as T
import collections
import operator

class multi_net(object):

	def __init__(self):
		self.data_list=[]
		self.net_list = []
		self.task_id=1
		self.epochs=5
		self.batch_size=40
		self.data_set={}
		self.cnn = ConvolutionalNeuralNetwork()
		self.cnn.initialize_mnist()	
		self.trX_spf, self.trY_spf = self.split_data (self.cnn.trX, self.cnn.trY, 0)
		self.data_set[0] = [self.trX_spf, self.trY_spf]
		self.signal = True
		self.hidden_resp = []

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

	def get_sub_test_data(self, cover_range = True):

		teX = self.cnn.teX
		teY = self.cnn.teY
		X = []
		Y = []
		for i in range(len(teX)):
			index = np.argmax(teY[i])
			if cover_range:
				if index <= self.task_id:
					X.append(teX[i,:,:,:])
					Y.append(teY[i])
			else:
				if index > self.task_id:
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
			trX = self.crossing_share_X(trX, trX_1, .4, trX)
			trY = self.crossing_share_Y(trY, trY_1, .4, trY)
			# task_number.append(self.task_id-1)

		if self.task_id >= 3:
			trX_2, trY_2 = self.data_set[self.task_id-3]
			trX = self.crossing_share_X(trX, trX_2, .4, trX)
			trY = self.crossing_share_Y(trY, trY_2, .4, trY)
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
			trX = self.crossing_share_X(trX, trX_4, .4, trX)
			trY = self.crossing_share_Y(trY, trY_4, .4, trY)
		# trX, trY = self.get_train_data()
		teX, teY = self.get_sub_test_data()
		return trX, trY, teX, teY

	# def get_train_data(self):
	# 	teX = self.cnn.teX
	# 	teY = self.cnn.teY
	# 	X = []
	# 	Y = []
	# 	for i in range(len(teX)):
	# 		index = np.argmax(teY[i])
	# 		if index <= self.task_id:
	# 			X.append(teX[i,:,:,:])
	# 			Y.append(teY[i])
	# 	return X, Y

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
		signals=[]
		for start, end in zip(range(0, len(trX), self.batch_size), range(self.batch_size, len(trX), self.batch_size)):
			signal = self.cnn.activate(trX[start:end])
			signals.extend(signal)
		trX1, _ = self.get_sub_test_data(False)
		
		neg_signals = []
		for start, end in zip(range(0, len(trX1), self.batch_size), range(self.batch_size, len(trX1), self.batch_size)):
			neg_signal = self.cnn.activate(trX1[start:end])
			neg_signals.extend(neg_signal)				
		
		signals = np.asarray(signals)
		neg_signals = np.asarray(neg_signals)



		sum_1 = np.sum(signals[:,0:self.task_id+1],axis=1)
		sum_2 = np.sum(neg_signals[:,0:self.task_id+1],axis=1)

		mean1 = np.median(sum_1) + np.mean(sum_1)
		mean2 = np.median(sum_2) + np.mean(sum_2)

		self.hidden_resp.append((mean1, mean2))

		accuracies = []
		for start, end in zip(range(0, len(teX), self.batch_size), range(self.batch_size, len(teX), self.batch_size)):
			accuracy, num_net, betw_unit = self.calculate_accuracy(teX[start:end], teY[start:end])
			accuracies.append(accuracy)
		print("num_net: {0} step: {1} accuracy: {2}".format(num_net, betw_unit, np.mean(accuracies)))
			
	def calculate_accuracy(self, teX, teY):
		if self.task_id <= 5:
			self.betw_unit = 1
			start = 2
		elif self.task_id > 5 and self.task_id < 10:
			self.betw_unit = 2	
			start = 3

		result = []
		prediction = []
		for i in range(len(teX)):
			index = np.argmax(teY[i])
			predict = self.cnn.predict(teX[i:i+1])[0]
			result.append(predict)

			if self.task_id == 2:
				cnn = self.net_list[0]
				in_range = self.check_hidd_resp(cnn, 0, teX[i:i+1])
				if in_range:
					predict = cnn.predict(teX[i:i+1])[0]
					result.append(predict)

			else:
				for i in range(start, self.task_id+1, self.betw_unit):
					cnn = self.net_list[self.task_id-i]
					in_range = self.check_hidd_resp(cnn, self.task_id-i, teX[i:i+1])
					if in_range:
						predict = cnn.predict(teX[i:i+1])[0]
						result.append(predict)
			predict = self.combine_frequncy(result)
			prediction.append(predict)


		prediction = np.asarray(prediction)
		# prediction = np.argmax(prediction, axis = 1)
		# predict = self.combine_frequncy(result)
		accuracy = np.mean(np.argmax(teY, axis = 1) == prediction)
		return accuracy, self.task_id, self.betw_unit

	def check_hidd_resp(self, cnn, index, teX):
		resp = cnn.activate(teX)
		sum_1 = np.sum(resp[:,0:index+2], axis=1)
		mean = np.median(sum_1) + np.mean(sum_1)
		(mean1, mean2) = self.hidden_resp[index]
		return abs(float(mean - mean1)) < abs(float(mean - mean2))


	def combine_frequncy(self, result):
		output=[]
		for l in result:
			l = np.argmax(l)
			output.append(l)

		Counter_list = collections.Counter(output)
		max_value = max(Counter_list.iteritems(), key=operator.itemgetter(1))[0]
		fst = output[0]

		if Counter_list[fst] >= Counter_list[max_value]:
			max_value = fst
		return max_value

if __name__ == "__main__":
	mn=multi_net()
	mn.run()
	print "processing complete"