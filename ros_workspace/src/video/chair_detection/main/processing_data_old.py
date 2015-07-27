import numpy as np
import os
import glob
import cv2
import image_transformation_new


class process_data(object):
	def __init__(self):
		self.transform = image_transformation_new.image_transformations()

	def divide(self, data_x, data_y): 
		if len(data_x)%3!=0:
			data_x=data_x[1:]
			data_y=data_y[1:]

		half=(len(data_x)/3)*2

		trX=data_x[:half]
		teX=data_x[half:]
		
		
		half=(len(data_y)/3)*2

		trY=data_y[:half]
		teY=data_y[half:]

		return trX, trY, teX, teY

	def go_through(self):
		scale = 10
		x_paths = []
		y = []
		datasets_dir = "/".join(os.path.abspath(__file__).split("/")[:-3])+"/data_storage/chair_data/data_set/"

		for d in glob.glob(datasets_dir+"*/"):
			paths = glob.glob(d+"*-full.png")
			x_paths.extend(paths)
			for path in paths:
				y.append(0 if "out" in path else 1)
		l=[]
		for p in x_paths:
			img = cv2.imread(p)
			size = img.shape
			(h, w, _) = size
			m = scale
			size = (int(w/m),int(h/m))
			img = cv2.resize(img,size)
			l.append(img)

		x = np.asarray(l)
		y = np.asarray(y)
		x = self.rgb_to_gray(x)

		return x, y

	def run(self, scale = 10):
		x, y = self.go_through()

		trX, trY, teX, teY = self.divide(x, y)

		print "before transform"
		print "trX", trX.shape
		print "trY", trY.shape
		print "teX", teX.shape
		print "teY", teY.shape
		
		trX, trY= self.transform.run(trX, trY)
		teX, teY= self.transform.run(teX, teY, False)	
		print "after transform"
		print "trX", trX.shape
		print "trY", trY.shape
		print "teX", teX.shape
		print "teY", teY.shape

		np.save("trX1.npy", trX)
		np.save("trY1.npy", trY)
		np.save("teX1.npy", teX)
		np.save("teY1.npy", teY)

		print "finish saving"
		
	def rgb_to_gray(self, image):
			# data_x=np.load("data_x.npy")
		data_x_new=[]
		for img in image:
			img= cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
			data_x_new.append(img)
		data_x_new = np.asarray(data_x_new)
		return data_x_new

	def one_hot(self, x,n):
		if type(x) == list:
				x = np.array(x)
		x = x.flatten()
		o_h = np.zeros((len(x),n))
		for i in range(len(x)):
				o_h[i, int(bool(x[i]))] = 1
		return o_h

	def form_square(self, data):
		l=[]
		for i in range(len(data)):
				(w,h)=data[i].shape
				if w > h:
						offset = (w-h)/2
						tmp = data[i][offset:w-offset, :]
				elif w < h:
						offset = (h-w)/2
						tmp = data[i][:, offset:h-offset]
				l.append(tmp)
		l=np.asarray(l)
		return l


	def load_data(self):

		trX = np.load("trX1.npy")
		trY = np.load("trY1.npy")
		teX = np.load("teX1.npy")
		teY = np.load("teY1.npy")   

		print trX.shape
		print teX.shape

		trY = self.one_hot(trY, 10)
		teY = self.one_hot(teY, 10)
		
		return trX,trY,teX,teY

if __name__ == "__main__":
	process = process_data()
	process.run()

