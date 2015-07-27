import numpy as np
import os
import glob
import cv2


def get_data():
	x_paths = []
	y = []
	datasets_dir = "/".join(os.path.abspath(__file__).split("/")[:-3])+"/data_storage/chair_data/data_set/data_test"
	for d in glob.glob(datasets_dir+"*/"):
		print d
		paths = glob.glob(d+"*-full.png")
		
		x_paths.extend(paths)
		for path in paths:
			y.append(0 if "out" in path else 1)
	l=[]
	for p in x_paths:
		img=cv2.imread(p)
		image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		image = cv2.GaussianBlur(image, (5, 5), 0)
		canny = cv2.Canny(image, 40, 100)

		# size=img.shape
		# m=10
		# size=(size[1]/m,size[0]/m)
		# img=cv2.resize(img,size)
		l.append(canny)
	x = np.asarray(l)
	y = np.asarray(y)
	return x, y


def rescale():
	np_img=np.load("data_x.npy")
	j=[]
	print np_img.shape
	for i in np_img:	
		size=i.shape
		print size
		m=10
		splitze=(size[1]/m,size[0]/m)
		print size
		after_np_img=cv2.resize(i,size)
		j.append(i)
	j = np.asarray(j)
	print j.shape


	# after_np_img=np_img
	# after_np_img=np.resize(after_np_img, size)
	# after_np_img.resize(size)

	# cv2.namedWindow("after_img",0)
	# cv2.resizeWindow("after_img", 640,480)
	# cv2.imshow("img", np_img)
	# cv2.imshow("after_img", after_np_img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()


if __name__ == "__main__":
	# x, y = get_data()
	# np.save("data_x.npy", x)
	# np.save("data_y.npy", y)
	rescale()
