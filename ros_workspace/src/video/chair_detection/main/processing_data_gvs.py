import numpy as np
import os
import glob
import cv2


def get_data(scale = 10):
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
		img=cv2.imread(p)
		size=img.shape
		# print "---"
		# print size
		m=scale
		size=(int(size[1]/m),int(size[0]/m))
		img=cv2.resize(img,size)
		# print img.shape
		l.append(img)
	# print l
	x = np.asarray(l)
	y = np.asarray(y)
	np.save("data_x.npy", x)
	np.save("data_y.npy", y)

	print "size:", size
	return size


if __name__ == "__main__":
	get_data()


