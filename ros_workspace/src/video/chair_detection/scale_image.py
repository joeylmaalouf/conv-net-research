import numpy as np
import os
import glob
import cv2


def get_data():
	x_paths = []
	y = []
	datasets_dir = "/".join(os.path.abspath(__file__).split("/")[:-1])+"/data_set/"
	d=glob.glob(datasets_dir+"*/")
	f=glob.glob(d[1]+"*")
	return f[2761]


np_img=cv2.imread(get_data())
size=np_img.shape
m=10
size=(size[1]/m,size[0]/m)
print size
after_np_img=cv2.resize(np_img,size)
# after_np_img=np_img
# after_np_img=np.resize(after_np_img, size)
# after_np_img.resize(size)
print after_np_img.shape
cv2.namedWindow("after_img",0)
cv2.resizeWindow("after_img", 640,480)
cv2.imshow("img", np_img)
cv2.imshow("after_img", after_np_img)
cv2.waitKey(0)
cv2.destroyAllWindows()