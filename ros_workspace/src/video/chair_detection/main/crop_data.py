import itertools
import numpy as np
import cv2

def corner_sampling(original, cropped_size):
	corners = [(0, 0),
			   (0, original.shape[1]-cropped_size[1]),
			   (original.shape[0]-cropped_size[0], 0),
			   (original.shape[0]-cropped_size[0], original.shape[1]-cropped_size[1]),
			   (original.shape[0]/2-cropped_size[0]/2, original.shape[1]/2-cropped_size[1]/2)]
	crops = []
	for corner in corners:
		crops.append(original[corner[0]:(corner[0]+cropped_size[0]), corner[1]:(corner[1]+cropped_size[1])])
	return np.asarray(crops)


if __name__ == "__main__":
	ratio = .8
	original_img = cv2.imread("chair.png")
	(h, w, _) = original_img.shape
	print w, h
	m = 5
	original_img = cv2.resize(original_img, (int(w/m), int(h/m)))
	(h, w, _) = original_img.shape
	print w, h
	crop_imgs = (corner_sampling(original_img, (int(round(ratio*h)), int(round(ratio*w)))))
	(_, h, w, _) = crop_imgs.shape
	print w, h

	cv2.imshow("original_img", original_img)
	for num, img in enumerate(crop_imgs):
		cv2.imshow(str(num), img)
	cv2.waitKey(0)
