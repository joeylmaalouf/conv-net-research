import itertools
import numpy as np


def binarize(data, val, dtype = np.uint8):
	""" Return a binarized version of an input list
		with values based on whether or not its contents
		are equal to some value.
	"""
	return (np.asarray(data) == val).astype(dtype)


def crop_sampling(original, cropped_size):
	""" Given a 2D NumPy array and a cropped size <= the size of the
		original array, return a list of all possible cropped variations.
	"""
	corners = itertools.product(range(0, 1+original.shape[0]-cropped_size[0]), range(0, 1+original.shape[1]-cropped_size[1]))
	return np.asarray([original[c[0]:(c[0]+cropped_size[0]), c[1]:(c[1]+cropped_size[1])] for c in corners])


if __name__ == "__main__":
	l = [0, 6, 1, 3, 4, 1, 5, 2, 5, 3]
	v = 3
	print("{0} == {1}?".format(np.asarray(l), v))
	print(binarize(l, v))

	image = np.reshape(np.arange(16), (4, 4))
	print("\nOriginal array:")
	print(image)
	print "Cropped sub-arrays:"
	for c in crop_sampling(image, (3, 2)):
		print(c)
