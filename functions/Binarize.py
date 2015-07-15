import numpy as np


def binarize(l, v, dtype = np.uint8):
	""" Return a binarized version of an input list l,
		with values based on whether or not its contents
		are equal to some value v.
	"""
	return (np.asarray(l) == v).astype(dtype)


if __name__ == "__main__":
	l = [0, 6, 1, 3, 4, 1, 5, 2, 5, 3]
	v = 3
	print("{0} == {1}?".format(np.asarray(l), v))
	print(binarize(l, v))
