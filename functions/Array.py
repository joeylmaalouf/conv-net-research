import itertools
import numpy as np


def binarize(data, val, dtype = np.uint8):
	""" Return a binarized version of an input list
		with values based on whether or not its contents
		are equal to some value.
	"""
	return (np.asarray(data) == val).astype(dtype)


def crop_sampling(original, cropped_size, im_dims = (0, 1)):
	""" Given a NumPy array and a cropped size that is less than or equal to the
		size of the original array return a list of all possible cropped variations.
		The cropped_size variable must have the same number of dimensions as the
		original input array. The im_dims variable can be used to specify the
		dimensions to crop across. This function supports any number of dimensions
		for the input array, as well as any number of dimensions to crop across.

		original: n-dimensional NumPy array
		cropped_size: n-dimensional iteratable
		im_dims: m-dimensional iteratable (m <= n)
	"""
	ranges = [range(0, 1+original.shape[dim]-cropped_size[dim]) for dim in im_dims]
	crops = []
	for corner in itertools.product(*ranges):
		extended = [None]*len(original.shape)
		for ind, dim in zip(range(len(corner)), im_dims):
			extended[dim] = corner[ind]
		indices = [Ellipsis]*len(original.shape)
		for dim in im_dims:
			indices[dim] = slice(extended[dim], (extended[dim]+cropped_size[dim]))
		crops.append(original[indices])
	return np.asarray(crops)


if __name__ == "__main__":
	l = [0, 6, 1, 3, 4, 1, 5, 2, 5, 3]
	v = 3
	print("{0} == {1}?".format(np.asarray(l), v))
	print(binarize(l, v))

	image = np.reshape(np.arange(16), (4, 4))
	print("\n\nOriginal 2D array:")
	print(image)
	print("\nCropped sub-arrays across dimensions 0, 1:")
	print(crop_sampling(image, (3, 2)))

	image = np.reshape(np.arange(48), (3, 4, 4))
	print("\n\nOriginal 3D array:")
	print(image)
	print("\nCropped sub-arrays across dimensions 1, 2:")
	print(crop_sampling(image, cropped_size = (3, 3, 2), im_dims = (1, 2)))
