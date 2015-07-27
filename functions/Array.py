import itertools
import numpy as np


def binarize(data, val, dtype = np.int64):
	""" Return a binarized version of an input list
		with values based on whether or not its contents
		are equal to some value.
	"""
	return (np.asarray(data) == val).astype(dtype)


def crop_sampling(original, cropped_size, crop_dims = (0, 1), steps = (1, 1)):
	""" Given a NumPy array and a cropped size that is less than or equal to the
		size of the original array, return an array of all possible cropped variations
		that can be created by shifting each dimension by the given step size. The
		cropped_size variable must be equal in length to the crop_dims variable
		because each dimension length value in cropped_size corresponds to a dimension
		to crop across specified in crop_dims. This function supports any number of
		dimensions for the input array, as well as any number of dimensions to crop
		across that is less than or equal to the number of dimensions in the input.

		original: m-dimensional NumPy array
		cropped_size: n-length tuple (n <= m)
		crop_dims: n-length tuple
		steps: n-length tuple
	"""
	if type(cropped_size) == type(0):
		cropped_size = (cropped_size,)
	if type(crop_dims) == type(0):
		crop_dims = (crop_dims,)
	ranges = [range(0, 1+original.shape[dim]-size, step) for size, dim, step in zip(cropped_size, crop_dims, steps)]
	crops = []
	for corner in itertools.product(*ranges):
		indices = [Ellipsis]*len(original.shape)
		for ind, dim in enumerate(crop_dims):
			indices[dim] = slice(corner[ind], (corner[ind]+cropped_size[ind]))
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
	print("\nCropped sub-arrays across dimensions 0, 1, stepping over every row and every other column:")
	print(crop_sampling(image, (3, 2), steps = (1, 2)))

	image = np.reshape(np.arange(48), (3, 4, 4))
	print("\n\nOriginal 3D array:")
	print(image)
	print("\nCropped sub-arrays across dimensions 1, 2:")
	print(crop_sampling(image, cropped_size = (3, 2), crop_dims = (1, 2)))

	image = np.reshape(np.arange(240), (5, 3, 4, 4))
	print("\n\nOriginal 4D array shape:")
	print("(index, channel, row, column)")
	print(image.shape)
	# you want each element in the output array to be of size 1 (down from, say, 3) in the specified
	# dimension, and you are referring to dimension 1 in the input array as the one to crop across
	crops = crop_sampling(image, cropped_size = 1, crop_dims = 1)
	print("\nShape output from splitting color channels in a list of images:")
	for crop in crops:
		print(crop.shape)
	print("Overall shape of {0}".format(crops.shape))
