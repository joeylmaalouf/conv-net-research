import numpy as np


def remove_class(data_set, data_labels, task, condense = False):
	# condense the data from a binary array to a single value if necessary
	classes = np.argmax(data_labels, axis = 1) if condense else data_labels
	# find the indices corresponding (or not) to the task to be removed
	nonmatching = np.nonzero(classes != task)[0]
	matching = np.nonzero(classes == task)[0]
	# return the split data using these indices
	return data_set[nonmatching], data_labels[nonmatching], data_set[matching], data_labels[matching]


def random_sampling(data_set, data_labels, p_kept = 0.5, to_keep = None):
	# temporarily remove the task that we want to keep all examples of
	if to_keep:
		data_set, data_labels, kept_set, kept_labels = remove_class(data_set, data_labels, to_keep)
	# pick random elements from leftover dataset, up to calculated limit
	length = len(data_set)
	limit = int(p_kept*length)
	indices = np.random.permutation(length)[:limit]
	data_set, data_labels = data_set[indices], data_labels[indices]
	# add back the kept task
	if to_keep:
		data_set = np.concatenate((data_set, kept_set), 0)
		data_labels = np.concatenate((data_labels, kept_labels), 0)
	# reshuffle after adding back the fully kept task
	indices = np.random.permutation(len(data_set))
	data_set, data_labels = data_set[indices], data_labels[indices]	
	return data_set, data_labels


if __name__ == "__main__":
	data = np.asarray([x**2 for x in range(1000)])
	labels = np.asarray([x%10 for x in range(1000)])
	print data.shape, labels.shape
	data, labels = random_sampling(data, labels, 0.5)
	print data.shape, labels.shape
	data08, labels08, data_9, labels_9 = remove_class(data, labels, 9)
	print data08.shape, labels08.shape
	print data_9.shape, labels_9.shape
