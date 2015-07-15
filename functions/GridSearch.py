import itertools
import random


def grid_search(model, params):
	""" Simple parameter optimization by iterate through
		all possible combinations of the given parameters
		for the model and returning the combination that
		results in the highest evaluation.
	"""
	names = params.keys()
	vals = params.values()
	best = (None, [], -float("Inf"))
	for valuelist in itertools.product(*vals):
		params = { names[i]: valuelist[i] for i in range(len(valuelist)) }
		m = model(**params)
		acc = m.eval()
		if acc > best[2]:
			best = (m, params, acc)
	return best


class DummyModel(object):
	""" An example model to demonstrate
		the functionality of grid_search.
	"""
	def __init__(self, par0 = 0, par1 = 1, par2 = 2):
		super(DummyModel, self).__init__()
		self.par0 = par0
		self.par1 = par1
		self.par2 = par2

	def __str__(self):
		return "DummyModel with parameters {0}, {1}, and {2}".format(self.par0, self.par1, self.par2)

	def eval(self):
		return random.random()


if __name__ == "__main__":
	params = {
		"par0": ["rbf", "poly"],
		"par1": [1e3, 1e2, 1e1],
		"par2": [1e0, 1e-1, 1e-2, 1e-3]
	}
	best = grid_search(DummyModel, params)
	print best
