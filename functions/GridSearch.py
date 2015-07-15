import itertools
import random


def grid_search(model, params):
	""" Simple parameter optimization by iterate through
		all possible combinations of the given parameters
		for the model and returning the combination that
		results in the highest evaluation. If the given model
		is a class type, grid_search will initialize a new
		model for each evaluation; if it is a class instance,
		grid_search will use that instance, re-setting
		just the attributes passed in for testing.
	"""
	best = ({}, -float("Inf"))
	names = params.keys()
	vals = params.values()

	for valuelist in itertools.product(*vals):
		params = { names[i]: valuelist[i] for i in range(len(valuelist)) }

		if type(model) == type(object):
			m = model()
			instance = m
		else:
			instance = model

		for k, v in params.items():
			setattr(instance, k, v)

		acc = instance.eval()

		if acc > best[1]:
			best = (params, acc)
	return best[0]


class DummyModel(object):
	""" An example model to demonstrate
		the functionality of grid_search.
	"""
	def __init__(self, par0 = 0, par1 = 1, par2 = 2, par3 = 3):
		super(DummyModel, self).__init__()
		self.par0 = par0
		self.par1 = par1
		self.par2 = par2
		self.par3 = par3

	def __str__(self):
		return "DummyModel with parameters {0}, {1}, {2}, and {3}".format(self.par0, self.par1, self.par2, self.par3)

	def eval(self):
		return random.random()


if __name__ == "__main__":
	params = {
		"par0": ["rbf", "poly"],
		"par1": [1e3, 1e2, 1e1],
		"par2": [1e0, 1e-1, 1e-2, 1e-3]
	}

	best = grid_search(DummyModel, params)
	print("Given a class type:      {0}".format(best))

	dm = DummyModel(par3 = 10)
	best = grid_search(dm, params)
	print("Given a class instance:  {0}".format(best))
