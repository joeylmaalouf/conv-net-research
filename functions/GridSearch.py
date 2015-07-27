import itertools
import random


def grid_search(model, params, verbose = False):
	""" Simple parameter optimization by iterate through
		all possible combinations of the given parameters
		for the model and returning the combination that
		results in the highest evaluation. If the given model
		is a class type, grid_search will initialize a new
		model for each evaluation; if it is a class instance,
		grid_search will use that instance, re-setting
		just the attributes passed in for testing. The only
		requirement for the model is that it has an eval()
		method that returns a number; if there is any code that
		needs to be run after __init__() and before eval(), it
		can go in a prep() method that will be run if it exists.
	"""
	best = ({}, -float("Inf"))
	names = params.keys()
	vals = params.values()

	for valuelist in itertools.product(*vals):
		params = { names[i]: valuelist[i] for i in range(len(valuelist)) }

		instance = model() if type(model) == type(object) else model
		for k, v in params.items():
			setattr(instance, k, v)

		if hasattr(instance, "prep"):
			instance.prep()

		val = instance.eval()
		if verbose:
			print("Parameter combination {0} yields an evaluation of {1}".format(params, val))
		if val > best[1]:
			best = (params, val)

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

	def prep(self):
		""" Do any work that, for some reason, needs to
			be done before eval and can't be combined with it.
			This method does not need to exist in your object.
		"""
		pass

	def eval(self):
		""" Return the model's evaluation (usually accuracy) as a
			single number. This method must exist in your object.
		"""
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
