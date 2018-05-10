import numpy as np

class Optimizer(object):
    def __init__(self, budjet, param_config, fn, **kwargs):
        self.out = np.zeros(budjet, dtype=np.float32)
        self.inputs = []
        self.budjet = budjet
        self.param_config = param_config
        self.fn = fn

class RandomOptimizer(Optimizer):
    def optimize(self):
        for i in range(self.budjet):
            self._random_sample(i)

        best_index = np.argmax(self.out)
        return self.inputs[best_index], self.out[best_index]

    def _random_sample(self, i):
        args = self.param_config.sample_uniform()
        self.inputs.append(args)
        self.out[i] = self.fn(args)
