import numpy as np
from scipy import optimize

class RandomRegression(object):
    def __init__(self):
        self.d_in = 2
        self.expand = 7
        self.w = np.random.uniform(-0.01, 0.01, size=(self.expand * self.d_in, 1))
        self.min = -10
        self.max = 10

    def __call__(self, arg):
        value = (self._expand(arg.x[None]) @ self.w)[0]
        return value + np.random.normal(loc=0.0, scale=10.0)

    def _expand(self, x):
        out = np.zeros((x.shape[0], x.shape[1] * self.expand))
        for i in range(x.shape[0]):
            out[i, :] = np.power(x[i][None].T, np.arange(self.expand)).ravel()
        return out

    def _gaussian(self, x, mu, sigma2):
        return np.exp(-(x - mu)**2/(2 * sigma2))

    def optimal_input(self):
        x0s = np.random.uniform([-10, -10], [10, 10], size=(250, self.d_in))
        best_output = np.inf
        best_input = None
        function = lambda x: -(self._expand(x[None]) @ self.w)[0][0]

        for x0 in x0s:
            result = optimize.minimize(function, x0, bounds=((-10, 10), (-10, 10)))
            if result.fun < best_output:
                best_input = result.x
                best_output = result.fun
        return best_input, -best_output
