from functools import reduce
from collections import OrderedDict
import numpy as np
from tablib import Dataset
from scipy import optimize

class FnArg(object):

    def __init__(self):
        self.attributes = OrderedDict()

    def setattr(self, attr, value):
        self.attributes[attr] = value
        setattr(self, attr, value)

    def __repr__(self):
        return self.attributes.__repr__()

    def __getitem__(self, i):
        param = [*self.attributes.keys()][i]
        return self.attributes[param]

    def get_parameters(self, params):
        out = []
        for param in params:
            out.append(getattr(self, param))

        return out

    def to_array(self):
        values = [v.ravel() for v in self.attributes.values()]
        return np.concatenate(values)

class Config(object):

    def __init__(self, filename):
        with open(filename, 'rt') as (f):
            self.config = Dataset().load(f.read()).dict
        self._parse_config()

    def _parse_config(self):
        self.params = [p['name'] for p in self.config]

    def sample_uniform(self):
        args = FnArg()
        for param in self.config:
            if 'int' in param['type']:
                value = self._sample_int_param(param)
            else:
                value = self._sample_float_param(param)

            args.setattr(param['name'], value)
        return args

    def function_parameters(self):
        return self.params

    def input_dim(self):
        shapes = np.array([np.prod(v['shape']) for v in self.config])
        return np.sum(shapes)

    def param_from_array(self, x):
        fn_arg = FnArg()
        used_items = 0
        for param in self.config:
            size = reduce(lambda a, b: a * b, param['shape'])

            value = x[used_items:used_items + size].reshape(param['shape'])

            if 'int' in param['type']:
                value = np.round(value).astype(getattr(np, param['type']))

            fn_arg.setattr(param['name'], value)
            used_items += value.size

        return fn_arg

    def min(self):
        args = FnArg()
        for param in self.config:
            value = np.tile(param['min'], param['shape'])
            args.setattr(param['name'], value)
        return args

    def max(self):
        args = FnArg()
        for param in self.config:
            value = np.tile(param['max'], param['shape'])
            args.setattr(param['name'], value)
        return args

    def _sample_float_param(self, param):
        return np.random.uniform(param['min'], param['max'], size=param['shape'])

    def _sample_int_param(self, param):
        return np.random.randint(param['min'], param['max'], size=param['shape'])

class AcquisitionOptimizer(object):
    def __init__(self, gp, acquisition_fn, min_value, max_value):
        self.gp = gp
        self.acquisition_fn = acquisition_fn
        self.lower_bound = min_value
        self.upper_bound = max_value
        self.random_starts = 250

    def maximize(self):
        random_starts = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.random_starts, self.lower_bound.size))
        bounds = np.stack([self.lower_bound, self.upper_bound]).T
        best_y = np.inf
        best_input = None
        for start in random_starts:
            out = optimize.minimize(self._flipped_fn,
                x0=start,
                bounds=bounds)
            if out.fun < best_y:
                best_input = out.x
                best_y = out.fun

        return best_input

    def _flipped_fn(self, x):
        mean, var = self.gp.predict(x.reshape(1, -1))
        return -self.acquisition_fn(mean, var)


def get_module(module_path):
    components = module_path.split('.')
    module = __import__(components[0])
    for component in components[1:]:
        module = getattr(module, component)

    return module


if __name__ == '__main__':
    config = Config('./nn.yaml')
    param = config.param_from_array(np.array([64.6, 0.001]))
    assert (param.hidden_size == 65).all()
    assert (param.lr == 0.001).all()
