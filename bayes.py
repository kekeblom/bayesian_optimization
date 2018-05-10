import numpy as np
import GPy
from sklearn import preprocessing
from matplotlib import pyplot as plt
from matplotlib import colors
from random_opt import RandomOptimizer
from utils import AcquisitionOptimizer
import matplotlib

class BayesianOptimizer(RandomOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X_scaler = preprocessing.StandardScaler()
        self.Y_scaler = preprocessing.StandardScaler()
        self.random_samples = 3
        self.kappa = 2.5
        self.plot = kwargs.get('plot', False)

    def optimize(self):
        for i in range(self.random_samples):
            self._random_sample(i)

        input_dim = self.param_config.input_dim()
        kernel = GPy.kern.Matern32(input_dim=input_dim, ARD=True)
        for i in range(self.random_samples, self.budjet):
            X, Y = self._get_examples()
            model = GPy.models.GPRegression(X, Y, kernel)
            model.optimize_restarts(num_restarts=15)
            self._plot(model, X, Y, i)
            self._acquire_sample(model, i)
        best_index = np.argmax(self.out)
        return self.inputs[best_index], self.out[best_index]

    def _acquire_sample(self, model, i):
        next_sample = self._next_sample(model)
        arguments = self.param_config.param_from_array(next_sample)
        self.inputs.append(arguments)
        self.out[i] = self.fn(arguments)

    def _acquisition_function(self, mean, var):
        return mean + self.kappa * np.sqrt(var)

    def _next_sample(self, model):
        min_value = self.param_config.min().to_array()
        max_value = self.param_config.max().to_array()
        min_value = self.X_scaler.transform(min_value[np.newaxis])[0]
        max_value = self.X_scaler.transform(max_value[np.newaxis])[0]

        optimizer = AcquisitionOptimizer(model, self._acquisition_function, min_value, max_value)
        next_x = optimizer.maximize()
        return self.X_scaler.inverse_transform(next_x[np.newaxis])[0]

    def _get_examples(self):
        count = len(self.inputs)
        X = np.stack([arg.to_array() for arg in self.inputs])
        Y = self.out[0:count].reshape(-1, 1)
        self.X_scaler.fit(X)
        self.Y_scaler.fit(Y)
        return self.X_scaler.transform(X), self.Y_scaler.transform(Y)

    def _plot(self, model, X, Y, i):
        if not self.plot:
            return
        input_dim = self.param_config.input_dim()
        if input_dim == 1:
            self._plot1d(model, X, Y, i)
        elif input_dim == 2:
            self._plot2d(model, X, Y, i)

    def _plot1d(self, model, X, Y, i):
        fig = plt.figure(figsize=(7, 3.5))
        ax = plt.subplot(2, 1, 1)

        actual_x = np.linspace(self.param_config.min().x[0, 0], self.param_config.max().x[0, 0], num=100)
        x = self.X_scaler.transform(actual_x.reshape(-1, 1))
        actual_X = self.X_scaler.inverse_transform(X)
        actual_Y = self.Y_scaler.inverse_transform(Y)

        mean, var = model.predict(x)
        actual_mean = self.Y_scaler.inverse_transform(mean.reshape(-1, 1))[:, 0]

        ax.plot(actual_x, actual_mean, label='Mean')
        actual_lower_confidence = self.Y_scaler.inverse_transform(mean - 2 * var)[:, 0]
        actual_upper_confidence = self.Y_scaler.inverse_transform(mean + 2 * var)[:, 0]
        ax.plot(actual_x, actual_lower_confidence, color='b', alpha=0.5)
        ax.plot(actual_x, actual_upper_confidence, color='b', alpha=0.5)

        transparent_blue = colors.to_rgba('b', alpha=0.1)
        ax.fill_between(actual_x, actual_lower_confidence, actual_upper_confidence, color=transparent_blue, label=r'$2\sigma^2$ confidence interval')
        ax.scatter(actual_X[:, 0], actual_Y[:, 0], label='Data point')
        ax.legend()

        ax = plt.subplot(2, 1, 2, sharex=ax)

        acquisition_function = self.Y_scaler.inverse_transform(self._acquisition_function(mean, var))[:, 0]
        ax.plot(actual_x, acquisition_function)
        ax.set_title("Upper confidence bound")

        fig.tight_layout()
        fig.savefig('./plots/gp_fit_{}.png'.format(i))
        plt.close(fig)

    def _plot2d(self, model, X, Y, i):
        model.plot()
        plt.tight_layout()
        plt.savefig('./plots/gp_fit_{}.png'.format(i))
        plt.close()

