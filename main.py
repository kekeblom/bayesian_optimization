import argparse
import numpy as np
from tablib import Dataset
from utils import get_module, Config

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='simple_fn.yaml')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fn', type=str, default='tasks.regression.RandomRegression')
    parser.add_argument('--optimizer', default='bayes.BayesianOptimizer', type=str)
    parser.add_argument('--plot', action='store_true', default=False)
    return parser.parse_args()

def write_output(cli_args, param_config, optimizer, fn):
    fn_params = param_config.function_parameters()
    out_file = './out.csv'
    headers = ['Seed', 'Function', 'Attempt', 'Optimal Input', 'Optimal Output'] + fn_params + ['Output']
    output = Dataset(headers=headers)
    try:
        with open(out_file, 'rt') as f:
            output.load(f.read())
    except FileNotFoundError:
        pass

    seed = cli_args.seed
    fn_name = cli_args.fn
    optimal_in, optimal_out = fn.optimal_input()
    for i in range(optimizer.budjet):
        input_arguments = optimizer.inputs[i]
        params = input_arguments.get_parameters(fn_params)
        output.append([seed, fn_name, i, optimal_in, optimal_out] + params + [optimizer.out[i]])

    with open(out_file, 'w') as f:
        f.write(output.csv)


def main():
    cli_args = read_args()
    param_config = Config(cli_args.config)

    N_TRIES = 25

    np.random.seed(cli_args.seed)

    fn = get_module(cli_args.fn)()

    optimizer = get_module(cli_args.optimizer)(N_TRIES, param_config, fn, plot=cli_args.plot)

    best_input, best_output = optimizer.optimize()

    optimal_in, optimal_out = fn.optimal_input()
    print("max value: {} with input: {}, optimal output: {opt_out} with input: {opt_in} diff: {diff}".format(best_output, best_input,
        opt_out=optimal_out,
        opt_in=optimal_in,
        diff=optimal_out - best_output))

    write_output(cli_args, param_config, optimizer, fn)


if __name__ == "__main__":
    main()
