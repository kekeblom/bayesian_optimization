import argparse
import numpy as np
from tasks.cnn import Cifar10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--layers', default=[1, 1, 1, 1])
    parser.add_argument('--conv-dropout', default=0.1, type=float)
    parser.add_argument('--fc-dropout', default=0.1, type=float)
    flags = parser.parse_args()

    task = Cifar10()
    accuracy = task(flags)
    print("Accuracy: ", accuracy)



if __name__ == '__main__':
    main()