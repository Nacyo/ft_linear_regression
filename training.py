import sys
import argparse
import pandas as pd
import numpy as np
import os.path
from linear_regression import *
import matplotlib.pyplot as plt


def main():

        parser = argparse.ArgumentParser()
        parser.add_argument("-m", "--maxiter",
                            help="set maximum iterations", type=int)
        parser.add_argument("-a", "--alpha",
                            help="set step size", type=float)
        parser.add_argument("-d", "--display", action='store_true',
                            help="display plot of gradient descent")
        parser.add_argument("-c", "--cost", action='store_true',
                            help="display plot of cost")
        parser.add_argument("file",
                            help="data file", type=argparse.FileType('r'))

        linear = LinearRegression()

        args = parser.parse_args()
        if args.maxiter:
            linear.max_iters = args.maxiter
            print("maxiter set at {:d}".format(linear.max_iters))
        if args.alpha:
            linear.alpha = args.alpha
            print("alpha set at {:f}".format(linear.alpha))
        if args.display:
            linear.disp = True
        if args.cost:
            linear.disp_cost = True

        data = pd.read_csv(args.file)
        x = data['km'].values
        y = data['price'].values
        linear.train(x, y)


if __name__ == '__main__':
    main()
