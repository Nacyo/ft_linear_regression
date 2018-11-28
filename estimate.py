import sys
import pandas as pd
import os.path
import pickle
import argparse
from linear_regression import *


def display_final(loaded_model, estimate, mileage):
    data = pd.read_csv("data.csv")
    km = data['km'].values
    price = data['price'].values
    plt.title('Data visualisation price / mileage')
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.grid(True)
    plt.scatter(km, price)
    km_norm = loaded_model.normalize(km)
    plt.plot(km, loaded_model.h(km_norm), lw=2, c="k")
    plt.plot([mileage], [estimate], marker='o', markersize=3, color="red")
    plt.show()


def main():

        parser = argparse.ArgumentParser()
        parser.add_argument("-d", "--display", action='store_true',
                            help="display plot of gradient descent")

        parser.add_argument("-e", "--erase", action='store_true',
                            help="reinit trained model")

        parser.add_argument("mileage",
                            help="Mileage to estimate", type=float)

        linear = LinearRegression()
        filename = 'finalized_model.sav'

        args = parser.parse_args()
        if args.display:
            linear.disp = True

        if args.erase is False and os.path.isfile(filename):
            loaded_model = pickle.load(open(filename, 'rb'))
        elif args.erase and os.path.isfile(filename) is False:
            print("Model does not exist cannot erase")
            sys.exit()
        else:
            if args.erase:
                os.remove(filename)
            loaded_model = LinearRegression()
            print("New model not trained")

        mileage = args.mileage

        estimate = loaded_model.estimation(mileage)
        if linear.disp is True:
            display_final(loaded_model, estimate, mileage)


if __name__ == '__main__':
    main()
