# Linear regression

The aim of this project is to introduce you to the basic concept behind machine learning.
For this project, you will have to create a program that predicts the price of a car
by using a linear function train with a gradient descent algorithm.
We will work on a precise example for the project, but once you’re done you will be
able to use the algorithm with any other dataset.

## Implementation
You will implement a simple linear regression with a single feature - in this case, the
mileage of the car.
To do so, you need to create two programs :
* The first program will be used to predict the price of a car for a given mileage.
  When you launch the program, it should prompt you for a mileage, and then give
  you back the estimated price for that mileage.
  Before the run of the training program, theta0 and theta1 will be set to 0.
* The second program will be used to train your model. It will read your dataset
  file and perform a linear regression on the data.
  Once the linear regression has completed, you will save the variables theta0 and
  theta1 for use in the first program.

## Usage
Training
```
usage: training.py [-h] [-m MAXITER] [-a ALPHA] [-d] [-c] file

positional arguments:
  file                  data file

optional arguments:
  -h, --help            show this help message and exit
  -m MAXITER, --maxiter MAXITER
                        set maximum iterations
  -a ALPHA, --alpha ALPHA
                        set step size
  -d, --display         display plot of gradient descent
  -c, --cost            display plot of cost
```
Estimate
```
usage: estimate.py [-h] [-d] [-e] mileage

positional arguments:
  mileage        Mileage to estimate

optional arguments:
  -h, --help     show this help message and exit
  -d, --display  display plot of gradient descent
  -e, --erase    reinit trained model
```

## Resources
* http://mccormickml.com/2014/03/04/gradient-descent-derivation/
* https://www.coursera.org/learn/machine-learning
* https://en.wikipedia.org/wiki/Linear_function
* https://en.wikipedia.org/wiki/Gradient_descent
* https://towardsdatascience.com/linear-regression-from-scratch-cd0dee067f72
* https://en.wikipedia.org/wiki/Coefficient_of_determination
* https://towardsdatascience.com/linear-regression-from-scratch-cd0dee067f72
