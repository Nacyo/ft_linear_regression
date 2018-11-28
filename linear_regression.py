import matplotlib.pyplot as plt
import numpy as np
import pickle


class LinearRegression:

    def __init__(self, alpha=0.01, max_iters=15000, disp=False, disp_cost=False):
        self.thetas = [0, 0]
        self.min_x = 0
        self.max_x = 1
        self.size = 0
        self.iters_hist = [0]
        self.cost_hist = []
        self.list_h = []
        self.disp = disp
        self.disp_cost = disp_cost
        self.alpha = alpha
        self.max_iters = max_iters

    def update_thetas(self, X, y):
        error_np_arr = self.h(X) - y
        m = self.size
        tmp0 = (self.alpha / m) * np.sum(error_np_arr)
        tmp1 = (self.alpha / m) * np.sum(error_np_arr * X)
        self.thetas[0] = self.thetas[0] - tmp0
        self.thetas[1] = self.thetas[1] - tmp1

    def cost_func(self, X, y):
        squared_error = np.power(self.h(X) - y, 2)
        m = self.size
        cost = np.sum(squared_error) / (2 * m)
        return cost

    def h(self, X):
        return self.thetas[0] + self.thetas[1] * X

    def normalize(self, vect):
        self.min_x = np.min(vect)
        self.max_x = np.max(vect)
        self.size = len(vect)
        normalized_vec = (vect - self.min_x) / (self.max_x - self.min_x)
        return normalized_vec

    def normalize_exemple(self, x):
        normalize_exemple = (float(x) - self.min_x) / (self.max_x - self.min_x)
        return normalize_exemple

    def estimation(self, mileage):
        mileage = self.normalize_exemple(mileage)
        estimate = self.h(mileage)
        if estimate < 0:
            estimate = 0
        print("Your car is estimated at {:.2f}".format(estimate))
        return estimate

    def r_2_rmse_score(self, X, y):
        y_pred = self.thetas[0] + self.thetas[1] * X
        sumofresiduals = np.sum(np.square((y - y_pred)))
        sumofsquares = np.sum(np.square((y - np.mean(y))))
        r_2_score = 1 - (sumofresiduals/sumofsquares)
        rmse = (sumofresiduals / len(X)) ** (1 / 2)
        print("R² score : ", r_2_score, "\nRMSE score : ", rmse)

    def display_cost(self):
        plt.figure(2)
        plt.title('cost function')
        plt.xlabel('Iteration')
        plt.ylabel('sqrt of Cost')
        plt.grid(True)
        axes = plt.gca()
        axes.set_xlim([0, self.max_iters])
        axes.set_ylim([self.cost_hist[-1], self.cost_hist[0]])
        Ln_cost, = plt.plot(range(1), self.cost_hist[0], 'b.')

        plt.figure(2)
        for i in range(len(self.cost_hist)):
            Ln_cost.set_ydata(self.cost_hist[:i])
            Ln_cost.set_xdata(self.iters_hist[:i])
            plt.pause(0.005)
        plt.show()

    def display_linear(self, x, y):
        plt.figure(1)
        plt.title('Price / mileage')
        plt.xlabel('Mileage')
        plt.ylabel('Price')
        plt.grid(True)
        plt.scatter(x, y)
        Ln, = plt.plot(x, self.list_h[0], lw=2, c="k")

        plt.figure(1)
        for h in self.list_h[1:]:
            Ln.set_ydata(h)
            plt.pause(0.005)
        plt.show()
        if self.disp_cost:
            plt.close()

    def train(self, x, y):
        print("Training in process: ")
        X_norm = self.normalize(x)
        iters = 0

        while iters < self.max_iters:
            prev_thet = self.thetas
            self.update_thetas(X_norm, y)
            if iters % 50 == 0:
                if self.disp:
                    self.list_h.append(self.h(X_norm))
                if self.disp_cost:
                    self.cost_hist.append((self.cost_func(X_norm, y)) ** (1/2))
                    self.iters_hist.append(iters)
            iters += 1
        print("Iterations: ", iters)
        self.r_2_rmse_score(X_norm, y)
        print("DONE")

        filename = 'finalized_model.sav'
        pickle.dump(self, open(filename, 'wb'))

        if self.disp is True and self.max_iters > 9:
            self.display_linear(x, y)

        if self.disp_cost is True and self.max_iters > 9:
            self.display_cost()
