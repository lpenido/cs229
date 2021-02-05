import util
import numpy as np
import matplotlib.pyplot as plt
import os

np.seterr(all='raise')


factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """
        Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***

        # J(theta): Least squares cost function
        # Square the differences between h(x)_i and y_i, for each i
        sq_diffs = []
        for i in range(X.shape[0]):
            diff = self.theta*X[i][1] - y[i]
            sq_diffs.append(diff ** 2)

        mapd = list(map(lambda x, y:  ))
        ssd = sum(sq_diffs)
        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***

        # Separate x from X
        x = np.array_split(X, 2, axis=1)[1]

        pwrs = [i+1 for i in range(k)]
        print(pwrs)
        # Take each x and return an array for the ks
        quik_maths = lambda x: np.array([x**3, x**2, x**1, 1])
        # print(quik_maths(x))


        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        # *** END CODE HERE ***


def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    train_x, train_y = util.load_dataset(train_path, add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)

    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***
        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 1], plot_y, label='k=%d' % k)

    plt.legend()
    plt.savefig(filename)
    plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Run all expetriments
    '''
    # *** START CODE HERE ***
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv')

    train_path = "/Users/lucaspenido/Downloads/CS221/ps1/src/featuremaps/train.csv"
    train_x, train_y = util.load_dataset(train_path, add_intercept=True)


    lm = LinearModel(theta=0.0001)
    samp = np.array([
        [1,2],
        [3,4],
        [5,6]
        ])
    lm.fit(train_x, train_y)

    lm.create_poly(k=3, X=samp)
