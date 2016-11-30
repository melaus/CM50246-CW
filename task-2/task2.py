import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, det
from scipy.optimize import minimize


def generate_data(num_pts, a, b, list_range, delta_mode):
    """
    generate some random data
    :param num_pts:
    :param a:
    :param b:
    :param list_range: list. [lower_limit, upper limit]
    :param delta_mode: string. 'uniform' or 'normal' distribution
    :return: np.array. x and y coordinates 2D points (transposed)
    """
    num_pts_list = np.random.uniform(list_range[0], list_range[1], size=(num_pts,))

    if delta_mode == 'normal':
        delta = np.random.randn(num_pts)
    elif delta_mode == 'uniform':
        delta = np.random.uniform(list_range[0], list_range[1], size=(num_pts,))

    return np.array([num_pts_list, a*num_pts_list + b + delta])


def ml_est_phi(x_vec, y_vec):
    """
    compute the estimated phi values with maximum likelihood
    :param x_vec:
    :param y_vec:
    :return:
    """
    return inv(x_vec * x_vec.T) * x_vec * y_vec


def ml_est_covar(x_vec, y_vec, phi, num_pts):
    return ((y_vec - x_vec.T * phi).T * (y_vec - x_vec.T * phi)) / num_pts


def map_est_phi(x_vec, y_vec, covar, p_covar):
    """

    :param x_vec:
    :param y_vec:
    :param covar:
    :param p_covar: prior covariance
    :return:
    """
    return inv(x_vec * x_vec.T + (covar / p_covar) * np.mat(np.identity(1))) * x_vec * y_vec


def bayes_a(x_vec, covar, p_covar):
    """
    the A component to Bayes posterior/ estimation
    :param x_vec:
    :param covar:
    :return:
    """
    return (x_vec * x_vec.T) / covar + 1.0 / np.mat([0.7]) * np.identity(1)


def bayes_posterior(x_vec, y_vec, phi, covar):
    a = bayes_a(x_vec, covar)

    bayes_phi = inv(a) * x_vec * y_vec / covar
    bayes_covar = inv(a)


def lin_reg(pts, num_pts, mode, p_covar=2.0,
            is_plot=False, title='', fig_num=1):
    """
    perform linear regression (ML or MAP)
    :param pts: np.array. datapoints and targets
    :param num_pts: int. Number of points
    :param mode: string. 'ml' or 'map'
    :param p_covar: double. prior covariance value
    :param is_plot: (optional) True if a plot is required
    :param title: (optional) Title for the plot
    :param fig_num: (optional) On which figure to plot data on
    :return:
    """
    x_vec = np.insert(np.mat(pts[0]), 0, [1 for i in range(num_pts)], axis=0)  # datapoints with prepended 1's
    print x_vec.shape
    y_vec = np.mat(pts[1]).T  # target
    dim = y_vec.shape[1]

    if mode == 'ml':
        est_phi = ml_est_phi(x_vec, y_vec)

    elif mode == 'map':
        est_phi = ml_est_phi(x_vec, y_vec)
        est_covar = ml_est_covar(x_vec, y_vec, est_phi, num_pts)
        est_phi = map_est_phi(x_vec, y_vec, est_covar, p_covar)

    # print to check output
    print '{0} est_phi:\n{1}'.format(mode, est_phi)

    if is_plot:
        plot_output(title, pts, num_pts, est_phi, fig_num)


def plot_output(title, pts, num_pts, phi, fig_num=1):
    """
    :param title
    :param pts
    :param num_pts:
    :param phi:
    :param fig_num: (optional) int. Default is 1. Which figure to plot things on
    """
    plt.figure(fig_num)
    plt.title(title)
    plt.plot(pts[0], pts[1], 'bs')  # plot the datapoints
    plt.plot(pts[0], np.array(phi[0] + phi[1] * pts[0]).reshape(num_pts, ), 'r')  # plot the fitted line
    plt.draw()


if __name__ == '__main__':
    num_pts = 50
    data = generate_data(num_pts, 0.4, 3, [0, 40], 'uniform')

    # perform an ML linear regression
    lin_reg(data, num_pts, 'ml',  is_plot=True, title='ML linear regression (normal)',  fig_num=1)
    lin_reg(data, num_pts, 'map', is_plot=True, title='MAP linear regression (normal)', fig_num=2)
    plt.show()
