"""
Created on 09.09.2020

@author: pamueller

A script containing arbitrary fitting routines. Currently implemented

linear regression algorithms:
    - york(); [York et al., Am. J. Phys. 72, 367 (2004)]

"""

import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt


def straight(x, a, b):
    """
    :param x: The x values.
    :param a: The y-intercept.
    :param b: The slope.
    :returns: The y values resulting from the 'x' values via the given linear relation.
    """
    return a + b * x


def weight(sigma):
    """
    :param sigma: The 1-sigma uncertainty.
    :returns: The weight corresponding to the 1-sigma uncertainty 'sigma'.
    """
    return 1. / sigma ** 2


def york(x, y, sigma_x=None, sigma_y=None, corr=None, iter_max=200, report=True, show=False):
    """
    A linear regression algorithm to find the best straight line, given normally distributed errors for x and y
     and correlation coefficients between errors in x and y. The algorithm is described in
     ['Unified equations for the slope, intercept, and standard errors of the best straight line',
     York et al., American Journal of Physics 72, 367 (2004)]. See the comments to compare the individual steps.
    :param x: The x data.
    :param y: The y data.
    :param sigma_x: The 1-sigma uncertainty of the x data.
    :param sigma_y: The 1-sigma uncertainty of the y data.
    :param corr: The correlation coefficients between errors in 'x' and 'y'.
    :param iter_max: The maximum number of iterations to find the best slope.
    :param report: Whether to print the result of the fit.
    :param show: Whether to plot the fit result.
    :returns: a, b, sigma_a, sigma_b, corr_ab. The best y-intercept and slope,
     their respective 1-sigma uncertainties and their correlation coefficient.
    """
    x, y = np.asarray(x), np.asarray(y)
    if sigma_x is None:
        sigma_x = np.full_like(x, 1.)
        if report:
            print('\nNo uncertainties for \'x\' were given. Assuming \'sigma_x\'=1.')
    if sigma_y is None:
        sigma_y = np.full_like(y, 1.)
        if report:
            print('\nNo uncertainties for \'y\' were given. Assuming \'sigma_y\'=1.')
    if corr is None:
        corr = 0.
    sigma_x, sigma_y, corr = np.asarray(sigma_x), np.asarray(sigma_y), np.asarray(corr)

    # noinspection PyTypeChecker
    p_opt, _ = curve_fit(straight, x, y, p0=[0., 1.], sigma=sigma_y)  # (1)  # Update scipy to remove PyCharm warning.
    b_init = p_opt[1]
    b = b_init

    w_x, w_y = weight(sigma_x), weight(sigma_y)  # w(X_i), w(Y_i), (2)
    alpha = np.sqrt(w_x * w_y)

    n = 0
    r_tol = 1e-15
    tol = 1.
    mod_w = None
    x_bar, y_bar = None, None
    beta = None
    while tol > r_tol and n < iter_max:  # (6)
        b_init = b
        mod_w = w_x * w_y / (w_x + b_init ** 2 * w_y - 2. * b_init * corr * alpha)  # W_i, (3)
        x_bar, y_bar = np.average(x, weights=mod_w), np.average(y, weights=mod_w)  # (4)
        u = x - x_bar
        v = y - y_bar
        beta = mod_w * (u / w_x + v * b_init / w_y - straight(u, v, b_init) * corr / alpha)
        b = np.sum(mod_w * beta * v) / np.sum(mod_w * beta * u)  # (5)
        tol = abs((b - b_init) / b)
        n += 1  # (6)

    a = y_bar - b * x_bar  # (7)
    x_i = x_bar + beta  # (8)
    x_bar = np.average(x_i, weights=mod_w)  # (9)
    u = x_i - x_bar
    sigma_b = np.sqrt(1. / np.sum(mod_w * u ** 2))  # (10)
    sigma_a = np.sqrt(1. / np.sum(mod_w) + x_bar ** 2 * sigma_b ** 2)
    corr_ab = -x_bar * sigma_b / sigma_a
    chi2 = np.sum(mod_w * (y - b * x - a) ** 2) / (x.size - 2)

    if report:
        if n == iter_max:
            print('\nMaximum number of iterations ({}) was reached.'.format(iter_max))
        print('\nFit result:\na: {} +- {}\nb: {} +- {}\ncorr_ab: {}\nchi2: {}'
              .format(a, sigma_a, b, sigma_b, corr_ab, chi2))
    if show:
        x_cont = np.linspace(np.min(x) - 0.1 * (np.max(x) - np.min(x)), np.max(x) + 0.1 * (np.max(x) - np.min(x)), 1001)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.errorbar(x, y, xerr=sigma_x, yerr=sigma_y, fmt='k.', label='Data')
        plt.plot(x_cont, straight(x_cont, a, b), 'b-', label='Fit')
        y_min = np.min(np.array([straight(x_cont, a, b - sigma_b), straight(x_cont, a, b + sigma_b)]), axis=0)
        y_max = np.max(np.array([straight(x_cont, a, b - sigma_b), straight(x_cont, a, b + sigma_b)]), axis=0)
        plt.fill_between(x_cont, y_min, y_max, color='b', alpha=0.3, antialiased=True)
        plt.legend(loc='best', numpoints=1)
        plt.show()

    return a, b, sigma_a, sigma_b, corr_ab


# TODO: Replace york algorithm in 'alpha'-method with arbitrary linear regression routine.
def york_alpha(x, y, sigma_x=None, sigma_y=None, corr=None, alpha=0., find_alpha=True,
               iter_max=200, report=True, show=False):
    """
    :param x: The x data.
    :param y: The y data.
    :param sigma_x: The 1-sigma uncertainty of the x data.
    :param sigma_y: The 1-sigma uncertainty of the y data.
    :param corr: The correlation coefficients between errors in 'x' and 'y'.
    :param alpha: An x-axis offset to reduce the correlation coefficient between the y-intercept and the slope.
    :param find_alpha: Whether to search for the best 'alpha'. Uses the given 'alpha' as a starting point.
     May not give the desired result if 'alpha' was initialized to far from its optimal value.
    :param iter_max: The maximum number of iterations to find the best slope.
    :param report: Whether to print the result of the fit.
    :param show: Whether to plot the fit result.
    :returns: a, b, sigma_a, sigma_b, corr_ab, alpha. The best y-intercept and slope,
     their respective 1-sigma uncertainties, their correlation coefficient and the used alpha.
    """
    def cost(x0):
        _, _, _, _, c = york(x - x0[0], y, sigma_x=sigma_x, sigma_y=sigma_y, corr=corr, report=False)
        return c ** 2

    if find_alpha:
        alpha = minimize(cost, np.array([alpha])).x[0]

    a, b, sigma_a, sigma_b, corr_ab =\
        york(x - alpha, y, sigma_x=sigma_x, sigma_y=sigma_y, corr=corr, iter_max=iter_max, report=report, show=show)
    if report:
        print('alpha: {}'.format(alpha))
    return a, b, sigma_a, sigma_b, corr_ab, alpha


def test_york():
    """
    A small test of the fit with and without alpha.
    :returns: None.
    """
    x = np.linspace(90., 91., 1001)
    y = straight(x, -1., 2.)
    sigma_x = np.random.normal(loc=0.01, scale=0.002, size=1001)
    sigma_y = np.random.normal(loc=0.01, scale=0.002, size=1001)
    x = np.array([np.random.normal(loc=x_i, scale=s_x) for x_i, s_x in zip(x, sigma_x)])
    y = np.array([np.random.normal(loc=y_i, scale=s_y) for y_i, s_y in zip(y, sigma_y)])
    york(x, y, sigma_x=sigma_x, sigma_y=sigma_y, show=True)
    york_alpha(x, y, sigma_x=sigma_x, sigma_y=sigma_y, alpha=87., show=True)
