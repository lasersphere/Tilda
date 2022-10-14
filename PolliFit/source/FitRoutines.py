"""
Created on 09.09.2020

@author: pamueller

A script containing arbitrary fitting routines. Currently implemented

linear regression algorithms:
    - york(); [York et al., Am. J. Phys. 72, 367 (2004)]

"""

import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as plt


def const(_, a):
    """
    :param _: The x values.
    :param a: The constant.
    :returns: a.
    """
    return a


def straight(x, a, b):
    """
    :param x: The x values.
    :param a: The y-intercept.
    :param b: The slope.
    :returns: The y values resulting from the 'x' values via the given linear relation.
    """
    return a + b * x


def straight_std(x, sigma_a, sigma_b, corr_ab):
    """
    :param x: The x values.
    :param sigma_a: The standard deviation of the y-intercept.
    :param sigma_b: The standard deviation of the slope.
    :param corr_ab: The correlation coefficient between the slope and y-intercept.
    :returns: The standard deviation of a straight line where the x values do not have uncertainties.
    """
    return np.sqrt(sigma_a ** 2 + (x * sigma_b) ** 2 + 2 * x * sigma_a * sigma_b * corr_ab)


def straight_x_std(x, b, sigma_x, sigma_a, sigma_b, corr_ab):
    """
    :param x: The x values.
    :param b: The slope.
    :param sigma_x: The standard deviation of the x values.
    :param sigma_a: The standard deviation of the y-intercept.
    :param sigma_b: The standard deviation of the slope.
    :param corr_ab: The correlation coefficient between the slope and y-intercept.
    :returns: The standard deviation of a straight line where all input values have uncertainties.
    """
    return np.sqrt(sigma_a ** 2 + (x * sigma_b) ** 2 + 2 * x * sigma_a * sigma_b * corr_ab
                   + (b * sigma_x) ** 2 + (sigma_b * sigma_x) ** 2)


def weight(sigma):
    """
    :param sigma: The 1-sigma uncertainty.
    :returns: The weight corresponding to the 1-sigma uncertainty 'sigma'.
    """
    return 1. / sigma ** 2


def floor_log10(x):
    """
    :param x: Scalar values.
    :returns: The closest integer values to the logarithm with basis 10 of the absolute value of 'x' that are smaller.
    """
    if x == 0:
        return 0
    return int(np.floor(np.log10(np.abs(x))))


def ellipse2d(x, y, scale_x, scale_y, phi, corr):
    """
    :param x: The x-component of the position of the ellipse.
    :param y: The y-component of the position of the ellipse.
    :param scale_x: The amplitude of the x-component.
    :param scale_y: The amplitude of the y-component.
    :param phi: The angle between the vector to the point on the ellipse and the x-axis.
    :param corr: The correlation coefficient between the x and y data.
    :returns: A point on an ellipse in 2d-space with amplitudes 'x', 'y'
     and correlation 'corr' between x- and y-component.
    """
    x, y, scale_x, scale_y, corr = np.asarray(x), np.asarray(y), \
        np.asarray(scale_x), np.asarray(scale_y), np.asarray(corr)
    return x + scale_x * np.cos(phi), y + scale_y * (corr * np.cos(phi) + np.sqrt(1 - corr ** 2) * np.sin(phi))


def draw_sigma2d(x, y, sigma_x, sigma_y, corr, n, **kwargs):
    """
    :param x: The x data.
    :param y: The y data.
    :param sigma_x: The 1-sigma uncertainties of the x data.
    :param sigma_y: The 1-sigma uncertainties of the y data.
    :param corr: The correlation coefficients between the x and y data.
    :param n: The maximum sigma region to draw
    :param kwargs: Additional keyword arguments are passed to plt.plot().
    :returns: None. Draws the sigma-bounds of the given data points (x, y) until the n-sigma region.
    """
    phi = np.arange(0., 2 * np.pi, 0.001)
    for x_i, y_i, s_x, s_y, r in zip(x, y, sigma_x, sigma_y, corr):
        for i in range(1, n + 1, 1):
            _x, _y = ellipse2d(x_i, y_i, i * s_x, i * s_y, phi, r)
            plt.plot(_x, _y, 'k-', **kwargs)


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
     If not None, The errorbars are circles indicating the 2-dimensional 1-sigma region.
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
    sigma_2d = True
    if corr is None:
        sigma_2d = False
        corr = 0.
    sigma_x, sigma_y, corr = np.asarray(sigma_x), np.asarray(sigma_y), np.asarray(corr)

    p_opt, _ = so.curve_fit(straight, x, y, p0=[0., 1.], sigma=sigma_y)
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
        beta = mod_w * (u / w_y + v * b_init / w_x - straight(u, v, b_init) * corr / alpha)
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
        if sigma_2d:
            plt.plot(x, y, 'k.', label='Data')
            draw_sigma2d(x, y, sigma_x, sigma_y, corr, n=1)
        else:
            plt.errorbar(x, y, xerr=sigma_x, yerr=sigma_y, fmt='k.', label='Data')
        plt.plot(x_cont, straight(x_cont, a, b), 'b-', label='Fit')
        y_min = straight(x_cont, a, b) - straight_std(x_cont, sigma_a, sigma_b, corr_ab)
        y_max = straight(x_cont, a, b) + straight_std(x_cont, sigma_a, sigma_b, corr_ab)
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
     May not give the desired result if 'alpha' was initialized too far from its optimal value.
    :param iter_max: The maximum number of iterations to find the best slope.
    :param report: Whether to print the result of the fit.
    :param show: Whether to plot the fit result.
    :returns: a, b, sigma_a, sigma_b, corr_ab, alpha. The best y-intercept and slope,
     their respective 1-sigma uncertainties, their correlation coefficient and the used alpha.
    """
    n = floor_log10(alpha)

    def cost(x0):
        _, _, _, _, c = york(x - x0[0], y, sigma_x=sigma_x, sigma_y=sigma_y, corr=corr, report=False)
        return c ** 2 * 10. ** (n + 1)

    if find_alpha:
        alpha = so.minimize(cost, np.array([alpha])).x[0]

    a, b, sigma_a, sigma_b, corr_ab =\
        york(x - alpha, y, sigma_x=sigma_x, sigma_y=sigma_y, corr=corr, iter_max=iter_max, report=report, show=show)
    if report:
        print('alpha: {}'.format(alpha))
    return a, b, sigma_a, sigma_b, corr_ab, alpha


def curve_fit(f, x, y, p0=None, p0_fixed=None, sigma=None, absolute_sigma=False, check_finite=True,
              bounds=(-np.inf, np.inf), method=None, jac=None, report=False, **kwargs):
    """
    :param f: The model function to fit to the data.
    :param x: The x data.
    :param y: The y data.
    :param p0: A numpy array or an Iterable of the initial guesses for the parameters.
     Must have at least the same length as the minimum number of parameters required by the function 'f'.
     If 'p0' is None, 1 is taken as an initial guess for all non-keyword parameters.
    :param p0_fixed: A numpy array or an Iterable of bool values specifying, whether to fix a parameter.
     Must have the same length as p0.
    :param sigma: The 1-sigma uncertainty of the y data.
    :param absolute_sigma: See scipy.optimize.curve_fit.
    :param check_finite: See scipy.optimize.curve_fit.
    :param bounds: See scipy.optimize.curve_fit.
    :param method: See scipy.optimize.curve_fit.
    :param jac: See scipy.optimize.curve_fit.
    :param report: Whether to print the result of the fit.
    :param kwargs: See scipy.optimize.curve_fit.
    :returns: popt, pcov. The optimal parameters and their covariance matrix.
    """
    if p0_fixed is None or all(not p for p in p0_fixed):
        popt, pcov = so.curve_fit(f, x, y, p0=p0, sigma=sigma, absolute_sigma=absolute_sigma,
                                  check_finite=check_finite, bounds=bounds, method=method, jac=jac, **kwargs)
    elif p0 is None:
        raise ValueError('Please specify the initial parameters when any of the parameters shall be fixed.')
    else:
        p0, p0_fixed = np.asarray(p0), np.asarray(p0_fixed).astype(bool)
        if p0_fixed.shape != p0.shape:
            raise ValueError('\'p0_fixed\' must have the same shape as \'p0\'.')
        _p0 = p0[~p0_fixed]
        _bounds = (np.asarray(bounds[0]), np.asarray(bounds[1]))
        if len(_bounds[0].shape) > 0 and _bounds[0].size == p0.size:
            _bounds = (_bounds[0][~p0_fixed], _bounds[1][~p0_fixed])

        def _f(_x, *args):
            _args = p0
            _args[~p0_fixed] = np.asarray(args)
            return f(_x, *_args)

        popt, pcov = np.ones_like(p0, dtype=float), np.zeros((p0.size, p0.size), dtype=float)
        pcov_mask = ~(np.expand_dims(p0_fixed, axis=1) + np.expand_dims(p0_fixed, axis=0))
        popt[p0_fixed] = p0[p0_fixed]
        _popt, _pcov = so.curve_fit(_f, x, y, p0=_p0, sigma=sigma, absolute_sigma=absolute_sigma,
                                    check_finite=check_finite, bounds=_bounds, method=method, jac=jac, **kwargs)
        popt[~p0_fixed] = _popt
        pcov[pcov_mask] = _pcov.flatten()

    if report:
        print('curve_fit result:')
        for i, (val, err) in enumerate(zip(popt, np.sqrt(np.diag(pcov)))):
            print('{}: {} +/- {}'.format(i, val, err))
    return popt, pcov
