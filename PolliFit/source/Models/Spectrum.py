"""
Created on 25.02.2022

@author: Patrick Mueller
"""
import numpy as np
from scipy.stats import norm, cauchy
from scipy.special import voigt_profile

from Physics import source_energy_pdf
from Models.Base import Model


# The names of the spectra. Includes all spectra that appear in the GUI.
SPECTRA = ['Gauss', 'Lorentz', 'Voigt', 'GaussChi2']


class Spectrum(Model):
    def __init__(self):
        super().__init__(model=None)
        self.type = 'Spectrum'

    def evaluate(self, x, *args, **kwargs):
        return np.zeros_like(x)

    @property
    def dx(self):
        return self.fwhm() * 0.02

    def fwhm(self):
        return 1.

    def min(self):
        return -5 * self.fwhm()

    def max(self):
        return 5 * self.fwhm()


class Lorentz(Spectrum):
    def __init__(self):
        super().__init__()
        self.type = 'Lorentz'

        self._add_arg('Gamma', 1., False, False)

    def evaluate(self, x, *args, **kwargs):
        scale = 0.5 * args[0]
        return np.pi * scale * cauchy.pdf(x, loc=0, scale=scale)

    def fwhm(self):
        return abs(self.vals[self.p['Gamma']])

    def min(self):
        return -10 * self.fwhm()

    def max(self):
        return 10 * self.fwhm()


class Gauss(Spectrum):
    def __init__(self):
        super().__init__()
        self.type = 'Gauss'

        self._add_arg('sigma', 1., False, False)

    def evaluate(self, x, *args, **kwargs):
        return np.sqrt(2 * np.pi) * args[0] * norm.pdf(x, loc=0, scale=args[0])

    def fwhm(self):
        return abs(np.sqrt(8 * np.log(2)) * self.vals[self.p['sigma']])

    def min(self):
        return -2.5 * self.fwhm()

    def max(self):
        return 2.5 * self.fwhm()


class Voigt(Spectrum):
    def __init__(self):
        super().__init__()
        self.type = 'Voigt'

        self._add_arg('Gamma', 1., False, False)
        self._add_arg('sigma', 1., False, False)

    def evaluate(self, x, *args, **kwargs):
        return voigt_profile(x, args[1], 0.5 * args[0]) / voigt_profile(0, args[1], 0.5 * args[0])

    def fwhm(self):
        f_l = self.vals[self.p['Gamma']]
        f_g = np.sqrt(8 * np.log(2)) * self.vals[self.p['sigma']]
        return abs(0.5346 * f_l + np.sqrt(0.2166 * f_l ** 2 + f_g ** 2))


def _gauss_chi2_limits(sigma, xi, a, b, c, d):
    return a * sigma ** b + c * xi ** d


class GaussChi2(Spectrum):  # TODO: The bounds of GaussChi2 are not fitting very well for xi < sigma yet.
    def __init__(self):
        super().__init__()
        self.type = 'GaussBoltzmann'

        self._add_arg('sigma', 1., False, False)
        self._add_arg('xi', 1., False, False)

    def evaluate(self, x, *args, **kwargs):
        return source_energy_pdf(x, 0, args[0], args[1], collinear=True)

    def fwhm(self):
        xi = np.abs(self.vals[self.p['xi']])
        sigma = self.vals[self.p['sigma']]
        if xi == 0:
            return np.sqrt(8 * np.log(2)) * sigma
        a, b, c, d = (1.83242409, 1.18078908, 0.20052479, 1.37141081)
        return _gauss_chi2_limits(sigma, xi, a, b, c, d)

    def min(self):
        # return -5 * np.sqrt(8 * np.log(2)) * self.vals[self.p['sigma']]
        xi = np.abs(self.vals[self.p['xi']])
        sigma = self.vals[self.p['sigma']]
        if xi == 0:
            return -2.5 * np.sqrt(8 * np.log(2)) * sigma
        a, b, c, d = (13.06659493, -0.80613167, -26.2198051, 0.75698759)
        if self.vals[self.p['xi']] < 0:
            a, b, c, d = (-3.77923609, 0.98956488, -3.54807923, -0.65252397)
        return _gauss_chi2_limits(sigma, xi, a, b, c, d)

    def max(self):
        # return 5 * np.sqrt(8 * np.log(2)) * self.vals[self.p['sigma']]
        xi = np.abs(self.vals[self.p['xi']])
        sigma = self.vals[self.p['sigma']]
        if xi == 0:
            return 2.5 * np.sqrt(8 * np.log(2)) * sigma
        a, b, c, d = (-13.06659493, -0.80613167, 26.2198051, 0.75698759)
        if self.vals[self.p['xi']] > 0:
            a, b, c, d = (3.77923609, 0.98956488, 3.54807923, -0.65252397)
        return _gauss_chi2_limits(sigma, xi, a, b, c, d)
