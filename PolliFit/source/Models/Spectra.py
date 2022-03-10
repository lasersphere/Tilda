"""
Created on 25.02.2022

@author: Patrick Mueller
"""

from scipy.stats import norm, cauchy
from scipy.special import voigt_profile

from Models.Base import *


# The names of the spectra. Includes all spectra that appear in the GUI.
SPECTRA = ['Gauss', 'Lorentz', 'Voigt']


class Spectrum(Model):
    def __init__(self):
        super().__init__(model=None)
        self.type = 'Spectrum'

    def evaluate(self, x, *args, **kwargs):
        return np.zeros_like(x)

    @property
    def dx(self):
        return self.fwhm() * 1e-2

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
        return self.vals[self.p['Gamma']]


class Gauss(Spectrum):
    def __init__(self):
        super().__init__()
        self.type = 'Gauss'
        self._add_arg('sigma', 1., False, False)

    def evaluate(self, x, *args, **kwargs):
        return np.sqrt(2 * np.pi) * args[0] * norm.pdf(x, loc=0, scale=args[0])

    def fwhm(self):
        return np.sqrt(8 * np.log(2)) * self.vals[self.p['sigma']]


class Voigt(Spectrum):
    def __init__(self):
        super().__init__()
        self.type = 'Voigt'
        self._add_arg('Gamma', 1., False, False)
        self._add_arg('sigma', 1., False, False)

    def evaluate(self, x, *args, **kwargs):
        # z = (x + 1j * args[0] / 2.) / (np.sqrt(2.) * args[1])
        # return np.real(wofz(z)) / (args[1] * np.sqrt(2. * np.pi))
        return voigt_profile(x, args[1], 0.5 * args[0]) / voigt_profile(0, args[1], 0.5 * args[0])

    def fwhm(self):
        f_l = self.vals[self.p['Gamma']]
        f_g = np.sqrt(8 * np.log(2)) * self.vals[self.p['sigma']]
        return 0.5346 * f_l + np.sqrt(0.2166 * f_l ** 2 + f_g ** 2)
