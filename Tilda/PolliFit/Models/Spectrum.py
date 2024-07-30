"""
Created on 30.07.2024

@author: Patrick Mueller
"""

import numpy as np
import qspec.models as mod


# The names of the spectra. Includes all spectra that appear in the GUI.
SPECTRA = ['TestShape']


class TestShape(mod.Spectrum):
    """
    This is an example for a custom lineshape model created in PolliFit.
    """
    def __init__(self):
        super().__init__()
        self.type = 'TestShape'

        self._add_arg('freq', 1., False, False)
        self._add_arg('ratio', 1., False, False)
        self._add_arg('phase', 1., False, False)

    def evaluate(self, x, *args, **kwargs):  # Normalize to the maximum.
        return np.sin(2 * np.pi * args[0] * x) * np.cos(2 * np.pi * args[0] * args[1] * (x - args[2]))

    def fwhm(self):
        f = self.vals[self.p['freq']]
        r = self.vals[self.p['ratio']]
        return abs(np.max([1 / f, 1 / (f * r)]))

    def min(self):
        return -2.5 * self.fwhm()

    def max(self):
        return 2.5 * self.fwhm()
