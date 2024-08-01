"""
Created on 30.07.2024

@author: Patrick Mueller
"""

import numpy as np
import qspec.models as mod


# The names of the spectra. Includes all spectra that appear in the GUI.
SPECTRA = ['TestShape']


class TestShape(mod.Gauss):
    """
    This is an example for a custom lineshape model created in PolliFit. The fwhm, min and max functions are the same
    as in the parent class and could be omitted here.
    """
    def __init__(self):
        super().__init__()
        self.type = 'TestShape'

        self._add_arg('wobble', 1., False, False)
        self._add_arg('wiggle', 1., False, False)
        self._add_arg('phase', 0., False, False)

    def evaluate(self, x, *args, **kwargs):
        return (1 + args[1] * np.cos(2 * np.pi * args[2] / args[0] * (x + args[3]))) * super().evaluate(x, args[0])

    def fwhm(self):
        return super().fwhm()

    def min(self):
        return super().min()

    def max(self):
        return super().max()
