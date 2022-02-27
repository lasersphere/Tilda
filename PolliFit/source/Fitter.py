"""
Created on 20.02.2022

@author: Patrick Mueller
"""


import numpy as np
import Physics as Ph

from FitRoutines import curve_fit


class Fitter:
    def __init__(self, model, meas, st, iso):
        self.model = model
        self.meas = meas
        self.st = st
        self.iso = iso

        self.routines = {'curve_fit', }

        self.x, self.y, self.yerr = self.gen_data()

    def get_pars(self):
        return self.model.get_pars()

    def _volt_to_freq(self, volt):
        freq = np.zeros_like(volt)
        for i, (u, meas, iso) in enumerate(zip(volt, self.meas, self.iso)):
            pm = -1 if meas.col else 1
            v = pm * Ph.relVelocity(Ph.qe * iso.q * u, iso.mass * Ph.u)
            freq[i] = Ph.relDoppler(meas.laserFreq, v) - iso.freq
        return freq

    def _transform_x(self, x):
        """
        :param x: The x data.
        :returns: The x data as defined by the Fitter object.
        """
        return self._volt_to_freq(x)

    def _transform_y(self, y):
        """
        :param y: The y data.
        :returns: The y data as defined by the Fitter object.
        """
        return y

    def _transform_yerr(self, y, yerr):
        """
        :param y: The y data.
        :param yerr: The uncertainties in the y-axis.
        :returns: The uncertainties as defined by the Fitter object.
        """
        return np.sqrt(y)

    def gen_data(self):
        """
        :returns: x, y, xerr, yerr. The combined sorted data of the given measurements and fitting options. TODO
        """
        data = [meas.getArithSpec(*self.st, function=None, eval_on=True) for meas in self.meas]
        data = np.transpose(data, axes=[1, 0, 2])
        data[0] = self._transform_x(data[0])
        data[1] = self._transform_y(data[1])
        data[2] = self._transform_yerr(data[1], data[2])
        return data

    def fit(self, routine='curve_fit'):
        """
        :param routine: The fit routine that is used.
        :returns: popt, pcov. The optimal parameters and their covariance matrix.
        :raises ValueError: If 'routine' is not in {'curve_fit', }.
        """
        if routine == 'curve_fit':
            popt, pcov = curve_fit(self.model, self.x[0], self.y[0], p0=self.model.vals, p0_fixed=self.model.fixes,
                                   sigma=self.yerr[0])  # TODO multi-file support.
        else:
            raise ValueError('The fit routine {} is not one of the available routines {}.'
                             .format(routine, self.routines))

        self.model.set_vals(popt)
        return popt, pcov
