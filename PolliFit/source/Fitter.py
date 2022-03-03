"""
Created on 20.02.2022

@author: Patrick Mueller
"""


import os
import numpy as np
import Physics as Ph

# noinspection PyUnresolvedReferences
from FitRoutines import curve_fit
from Tools import print_colored


class Fitter:
    def __init__(self, models, meas, st, iso, config):
        """
        :param models: A list of models.
        :param meas: A list of SpecData objects.
        :param st: A list of scaler and track info.
        :param iso: A list of isotopes used for the axis conversion of each SpecData object.
        """
        self.models = models
        self.meas = meas
        self.st = st
        self.iso = iso
        self.config = config
        self.size = len(self.meas)

        self.kepco = False

        self.routines = {'curve_fit', }

        self.sizes = []
        self.x_volt, self.x, self.y, self.yerr = [], [], [], []
        self.gen_data()

    def get_pars(self, i):
        return self.models[i].get_pars()

    def set_val(self, i, j, val):
        self.models[i].vals[j] = val

    def set_fix(self, i, j, fix):
        self.models[i].fixes[j] = fix

    def set_link(self, i, j, link):
        self.models[i].links[j] = link

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
        :returns: x_volt, x, y, yerr. The combined sorted data of the given measurements and fitting options.
        """
        self.sizes = []
        self.x_volt, self.x, self.y, self.yerr = [], [], [], []
        for meas, st, iso in zip(self.meas, self.st, self.iso):
            data = meas.getArithSpec(*st, function=None, eval_on=True)
            self.sizes.append(data[0].size)
            self.x_volt.append(data[0])
            self.x.append(Ph.volt_to_rel_freq(data[0], iso.q, iso.mass, meas.laserFreq, iso.freq, meas.col))
            self.y.append(self._transform_y(data[1]))
            self.yerr.append(self._transform_yerr(data[1], data[2]))

    def get_routine(self):
        if self.config['routine'] not in self.routines:
            raise ValueError('The fit routine {} is not one of the available routines {}.'
                             .format(self.config['routine'], self.routines))
        return eval(self.config['routine'])

    def reduced_chi2(self, i=None):
        """ Calculate the reduced chi square """
        if i is None:
            return [self.reduced_chi2(i) for i in range(self.size)]
        else:
            return np.sum(self.residuals(i) ** 2 / self.yerr[i] ** 2) / self.n_dof(i)

    def n_dof(self, i=None):
        """ Calculate number of degrees of freedom """
        if i is None:
            return [self.n_dof(i) for i in range(self.size)]
        else:
            # if bounds are given instead of boolean, write False to fixed bool list.
            fixed_sum = sum(f if isinstance(f, bool) else False for f in self.models[i].fixes)
            return self.x_volt[i].size - (self.models[i].size - fixed_sum)

    def residuals(self, i=None):
        """ Calculate the residuals of the current parameter set """
        if i is None:
            return [self.residuals(i) for i in range(self.size)]
        else:
            model = self.models[i]
            y_model = model(self.x[i], *model.vals)
            return self.y[i] - y_model

    def fit_batch(self):
        """
        Fit all SpecData objects sequentially.

        :returns: popt, pcov, info.
        """
        routine = self.get_routine()
        warn = []
        errs = []
        chi2 = []
        popt, pcov = [], []
        for i, (model, x, y, yerr) in enumerate(zip(self.models, self.x, self.y, self.yerr)):
            try:
                pt, pc = routine(model, x, y, p0=model.vals, p0_fixed=model.fixes, sigma=yerr,
                                 absolute_sigma=self.config['absolute_sigma'], report=False)
                chi2.append(self.reduced_chi2(i))
                for name, val, err in zip(model.names, pt, np.sqrt(np.diag(pc))):
                    print('{}: {} +/- {}'.format(name, val, err))
                print('Red. chi2: {}'.format(chi2[-1]))
                popt.append(pt)
                pcov.append(pc)
                model.set_vals(pt)
                if np.any(np.isinf(pc)):
                    warn.append(i)
                    print_colored('WARNING', 'Failed to estimate uncertainties for file number {}.'.format(i + 1))
                else:
                    print_colored('OKGREEN', 'Successfully fitted file number {}.'.format(i + 1))
            except ValueError as e:
                print_colored('FAIL', 'Error while fitting file number {}: {}.'.format(i + 1, e))
                warn.append(i)
                errs.append(i)
                chi2.append(0.)
                popt.append(np.array(model.vals))
                pcov.append(np.zeros((popt[-1].size, popt[-1].size)))
        color = 'OKGREEN'
        if len(warn) > 0:
            color = 'WARNING'
        if len(errs) > 0:
            color = 'FAIL'
        print_colored(color, 'Fits completed, success in {} / {}.'.format(self.size - len(warn), self.size))
        info = dict(warn=warn, errs=errs, chi2=chi2)
        return popt, pcov, info

    def fit_summed(self):
        return None, None, {}

    def fit_linked(self):
        return None, None, {}

    def fit(self):
        """
        :returns: popt, pcov. The optimal parameters and their covariance matrix.
        :raises ValueError: If 'routine' is not in {'curve_fit', }.
        """
        if self.config['summed']:
            popt, pcov, info = self.fit_summed()
        elif self.config['linked']:
            popt, pcov, info = self.fit_linked()
        else:
            popt, pcov, info = self.fit_batch()
        return popt, pcov, info
