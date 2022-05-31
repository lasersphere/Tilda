"""
Created on 20.02.2022

@author: Patrick Mueller
"""

import numpy as np
import itertools as it
from PyQt5.QtCore import QObject, pyqtSignal

# noinspection PyUnresolvedReferences
from FitRoutines import curve_fit
import Physics as Ph
from Tools import print_colored, print_cov
from Models.Collection import Linked


class Xlist:  # Custom list to trick 'curve_fit' for linked fitting of files with different x-axis sizes.
    def __init__(self, x):
        self.x = x

    def __iter__(self):
        for _x in self.x:
            yield _x

    def __getitem__(self, key):
        return self.x[key]

    def __setitem__(self, key, value):
        self.x[key] = value


class Fitter(QObject):

    finished = pyqtSignal()

    def __init__(self, models, meas, st, iso, config):
        """
        :param models: A list of models.
        :param meas: A list of SpecData objects.
        :param st: A list of scaler and track info.
        :param iso: A list of isotopes used for the axis conversion of each SpecData object.
        :param config: A dictionary with information for the fit.
        """
        super().__init__()
        self.models = models
        self.meas = meas
        self.st = st
        self.iso = iso
        self.config = config
        self.size = len(self.meas)
        self.n_scaler = min(min(meas.nrScalers if isinstance(meas.nrScalers, list) else [meas.nrScalers])
                            for meas in self.meas)  # The minimum number of scalers for all files and tracks.

        self.routines = {'curve_fit', }

        self.sizes = []
        self.x_volt, self.x, self.y, self.yerr = [], [], [], []
        self.popt, self.pcov, self.info = [], [], []
        self.gen_data()

    def get_pars(self, i):
        return self.models[i].get_pars()

    def set_val(self, i, j, val):
        self.models[i].set_val(j, val)

    def set_fix(self, i, j, fix):
        self.models[i].set_fix(j, fix)

    def set_link(self, i, j, link):
        self.models[i].set_link(j, link)

    def get_meas_x_in_freq(self, i):
        meas, iso = self.meas[i], self.iso[i]
        return [[Ph.volt_to_rel_freq(x, iso.q, iso.mass, meas.laserFreq, iso.freq, meas.col)
                 for x in x_track] for x_track in meas.x]

    def _gen_yerr(self, meas, st, data, n=10000):
        """
        This should only be called for real spectra once per meas in 'gen_data'.

        :param meas: A spec_data object.
        :param st: The scaler and track info.
        :param data: The data as returned by spec_data.getArithSpec.
        :param n: The number of samples to estimate the uncertainties in the 'function' mode.
        :returns: None.
        """
        if self.config['arithmetics'] is None or 's' not in self.config['arithmetics']:
            self.yerr.append(np.sqrt(data[1]))
        else:
            if st[1] == -1:
                cts = [np.array([i for i in it.chain(*(t[scaler] for t in meas.cts))]) for scaler in st[0]]
            else:
                cts = [np.array(meas.cts[st[1]][scaler]) for scaler in st[0]]
            y_samples = {'s{}'.format(i): np.random.normal(loc=cts[i], scale=np.sqrt(cts[i]), size=(n, cts[i].size))
                         for i in range(self.n_scaler) if 's{}'.format(i) in self.config['arithmetics']}
            self.yerr.append(np.std(eval(self.config['arithmetics'], y_samples), axis=0, ddof=1))

    def gen_data(self):
        """
        :returns: x_volt, x, y, yerr. The combined sorted data of the given measurements and fitting options.
        """
        self.sizes = []
        self.x_volt, self.x, self.y, self.yerr = [], [], [], []
        for meas, st, iso in zip(self.meas, self.st, self.iso):
            data = meas.getArithSpec(*st, function=self.config['arithmetics'], eval_on=True)
            self.sizes.append(data[0].size)
            self.x_volt.append(data[0])
            self.y.append(data[1])
            if meas.seq_type == 'kepco':
                self.x.append(data[0])
                self.yerr.append(data[2])
            else:
                if 'CounterDrift' in meas.scan_dev_dict_tr_wise[0]['name']:
                    self.x.append(Ph.volt_to_rel_freq(meas.accVolt, iso.q, iso.mass, data[0], iso.freq, meas.col))
                else:
                    self.x.append(Ph.volt_to_rel_freq(data[0], iso.q, iso.mass, meas.laserFreq, iso.freq, meas.col))
                self._gen_yerr(meas, st, data)
        if all(model.type == 'Offset' for model in self.models):
            self.gen_x_cuts()

    def gen_x_cuts(self):
        for model, meas, iso, x in zip(self.models, self.meas, self.iso, self.x):
            if not model.x_cuts:
                model.gen_offset_masks(x)
                continue
            x_min = np.array([_x[0] if _x[0] <= _x[-1] else _x[-1] for _x in meas.x])
            # Array of the tracks lowest voltages
            x_max = np.array([_x[-1] if _x[0] <= _x[-1] else _x[0] for _x in meas.x])
            # Array of the tracks highest voltages
            order = np.argsort(x_min)  # Find ascending order of the lowest voltages
            x_min = x_min[order]  # apply order to the lowest voltages
            x_max = x_max[order]  # apply order to the highest voltages
            # cut at the mean between the highest voltage and the corresponding lowest voltage of the next track.
            # Iteration goes over the sorted tracks and only non-overlapping tracks get a unique offset parameter.
            x_cuts = [0.5 * float(x_max[i] + x_min[i + 1]) for i in range(len(model.x_cuts))]
            if any(x0 > x1 for x0, x1 in zip(x_cuts[:-1], x_cuts[1:])):
                print_colored('WARNING', 'Tracks are overlapping in file {}.'
                                         ' Cannot use \'offset per track\' option'.format(meas.file))
                continue
            x_cuts = [Ph.volt_to_rel_freq(_x, iso.q, iso.mass, meas.laserFreq, iso.freq, meas.col) for _x in x_cuts]
            model.set_x_cuts(x_cuts)

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
                if model.error:
                    raise ValueError(model.error)
                if model.type == 'Offset':
                    model.update_on_call = False
                    model.gen_offset_masks(x)
                    if self.config['guess_offset']:
                        model.guess_offset(x, y)
                fixed, bounds = model.fit_prepare()
                if self.config['cov_mc']:
                    print('-----------------MC-----------------')
                    y_samples = np.random.normal(y, yerr, size=(self.config['samples_mc'], y.size))
                    ps = np.array([routine(model, x, y_sample, p0=model.vals, p0_fixed=fixed, sigma=yerr,
                                           absolute_sigma=self.config['absolute_sigma'], bounds=bounds, report=False,
                                           maxfev=10000)[0] for y_sample in y_samples], dtype=float)
                    pt = np.mean(ps, axis=0)
                    pc = np.zeros((pt.size, pt.size))
                    indices = np.array([i for i, fix in enumerate(model.fixes) if not fix])
                    mask = (indices[:, None], indices)
                    pc[mask] = np.cov(ps[:, indices], rowvar=False)
                    pt = np.array(model.update_args(pt))
                else:
                    pt, pc = routine(model, x, y, p0=model.vals, p0_fixed=fixed, sigma=yerr,
                                     absolute_sigma=self.config['absolute_sigma'], bounds=bounds, report=False,
                                     maxfev=10000)
                    pt = np.array(model.update_args(pt))
                model.set_vals(pt, force=True)
                chi2.append(self.reduced_chi2(i))  # Calculate chi2 after the vals are set.
                for name, val, err in zip(model.names, pt, np.sqrt(np.diag(pc))):
                    print('{}: {} +/- {}'.format(name, val, err))
                print('Cov. Matrix:')
                print_cov(pc, normalize=True, decimals=2)
                print('Red. chi2: {}'.format(chi2[-1]))
                popt.append(pt)
                pcov.append(pc)
                if np.any(np.isinf(pc)):
                    warn.append(i)
                    print_colored('WARNING', 'Failed to estimate uncertainties for file number {}.'.format(i + 1))
                else:
                    print_colored('OKGREEN', 'Successfully fitted file number {}.'.format(i + 1))
            except (ValueError, RuntimeError) as e:
                print_colored('FAIL', 'Error while fitting file number {}: {}.'.format(i + 1, e))
                warn.append(i)
                errs.append(i)
                chi2.append(0.)
                popt.append(np.array(model.vals))
                pcov.append(np.zeros((popt[-1].size, popt[-1].size)))
            if model.type == 'Offset':
                model.update_on_call = True  # Reset the offset model to be updated on call.
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
        """
        Fit all SpecData objects simultaneously.

        :returns: popt, pcov, info.
        """
        routine = self.get_routine()
        model = Linked(self.models)
        warn = []
        errs = []
        y, yerr = np.concatenate(self.y, axis=0), np.concatenate(self.yerr, axis=0)
        try:
            if model.error:
                raise ValueError(model.error)
            for _model, _x, _y in zip(self.models, self.x, self.y):
                # Handle offsets for all linked models.
                if _model.type == 'Offset':
                    _model.update_on_call = False
                    _model.gen_offset_masks(_x)
                    if self.config['guess_offset']:
                        _model.guess_offset(_x, _y)
            model.inherit_vals()  # Inherit values of the linked models afterwards.
            fixed, bounds = model.fit_prepare()
            # curve_fit wants to convert lists and tuples to arrays -> Use custom list type.
            pt, pc = routine(model, Xlist(self.x), y, p0=model.vals, p0_fixed=fixed, sigma=yerr,
                             absolute_sigma=self.config['absolute_sigma'], bounds=bounds, report=False)
            pt = np.array(model.update_args(pt))
            model.set_vals(pt, force=True)  # Set the vals of the model (auto sets the vals of the linked models).
            chi2 = self.reduced_chi2()  # Calculate chi2 after the vals are set.
            for name, val, err in zip(model.names, pt, np.sqrt(np.diag(pc))):
                print('{}: {} +/- {}'.format(name, val, err))
            for i, _chi2 in enumerate(chi2):
                print('{} Red. chi2: {}'.format(str(i).zfill(int(np.log10(self.size))), _chi2))
            popt = [pt[_slice] for _slice in model.slices]
            pcov = [pc[_slice, _slice] for _slice in model.slices]
        except (ValueError, RuntimeError) as e:
            print_colored('FAIL', 'Error while fitting linked files: {}.'.format(e))
            warn = list(range(self.size))  # Issue warnings for all files.
            errs = list(range(self.size))
            chi2 = [0., ] * self.size
            popt = [np.array(model.vals) for model in self.models]
            pcov = [np.zeros((popt[-1].size, popt[-1].size)) for _ in self.models]
        for _model in self.models:
            if _model.type == 'Offset':
                _model.update_on_call = True  # Reset all offset models to be updated on call.
        if len(warn) == len(errs) == 0:
            print_colored('OKGREEN', 'Linked fit completed, success.')
        elif len(errs) > 0:
            print_colored('FAIL', 'Linked fit completed, failed.')
        elif len(warn) > 0:
            print_colored('WARNING', 'Linked fit completed, warning.')
        info = dict(warn=warn, errs=errs, chi2=chi2)
        return popt, pcov, info

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
        self.popt, self.pcov, self.info = popt, pcov, info
        self.finished.emit()
