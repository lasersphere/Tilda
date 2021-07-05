"""
Created on 04.03.2020

@author: pamueller based on
    [Gebert et al., Phys. Rev. Lett. 115, 053003 (2015)] and [PolliFit.KingFitter]

Multidimensional MonteCarlo-KingFitter:
The fit also considers the correlations between the modified observables
 which appear when the uncertainties of the masses are comparable or larger
 than those of the isotope shifts or changes of mean square nuclear charge radii.

Optionally, the two isotopes which define the straight can be specified
 or automatically determined for the fit to be most efficient.

A parameter alpha, which shifts the single x-axis of the fit, can be specified
 or automatically determined to decrease the correlation between the slopes and intercepts of the fit.

Note that with 32-bit Python the number of samples cannot be much larger than 1,000,000.
 Always check the number of ACCEPTED samples.
"""

import ast
import os
import sqlite3
import numpy as np
import scipy.constants as sc
import scipy.stats as st
import scipy.integrate as si
import matplotlib.pyplot as plt

import TildaTools as TiTs
import FitRoutines as Fr

m_e_u = sc.physical_constants['electron mass in u']


def import_iso_shifts(db, iso_shifts: dict):
    con = sqlite3.connect(db)
    cur = con.cursor()
    for iso, line_dict in iso_shifts.items():
        for line, val in line_dict.items():
            cur.execute('INSERT OR REPLACE INTO Combined (iso, parname, run) VALUES (?, ?, ?)',
                        (iso, 'shift', line))
            con.commit()
            cur.execute('UPDATE Combined SET config=?, final=?, '
                        'val = ?, statErr = ?, statErrForm=?, systErr = ?, systErrForm=? '
                        'WHERE iso = ? AND parname = ? AND run = ?',
                        ([], 0, val[0], val[1], 'err', val[2] if len(val) > 2 else 0., 0, iso, 'slope', line))
            con.commit()
    con.close()


def get_number(string):
    number = ''
    for s in string:
        if s.isdigit():
            number += s
    return number


def get_average(a, axis=0):
    a = np.asarray(a, dtype=float)
    av = np.average(a, axis=axis)
    dav = np.std(a, axis=axis, ddof=1)
    return av, dav


def get_weighted_average(a, da, correlated=False):
    a = np.asarray(a, dtype=float)
    da = np.asarray(da, dtype=float)
    wav, dwav = np.average(a, weights=1/(da**2), returned=True)
    if correlated:
        dcorr = dwav + np.sum([[1/(da1*da2) for da2 in da if da2 != da1] for da1 in da])
        dwav = np.sqrt(dcorr)/dwav
    else:
        dwav = np.sqrt(1/dwav)
    return wav, dwav


def product_pdf(z, pdf_1=st.norm.pdf, pdf_2=st.norm.pdf, loc_1=0., scale_1=1., loc_2=0., scale_2=1.):
    arg = np.asarray(z)
    if arg.shape == ():
        arg = np.expand_dims(arg, axis=0)
    arg = np.expand_dims(arg, axis=-1)

    def kernel(x):
        return pdf_1(x, loc=loc_1, scale=scale_1) * pdf_2(arg / x, loc=loc_2, scale=scale_2) / abs(x)

    k = 11
    x_range = np.linspace(loc_1 - 10. * scale_1, loc_1 + 10. * scale_1, 2 ** k + 1)
    dx = float(x_range[1] - x_range[0])
    x_range = np.expand_dims(x_range, axis=0)
    y = kernel(x_range)
    result = si.romb(y, dx, axis=1)
    return result


def calc_mass_factor(am, am_d, ref_am, ref_am_d):
    mu = (am + m_e_u[0]) * ref_am / (am - ref_am)
    mu_d = np.square(-(ref_am + m_e_u[0]) * ref_am / ((am - ref_am) ** 2) * am_d)
    mu_d += np.square((am + m_e_u[0]) * am / ((am - ref_am) ** 2) * ref_am_d)
    mu_d += np.square(ref_am / (am - ref_am) * m_e_u[2])
    return [mu, np.sqrt(mu_d)]


def straight(x, m, b):
    return m*x + b


def calc_f_k_factors(p_0, p_1, alpha=0.):
    f = np.array([(p_1[:, i] - p_0[:, i])/(p_1[:, -1] - p_0[:, -1]) for i in range(p_0.shape[1] - 1)]).T
    k = np.array([p_0[:, i] - f[:, i]*(p_0[:, -1] - alpha) for i in range(p_0.shape[1] - 1)]).T
    return f, k


def calc_x_from_fit(iso_shift, mu, f, k, alpha):
    return (iso_shift - k/mu)/f + alpha/mu


def x_alpha(x, alpha):
    return x*np.cos(alpha)


def y_alpha(y, alpha):
    return y*np.sin(alpha)


def y_ell(y, alpha, rho):
    return y*(rho*np.cos(alpha) + np.sqrt(1 - rho**2)*np.sin(alpha))


class MCFitter(object):
    def __init__(self, db, runs=None, ref_run=-1, subtract_electrons=0., add_ionization_energy=0., plot_folder=None,
                 popup=True):
        self.db = db
        self.runs = runs
        if self.runs is None:
            self.runs = [run[0] for run in TiTs.select_from_db(self.db, 'refRun', 'Lines', caller_name=__name__)]
        self.n_dim = len(self.runs)
        self.ref_run = ref_run
        self.subtract_electrons = subtract_electrons
        self.add_ionization_energy = add_ionization_energy
        self.n_sample = 1000000
        self.label = 'val'
        self.config = ''
        self.xlabel = 'x-axis'
        self.unit_factors = {'MHz': 1., 'GHz': 1e-3}
        self.fontsize = 12

        self.popup = popup  # Set to false to disable plt.show
        self.store_loc = None  # For now only to store the Graphs if a plot folder is specified.
        if plot_folder is not None:
            self.store_loc = plot_folder
            if not os.path.isdir(self.store_loc):
                os.mkdir(self.store_loc)

        try:
            if ref_run == -1:
                self.iso_ref = TiTs.select_from_db(self.db, 'reference', 'Lines', caller_name=__name__)[0][0]
            else:
                self.iso_ref = TiTs.select_from_db(self.db, 'reference', 'Lines',
                                                   [['refRun'], [ref_run]], caller_name=__name__)[0][0]
            self.mass_ref = TiTs.select_from_db(self.db, 'mass', 'Isotopes',
                                                [['iso'], [self.iso_ref]], caller_name=__name__)[0][0]
            self.mass_ref -= self.subtract_electrons * m_e_u[0]
            self.mass_ref += self.add_ionization_energy * sc.e / (sc.atomic_mass * sc.c ** 2)
            self.mass_ref_d = TiTs.select_from_db(self.db, 'mass_d', 'Isotopes',
                                                  [['iso'], [self.iso_ref]], caller_name=__name__)[0][0]
            self.mass_ref_d = np.sqrt(self.mass_ref_d ** 2 + (self.subtract_electrons * m_e_u[2]) ** 2)
        except Exception as e:
            print('error: %s  \n\t-> Kingfitter could not find a reference isotope from'
                  ' Lines in database or mass of this reference Isotope in Isotopes' % e)

        self.mass_number_ref = get_number(self.iso_ref)

        self.isotopes = None
        self.masses = None
        self.mass_factors = None
        self.isotope_shifts = None
        self.iso_index = None
        self.order_best = None
        self.order_plot = None

        self.mass_numbers = None
        self.mass_factors_array = None
        self.rand_m_factors = None

        self.mean0 = None
        self.std0 = None
        self.cov0 = None
        self.mean = None
        self.std = None
        self.cov = None

        self.alpha = 0.
        self.alpha_k = 0.
        self.alpha_step = 1.

        self.correlation_best = 1.

        self.fixed_axis = 0

        self.accepted = None
        self.m_factors_accepted = None
        self.f_accepted = None
        self.k_accepted = None
        self.p_accepted = None
        self.p_x_accepted = None

        self.f = None
        self.f_d = None
        self.k = None
        self.k_d = None
        self.p = None
        self.p_d = None
        self.p_x = None
        self.p_x_d = None

        self.p_x_calc_accepted = None

        self.p_x_calc = None
        self.p_x_calc_d = None

    def write_results_to_db(self, mark_as_mc):
        mc0, mc1 = '', ''
        if mark_as_mc:
            mc0, mc1 = 'MC(', ')'
        mc_runs = mc0 + str(self.runs).replace("'", "") + mc1
        
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        for i in range(self.n_dim - 1):
            mc_run = mc0 + self.runs[i] + mc1
            cur.execute('INSERT OR IGNORE INTO Runs (run, lineVar, isoVar, scaler, track, '
                        'softwGates, softwGateWidth, softwGateDelayList) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                        (mc_run, self.runs[i], str([0]), -1, None, str([]), 0, str([])))

            cur.execute('INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)',
                        (self.label, 'intercept', mc_run))
            con.commit()
            cur.execute('UPDATE Combined SET val = ?, statErr = ?, systErr = ?, config = ? '
                        'WHERE iso = ? AND parname = ? AND run = ?',
                        (self.k[i], 0., self.k_d[i], self.config, self.label, 'intercept', mc_run))
            con.commit()

            cur.execute('INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)',
                        (self.label, 'slope', mc_run))
            con.commit()
            cur.execute('UPDATE Combined SET val = ?, statErr = ?, systErr = ?, config=? '
                        'WHERE iso = ? AND parname = ? AND run = ?',
                        (self.f[i], 0., self.f_d[i], self.config, self.label, 'slope', mc_run))
            con.commit()

            if not mark_as_mc:
                cur.execute('INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)',
                            (self.label, 'alpha', self.runs[i]))
                con.commit()
                cur.execute('UPDATE Combined SET val = ?, config=? WHERE iso = ? AND parname = ? AND run = ?',
                            (self.alpha_k, self.config, self.label, 'alpha', self.runs[i]))
                con.commit()

        cur.execute('INSERT OR IGNORE INTO Runs (run, lineVar, isoVar, scaler, track, '
                    'softwGates, softwGateWidth, softwGateDelayList) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                    (mc_runs, str(self.runs).replace("'", ""), str([0]), -1, None, str([]), 0, str([])))
        cur.execute('INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)',
                    (self.label, 'alpha', mc_runs))
        con.commit()
        cur.execute('UPDATE Combined SET val = ?, config=? WHERE iso = ? AND parname = ? AND run = ?',
                    (self.alpha, self.config, self.label, 'alpha', mc_runs))
        con.commit()
        con.close()
        return mc_runs

    def apply_alpha(self, alpha):
        self.alpha = alpha

    def get_axis_point_on_straight(self, p_i_fixed_axis, f, k):
        p_ret = []
        if self.fixed_axis == self.mean.shape[1] - 1:
            t = p_i_fixed_axis
        else:
            t = (p_i_fixed_axis - k[:, self.fixed_axis]) / f[:, self.fixed_axis]
        for axis in range(self.mean.shape[1] - 1):
            if axis == self.fixed_axis:
                p_ret.append(p_i_fixed_axis)
            else:
                p_ret.append(f[:, axis] * t + k[:, axis])
        p_ret.append(t)
        return np.array(p_ret).T

    def get_collinear_points(self, n_sample, print_info=True):
        self.rand_m_factors = np.array([st.norm.rvs(loc=self.mass_factors[iso][0], scale=self.mass_factors[iso][1],
                                                    size=n_sample) for iso in self.isotopes])
        p_0 = st.multivariate_normal.rvs(mean=self.mean0[0, :], cov=self.cov0[0, :], size=n_sample)
        p_1 = st.multivariate_normal.rvs(mean=self.mean0[-1, :], cov=self.cov0[-1, :], size=n_sample)
        p_0 *= np.expand_dims(self.rand_m_factors[0, :], axis=1)
        p_1 *= np.expand_dims(self.rand_m_factors[-1, :], axis=1)
        f, k = calc_f_k_factors(p_0, p_1, self.alpha)
        p_i = [p_0, ]
        p_i_fixed_axis = []
        for i in range(1, self.mean0.shape[0] - 1):
            p_i_fixed_axis.append(st.norm.rvs(loc=self.mean0[i, self.fixed_axis],
                                              scale=self.std0[i, self.fixed_axis], size=n_sample)
                                  * self.rand_m_factors[i, :])
            p_i.append(self.get_axis_point_on_straight(p_i_fixed_axis[-1], f, k - f * self.alpha))
        p_i.append(p_1)
        p_i = np.array(p_i)
        
        # isotope = '44_Ca'
        # index = self.iso_index[isotope]
        # if index not in [0, self.isotopes.size - 1]:
        #     x = np.sort(p_i_fixed_axis[index-1])
        #     p_pdf = product_pdf(x, loc_1=self.mass_factors[isotope][0], scale_1=self.mass_factors[isotope][1],
        #                         loc_2=self.mean0[index, self.fixed_axis], scale_2=self.std0[index, self.fixed_axis])
        #     n_pdf = st.norm.pdf(x, loc=self.mass_factors[isotope][0]*self.mean0[index, self.fixed_axis],
        #                         scale=np.sqrt((self.mass_factors[isotope][0]*self.std0[index, self.fixed_axis])**2
        #                                       + (self.mass_factors[isotope][1]
        #                                          * self.mean0[index, self.fixed_axis])**2))
        #     n2_pdf = st.norm.pdf(x, loc=self.mean[index, self.fixed_axis], scale=self.std[index, self.fixed_axis])
        #     plt.plot(x, p_pdf, label='product dist.')
        #     plt.plot(x, n_pdf, label='normal dist.')
        #     plt.plot(x, n2_pdf, label='normal2 dist.')
        #     plt.legend()
        #     plt.show()
        
        u = np.random.random(size=n_sample)
        # u_criteria = np.prod([st.multivariate_normal.pdf(p_i[i, :, :], mean=self.mean[i, :], cov=self.cov[i, :, :])
        #                       / product_pdf(p_i_fixed_axis[i-1], loc_1=self.mass_factors[self.isotopes[i]][0],
        #                                     scale_1=self.mass_factors[self.isotopes[i]][1],
        #                                     loc_2=self.mean0[i, self.fixed_axis],
        #                                     scale_2=self.std0[i, self.fixed_axis])
        #                       for i in range(1, self.mean0.shape[0] - 1)], axis=0)
        u_criteria = np.prod([st.multivariate_normal.pdf(p_i[i, :, :], mean=self.mean[i, :], cov=self.cov[i, :, :])
                              / st.norm.pdf(p_i_fixed_axis[i-1], loc=self.mean[i, self.fixed_axis],
                                            scale=self.std[i, self.fixed_axis])
                              for i in range(1, self.mean0.shape[0] - 1)], axis=0)
        u_criteria /= np.max(u_criteria)
        accepted = np.argwhere(u < u_criteria).T[0]
        if print_info:
            print('Accepted samples (Absolute, Relative): '
                  + str(accepted.size) + ', ' + str(np.around(100*accepted.size/n_sample, decimals=2)) + ' %')
        return f, k, p_i, accepted

    def switch_order(self, iso_order):
        order = np.array([self.iso_index[iso] for iso in iso_order])
        self.isotopes = self.isotopes[order]
        self.iso_index = {iso: i for i, iso in enumerate(self.isotopes)}
        self.mass_numbers = self.mass_numbers[order]
        self.mass_factors_array = self.mass_factors_array[order]
        if self.rand_m_factors is not None:
            self.rand_m_factors = self.rand_m_factors[order, :]
        self.mean0 = self.mean0[order, :]
        self.std0 = self.std0[order, :]
        self.cov0 = self.cov0[order, :]
        self.mean = self.mean[order, :]
        self.std = self.std[order, :]
        self.cov = self.cov[order, :, :]
        return order

    def find_best_order(self, n_sample=100000):
        print('\nSearching most efficient isotope order and fixed axis to constrain straight lines ...')
        border_isotopes = [[iso_0, iso_1] for i, iso_0 in enumerate(self.isotopes) for iso_1 in self.isotopes[(i+1):]]
        iso_orders = [[iso_b[0], ] + [iso for iso in self.isotopes if iso not in iso_b] + [iso_b[1], ]
                      for i, iso_b in enumerate(border_isotopes)]
        order_best = iso_orders[0]
        axis_best = self.fixed_axis
        acceptance_best = 0
        for iso_order in iso_orders:
            self.switch_order(iso_order)
            for i in range(self.n_dim):
                print('\nIsotope order: ' + str(iso_order) + ', fixed axis: ' + str(i))
                self.fixed_axis = i
                _, _, _, accepted = self.get_collinear_points(n_sample)
                acceptance = accepted.size
                if acceptance > acceptance_best:
                    acceptance_best = acceptance
                    order_best = iso_order
                    axis_best = i
        self.order_best = order_best
        self.fixed_axis = axis_best
        print('\nBest isotope order: ' + str(order_best))
        print('Best fixed axis: ' + str(axis_best))
        print('Best acceptance: ' + str(np.around(100*acceptance_best/n_sample, decimals=2)) + ' %')

    def find_fixed_axis(self, iso_order, n_sample=100000):
        print('\nSearching most efficient fixed axis to constrain straight lines ...')
        self.order_best = iso_order
        self.switch_order(iso_order)
        acceptance_best = 0
        axis_best = self.fixed_axis
        for i in range(self.n_dim):
            print('\nIsotope order: ' + str(self.order_best) + ', fixed axis: ' + str(i))
            self.fixed_axis = i
            _, _, _, accepted = self.get_collinear_points(n_sample)
            acceptance = accepted.size
            if acceptance > acceptance_best:
                acceptance_best = acceptance
                axis_best = i
        self.fixed_axis = axis_best
        print('\nChosen isotope order: ' + str(self.order_best))
        print('Best fixed axis: ' + str(axis_best))
        print('Best acceptance: ' + str(np.around(100*acceptance_best/n_sample, decimals=2)) + ' %')

    def correlation(self, alpha_opt, n_sample, norm):
        self.apply_alpha(alpha_opt)
        f_alpha, k_alpha, _, accepted_alpha = self.get_collinear_points(n_sample)
        f_accepted = f_alpha[accepted_alpha, :]
        k_accepted = k_alpha[accepted_alpha, :]
        corrcoef = np.corrcoef(f_accepted.T, y=k_accepted.T, ddof=1)
        return np.sum(np.abs(corrcoef[(self.n_dim - 1):, :(self.n_dim - 1)]))/norm

    def find_best_alpha(self, alpha, n_sample, show=True):
        print('\nSearching best alpha to reduce the correlation between F and K ...')
        norm = (self.n_dim - 1)**2

        print('\nalpha: ' + str(alpha))
        alpha_best = alpha
        alpha_list = [alpha, ]
        alpha_found = False
        correlation_best = self.correlation(alpha_best, n_sample, norm)
        corr_list = [correlation_best, ]
        print('mean correlation: ' + str(np.around(correlation_best, decimals=3)))
        while not alpha_found:
            print('\nalpha: ' + str(self.alpha + self.alpha_step))
            corr = self.correlation(self.alpha + self.alpha_step, n_sample, norm)
            print('Mean correlation: ' + str(np.around(corr, decimals=3)))
            if corr < correlation_best:
                correlation_best = corr
                corr_list.append(corr)
                alpha_best = self.alpha
                alpha_list.append(self.alpha)
            else:
                if self.alpha_step > 0:
                    self.alpha_step = -self.alpha_step
                    self.apply_alpha(alpha_best)
                else:
                    alpha_found = True
        self.alpha_step = np.abs(self.alpha_step)

        self.correlation_best = correlation_best

        print('\nBest alpha: ' + str(alpha_best))
        print('Best mean correlation (F, K): ' + str(np.around(correlation_best, decimals=3)))
        if show:
            plt.xlabel(r'$\alpha_\mathrm{best}')
            plt.ylabel('mean correlation (F, K)')
            plt.plot(alpha_list, corr_list)
            if self.store_loc is not None:
                f_name = 'MC_king_fit_alpha_%d_find_alpha' % self.alpha
                plt.savefig(os.path.join(self.store_loc, f_name + '.pdf'))
                plt.savefig(os.path.join(self.store_loc, f_name + '.png'))
            if self.popup:
                plt.show()
            plt.gcf().clear()
        return alpha_best

    def plot_results(self, mod_iso_shift_scale):
        pass

    def king_fit(self, n_sample=1000000, isotope_order=None, alpha=0., find_best_alpha=False, results_to_db=True,
                 mark_as_mc=True, show=True, mod_iso_shift_scale='GHz'):
        if isotope_order is None:
            self.find_best_order()
        else:
            self.find_fixed_axis(isotope_order)
        self.switch_order(self.order_best)
        if find_best_alpha:
            self.alpha_k = self.find_best_alpha(alpha, n_sample)
        else:
            self.alpha_k = alpha
            self.correlation_best = self.correlation(alpha, n_sample, (self.n_dim - 1)**2)

        self.apply_alpha(self.alpha_k)

        print('\nPerforming King-Fit ...')

        f, k, p_i, accepted = self.get_collinear_points(n_sample)
        order = self.switch_order(self.order_plot)
        p_i = p_i[order, :, :]

        self.accepted = accepted
        self.m_factors_accepted = self.rand_m_factors[:, accepted]
        self.f_accepted = f[accepted, :]
        self.k_accepted = k[accepted, :]
        self.p_accepted = p_i[:, accepted, :]
        self.p_x_accepted = self.p_accepted[:, :, -1] / self.m_factors_accepted

        self.f = np.average(self.f_accepted, axis=0)
        self.f_d = np.std(self.f_accepted, axis=0, ddof=1)
        self.k = np.average(self.k_accepted, axis=0)
        self.k_d = np.std(self.k_accepted, axis=0, ddof=1)
        self.p = np.average(self.p_accepted, axis=1)
        self.p_d = np.std(self.p_accepted, axis=1, ddof=1)
        self.p_x = np.average(self.p_x_accepted, axis=1)
        self.p_x_d = np.std(self.p_x_accepted, axis=1, ddof=1)

        if results_to_db:
            self.write_results_to_db(mark_as_mc)

        if show:
            self.plot_results(mod_iso_shift_scale)

    def correlation_coefficient(self):
        corr_param = np.corrcoef(self.f_accepted.T, y=self.k_accepted.T, ddof=1)
        corr_linear = np.corrcoef(self.mean[:, -1].T, self.mean[:, :-1].T)
        print('\nParameter correlation:\n', corr_param)
        print('\nLinearity:\n', corr_linear)
        return corr_param

    def calc_x_from_fit(self, run=-1, n_sample=1000000):
        correlations = {}
        x_results = {}
        x_ret = {}

        if run == -1:
            f_data = TiTs.select_from_db(self.db, 'iso, run, config, val, statErr, systErr', 'Combined',
                                         [['parname'], ['slope']], caller_name=__name__)
        else:
            f_data = TiTs.select_from_db(self.db, 'iso, run, config, val, statErr, systErr', 'Combined',
                                         [['parname', 'run'], ['slope', run]], caller_name=__name__)

        for data in f_data:
            (iso, run_f, config, f, f_stat, f_sys) = data
            if iso != self.label or run_f == '-1' or config[config.find('}'):] != self.config[config.find('}'):]:
                continue
            x_results[run_f] = {}
            (k, k_stat, k_sys) = TiTs.select_from_db(self.db, 'val, statErr, systErr', 'Combined',
                                                     [['iso', 'parname', 'run'], [iso, 'intercept', run_f]],
                                                     caller_name=__name__)[0]
            alpha = TiTs.select_from_db(self.db, 'val', 'Combined',
                                        [['iso', 'parname', 'run'], [iso, 'alpha', run_f]], caller_name=__name__)[0][0]
            f_dist = st.norm.rvs(loc=f, scale=np.sqrt(f_stat**2 + f_sys**2), size=n_sample)
            k_dist = st.norm.rvs(loc=k, scale=np.sqrt(k_stat**2 + k_sys**2), size=n_sample)
            iso_shift_data = TiTs.select_from_db(self.db, 'iso, val, statErr, systErr', 'Combined',
                                                 [['parname', 'run'], ['shift', run_f]], caller_name=__name__)
            for iso_data in iso_shift_data:
                (iso, iso_shift, iso_shift_stat, iso_shift_sys) = iso_data
                if iso == self.iso_ref:
                    continue
                x_ret[iso] = (0., 0.)
                mass = TiTs.select_from_db(self.db, 'mass, mass_d', 'Isotopes',
                                           [['iso'], [iso]], caller_name=__name__)[0]
                mass = [mass[0] - self.subtract_electrons * m_e_u[0],
                        np.sqrt(mass[1]**2 + (self.subtract_electrons * m_e_u[2])**2)]
                mass_factor = calc_mass_factor(mass[0], mass[1], self.mass_ref, self.mass_ref_d)
                mass_factor_dist = st.norm.rvs(loc=mass_factor[0], scale=mass_factor[1], size=n_sample)
                iso_shift_dist = st.norm.rvs(loc=iso_shift, scale=np.sqrt(iso_shift_stat**2 + iso_shift_sys**2),
                                             size=n_sample)
                x_dist = calc_x_from_fit(iso_shift_dist, mass_factor_dist, f_dist, k_dist, alpha)
                if iso not in correlations.keys():
                    correlations[iso] = [x_dist, ]
                else:
                    correlations[iso].append(x_dist)
                x_results[run_f][iso] = get_average(x_dist)
        x_ret = {iso: get_weighted_average([iso_dict[iso][0] for run_i, iso_dict
                                            in x_results.items() if iso in iso_dict.keys()],
                                           [iso_dict[iso][1] for run_i, iso_dict
                                            in x_results.items() if iso in iso_dict.keys()], correlated=True)
                 for iso in x_ret.keys()}

        isotopes = np.array([iso for iso in x_ret.keys()])
        a = np.array([get_number(iso) for iso in isotopes]).astype(int)
        a = np.insert(a, 0, int(self.mass_number_ref))
        y = np.array([x_ret[iso][0] for iso in isotopes])
        yerr = np.array([x_ret[iso][1] for iso in isotopes])
        y = np.insert(y, 0, 0.)
        yerr = np.insert(yerr, 0, 0.)
        a_order = a.argsort()
        a = a[a_order]
        y, yerr = y[a_order], yerr[a_order]

        plt.errorbar(a, y, yerr=yerr, fmt='ro', linestyle='-', label='Data')
        plt.xticks(fontsize=self.fontsize)
        plt.yticks(fontsize=self.fontsize)
        plt.xlabel(r'mass number $A$', fontsize=self.fontsize)
        plt.ylabel(self.xlabel, fontsize=self.fontsize)
        plt.legend(numpoints=1, loc='best', fontsize=self.fontsize)
        plt.margins(0.1)
        plt.gcf().set_facecolor('w')
        plt.tight_layout()
        if self.store_loc is not None:
            f_name = 'MC_king_fit_alpha_%d_calc_x' % self.alpha
            plt.savefig(os.path.join(self.store_loc, f_name + '.pdf'))
            plt.savefig(os.path.join(self.store_loc, f_name + '.png'))
        if self.popup:
            plt.show()
        plt.gcf().clear()

        return x_ret


class KingFitter(MCFitter):
    def __init__(self, db, runs=None, ref_run=-1, litvals=None, subtract_electrons=0., add_ionization_energy=0.,
                 plot_folder=None, popup=True):
        super().__init__(db, runs=runs, ref_run=ref_run, subtract_electrons=subtract_electrons,
                         add_ionization_energy=add_ionization_energy, plot_folder=plot_folder, popup=popup)
        self.n_dim = len(self.runs) + 1
        self.litvals = litvals
        if self.litvals is None:
            if self.ref_run == -1:
                config = TiTs.select_from_db(self.db, 'config', 'Combined',
                                             [['parname'], ['slope']], caller_name=__name__)[0][0]
            else:
                config = TiTs.select_from_db(self.db, 'config', 'Combined',
                                             [['parname', 'run'], ['slope', self.ref_run]], caller_name=__name__)[0][0]
            self.litvals = ast.literal_eval(config[:(config.find('}') + 1)])
        self.label = 'kingVal'
        self.config = str(self.litvals) + str(', incl_projected = ') + str(False)
        self.xlabel = r'$\delta\langle r^2\rangle^{' + self.mass_number_ref + r', A}\quad(\mathrm{fm}^2)$'
        self.isotopes = []
        self.delta_r = []
        for iso, delta_r in self.litvals.items():
            self.isotopes.append(iso)
            self.delta_r.append(delta_r[0])
        self.isotopes = np.array(self.isotopes)
        self.delta_r = np.array(self.delta_r)

        self.masses = {iso: [TiTs.select_from_db(self.db, 'mass', 'Isotopes', [['iso'], [iso]],
                                                 caller_name=__name__)[0][0]
                             - self.subtract_electrons * m_e_u[0]
                             + self.add_ionization_energy * sc.e / (sc.atomic_mass * sc.c ** 2),
                             np.sqrt(TiTs.select_from_db(self.db, 'mass_d', 'Isotopes', [['iso'], [iso]],
                                                         caller_name=__name__)[0][0] ** 2
                                     + (self.subtract_electrons * m_e_u[2]) ** 2)]
                       for iso in self.isotopes}

        self.mass_factors = {iso: calc_mass_factor(self.masses[iso][0], self.masses[iso][1],
                                                   self.mass_ref, self.mass_ref_d)
                             for iso in self.isotopes if iso != self.iso_ref}

        self.isotope_shifts = {iso: {run: TiTs.select_from_db(self.db, 'val, statErr, systErr', 'Combined',
                                                              [['iso', 'parname', 'run'], [iso, 'shift', run]],
                                                              caller_name=__name__)[0]
                                     for run in self.runs}
                               for iso in self.isotopes if iso != self.iso_ref}

        self.mod_delta_r = np.array([self.delta_r[i]*self.mass_factors[iso][0] for i, iso in enumerate(self.isotopes)])
        order = np.argsort(self.mod_delta_r)
        self.mod_delta_r = self.mod_delta_r[order]
        self.isotopes = self.isotopes[order]
        self.iso_index = {iso: i for i, iso in enumerate(self.isotopes)}
        self.order_best = self.isotopes
        self.order_plot = self.isotopes

        self.mass_numbers = np.array([get_number(iso) for iso in self.isotopes])
        self.mass_factors_array = np.array([self.mass_factors[iso][0] for iso in self.isotopes])

        self.mean0 = np.array([[self.isotope_shifts[iso][run][0] for run in self.runs]
                               + [self.litvals[iso][0]] for iso in self.isotopes])
        self.std0 = np.array([[self.isotope_shifts[iso][run][1] for run in self.runs]
                              + [self.litvals[iso][1]] for iso in self.isotopes])
        self.cov0 = self.std0 ** 2

        self.mean = self.mean0 * np.expand_dims(self.mass_factors_array, axis=1)

        rand_m_factors = np.array([st.norm.rvs(loc=self.mass_factors[iso][0], scale=self.mass_factors[iso][1],
                                               size=self.n_sample) for iso in self.isotopes])
        p_temp = [st.multivariate_normal.rvs(mean=self.mean0[i, :], cov=self.cov0[i, :], size=self.n_sample)
                  for i in range(len(self.isotopes))]
        self.cov = np.array([np.cov(np.expand_dims(rand_m_factors[i, :], axis=0) * p.T, ddof=1)
                             for i, p in enumerate(p_temp)])
        self.std = np.array([np.sqrt(np.diag(cov)) for cov in self.cov])
        self.fixed_axis = np.argmin(np.sum(np.abs(self.std/self.mean), axis=0))

    def write_results_to_db(self, mark_as_mc):
        mc_runs = super().write_results_to_db(mark_as_mc)
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        for i, iso in enumerate(self.isotopes):
            cur.execute('INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)',
                        (iso, 'delta_r_square', mc_runs))
            con.commit()
            cur.execute('UPDATE Combined SET val = ?, statErr = ?, systErr = ? '
                        'WHERE iso = ? AND parname = ? AND run = ?',
                        (self.p_x[i], 0., self.p_x_d[i], iso, 'delta_r_square', mc_runs))
            con.commit()
        con.close()

    def plot_results(self, mod_iso_shift_scale):
        if mod_iso_shift_scale in self.unit_factors.keys():
            y_s = self.unit_factors[mod_iso_shift_scale]
        else:
            mod_iso_shift_scale, y_s = 'GHz', self.unit_factors['GHz']
        phi = np.arange(0., 2*np.pi, 0.001)
        c_ab = self.correlation_coefficient()
        for i in range(self.n_dim - 1):
            x_cont = np.linspace(self.p[0, -1] - 0.1*(self.p[-1, -1] - self.p[0, -1]),
                                 self.p[-1, -1] + 0.1*(self.p[-1, -1] - self.p[0, -1]), 1001)
            x_cont -= self.alpha_k
            plt.plot(x_cont, straight(x_cont, self.f[i], self.k[i])*y_s
                     - Fr.straight_std(x_cont, self.k_d[i], self.f_d[i], c_ab[i, (self.n_dim - 1) + i])*y_s, 'b-',
                     label='Slope uncertainty')
            plt.plot(x_cont, straight(x_cont, self.f[i], self.k[i])*y_s
                     + Fr.straight_std(x_cont, self.k_d[i], self.f_d[i], c_ab[i, (self.n_dim - 1) + i])*y_s, 'b-')
            plt.plot(x_cont, straight(x_cont, self.f[i], self.k[i])*y_s, 'r-', label='King fit')
            plt.plot(self.mean[:, -1] - self.alpha_k, self.mean[:, i]*y_s, 'ko', label='Data')
            print('\ndr²-{} correlation of:'.format(self.runs[i]))
            for j, iso in enumerate(self.isotopes):
                plt.text(self.p[j, -1] - self.alpha_k, (self.p[j, i] + np.max(self.std[:, i])*3)*y_s, iso)

                shift_x = st.norm.rvs(loc=self.mean0[j, -1], scale=self.std0[j, -1], size=self.n_sample)
                shift = st.norm.rvs(loc=self.mean0[j, i], scale=self.std0[j, i], size=self.n_sample) * y_s
                m_fac = st.norm.rvs(loc=self.mass_factors[iso][0], scale=self.mass_factors[iso][1], size=self.n_sample)
                rho = np.corrcoef(shift_x * m_fac, shift * m_fac, ddof=1)[0, -1]
                print('{}: {}'.format(iso, rho))

                plt.plot(self.mean[j, -1] - self.alpha_k + x_alpha(self.std[j, -1], phi),
                         self.mean[j, i] * y_s + y_ell(self.std[j, i] * y_s, phi, rho), 'g-')
                plt.plot(self.mean[j, -1] - self.alpha_k + x_alpha(self.std[j, -1] * 2, phi),
                         self.mean[j, i] * y_s + y_ell(self.std[j, i] * y_s * 2, phi, rho), 'g-')

            plt.xticks(fontsize=self.fontsize)
            plt.yticks(fontsize=self.fontsize)
            plt.xlabel(r'$\mu\ \delta\langle r^2\rangle^{' + self.mass_number_ref
                       + r', A} - \alpha\quad(\mathrm{u}\ \mathrm{fm}^2)$', fontsize=self.fontsize)
            plt.ylabel(r'$\mu\ \delta\nu_\mathrm{' + self.runs[i] + r'}^{' + self.mass_number_ref
                       + r', A}\quad(\mathrm{u}\ \mathrm{' + mod_iso_shift_scale + r'})$',
                       fontsize=self.fontsize)
            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend(handles[::-1], labels[::-1], numpoints=1, loc='best', fontsize=self.fontsize)
            plt.margins(0.1)
            plt.gcf().set_facecolor('w')
            plt.tight_layout()

            if self.store_loc is not None:
                f_name = 'MC_king_fit_alpha_%d' % self.alpha
                plt.savefig(os.path.join(self.store_loc, f_name + '.pdf'))
                plt.savefig(os.path.join(self.store_loc, f_name + '.png'))

                # store results to file when showing the plot
                # (.fit() is called very often without showing the plot when optimitzing self.c(=alpha))
                with open(os.path.join(self.store_loc, f_name + '_fit_points' + '.txt'), 'w') as king_f:
                    king_f.write('# intercept: %.3f +/- %.3f\n' % (self.k[i], self.k_d[i]))
                    king_f.write('# slope: %.3f +/- %.3f\n' % (self.f[i], self.f_d[i]))
                    king_f.write('# correlation: %.3f\n' % self.correlation_best)
                    king_f.write('# %s\t%s\t%s\t%s\t%s \n' % ('x', 'x_err', 'y', 'y_stat_err', 'y_err_total'))
                    for num, x in enumerate(self.mean[:, -1] - self.alpha_k):
                        king_f.write('%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n'
                                     % (x, self.std[num, -1], self.mean[num, i] * y_s,
                                        self.std[num, i] * y_s, self.std[num, i] * y_s))
            if self.popup:
                plt.show()
            plt.gcf().clear()

        a = self.mass_numbers.astype(int)
        a = np.insert(a, 0, int(self.mass_number_ref))
        a_order = a.argsort()
        y = np.insert(self.p_x, 0, 0.)
        yerr = np.insert(self.p_x_d, 0, 0.)
        plt.errorbar(a[a_order], y[a_order], yerr=yerr[a_order], fmt='ro', linestyle='-', label='Data')

        plt.xticks(fontsize=self.fontsize)
        plt.yticks(fontsize=self.fontsize)
        plt.xlabel(r'mass number $A$', fontsize=self.fontsize)
        plt.ylabel(r'$\delta\langle r^2\rangle^{' + self.mass_number_ref + r', A}\quad(\mathrm{fm}^2)$',
                   fontsize=self.fontsize)
        plt.legend(numpoints=1, loc='best', fontsize=self.fontsize)
        plt.margins(0.1)
        plt.gcf().set_facecolor('w')
        plt.tight_layout()

        if self.store_loc is not None:
            f_name = 'MC_king_fit_alpha_%d_radii' % self.alpha
            plt.savefig(os.path.join(self.store_loc, f_name + '.pdf'))
            plt.savefig(os.path.join(self.store_loc, f_name + '.png'))
        if self.popup:
            plt.show()
        plt.gcf().clear()


class KingFitterRatio(MCFitter):
    def __init__(self, db, runs=None, ref_run=-1, isotopes=None, x_run=-1, subtract_electrons=0.,
                 add_ionization_energy=0.):
        super().__init__(db, runs=runs, ref_run=ref_run, subtract_electrons=subtract_electrons,
                         add_ionization_energy=add_ionization_energy)
        self.alpha_step = 2000.
        self.isotopes = isotopes
        if self.isotopes is None:
            if self.ref_run == -1:
                self.isotopes = [iso[0] for iso in TiTs.select_from_db(
                    self.db, 'iso', 'Combined', [['parname', 'run'], ['shift', self.runs[0]]], caller_name=__name__)]
            else:
                self.isotopes = [iso[0] for iso in TiTs.select_from_db(
                    self.db, 'iso', 'Combined', [['parname', 'run'], ['shift', ref_run]], caller_name=__name__)]
        self.label = 'kingRatioVal'
        i = x_run if isinstance(x_run, int) else self.runs.index(x_run)
        self.x_run = self.runs.pop(i)
        self.runs.append(self.x_run)

        self.masses = {iso: [TiTs.select_from_db(self.db, 'mass', 'Isotopes', [['iso'], [iso]],
                                                 caller_name=__name__)[0][0]
                             - self.subtract_electrons * m_e_u[0]
                             + self.add_ionization_energy * sc.e / (sc.atomic_mass * sc.c ** 2),
                             np.sqrt(TiTs.select_from_db(self.db, 'mass_d', 'Isotopes', [['iso'], [iso]],
                                                         caller_name=__name__)[0][0] ** 2
                                     + (self.subtract_electrons * m_e_u[2]) ** 2)]
                       for iso in self.isotopes}

        self.mass_factors = {iso: calc_mass_factor(self.masses[iso][0], self.masses[iso][1],
                                                   self.mass_ref, self.mass_ref_d)
                             for iso in self.isotopes if iso != self.iso_ref}

        self.isotope_shifts = {iso: {run: TiTs.select_from_db(self.db, 'val, statErr, systErr', 'Combined',
                                                              [['iso', 'parname', 'run'], [iso, 'shift', run]],
                                                              caller_name=__name__)[0]
                                     for run in self.runs}
                               for iso in self.isotopes if iso != self.iso_ref}
        
        self.config = {iso: (run_dict[self.x_run][0], np.sqrt(run_dict[self.x_run][1]**2 + run_dict[self.x_run][2]**2))
                       for iso, run_dict in self.isotope_shifts.items()}
        self.config = str(self.config) + ', x_run=' + self.x_run
        self.xlabel = r'$\delta\nu_\mathrm{' + self.x_run + r'}^{' + self.mass_number_ref + r', A}\quad(\mathrm{MHz})$'

        self.mod_isotope_shift_x = np.array([self.isotope_shifts[iso][self.x_run][0]*self.mass_factors[iso][0]
                                             for iso in self.isotopes])
        order = np.argsort(self.mod_isotope_shift_x)
        self.mod_isotope_shift_x = self.mod_isotope_shift_x[order]
        self.isotopes = np.array(self.isotopes)[order]
        self.iso_index = {iso: i for i, iso in enumerate(self.isotopes)}
        self.order_best = self.isotopes
        self.order_plot = self.isotopes

        self.mass_numbers = np.array([get_number(iso) for iso in self.isotopes])
        self.mass_factors_array = np.array([self.mass_factors[iso][0] for iso in self.isotopes])

        self.mean0 = np.array([[self.isotope_shifts[iso][run][0] for run in self.runs] for iso in self.isotopes])
        self.std0 = np.array([[np.sqrt(self.isotope_shifts[iso][run][1]**2 + self.isotope_shifts[iso][run][2]**2)
                               for run in self.runs] for iso in self.isotopes])
        self.cov0 = self.std0 ** 2

        self.mean = self.mean0*np.expand_dims(self.mass_factors_array, axis=1)

        rand_m_factors = np.array([st.norm.rvs(loc=self.mass_factors[iso][0], scale=self.mass_factors[iso][1],
                                               size=self.n_sample) for iso in self.isotopes])
        p_temp = [st.multivariate_normal.rvs(mean=self.mean0[i, :], cov=self.cov0[i, :], size=self.n_sample)
                  for i in range(len(self.isotopes))]
        self.cov = np.array([np.cov(np.expand_dims(rand_m_factors[i, :], axis=0) * p.T, ddof=1)
                             for i, p in enumerate(p_temp)])
        self.std = np.array([np.sqrt(np.diag(cov)) for cov in self.cov])
        self.fixed_axis = np.argmin(np.sum(np.abs(self.std / self.mean), axis=0))

    def write_results_to_db(self, mark_as_mc):
        mc_runs = super().write_results_to_db(mark_as_mc)
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        for i, iso in enumerate(self.isotopes):
            cur.execute('INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)',
                        (iso, 'new_{}_shift'.format(self.x_run), mc_runs))
            con.commit()
            cur.execute('UPDATE Combined SET val = ?, statErr = ?, systErr = ? '
                        'WHERE iso = ? AND parname = ? AND run = ?',
                        (self.p_x[i], 0., self.p_x_d[i], iso, 'new_{}_shift'.format(self.x_run), mc_runs))
            con.commit()
        con.close()

    def plot_results(self, mod_iso_shift_scale, save_iso=None):
        if save_iso is None:
            save_iso = []
        if mod_iso_shift_scale in self.unit_factors.keys():
            y_s = self.unit_factors[mod_iso_shift_scale]
        else:
            mod_iso_shift_scale, y_s = 'GHz', self.unit_factors['GHz']
        phi = np.arange(0., 2*np.pi, 0.001)
        c_ab = self.correlation_coefficient()
        for i in range(self.n_dim - 1):
            x_cont = np.linspace(self.p[0, -1] - 0.1*(self.p[-1, -1] - self.p[0, -1]),
                                 self.p[-1, -1] + 0.1*(self.p[-1, -1] - self.p[0, -1]), 1001)
            x_cont -= self.alpha_k
            x_plot = np.multiply(x_cont, y_s)
            plt.plot(x_plot, straight(x_cont, self.f[i], self.k[i])*y_s
                     - Fr.straight_std(x_cont, self.k_d[i], self.f_d[i], c_ab[i, (self.n_dim - 1) + i])*y_s, 'b-',
                     label='Slope uncertainty')
            plt.plot(x_plot, straight(x_cont, self.f[i], self.k[i])*y_s
                     + Fr.straight_std(x_cont, self.k_d[i], self.f_d[i], c_ab[i, (self.n_dim - 1) + i])*y_s, 'b-')
            plt.plot(x_plot, straight(x_cont, self.f[i], self.k[i])*y_s, 'r-', label='King fit')
            plt.plot((self.mean[:, -1] - self.alpha_k)*y_s, self.mean[:, i]*y_s, 'ko', label='Data')
            print('\n{}-{} correlation of:'.format(self.x_run, self.runs[i]))
            for j, iso in enumerate(self.isotopes):
                plt.text((self.p[j, -1] - self.alpha_k - np.max(self.std[:, -1])*5)*y_s,
                         (self.p[j, i] + np.max(self.std[:, i])*5)*y_s, iso)

                shift_x = st.norm.rvs(loc=self.mean0[j, -1], scale=self.std0[j, -1], size=self.n_sample) * y_s
                shift = st.norm.rvs(loc=self.mean0[j, i], scale=self.std0[j, i], size=self.n_sample) * y_s
                m_fac = st.norm.rvs(loc=self.mass_factors[iso][0], scale=self.mass_factors[iso][1], size=self.n_sample)
                rho = np.corrcoef(shift_x*m_fac, shift*m_fac, ddof=1)[0, -1]
                print('{}: {}'.format(iso, rho))

                plt.plot((self.mean[j, -1] - self.alpha_k) * y_s + x_alpha(self.std[j, -1] * y_s, phi),
                         self.mean[j, i] * y_s + y_ell(self.std[j, i] * y_s, phi, rho), 'k-')
                plt.plot((self.mean[j, -1] - self.alpha_k) * y_s + x_alpha(self.std[j, -1] * y_s * 2, phi),
                         self.mean[j, i] * y_s + y_ell(self.std[j, i] * y_s * 2, phi, rho), 'k-')

                if iso in save_iso:
                    out = np.array([phi, self.mean[j, -1] - self.alpha_k + x_alpha(self.std[j, -1], phi),
                                    self.mean[j, i] + y_ell(self.std[j, i], phi, rho),
                                    self.mean[j, -1] - self.alpha_k + x_alpha(self.std[j, -1]*2, phi),
                                    self.mean[j, i] + y_ell(self.std[j, i]*2, phi, rho)]).T
                    np.savetxt('ErrorCircle_{}.txt'.format(iso), out, fmt='%.7f')

            plt.xticks(fontsize=self.fontsize)
            plt.yticks(fontsize=self.fontsize)
            plt.xlabel(r'$\mu\ \delta\nu_\mathrm{' + self.x_run + r'}^{' + self.mass_number_ref
                       + r', A}\quad(\mathrm{u}\ \mathrm{' + mod_iso_shift_scale + r'})$', fontsize=self.fontsize)
            plt.ylabel(r'$\mu\ \delta\nu_\mathrm{' + self.runs[i] + r'}^{' + self.mass_number_ref
                       + r', A}\quad(\mathrm{u}\ \mathrm{' + mod_iso_shift_scale + r'})$',
                       fontsize=self.fontsize)

            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend(handles[::-1], labels[::-1], numpoints=1, loc='best', fontsize=self.fontsize)
            plt.margins(0.1)
            plt.gcf().set_facecolor('w')
            plt.tight_layout()

            if self.store_loc is not None:
                f_name = 'MC_king_fit_alpha_%d' % self.alpha
                plt.savefig(os.path.join(self.store_loc, f_name + '.pdf'))
                plt.savefig(os.path.join(self.store_loc, f_name + '.png'))
            if self.popup:
                plt.show()
            plt.gcf().clear()
