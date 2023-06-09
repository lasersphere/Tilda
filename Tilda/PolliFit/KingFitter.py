"""
Created on 23.08.2016

@author: gorges

"""

import ast
import os
import sqlite3

from operator import itemgetter
from itertools import *

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc
import Tilda.PolliFit.TildaTools as TiTs

m_e_u = sc.physical_constants['electron mass in u']  # (value, unit, uncertainty)


# noinspection PyPep8Naming,PyAttributeOutsideInit
class KingFitter(object):
    """
    The Kingfitter needs some (at least three) charge radii as input and calculates the kingfits and new charge radii
    from the isotopeshifts in the database. The fitting routine is based on
    ['Unified equations for the slope, intercept, and standard errors of the best straight line', York et al.,
    American Journal of Physics 72, 367 (2004)]
    The variable alpha can be varied to reduce the uncertainty in the intercept and thus in the charge radii,
    this is described in e.g. Hammen PhD Thesis 2013
    """

    def __init__(self, db, litvals=None, showing=True, plot_y_mhz=True, font_size=12, ref_run=-1, incl_projected=False,
                 subtract_electrons=0., add_ionization_energy=0., plot_folder=None, popup=True):
        """
        Import the litvals and initializes a KingFit, run can be specified, for run==-1 any shift results are chosen
        """
        self.showing = showing
        self.popup = popup  # While showing is used to suppress plot during find best alpha, this directly cuts plt.show
        self.fontsize = font_size  # fontsize used in plots
        self.plot_y_mhz = plot_y_mhz  # use False to plot y axis in gigahertz
        self.db = db
        self.a = 0
        self.b = 1
        self.a_b_correlation = 0

        self.litvals = litvals
        if self.litvals is None:
            self.litvals = {}
        self.incl_projected = incl_projected
        self.subtract_electrons = subtract_electrons
        self.add_ionization_energy = add_ionization_energy
        self.reset_y_values = True  # can be changed when calcRedVar() may not reset y redMasses (important for 3DKing)

        self.isotopes = []
        self.isotopeMasses = []
        self.massErr = []
        self.isotopeShifts = []
        self.isotopeShiftErr = []
        self.isotopeShiftStatErr = []
        self.isotopeShiftSystErr = []
        self.run = []

        if plot_folder is None:
            self.store_loc = os.path.normpath(os.path.join(os.path.split(db)[0], 'king_fits'))
        else:
            self.store_loc = plot_folder
        if not os.path.isdir(self.store_loc):
            os.mkdir(self.store_loc)

        try:
            if ref_run == -1:
                self.ref = TiTs.select_from_db(self.db, 'reference', 'Lines', caller_name=__name__)[0][0]
            else:
                self.ref = TiTs.select_from_db(self.db, 'reference', 'Lines',
                                               [['refRun'], [ref_run]],
                                               caller_name=__name__)[0][0]
            self.ref_mass = TiTs.select_from_db(self.db, 'mass', 'Isotopes',
                                                [['iso'], [self.ref]], caller_name=__name__)[0][0]
            self.ref_mass -= self.subtract_electrons * m_e_u[0]
            self.ref_mass += self.add_ionization_energy * sc.e / (sc.atomic_mass * sc.c ** 2)
            self.ref_massErr = TiTs.select_from_db(self.db, 'mass_d', 'Isotopes',
                                                   [['iso'], [self.ref]], caller_name=__name__)[0][0]
            self.ref_massErr = np.sqrt(self.ref_massErr ** 2 + (self.subtract_electrons * m_e_u[2]) ** 2)

        except Exception as e:
            print('error: %s  \n\t-> Kingfitter could not find a reference isotope from'
                  ' Lines in database or mass of this reference Isotope in Isotopes' % e)

    def kingFit(self, run=-1, alpha=0, findBestAlpha=True, find_slope_with_statistical_error=False,
                print_coeff=True, print_information=True, results_to_db=True):
        """
        For find_b_with_statistical_error=True:
        performs at first a KingFit with just statistical uncertainty to find out the slope, afterwards
        performing a KingFit with full error to obtain the y-intercept
        """

        self.calcRedVar(run=run, find_slope_with_statistical_error=find_slope_with_statistical_error,
                        findBestAlpha=findBestAlpha, alpha=alpha, reset_y_values=self.reset_y_values)

        slope_syst_err = intercept_syst_err = slope_stat_err = intercept_stat_err = 0

        if print_information:
            print('performing King fit!')

        """ Here we perform now several fits! Be aware that only the result of one will be written to the database,
         this will be stored in the variables final_a, final_b, ... """
        if find_slope_with_statistical_error:
            # first fit with only statistical error in x and y
            (self.a, self.b, self.aerr, self.berr, self.a_b_correlation) = self.fit(run, showplot=True)
            # self.yerr = self.yerr_total
            slope_stat_err = self.berr
            intercept_stat_err = self.aerr
            print('condition\t intercept (u MHz)\t err_int\t slope (MHz/fm^2)\t err_slope\t correlation coefficient')
            print('statistical y errors only\t %.0f \t %.0f \t %.3f \t %.3f \t %.4f'
                  % (self.a, self.aerr, self.b, self.berr, self.a_b_correlation))

            # the following fits are performed fit with total error in y
            self.yerr = self.yerr_total
            # self.xerr = self.xerr_total

            # fit with fixed slope, total error in y, correlation = 0
            (self.a, self.b, self.aerr, self.berr, self.a_b_correlation) = self.fit(run, showplot=False,
                                                                                    bFix=True, print_corr_coeff=False)
            print("fixed slope, full errors, error correlation = 0"
                  "\t %.0f \t %.0f \t %.3f \t %.3f \t %.4f"
                  % (self.a, self.aerr, self.b, self.berr, self.a_b_correlation))
            slope_syst_err = max(slope_syst_err, self.berr)
            intercept_syst_err = max(intercept_syst_err, self.aerr)
            final_a = self.a
            final_b = self.b

            # fit with slope free, total error in y, correlation = 0 (only for comparison)
            # the result of this fit will be carried into the database for the slope and the intercept
            # (not the uncertainties)
            (self.a, self.b, self.aerr, self.berr, self.a_b_correlation) = self.fit(run, showplot=self.showing,
                                                                                    bFix=False, print_corr_coeff=False)
            print("free slope, full errors, error correlation = 0"
                  "\t %.0f \t %.0f \t %.3f \t %.3f \t %.4f"
                  % (self.a, self.aerr, self.b, self.berr, self.a_b_correlation))
            slope_syst_err = max(slope_syst_err, self.berr)
            intercept_syst_err = max(intercept_syst_err, self.aerr)

            # print('King fits performed, final values:')
            # print('intercept: ', round(self.a), '(', round(intercept_stat_err),
            #  ') [', round(intercept_syst_err), '] u MHz',
            #  '\t percent systematic: %.2f' % (intercept_syst_err / self.a * 100))
            # print('slope: ', self.b, '(', slope_stat_err, ') [', slope_syst_err , '] MHz/fm^2',
            #   '\t percent systematic: %.2f' % (slope_syst_err / self.b * 100))

        else:
            # errors are systematic since they include the systematics but have been handled statistically
            #  --> be carefull
            self.yerr_total = self.yerr  # needed for printing of total error below
            (self.a, self.b, self.aerr, self.berr, self.a_b_correlation) = self.fit(run, showplot=self.showing,
                                                                                    print_corr_coeff=print_coeff)
            if print_information:
                print('King fit performed with full errors only, no correlation assumed,'
                      ' uncertainties might be too small.\n')
                print('final values:')
                print('intercept: ', self.a, '(', self.aerr, ') u MHz', '\t percent: %.2f' % (self.aerr / self.a * 100))
                print('slope: ', self.b, '(', self.berr, ') MHz/fm^2',  '\t percent: %.2f' % (self.berr / self.b * 100))
            slope_syst_err = self.berr
            intercept_syst_err = self.aerr
            final_a = self.a
            final_b = self.b

        if results_to_db:
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            cur.execute('INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)',
                        ('kingVal', 'intercept', run))
            con.commit()
            cur.execute('UPDATE Combined SET val = ?, statErr = ?,  systErr = ?, '
                        'config=? WHERE iso = ? AND parname = ? AND run = ?',
                        (final_a, intercept_stat_err, intercept_syst_err,
                         '{}, incl_projected = {}'.format(self.litvals, self.incl_projected),
                         'kingVal', 'intercept', run))
            con.commit()
            cur.execute('INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)',
                        ('kingVal', 'slope', run))
            con.commit()
            cur.execute('UPDATE Combined SET val = ?, statErr = ?, systErr = ?, '
                        'config=? WHERE iso = ? AND parname = ? AND run = ?',
                        (final_b, slope_stat_err, slope_syst_err,
                         '{}, incl_projected = {}'.format(self.litvals, self.incl_projected),
                         'kingVal', 'slope', run))
            con.commit()
            cur.execute('INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)',
                        ('kingVal', 'alpha', run))
            con.commit()
            cur.execute('UPDATE Combined SET val = ?, config=? WHERE iso = ? AND parname = ? AND run = ?',
                        (self.c, '{}, incl_projected = {}'.format(self.litvals, self.incl_projected),
                         'kingVal', 'alpha', run))
            con.commit()
            con.close()

    def fit(self, run=-1, showplot=True, bFix=False, plot_y_mhz=None, font_size=None, print_corr_coeff=True):
        if plot_y_mhz is None:
            plot_y_mhz = self.plot_y_mhz
        if font_size is None:
            font_size = self.fontsize
        i = 0
        totaldiff = 1
        omega_x = [1 / np.square(i) for i in self.xerr]
        omega_y = [1 / np.square(i) for i in self.yerr]
        sigma_a_square = 0.
        sigma_b_square = 0.
        a_b_correlation_coeff = 0.
        alpha = [np.sqrt(j * omega_y[i]) for i, j in enumerate(omega_x)]
        r = [0 for i in self.x]  # muonic data is not correlated to iso shift measurement results

        while totaldiff > 1e-10 and i < 200:
            w = [j * omega_y[i] / (j + np.square(self.b) * omega_y[i] - 2 * self.b * alpha[i] * r[i])
                 for i, j in enumerate(omega_x)]
            w_x = [j * w[i] for i, j in enumerate(self.x)]
            x_bar = sum(w_x) / sum(w)
            u = [i - x_bar for i in self.x]

            x_fit = [(i - self.a) / self.b for i in self.y]
            w_x_fit = [j * w[i] for i, j in enumerate(x_fit)]
            x_fit_bar = sum(w_x_fit) / sum(w)
            u_fit = [i - x_fit_bar for i in x_fit]
            w_u_fit_square = [np.square(j) * w[i] for i, j in enumerate(u_fit)]

            w_y = [j * w[i] for i, j in enumerate(self.y)]
            y_bar = sum(w_y) / sum(w)
            v = [i - y_bar for i in self.y]

            y_fit = [self.a + self.b * i for i in self.x]
            w_y_fit = [j * w[i] for i, j in enumerate(y_fit)]
            y_fit_bar = sum(w_y_fit) / sum(w)
            v_fit = [i - y_fit_bar for i in y_fit]  # v_fit == y_fit ???

            beta = [j * (u[i] / omega_y[i] + self.b * v[i] / omega_x[i]
                         - (self.b * u[i] + v[i]) * r[i] / alpha[i]) for i, j in enumerate(w)]
            betaW = [j * w[i] for i, j in enumerate(beta)]
            betaWV = [j * v[i] for i, j in enumerate(betaW)]
            betaWU = [j * u[i] for i, j in enumerate(betaW)]

            # covariance
            sigma_b_tilde_square = 1 / (sum([w_i * u_fit[i] ** 2 for i, w_i in enumerate(w)]))
            sigma_a_tilde_square = 1 / (sum([w_i for w_i in w])) + x_bar ** 2 * sigma_b_tilde_square
            # cov_a_b = - x_bar * sigma_b_tilde_square  # Covariance, not used at the moment
            # r_ab in paper:
            a_b_correlation_coeff = - x_bar * np.sqrt(sigma_b_tilde_square) / np.sqrt(sigma_a_tilde_square)

            if not bFix:
                self.b = sum(betaWV) / sum(betaWU)
            self.a = y_bar - self.b * x_bar

            sigma_b_square = 1 / sum(w_u_fit_square) if not bFix else 0.
            sigma_a_square = 1 / sum(w) + np.square(x_fit_bar) * sigma_b_square
            diff_x = np.abs(sum([np.abs(w_x_fit[i] - j) for i, j in enumerate(w_x)]))
            diff_y = np.abs(sum([np.abs(w_y_fit[i] - j) for i, j in enumerate(w_y)]))
            totaldiff = diff_x + diff_y
            i += 1

        # plt.subplots_adjust(bottom=0.2)
        plt.xticks(rotation=0)
        ax = plt.gca()
        if plot_y_mhz:
            ax.set_ylabel(r' $\mu$ $\delta$ $\nu^{60,A}$ / u MHz ', fontsize=font_size)
        else:
            ax.set_ylabel(r' $\mu$ $\delta$ $\nu^{60,A}$ / u GHz ', fontsize=font_size)
        if self.c == 0:
            ax.set_xlabel(r'$\mu$ $\delta$ $\langle$ r$_c$'+r'$^2$ $\rangle$ $^{60,A}$ / u fm $^2$', fontsize=font_size)
        else:
            ax.set_xlabel(r'$\mu$ $\delta$ < r'+r'$^2$ >$^{60,A}$ - $\alpha$ / u fm $^2$', fontsize=font_size)
        ax.set_xmargin(0.05)
        x_king = [min(self.x) - abs(min(self.x) - max(self.x)) * 0.2,
                  max(self.x) + abs(min(self.x) - max(self.x)) * 0.2]
        y_king = [self.a + self.b * i for i in x_king]
        if plot_y_mhz:
            plt.plot(x_king, y_king, 'r', label='King fit', linewidth=2)
        else:
            y_king_ghz = [each / 1000 for each in y_king]
            plt.plot(x_king, y_king_ghz, 'r', label='King fit', linewidth=2)

        if plot_y_mhz:
            plt.errorbar(self.x, self.y, self.yerr, self.xerr, fmt='k.', markersize=10)
        else:  # plot in Gigahertz
            y_ghz = [each / 1000 for each in self.y]
            y_err_ghz = [each / 1000 for each in self.yerr]
            plt.errorbar(self.x, y_ghz, y_err_ghz, self.xerr, fmt='k.', markersize=10)

        # print('x', self.x, self.xerr)
        # print('y', self.y, self.yerr)
        print('%s\t%s\t%s\t%s\t%s' % ('x', 'x_err', 'y', 'y_stat_err', 'y_err_total'))
        for i, x in enumerate(self.x):
            print('%.5f\t%.5f\t%.5f\t%.5f\t%.5f' % (x, self.xerr[i], self.y[i], self.yerr[i], self.yerr_total[i]))
        print('%s\t%s' % ('King_fit_x', 'King_fit_y'))
        print('%.5f\t%.5f' % (x_king[0], y_king[0]))
        print('%.5f\t%.5f' % (x_king[1], y_king[1]))

        plt.gcf().set_facecolor('w')
        plt.legend(fontsize=font_size)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.tight_layout()
        f_name = 'king_fit_alpha_%d' % self.c
        if showplot:
            plt.savefig(os.path.join(self.store_loc, f_name + '.pdf'))
            plt.savefig(os.path.join(self.store_loc, f_name + '.png'))
            if self.popup:
                plt.show()
        plt.gcf().clear()

        self.aerr = np.sqrt(sigma_a_square)
        if not bFix:
            self.berr = np.sqrt(sigma_b_square)
        if print_corr_coeff:
            print('correlation coefficient of a and b is: %.5f' % a_b_correlation_coeff)

        if showplot:
            # store results to file when showing the plot
            # (.fit() is called very often without showing the plot when optimitzing self.c(=alpha))
            with open(os.path.join(self.store_loc, f_name + '_fit_points' + '.txt'), 'w') as king_f:
                king_f.write('# intercept: %.3f +/- %.3f\n' % (self.a, self.aerr))
                king_f.write('# slope: %.3f +/- %.3f\n' % (self.b, self.berr))
                king_f.write('# correlation: %.3f\n' % a_b_correlation_coeff)
                king_f.write('# %s\t%s\t%s\t%s\t%s \n' % ('x', 'x_err', 'y', 'y_stat_err', 'y_err_total'))
                for i, x in enumerate(self.x):
                    king_f.write('%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n'
                                 % (x, self.xerr[i], self.y[i], self.yerr[i], self.yerr_total[i]))

            with open(os.path.join(self.store_loc, f_name + '_fit_data' + '.txt'), 'w') as king_f:
                king_f.write('# intercept: %.3f +/- %.3f\n' % (self.a, self.aerr))
                king_f.write('# slope: %.3f +/- %.3f\n' % (self.b, self.berr))
                king_f.write('# correlation: %.3f\n' % a_b_correlation_coeff)
                king_f.write('King_fit_x\tKing_fit_y\n')
                king_f.write('%.5f\t%.5f\n' % (x_king[0], y_king[0]))
                king_f.write('%.5f\t%.5f\n' % (x_king[1], y_king[1]))

        return self.a, self.b, self.aerr, self.berr, a_b_correlation_coeff

    def calcChargeRadii(self, isotopes=None, run=-1, plot_evens_seperate=False, incl_projected=False,
                        save_in_db=True, print_results=True, print_information=True, dash_missing_data=False):
        if print_information:
            print('calculating the charge radii...')
        self.isotopes = []
        self.isotopeMasses = []
        self.massErr = []
        self.isotopeShifts = []
        self.isotopeShiftErr = []
        self.isotopeShiftStatErr = []
        self.isotopeShiftSystErr = []
        self.run = []

        (self.b, self.berr) = TiTs.select_from_db(self.db, 'val, systErr', 'Combined',
                                                  [['parname', 'run'], ['slope', run]], caller_name=__name__)[0]
        (self.a, self.aerr) = TiTs.select_from_db(self.db, 'val, systErr', 'Combined',
                                                  [['parname', 'run'], ['intercept', run]], caller_name=__name__)[0]
        (self.c,) = TiTs.select_from_db(self.db, 'val', 'Combined',
                                        [['parname', 'run'], ['alpha', run]], caller_name=__name__)[0]
        f_name = 'charge_radii_alpha_%d' % self.c
        if run == -1:
            vals = TiTs.select_from_db(self.db, 'iso, val, statErr, systErr, run', 'Combined', [['parname'], ['shift']],
                                       caller_name=__name__)
        else:
            vals = TiTs.select_from_db(self.db, 'iso, val, statErr, systErr, run', 'Combined',
                                       [['parname', 'run'], ['shift', run]], caller_name=__name__)
        for i in vals:
            (name, val, statErr, systErr, run) = i
            if name != self.ref:
                if isotopes is None or name in isotopes:
                    self.isotopes.append(name)
                    mass = TiTs.select_from_db(self.db, 'mass, mass_d', 'Isotopes', [['iso'], [name]],
                                               caller_name=__name__)[0]
                    self.isotopeMasses.append(mass[0] - self.subtract_electrons * m_e_u[0]
                                              + self.add_ionization_energy * sc.e / (sc.atomic_mass * sc.c ** 2))
                    self.massErr.append(np.sqrt(mass[1] ** 2 + (self.subtract_electrons * m_e_u[2]) ** 2))
                    self.isotopeShifts.append(val)
                    self.isotopeShiftStatErr.append(statErr)
                    self.isotopeShiftSystErr.append(systErr)
                    self.run.append(run)

        # if there are projected isotope shifts from a different transition included,
        # these can be also used for the calculation
        if incl_projected:
            vals = TiTs.select_from_db(self.db, 'iso, val, statErr, systErr, run', 'Combined',
                                       [['parname', 'run'], ['projected IS', run]], caller_name=__name__)
            for i in vals:
                (name, val, statErr, systErr, run) = i
                if name != self.ref:
                    if isotopes is None or name in isotopes:
                        self.isotopes.append(name+str('_proj'))
                        mass = TiTs.select_from_db(self.db, 'mass, mass_d', 'Isotopes', [['iso'], [name]],
                                                   caller_name=__name__)[0]
                        self.isotopeMasses.append(mass[0] - self.subtract_electrons * m_e_u[0]
                                                  + self.add_ionization_energy * sc.e / (sc.atomic_mass * sc.c ** 2))
                        self.massErr.append(np.sqrt(mass[1] ** 2 + (self.subtract_electrons * m_e_u[2]) ** 2))
                        self.isotopeShifts.append(val)
                        self.isotopeShiftStatErr.append(statErr)
                        self.isotopeShiftSystErr.append(systErr)
                        self.run.append(run)

        self.isotopeRedMasses = [(i + m_e_u[0]) * self.ref_mass / (i - self.ref_mass) for i in self.isotopeMasses]
        # from error prop:
        self.isotopeRedMassesErr = [np.sqrt(
            ((i_d * (self.ref_mass + m_e_u[0]) * self.ref_mass) / (i - self.ref_mass) ** 2) ** 2
            + ((self.ref_massErr * (i + m_e_u[0]) * i) / (i - self.ref_mass) ** 2) ** 2
            + (m_e_u[2] * self.ref_mass / (i - self.ref_mass)) ** 2)
                                    for i, i_d in zip(self.isotopeMasses, self.massErr)]
        self.chargeradii = [(j - self.a / self.isotopeRedMasses[i]) / self.b + self.c / self.isotopeRedMasses[i]
                            for i, j in enumerate(self.isotopeShifts)]
        # self.chargeradiiStatErrs = [np.abs(i/self.b) for i in self.isotopeShiftStatErr]
        self.chargeradiiTotalErrs = [np.sqrt(
            np.square(self.isotopeShiftStatErr[i]/self.b) +
            np.square(self.aerr/(self.isotopeRedMasses[i]*self.b)) +
            np.square((self.a/self.isotopeRedMasses[i]-j)*self.berr/np.square(self.b)) +
            np.square((self.a/self.b-self.c) * self.isotopeRedMassesErr[i]/np.square(self.isotopeRedMasses[i]))
        )
                                     for i, j in enumerate(self.isotopeShifts)]
        errs_to_print = [(
                             self.isotopes[i],  # iso
                             self.isotopeShifts[i],  # shift
                             self.isotopeShiftStatErr[i],  # shift_stat_err
                             abs(self.isotopeShiftStatErr[i] / self.isotopeShifts[i]) * 100,  # rel. shift_stat_err
                             self.chargeradii[i],  # dr^2
                             self.chargeradiiTotalErrs[i],  # Delta dr^2
                             abs(self.chargeradiiTotalErrs[i] / self.chargeradii[i]) * 100,  # rel. Delta dr^2
                             abs(self.isotopeShiftStatErr[i] / self.b),  # Delta Is
                             abs(self.aerr / (self.isotopeRedMasses[i] * self.b)),  # Delta K
                             abs((self.a / self.isotopeRedMasses[i] - j) * self.berr / np.square(self.b)),  # Delta F
                             abs((self.a / self.b - self.c) * self.isotopeRedMassesErr[i]
                                 / np.square(self.isotopeRedMasses[i])),  # Delta M
                         )
                         for i, j in enumerate(self.isotopeShifts)]
        if print_results:
            # print error componenet that are combined to get the total charge radii uncertainty -> Gaussian error prop
            # e.g. Delta IS -> abs(self.isotopeShiftStatErr[i] / self.b) -> shift_stat_err / F
            # K -> mass shift factor -> self.a
            # F -> field shift factor -> self.b
            # alpha -> x-axis offset -> self.c
            print('iso\tshift\tshift_stat_err\trel. shift_stat_err\t'
                  'dr^2\tDelta dr^2\trel. Delta dr^2\tDelta IS\tDelta K\tDelta F\tDelta M')
            for each in errs_to_print:
                print('%s\t%.4f\t%.4f\t%.2f\t%.4f\t%.4f\t%.2f\t%.8f\t%.8f\t%.8f\t%.3E' % each)

            with open(os.path.join(self.store_loc, f_name + '_vals_errs.txt'), 'w') as ch_err_f:
                ch_err_f.write('#iso\tshift\tshift_stat_err\trel. shift_stat_err\t'
                               'dr^2\tDelta dr^2\trel. Delta dr^2\tDelta IS\tDelta K\tDelta F\tDelta M\n')
                for each in errs_to_print:
                    ch_err_f.write('%s\t%.4f\t%.4f\t%.2f\t%.4f\t%.4f\t%.2f\t%.8f\t%.8f\t%.8f\t%.3E\n' % each)

        finalVals = {}
        for i, j in enumerate(self.isotopes):
            finalVals[j] = [self.chargeradii[i], self.chargeradiiTotalErrs[i]]
            if save_in_db:
                con = sqlite3.connect(self.db)
                cur = con.cursor()
                cur.execute('INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)',
                            (j, 'delta_r_square', self.run[i]))
                con.commit()
                cur.execute(
                   'UPDATE Combined SET val = ?, statErr = ?, systErr = ? WHERE iso = ? AND parname = ? AND run = ?',
                   (self.chargeradii[i], 0, self.chargeradiiTotalErrs[i], j, 'delta_r_square', self.run[i]))
                con.commit()
                con.close()

        if self.showing:
            font_size = self.fontsize
            finalVals[self.ref] = [0, 0, 0]
            keyVals = sorted(finalVals)
            x = []
            y = []
            yerr = []
            print('iso\t ' + r'$\delta$ <r$^2$>[fm$^2$]')
            for i in keyVals:
                x.append(int(''.join(filter(str.isdigit, i))))
                y.append(finalVals[i][0])
                yerr.append(finalVals[i][1])
                print('%s\t%.3f(%.0f)\t%.1f' % (i, finalVals[i][0], finalVals[i][1] * 1000,
                                                finalVals[i][1] / max(abs(finalVals[i][0]), 0.000000000000001) * 100))
                # print("'" + str(i) + "'", ':[', np.round(finalVals[i][0], 3),
                #       ',' + str(np.round(np.sqrt(finalVals[i][1]**2 + finalVals[i][2]**2), 3))+'],')
            with open(os.path.join(self.store_loc, f_name + '_plot_vals.txt'), 'w') as plot_f:
                plot_f.write('#mass\tdRcharge2\terrdRcharge2\n')
                for x_i, y_i, y_i_err in zip(x, y, yerr):
                    plot_f.write('%d\t%.5f\t%.5f\n' % (x_i, y_i, y_i_err))

            # plt.subplots_adjust(bottom=0.2, left=0.19, right=0.97)
            plt.xticks(rotation=0)
            ax = plt.gca()
            ax.set_ylabel(r'$\delta$ < r'+r'$^2$ > (fm $^2$) ', fontsize=font_size)
            ax.set_xlabel('A', fontsize=font_size)
            if plot_evens_seperate:
                x_odd = [each for each in x if each % 2 != 0]
                y_odd = [each for i, each in enumerate(y) if x[i] % 2 != 0]
                y_odd_err = [each for i, each in enumerate(yerr) if x[i] % 2 != 0]
                x_even = [each for each in x if each % 2 == 0]
                y_even = [each for i, each in enumerate(y) if x[i] % 2 == 0]
                y_even_err = [each for i, each in enumerate(yerr) if x[i] % 2 == 0]

                plt.errorbar(x_even, y_even, y_even_err, fmt='ro', label='even', linestyle='-')
                plt.errorbar(x_odd, y_odd, y_odd_err, fmt='r^', label='odd', linestyle='--')
                # plt.legend(loc=2)
            elif dash_missing_data:
                # if some isotopes are missing, this dashes the line at these isotopes
                # has no effect when plot_evens_separate is True
                split_x_list = []
                for k, g in groupby(enumerate(x), lambda a: a[0]-a[1]):
                    split_x_list.append(list(map(itemgetter(1), g)))
                i = 0
                for each in split_x_list:
                    y_vals = y[i:i+len(each)]
                    yerr_vals = yerr[i:i + len(each)]
                    plt.errorbar(each, y_vals, yerr_vals, fmt='ro', linestyle='-')
                    # plot dashed lines between missing values
                    if len(x) > i+len(each):  # might be last value
                        x_gap = [x[i+len(each)-1], x[i+len(each)]]
                        y_gap = [y[i + len(each) - 1], y[i + len(each)]]
                        plt.plot(x_gap, y_gap, c='r', linestyle='--')
                    i = i+len(each)
            else:
                plt.errorbar(x, y, yerr, fmt='ro', linestyle='-')
            ax.set_xmargin(0.05)
            plt.margins(0.1)
            plt.gcf().set_facecolor('w')
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)
            plt.tight_layout()
            plt.savefig(os.path.join(self.store_loc, f_name + '.pdf'))
            plt.savefig(os.path.join(self.store_loc, f_name + '.png'))
            if self.popup:
                plt.show()

        return finalVals

    def findBestAlpha(self, run):
        self.x = [self.redmasses[i] * j - self.c for i, j in enumerate(self.x_origin)]
        (bestA, bestB, bestAerr, bestBerr, bestCorrelation) = self.fit(run, showplot=False)
        step = 1
        best = self.c
        end = False
        up = True
        self.c += step
        while not end:
            print('searching for the best alpha... Trying alpha = ', self.c)
            if np.abs(self.c) >= 2000:  # searching just between [-2000, 2000]
                self.c = - self.c + step
            self.x = [self.redmasses[i]*j - self.c for i, j in enumerate(self.x_origin)]
            (newA, newB, newAerr, newBerr, newCorrelation) = self.fit(run, showplot=False)
            if abs(newCorrelation) < abs(bestCorrelation):
                bestCorrelation = newCorrelation
                best = self.c
                self.c += step

            else:
                if up:
                    up = False
                    step = - step
                    self.c += 2*step
                else:
                    end = True
        self.c = best
        print('best alpha is: ', self.c)

    def calcRedVar(self, run=-1, find_slope_with_statistical_error=False,
                   alpha=0, findBestAlpha=False, reset_y_values=True):
        self.x_origin = []
        self.x = []
        self.xerr = []
        # Added this for 3DKingPlot since y-values and masses do not change, only x-values; saves time
        if reset_y_values:
            self.masses = []
            self.y = []
            self.yerr = []
            self.yerr_total = []

        self.c = alpha
        self.findBestAlphaTrue = findBestAlpha

        if self.litvals == {}:
            self.litvals = ast.literal_eval(TiTs.select_from_db(self.db, 'config', 'Combined',
                                            [['parname', 'run'], ['slope', run]], caller_name=__name__)[0][0])
        y = [0, 0, 0]
        for i in self.litvals.keys():
            if reset_y_values:
                self.masses.append(TiTs.select_from_db(self.db, 'mass', 'Isotopes', [['iso'], [i]],
                                                       caller_name=__name__)[0][0]
                                   - self.subtract_electrons * m_e_u[0]
                                   + self.add_ionization_energy * sc.e / (sc.atomic_mass * sc.c ** 2))
            if run == -1:
                y = TiTs.select_from_db(self.db, 'val, statErr, systErr', 'Combined',
                                        [['iso', 'parname'], [i, 'shift']], caller_name=__name__)[0]
            else:
                y = TiTs.select_from_db(self.db, 'val, statErr, systErr', 'Combined',
                                        [['iso', 'parname', 'run'], [i, 'shift', run]], caller_name=__name__)[0]
            self.y.append(y[0])
            if find_slope_with_statistical_error:
                self.yerr.append(y[1])  # statistical error
                self.yerr_total.append(np.sqrt(np.square(y[1])+np.square(y[2])))  # total error
            else:
                self.yerr.append(np.sqrt(np.square(y[1])+np.square(y[2])))  # total error
                self.yerr_total.append(np.sqrt(np.square(y[1])+np.square(y[2])))  # total error
            self.x_origin.append(self.litvals[i][0])
            self.xerr.append(self.litvals[i][1])

        if self.incl_projected:
            for i in self.litvals.keys():
                # Added this for 3DKingPlot since y-values and masses do not change, only x-values; saves time
                if reset_y_values:
                    self.masses.append(TiTs.select_from_db(self.db, 'mass', 'Isotopes', [['iso'], [i]],
                                                           caller_name=__name__)[0][0]
                                       - self.subtract_electrons * m_e_u[0]
                                       + self.add_ionization_energy * sc.e / (sc.atomic_mass * sc.c ** 2))
                    if run == -1:
                        y = TiTs.select_from_db(self.db, 'val, statErr, systErr', 'Combined',
                                                [['iso', 'parname'], [i, 'projected IS']],
                                                caller_name=__name__)[0]
                    else:
                        y = TiTs.select_from_db(self.db, 'val, statErr, systErr', 'Combined',
                                                [['iso', 'parname', 'run'], [i, 'projected IS', run]],
                                                caller_name=__name__)[0]
                self.y.append(y[0])
                if find_slope_with_statistical_error:
                    self.yerr.append(y[1])  # statistical error
                    self.yerr_total.append(np.sqrt(np.square(y[1])+np.square(y[2])))  # total error
                else:
                    self.yerr.append(np.sqrt(np.square(y[1])+np.square(y[2])))  # total error
                    self.yerr_total.append(np.sqrt(np.square(y[1])+np.square(y[2])))  # total error
                self.x_origin.append(self.litvals[i][0])
                self.xerr.append(self.litvals[i][1])

        if reset_y_values:
            self.redmasses = [(i + m_e_u[0]) * self.ref_mass / (i - self.ref_mass) for i in self.masses]
            self.y = [self.redmasses[i] * j for i, j in enumerate(self.y)]
            self.yerr = [np.abs(self.redmasses[i] * j) for i, j in enumerate(self.yerr)]
            self.yerr_total = [np.abs(self.redmasses[i] * j) for i, j in enumerate(self.yerr_total)]

        # the x-errors are only dependent on the reduced masses and thus constant for all alpha:
        self.xerr = [np.abs(self.redmasses[i] * j) for i, j in enumerate(self.xerr)]
        if self.findBestAlphaTrue:
            self.findBestAlpha(run)
        # now also calculate the final x in dependence of the best alpha
        self.x = [self.redmasses[i] * j - self.c for i, j in enumerate(self.x_origin)]
