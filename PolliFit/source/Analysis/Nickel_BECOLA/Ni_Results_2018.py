"""
Created on 2018-12-19

@author: fsommer

Module Description:  Analysis of the Nickel Data from BECOLA taken on 13.04.-23.04.2018
"""

import ast
import os
import sqlite3
from datetime import date, datetime, timedelta
import re
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mpdate
from scipy.optimize import curve_fit
from operator import itemgetter
from itertools import *

import BatchFit
import Physics
import Tools
import TildaTools as TiTs
from lxml import etree as ET
from XmlOperations import xmlWriteDict
from Measurement.XMLImporter import XMLImporter
from KingFitter import KingFitter

class NiAnalysis():
    def __init__(self):
        logging.getLogger().setLevel(logging.INFO)
        # Name this analysis run
        self.run_name = 'Ni_Results_2018'

        """
        ############################ Folders and Database !##########################################################
        Specify where files and db are located, and where results will be saved!
        """
        # working directory:
        # get user folder to access ownCloud
        user_home_folder = os.path.expanduser("~")
        # ownCould_path = 'ownCloud\\User\\Felix\\Measurements\\Nickel54_online_Becola20\\Analysis\\XML_Data'
        # ownCould_path = 'ownCloud\\User\\Felix\\Measurements\\Nickel54_postbeamtime_Becola20\\Analysis\\bunched'
        ownCould_path = 'ownCloud\\User\\Felix\\Measurements\\Nickel_online_Becola\\Analysis\\XML_Data'  # online 2018
        self.workdir = os.path.join(user_home_folder, ownCould_path)
        # data folder
        self.datafolder = os.path.join(self.workdir, 'SumsRebinned')
        # results folder
        analysis_start_time = datetime.now()
        self.results_name = self.run_name + '_' + analysis_start_time.strftime("%Y-%m-%d_%H-%M")
        results_path_ext = 'results\\' + self.results_name + '\\'
        self.resultsdir = os.path.join(self.workdir, results_path_ext)
        os.makedirs(self.resultsdir)
        # database
        self.db = os.path.join(self.workdir, 'Ni_Becola.sqlite')
        Tools.add_missing_columns(self.db)


        """
        ############################Analysis Parameters!##########################################################
        Specify how you want to run this Analysis!
        """
        # fit from scratch or use FitRes db?
        self.load_prev_results = True
        load_results_from = 'Ni_StandartizedAnalysis_2020-09-18_13-36.xml'  # load fit results from this file

        # Isotopes
        self.isotopes_single = ['56Ni', '58Ni', '60Ni']  # data is good enough for single file fitting
        self.isotopes_summed = ['55Ni']  # data must be summed in order to be fittable
        self.all_isotopes = self.isotopes_single + self.isotopes_summed

        # plot options
        self.save_plots_to_file = True  # if False plots will be displayed during the run for closer inspection
        self.isotope_colors = {60: 'b', 58: 'k', 56: 'g', 55: 'c', 54: 'm', 62: 'purple', 64: 'orange'}
        self.scaler_colors = {'scaler_0': 'navy', 'scaler_1': 'maroon', 'scaler_2': 'orangered',
                              'scaler_012': 'orange', 'scaler_12': 'yellow',
                              'scaler_c012': 'magenta', 'scaler_c0': 'purple', 'scaler_c1': 'grey', 'scaler_c2': 'fuchsia'}

        # note down main laser frequencies. These will be used for the summed files. Otherwise from file:
        self.laser_freqs = {'54Ni': 851248358.0,  # 425,624179 * 2
                            '55Ni': 851264686.7203143,  # 14197,56675,
                            '56Ni': 851253864.2125804,  # 14197.38625,
                            '58Ni': 851238644.9486578,  # 14197.13242,
                            '60Ni': 851224124.8007469,  # 14196.89025
                            '62Ni': 851209466,
                            '64Ni': 851196350
                            }

        # mean calibrated ion energy
        self.accVolt_set = 29850

        # Determine calibration parameters
        self.ref_iso = '60Ni'

        # Kingfit options
        self.KingFactorLit = 'Koenig 2020 60ref'  # which king fit factors to use? kaufm60, koenig60,koenig58

        """
        ############################ Physics and uncertainties! ########################################################
        Initialize other input
        """

        """
        ### Physics Input ###
        """
        ''' Masses '''
        # # Reference:   'The Ame2016 atomic mass evaluation: (II). Tables, graphs and references'
        # #               Chinese Physics C Vol.41, No.3 (2017) 030003
        # #               Meng Wang, G. Audi, F.G. Kondev, W.J. Huang, S. Naimi, Xing Xu
        self.masses = {
            '54Ni': (53957833.0, 0.5),
            '55Ni': (54951330.0, 0.8),
            '56Ni': (55942127.9, 0.5),
            '57Ni': (56939791.5, 0.6),
            '58Ni': (57935341.8, 0.4),
            '59Ni': (58934345.6, 0.4),
            '60Ni': (59930785.3, 0.4),
            '61Ni': (60931054.9, 0.4),
            '62Ni': (61928344.9, 0.5),
            '63Ni': (62929669.1, 0.5),
            '64Ni': (63927966.3, 0.5),
            '65Ni': (64930084.7, 0.5),
            '66Ni': (65929139.3, 1.5),
            '67Ni': (66931569, 3),
            '68Ni': (67931869, 3),
            '69Ni': (68935610, 4)
        }

        ''' Moments, Spin '''
        # Reference:    "Table of Nuclear Magnetic Dipole and Electric Quadrupole Moments",
        #               IAEA Nuclear Data Section, INDC(NDS)-0658, February 2014,
        #               N.J.Stone
        #               p.36
        # magnetic dipole moment µ in units of nuclear magneton µn
        # electric Quadrupolemoment Q in units of barn
        # Format: {'xxNi' : (IsoMass_A, IsoSpin_I, IsoDipMom_µ, IsoDipMomErr_µerr, IsoQuadMom_Q, IsoQuadMomErr_Qerr)}
        self.nuclear_spin_and_moments = {
            '55Ni': (55, -7 / 2, -0.98, 0.03, 0, 0),
            '57Ni': (57, -3 / 2, -0.7975, 0.0014, 0, 0),
            '61Ni': (61, -3 / 2, -0.75002, 0.00004, 0.162, 0.015)
            # even isotopes 56, 58, 60 Ni have Spin 0 and since they are all even-even nucleons also the moments are zero
        }

        ''' A and B Factors '''
        # Reference: COLLAPS 2016
        # Format: {'xxNi' : (Al, Al_d, Au, Au_d, Arat, Arat_d, Bl, Bl_d, Bu, Bu_d, Brat, Brat_d)}
        self.reference_A_B_vals = {
            '59Ni': (-452.70, 1.1, -176.1, 1.6, 0.389, 0.004, -56.7, 6.8, -31.5, 5.5, 0.556, 0.118),
            '61Ni': (-454.8, 0.4, -176.9, 0.6, 0.389, 0.001, -100.6, 2.8, -49.0, 2.3, 0.487, 0.015)
        }
        self.thisWork_A_B_vals = {
            '55Ni': (-452.70, 1.1,
                     -176.1, 1.6,
                     0.389, 0.004,
                     -56.7, 6.8,
                     -31.5, 5.5,
                     0.556, 0.118)
        }

        ''' restframe transition frequency '''
        # Reference: Various sources: Use Kristian col/acol
        # 58:
        # NIST: observed wavelength air 352.454nm corresponds to 850586060MHz
        # upper lvl 28569.203cm-1; lower lvl 204.787cm-1
        # resulting wavenumber 28364.416cm-1 corresponds to 850343800MHz
        # KURUCZ database: 352.4535nm, 850344000MHz, 28364.424cm-1
        # Some value I used in the excel sheet: 850347590MHz Don't remember where that came from...
        # Kristians col/acol value 2020: 850343678(2)(20) MHz. Still preliminary
        # 60Ni
        # Kristians col/acol value 2020: 850343673(7) MHz. Still preliminary
        # --> Resulting isoshift col/acol: 505 MHz
        self.restframe_trans_freq = {'58Ni': (850343678.0, 2.0), '60Ni': (850344183.0, 2.0)}

        ''' literature value IS 60-58'''
        iso_shifts_kaufm = {  # PRL 124, 132502, 2020
            '58Ni': (-509.1, 2.5 + 4.2),
            '59Ni': (-214.3, 2.5 + 2.0),  # From Simons Nickel_shift_results.txt(30.04.20), not published yet
            '60Ni': (0, 0),
            '61Ni': (280.8, 2.7 + 2.0),
            '62Ni': (503.9, 2.5 + 3.9),
            '63Ni': (784.9, 2.5 + 5.0),  # From Simons Nickel_shift_results.txt(30.04.20), not published yet
            '64Ni': (1027.2, 2.5 + 7.7),
            '65Ni': (1317.5, 2.5 + 9.0),  # From Simons Nickel_shift_results.txt(30.04.20), not published yet
            '66Ni': (1526.8, 2.5 + 11.0),  # From Simons Nickel_shift_results.txt(30.04.20), not published yet
            '67Ni': (1796.6, 2.5 + 13.0),  # From Simons Nickel_shift_results.txt(30.04.20), not published yet
            '68Ni': (1992.3, 2.7 + 14.7),
            '70Ni': (2377.2, 2.5 + 18.0),  # From Simons Nickel_shift_results.txt(30.04.20), not published yet
        }
        iso_shifts_steudel = {  # Z. Physik A - Atoms and Nuclei 296, 189 - 193 (1980)
            '58Ni': (Physics.freqFromWavenumber(-0.01694),  # 58-60
                     Physics.freqFromWavenumber(0.00009)),
            '60Ni': (0, 0),  # Shifts are given in pairs. Most ref to 60.
            '61Ni': (Physics.freqFromWavenumber(0.00916),  # 60-61
                     Physics.freqFromWavenumber(0.00010)),
            '62Ni': (Physics.freqFromWavenumber(0.01691),  # 60-62
                     Physics.freqFromWavenumber(0.00012)),
            '64Ni': (Physics.freqFromWavenumber(0.01691 + 0.01701),  # 60-62 and 62-64 combined.
                     Physics.freqFromWavenumber(np.sqrt(0.00012 ** 2 + 0.00026 ** 2)))}  # Quadr. error prop
        iso_shifts_koenig = {  # private com. excel sheet mid 2020
            '54Ni': (-1912.1, 7.3),  # preliminary from 2020 beamtime -1912.12(7.32) from Meeting 17.09.20
            '58Ni': (self.restframe_trans_freq['58Ni'][0] - self.restframe_trans_freq['60Ni'][0],
                     np.sqrt(self.restframe_trans_freq['58Ni'][1] ** 2 + self.restframe_trans_freq['60Ni'][1] ** 2)),
            '60Ni': (0, 0),
            '62Ni': (502.87, 3.43),
            '64Ni': (1026.14, 3.79)}
        iso_shifts_thisWork = {
            '55Ni': (-1432.8, 22.3),
            '56Ni': (-1001.6, 1.7),
            '58Ni': (-501.2, 4.0),
            '60Ni': (0, 0)}

        self.iso_shifts_lit = {'Kaufmann 2020 (incl.unbup.!)': {'data': iso_shifts_kaufm, 'color': 'green'},
                               # (incl.unbup.!)
                               'Steudel 1980': {'data': iso_shifts_steudel, 'color': 'black'},
                               'Koenig 2020': {'data': iso_shifts_koenig, 'color': 'blue'}}
        self.iso_shifts_thisWork = {'This Work': {'data': iso_shifts_thisWork, 'color': 'green'}}

        ''' literature Mass Shift and Field Shift constants '''
        self.king_literature = {
            'Kaufmann 2020 60ref': {'data': {'Alpha': 396, 'F': (-769, 60), 'Kalpha': (948000, 3000)},
                                    'color': 'green'},  # Kaufmann.2020 10.1103/PhysRevLett.124.132502
            'Koenig 2020 60ref': {'data': {'Alpha': 388, 'F': (-761.87, 89.22), 'Kalpha': (953881, 4717)},
                                  'color': 'red'},  # König.2020 private com
            'Koenig 2020 58ref': {'data': {'Alpha': 419, 'F': (-745.27, 96.79), 'Kalpha': (930263, 3009)},
                                  'color': 'black'},  # König.2020 private com
            # 'KingCombined60': {'data': {'Alpha': 371, 'F': (-810.58, 77.16), 'Kalpha': (966818, 3503)},
            #                    'color': 'blue'}
            }

        ''' literature radii '''
        delta_rms_kaufm = {'58Ni': (-0.275, 0.007),
                           '59Ni': (-0.180, 0.008),  # From Nickel_delta_r_square_results.txt(30.04.20), not published!
                           '60Ni': (0, 0),
                           '61Ni': (0.083, 0.005),
                           '62Ni': (0.223, 0.005),
                           '63Ni': (0.278, 0.007),  # From Nickel_delta_r_square_results.txt(30.04.20), not published!
                           '64Ni': (0.368, 0.009),
                           '65Ni': (0.386, 0.016),  # From Nickel_delta_r_square_results.txt(30.04.20), not published!
                           '66Ni': (0.495, 0.015),  # From Nickel_delta_r_square_results.txt(30.04.20), not published!
                           '67Ni': (0.516, 0.022),  # From Nickel_delta_r_square_results.txt(30.04.20), not published!
                           '68Ni': (0.620, 0.021),
                           '70Ni': (0.808, 0.022),  # From Nickel_delta_r_square_results.txt(30.04.20), not published!
                           }
        delta_rms_steudel = {'58Ni': (-0.218, 0.040),  # 58-60
                             '60Ni': (0, 0),  # Shifts are given in pairs. Most ref to 60.
                             '61Ni': (0.065, 0.017),  # 60-61
                             '62Ni': (0.170, 0.035),  # 60-62
                             '64Ni': (0.280, 0.041)}  # 60-62 and 62-64 combined. Quadratic error prop
        x, xx, r, rd, = self.extract_radius_from_factors('54Ni', '60Ni', isoshift=iso_shifts_koenig['54Ni'])
        delta_rms_koenig = {'54Ni': (r, rd),  # very preliminary!
                            '58Ni': (-0.275, 0.0082),  # private com. excel sheet mid 2020
                            '60Ni': (0, 0),
                            '62Ni': (0.2226, 0.0059),
                            '64Ni': (0.3642, 0.0095)}

        delta_rms_thisWork = {
            '55Ni': self.extract_radius_from_factors('55Ni', '60Ni', isoshift=iso_shifts_thisWork['55Ni'])[2:],
            '56Ni': self.extract_radius_from_factors('56Ni', '60Ni', isoshift=iso_shifts_thisWork['56Ni'])[2:],
            '58Ni': self.extract_radius_from_factors('58Ni', '60Ni', isoshift=iso_shifts_thisWork['58Ni'])[2:],
            '60Ni': (0, 0)}

        self.delta_rms_lit = {'Kaufmann 2020 (incl.unbup.!)': {'data': delta_rms_kaufm, 'color': 'green'},
                              # (incl.unbup.!)
                              'Steudel 1980': {'data': delta_rms_steudel, 'color': 'black'},
                              'Koenig 2020': {'data': delta_rms_koenig, 'color': 'blue'}}
        self.delta_rms_thisWork = {'This Work': {'data': delta_rms_thisWork, 'color': 'green'}}

        # from Landolt-Börnstein - Group I Elementary Particles, Nuclei and Atoms, Fricke 2004
        # http://materials.springer.com/lb/docs/sm_lbs_978-3-540-45555-4_30
        # Root mean square nuclear charge radii <r^2>^{1/2}_{0µe}
        # lit_radii = {
        #     '58_Ni': (3.770, 0.004),
        #     '60_Ni': (3.806, 0.002),
        #     '61_Ni': (3.818, 0.003),
        #     '62_Ni': (3.836, 0.003),
        #     '64_Ni': (3.853, 0.003)
        # }   # have ben calculated more accurately below

        # Model-independent Barrett equivalent radius from muonic x-ray measurements Rµkα
        baret_radii_lit = {
            '58Ni': (4.8386, np.sqrt(0.0009 ** 2 + 0.0019 ** 2)),
            '60Ni': (4.8865, np.sqrt(0.0008 ** 2 + 0.002 ** 2)),
            '61Ni': (4.9005, np.sqrt(0.0010 ** 2 + 0.0017 ** 2)),
            '62Ni': (4.9242, np.sqrt(0.0009 ** 2 + 0.002 ** 2)),
            '64Ni': (4.9481, np.sqrt(0.0009 ** 2 + 0.0019 ** 2))
        }

        # Ratio of radial moments V2 = Rkα <r^2>^–1/2 from elastic electron scattering.
        v2_lit = {
            '58Ni': 1.283517,
            '60Ni': 1.283944,
            '61Ni': 1.283895,
            '62Ni': 1.283845,
            '64Ni': 1.284133
        }

        # combined from above: <r^2>^1/2 = Rkα/V2
        lit_radii_calc = {iso: (val[0] / v2_lit[iso], val[1]) for iso, val in sorted(baret_radii_lit.items())}

        # using the more precise values by the self calculated one:
        lit_radii = lit_radii_calc

        self.delta_lit_radii_60 = {iso: [
            lit_vals[0] ** 2 - lit_radii['60Ni'][0] ** 2,
            np.sqrt(lit_vals[1] ** 2 + lit_radii['60Ni'][1] ** 2)]
            for iso, lit_vals in sorted(lit_radii.items())}
        self.delta_lit_radii_60.pop('60Ni')

        self.delta_lit_radii_58 = {iso: [
            lit_vals[0] ** 2 - lit_radii['58Ni'][0] ** 2,
            np.sqrt(lit_vals[1] ** 2 + lit_radii['58Ni'][1] ** 2)]
            for iso, lit_vals in sorted(lit_radii.items())}
        self.delta_lit_radii_58.pop('58Ni')

        """
        ##################### other initializations ##################################################################
        """
        # calculate differential doppler shifts
        self.diff_dopplers = {key: Physics.diffDoppler(self.restframe_trans_freq[self.ref_iso][0],
                                                       self.accVolt_set,
                                                       self.masses[key][0] / 1e6)
                              for key in self.masses.keys()}
        # adjust center fit estimations to accVoltage
        # self.adjust_center_ests_db()

        # define a global time reference
        self.ref_datetime = datetime.strptime('2018-04-13_13:08:55', '%Y-%m-%d_%H:%M:%S')  # run 6191, first 58 we use

        # create results dictionary:
        self.results = {}
        self.gate_analysis_res = {}

        # Set the scaler variable to a defined starting point:
        self.update_scalers_in_db('012')

        # import results
        if self.load_prev_results:
            # import the results from a previous run as sepcified in load_results_from
            self.import_results(load_results_from)


    def plot_results_of_fit(self, calibrated=False):
        add_sc = []
        isolist = self.isotopes_single
        if calibrated:
            add_sc = []  # ['scaler_c0', 'scaler_c1', 'scaler_c2'] !! not used any more !!
            isolist = ['{}_cal'.format(i) for i in isolist]

        # plot iso-results for each scaler combination
        for sc in self.scaler_combinations+['scaler_c012']+add_sc:
            # write the scaler to db for usage
            scaler = self.update_scalers_in_db(sc)
            # plot results of first fit
            self.plot_parameter_for_isos_and_scaler(isolist, [scaler], 'center_fits', folder='fit_res')
            self.all_centerFreq_to_scanVolt(isolist, [scaler])
            self.plot_parameter_for_isos_and_scaler(isolist, [scaler], 'center_scanvolt', unit='V', folder='fit_res')
            if scaler != 'scaler_c012':  # fitpars don't make sense for the calculated combined scaler
                self.plot_parameter_for_isos_and_scaler(isolist, [scaler], 'rChi', folder='fit_res')
                self.get_weighted_avg_linepars(isolist, [scaler])
                self.plot_parameter_for_isos_and_scaler(isolist, [scaler], 'all_fitpars:center', unit='', folder='fit_res')
                for par, vals in self.initial_par_guess.items():
                    used, fixed = self.check_par_in_lineshape(par)
                    if used and not vals[1]==True:  # only plot when used and not fixed
                        self.plot_parameter_for_isos_and_scaler(isolist, [scaler], 'all_fitpars:{}'.format(par), unit='', folder='fit_res')

        # plot all scaler-results for each isotope
        for iso in isolist:
            self.plot_parameter_for_isos_and_scaler([iso], self.scaler_combinations+['scaler_c012']+add_sc,
                                                    'center_fits', onlyfiterrs=True, folder='fit_res')
            self.plot_parameter_for_isos_and_scaler([iso], self.scaler_combinations+add_sc, 'all_fitpars:center', folder='fit_res')
            for par, vals in self.initial_par_guess.items():
                used, fixed = self.check_par_in_lineshape(par)
                if used and not vals[1]==True:  # only plot when used and not fixed
                    self.plot_parameter_for_isos_and_scaler([iso], self.scaler_combinations+add_sc, 'all_fitpars:{}'.format(par), unit='', folder='fit_res')

    def calculate_charge_radii(self, refiso, isolist=None, calibrated=False, summed=False):
        """
        Charge radii extraction.
        :return:
        """


        for iso in isolist:
            for sc in self.results[iso].keys():
                if 'scaler' in sc:  # is a scaler key
                    logging.info('\n'
                                 '## calculating charge radii for scaler {}'.format(sc))
                    # write the scaler to db for usage
                    scaler = self.update_scalers_in_db(sc)

                    delta_rms, delta_rms_d, avg_delta_rms, avg_delta_rms_d = self.extract_radius_from_factors(iso, refiso, scaler)
                    zeros = np.zeros(len(delta_rms)).tolist()
                    # write isoshift to results dict
                    delta_rms_dict = {iso: {scaler: {'delta_rms_iso-{}'.format(refiso[:2]): {'vals': delta_rms,
                                                                                             'd_stat': zeros,  # TODO: Count as syst or stat?
                                                                                             'd_syst': delta_rms_d},
                                                     'avg_delta_rms_iso-{}'.format(refiso[:2]): {'vals': [avg_delta_rms],
                                                                                                 'd_stat': [0],
                                                                                                 'd_syst': [avg_delta_rms_d]}
                                                     }}}
                    TiTs.merge_extend_dicts(self.results, delta_rms_dict, overwrite=True, force_overwrite=True)
            #self.plot_parameter_for_isos_and_scaler(isolist, [sc], 'delta_rms_iso-{}'.format(refiso[:2]), digits=2)

        # self.plot_parameter_for_isos_vs_scaler(isolist, self.scaler_combinations + ['scaler_c012'],
        #                                        'avg_delta_rms_iso-{}'.format(refiso[:2]), digits=2)

    def get_final_results(self):
        """
        Pick the isotope shifts to use for final results and calculate from there
        :return:
        """
        all_isos = self.all_isotopes

        # get final values
        for iso in all_isos:
            final_vals_iso = {'shift_iso-{}'.format(self.ref_iso[:2]):
                                      {'vals': [self.iso_shifts_thisWork['This Work']['data'][iso][0]],
                                       'd_stat': [self.iso_shifts_thisWork['This Work']['data'][iso][1]],
                                       'd_syst': [self.iso_shifts_thisWork['This Work']['data'][iso][2]]
                                       if len(self.iso_shifts_thisWork['This Work']['data'][iso]) > 3 else [0]},
                              'delta_rms_iso-{}'.format(self.ref_iso[:2]):
                                  {'vals': [self.delta_rms_thisWork['This Work']['data'][iso][0]],
                                   'd_stat': [self.delta_rms_thisWork['This Work']['data'][iso][1]],
                                   'd_syst': [self.delta_rms_thisWork['This Work']['data'][iso][2]]
                                   if len(self.delta_rms_thisWork['This Work']['data'][iso]) > 3 else [0]}
                              }

            self.results[iso]['final'] = final_vals_iso

        # TODO: also get final A, B values and calculate µ, Q

        self.make_final_plots()


    ''' db related '''
    def reset(self, db_like, reset):
        """
        Resets isotope name and acceleration voltage in Files database
        :param db_like: isotopes to be reset (e.g.
        :param reset:
        :return:
        """
        # Reset all calibration information so that pre-calib information can be extracted.
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        # Set type in files back to bare isotopes (56Ni, 58Ni, 60Ni)
        # Set accVolt in files back to nominal 29850
        cur.execute('''UPDATE Files SET accVolt = ?, type = ? WHERE type LIKE ? ''',
                    (reset[0], reset[1], db_like))
        con.commit()
        con.close()

    def get_iso_property_from_db(self, command, tup):
        """
        Query an entry for a given isotope from db and return.
        :param command: sql command to execute e.g. '''SELECT mass, mass_d FROM Isotopes WHERE iso = ?''')
        :param tup: tuple: specifying questionmarks
        :return: value
        """
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute(command, tup)  # get original isotope to copy from
        ret = cur.fetchone()
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def create_new_isotope_in_db(self, copy_iso, iso_new, acc_volt):
        """
        creates a new isotope of type 'iso_new' based on a copy of 'copy_iso'.
        :param copy_iso: str: the isotope to copy
        :param iso_new: str: name of the new isotope. Must be unique.
        :param acc_volt: int: acceleration voltage to be used. Used to calculate new center fit parameter.
        :return:
        """
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''SELECT * FROM Isotopes WHERE iso = ? ''', (copy_iso,))  # get original isotope to copy from
        copy_isopars = cur.fetchall()
        mass = copy_isopars[0][1]
        center_old = copy_isopars[0][4]
        center_new = center_old + (self.accVolt_set - acc_volt) * self.diff_dopplers[copy_iso[:4]]
        isopars_lst = list(copy_isopars[0])  # change into list to replace some values
        isopars_lst[0] = iso_new
        isopars_lst[4] = center_new
        isopars_new = tuple(isopars_lst)
        cur.execute('''INSERT OR REPLACE INTO Isotopes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                    isopars_new)
        con.commit()
        con.close()

    def write_to_db_lines(self, line, offset=None, sigma=None, gamma=None, asy=None, dispersive=None, centerAsym=None,
                          IntAsym=None, nPeaksAsym=None):
        """
        We might want to adjust some line-parameter before fitting. E.g. to the ones from reference.
        :param line: lineVar parameter to edit
        :param offset: new value for the offset parameter
        :return:
        """
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''SELECT * FROM Lines WHERE lineVar = ? ''', (line,))  # get original line to copy from
        copy_line = cur.fetchall()
        copy_shape = copy_line[0][6]
        copy_fixshape = copy_line[0][7]
        shape_dict = ast.literal_eval(copy_shape)
        fixshape_dict = ast.literal_eval(copy_fixshape)
        if offset is not None:
            shape_dict['offset'] = offset[0]
            fixshape_dict['offset'] = offset[1]
        if sigma is not None:
            shape_dict['sigma'] = sigma[0]
            fixshape_dict['sigma'] = sigma[1]
        if gamma is not None:
            shape_dict['gamma'] = gamma[0]
            fixshape_dict['gamma'] = gamma[1]
        if asy is not None and shape_dict.get('asy', None) is not None:  # only for VoigtAsy profiles
            shape_dict['asy'] = asy[0]
            fixshape_dict['asy'] = asy[1]
        if dispersive is not None and shape_dict.get('dispersive', None) is not None:  # only for FanoVoigt profiles
            shape_dict['dispersive'] = dispersive[0]
            fixshape_dict['dispersive'] = dispersive[1]
        if centerAsym is not None and shape_dict.get('centerAsym', None) is not None:  # only for AsymmetricVoigt
            shape_dict['centerAsym'] = centerAsym[0]
            fixshape_dict['centerAsym'] = centerAsym[1]
        if IntAsym is not None and shape_dict.get('IntAsym', None) is not None:  # only for AsymmetricVoigt
            shape_dict['IntAsym'] = IntAsym[0]
            fixshape_dict['IntAsym'] = IntAsym[1]
        if nPeaksAsym is not None and shape_dict.get('nPeaksAsym', None) is not None:  # only for AsymmetricVoigt
            shape_dict['nPeaksAsym'] = nPeaksAsym[0]
            fixshape_dict['nPeaksAsym'] = nPeaksAsym[1]
        copy_line_list = list(copy_line[0])
        copy_line_list[6] = str(shape_dict)
        copy_line_list[7] = str(fixshape_dict)
        line_new = tuple(copy_line_list)
        cur.execute('''INSERT OR REPLACE INTO Lines VALUES (?,?,?,?,?,?,?,?,?)''', line_new)
        con.commit()
        con.close()

    def check_par_in_lineshape(self, par):
        """ Check whether a parameter is part of the used run and return it or None """
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''SELECT * FROM Lines WHERE lineVar = ? ''', (self.run,))  # get original line to copy from
        copy_line = cur.fetchall()
        copy_shape = copy_line[0][6]
        copy_fixshape = copy_line[0][7]
        shape_dict = ast.literal_eval(copy_shape)
        fixshape_dict = ast.literal_eval(copy_fixshape)

        return shape_dict.get(par, False), fixshape_dict.get(par, False)

    def adjust_center_ests_db(self):
        """
        Write new center fit estimations for the standard isotopes into the db. use self.accVolt_set as reference
        :return:
        """
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        stand_ests = {'55Ni': -1060, '56Ni': -712, '58Ni': -225, '60Ni': 293}  # values that worked fine for 29850V
        ref_freq_dev = 850343800 - self.restframe_trans_freq[self.ref_iso][0]  # stand_ests are for 580343800MHz. Ref freq might be updated
        for iso, mass_tupl in stand_ests.items():
            cur.execute('''UPDATE Isotopes SET center = ? WHERE iso = ? ''',
                        (stand_ests[iso] + ref_freq_dev + (29850 - self.accVolt_set) * self.diff_dopplers[iso], iso))
        con.commit()
        con.close()

    def write_shift_to_combined_db(self, iso, run, shift_w_errs, config_str):
        """
        Writes the isotope shift of one isotope to database
        :param iso: str: isotope name
        :param run: str: run name
        :param shift_w_errs: tuple (shift_MHz, stat_err_MHz, syst_err_MHz)
        :param config_str: description of the configuration: Which files have been used, singly or combined?
        :return:
        """
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)''',
                    (iso, 'shift', run))
        con.commit()
        cur.execute(
            '''UPDATE Combined SET val = ?, statErr = ?,  systErr = ?, config=? WHERE iso = ? AND parname = ? AND run = ?''',
            (shift_w_errs[0], shift_w_errs[1], shift_w_errs[2], config_str, iso, 'shift', run))
        con.commit()
        con.close()

    def pick_files_from_db_by_type_and_num(self, type, selecttuple=None):
        """
        Searches the database for files with given type and numbers and returns names and numbers of found.
        :param type: str: type of files to be picked
        :param selecttuple: (int, int): lowest and highest file numbers to be included
        :return: list, list: file names, file numbers
        """
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute(
            '''SELECT file, date FROM Files WHERE type LIKE ? ORDER BY date ''', (type,))
        files = cur.fetchall()
        con.close()
        # convert into np array
        ret_files = []
        ret_file_nos = []
        ret_file_dates = []
        for file, date in files:
            fileno = int(re.split('[_.]', file)[1])
            file_date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
            if selecttuple is not None:
                if isinstance(selecttuple, tuple):
                    selecttuple = [selecttuple]
                select = []
                for tup in selecttuple:
                    select += range(tup[0], tup[1]+1, 1)
                if fileno in select:
                    ret_files.append(file)
                    ret_file_nos.append(fileno)
                    ret_file_dates.append(file_date)
            else:
                ret_files.append(file)
                ret_file_nos.append(fileno)
                ret_file_dates.append(file_date)
        return ret_files, ret_file_nos, ret_file_dates

    def update_scalers_in_db(self, scalers):
        '''
        Update the scaler parameter for all runs in the runs database
        :param scalers: prefer list as in db. Else int or str: either an int (0,1,2) if a single scaler is used or a string '0,1,2' for all
        :return: scaler name string
        '''
        if type(scalers) is list:
            scaler_db_string = str(scalers)
            scaler_name = ''.join([str(i) for i in scalers])
        elif 'scaler_' in str(scalers):
            scaler_db_string = ','.join(list(scalers.split('_')[-1])).join(('[', ']'))
            scaler_db_string = scaler_db_string.replace('c,', '')  # remove c if scaler_c012
            scaler_name = scalers.split('_')[-1]
        elif type(scalers) is int:
            scaler_db_string = str(scalers).join(('[',']'))
            scaler_name = str(scalers)
        else:
            scaler_db_string = '[0,1,2]'
            scaler_name = '012'
        self.scaler_name = 'scaler_{}'.format(scaler_name)
        if scalers == 'final':
            # special case! Final is used as a scaler here for plotting...
            self.scaler_name = scalers
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''UPDATE Runs SET Scaler = ?''', (scaler_db_string,))
        con.commit()
        con.close()
        return self.scaler_name

    def update_gates_in_db(self, iso, midtof=None, gatewidth=None, delaylist=None):
        '''
        Write all parameters relevant for the software gate position into the database
        :param midtof: float: center of software gate in µs
        :param gatewidth: float: width of software gate in µs
        :param delaylist: list of floats: list of delays in midtof for scalers 0,1,2
        :return:
        '''
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        if gatewidth is not None:
            cur.execute('''UPDATE Runs SET softwGateWidth = ? WHERE run = ?''', (gatewidth, self.run))
        if delaylist is not None:
            cur.execute('''UPDATE Runs SET softwGateDelayList = ? WHERE run = ?''', (str(delaylist), self.run))
        if midtof is not None:
            cur.execute('''UPDATE Isotopes SET midTof = ? WHERE iso = ?''', (midtof, iso))
        con.commit()
        con.close()

    def param_from_fitres_db(self, file, isostring, run, parameter):
        """
        Gets the 'center' fit parameter (or any other) for a given fit result (file, isotope and run)
        :param file: str: filename
        :param isostring: str: isotope name
        :param run: str: runname
        :param parameter: str: parameter to be extracted, e.g. 'center'
        :return: tuple: (center frequency, uncertainty, fixed)
        """
        # extract isotope shift and write to db combined
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        # Query fitresults for 55Ni
        cur.execute(
            '''SELECT pars FROM FitRes WHERE file = ? AND iso = ? AND run = ?''',
            (file, isostring, run))
        pars = cur.fetchall()
        con.close()
        parsdict = ast.literal_eval(pars[0][0])
        par = parsdict[parameter]  # tuple of (center frequency, Uncertainty, Fixed)

        return par

    ''' fitting and calibration '''

    def centerFreq_to_absVoltage(self, isostring, deltanu, nu_d, nu_dsyst, laserfreq=None):
        """
        Converts the center frequency parameter into a scan voltage again
        :return:
        """
        # get mass from database
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        # Query isotope parameters for isotope
        cur.execute(
            '''SELECT mass FROM Isotopes WHERE iso = ? ''', (isostring[:4], ))
        db_isopars = cur.fetchall()
        # Query laser frequency for isotope
        isostring_like = isostring + '%'
        cur.execute(
            '''SELECT laserFreq FROM Files WHERE type LIKE ? ''', (isostring_like, ))
        db_laserfreq = cur.fetchall()
        con.close()

        m = db_isopars[0][0]
        nuL = db_laserfreq[0][0] if laserfreq is None else laserfreq
        nuoff = self.restframe_trans_freq[self.ref_iso][0]

        velo = Physics.invRelDoppler(nuL, nuoff+deltanu)
        ener = Physics.relEnergy(velo, m*Physics.u)
        volt = ener/Physics.qe  # convert energy to voltage

        diffdopp = Physics.diffDoppler(nuoff + deltanu, volt, m)
        d = nu_d/diffdopp
        d_syst = nu_dsyst/diffdopp
        return volt, d, d_syst

    def absVoltage_to_centerFreq(self, isostring, volt, laserfreq=None, collinear=True):
        # get mass from database
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        # Query isotope parameters for isotope
        cur.execute(
            '''SELECT mass FROM Isotopes WHERE iso = ? ''', (isostring[:4],))
        db_isopars = cur.fetchall()
        if laserfreq is None:
            # Query laser frequency for isotope
            isostring_like = isostring + '%'
            cur.execute(
                '''SELECT laserFreq FROM Files WHERE type LIKE ? ''', (isostring_like,))
            db_laserfreq = cur.fetchall()
            laserfreq = db_laserfreq[0][0]
        con.close()

        m = db_isopars[0][0]

        # collinear or anticollinear?
        if collinear:
            ac = -1
        else:
            ac = 1
        rel_beta = Physics.relVelocity(volt*Physics.qe, m*Physics.u)/Physics.c
        restframe_f = laserfreq * np.sqrt((1+ac*rel_beta)/(1-ac*rel_beta))

        nuoff = self.restframe_trans_freq[self.ref_iso][0]
        nucenter = restframe_f - nuoff

        return nucenter

    def all_centerFreq_to_scanVolt(self, iso_list, scaler_list):
        """
        Converts all center frequencies of an isotope into scanVoltage values.
        :param iso_list: List of isotopes as named in results dict
        :return:
        """
        for isostring in iso_list:
            for sc in scaler_list:
                scaler = self.update_scalers_in_db(sc)
                if self.results[isostring].get(scaler, None) is not None:
                    center_freq = self.results[isostring][scaler]['center_fits']['vals']
                    center_freq_d_fit = self.results[isostring][scaler]['center_fits']['d_fit']

                    center_volt, center_volt_d, center_volt_dsyst = [], [], []
                    for nu, d in zip(center_freq, center_freq_d_fit):
                        v, v_d, v_ds = self.centerFreq_to_absVoltage(isostring, nu, d, 0)
                        center_volt.append(-v+self.accVolt_set)
                        center_volt_d.append(v_d)
                        center_volt_dsyst.append(v_ds)

                    voltdict = {'vals': center_volt,
                                'd_stat': center_volt_d,
                                'd_syst': center_volt_dsyst}
                    self.results[isostring][scaler]['center_scanvolt'] = voltdict

    ''' isoshift related: '''

    def plot_IS_results(self, iso, results_text):
        fig, ax = plt.subplots()
        # Plot Isotope shift iso-58 for all scalers to compare.
        lensc0 = len(self.results[iso]['file_numbers'])
        plt.errorbar(range(lensc0), self.results[iso]['scaler_0']['shift_iso-58']['vals'], c='b',
                     yerr=self.results[iso]['scaler_0']['shift_iso-58']['d_stat'],
                     label='scaler 0')
        lensc1 = len(self.results[iso]['file_numbers'])
        plt.errorbar(range(lensc1), self.results[iso]['scaler_1']['shift_iso-58']['vals'], c='g',
                     yerr=self.results[iso]['scaler_1']['shift_iso-58']['d_stat'],
                     label='scaler 1')
        lensc2 = len(self.results[iso]['file_numbers'])
        plt.errorbar(range(lensc2), self.results[iso]['scaler_2']['shift_iso-58']['vals'], c='r',
                     yerr=self.results[iso]['scaler_2']['shift_iso-58']['d_stat'],
                     label='scaler 2')
        # Plot weighted average and errorband for all scalers
        avg_is = self.results[iso]['scaler_0']['avg_shift_iso-58']['vals'][0]
        avg_is_d = self.results[iso]['scaler_0']['avg_shift_iso-58']['d_stat'][0]
        avg_is_d_sys = self.results[iso]['scaler_0']['avg_shift_iso-58']['d_syst'][0]
        plt.plot([-1, lensc0], [avg_is, avg_is], c='b')
        plt.fill([-1, lensc0, lensc0, -1],  # statistical error
                 [avg_is - avg_is_d, avg_is - avg_is_d, avg_is + avg_is_d, avg_is + avg_is_d], 'b', alpha=0.2)
        plt.fill([-1, lensc0, lensc0, -1],  # systematic error
                 [avg_is - avg_is_d - avg_is_d_sys, avg_is - avg_is_d - avg_is_d_sys, avg_is + avg_is_d + avg_is_d_sys,
                  avg_is + avg_is_d + avg_is_d_sys], 'b',
                 alpha=0.05)
        avg_is1 = self.results[iso]['scaler_1']['avg_shift_iso-58']['vals'][0]
        avg_is1_d = self.results[iso]['scaler_1']['avg_shift_iso-58']['d_stat'][0]
        avg_is1_dsys = self.results[iso]['scaler_1']['avg_shift_iso-58']['d_syst'][0]
        plt.plot([-1, lensc1], [avg_is1, avg_is1], c='g')
        plt.fill([-1, lensc1, lensc1, -1],  # statistical error
                 [avg_is1 - avg_is1_d, avg_is1 - avg_is1_d, avg_is1 + avg_is1_d, avg_is1 + avg_is1_d], 'g', alpha=0.2)
        plt.fill([-1, lensc1, lensc1, -1],  # systematic error
                 [avg_is1 - avg_is1_d - avg_is1_dsys, avg_is1 - avg_is1_d - avg_is1_dsys,
                  avg_is1 + avg_is1_d + avg_is1_dsys, avg_is1 + avg_is1_d + avg_is1_dsys], 'g',
                 alpha=0.05)
        avg_is2 = self.results[iso]['scaler_2']['avg_shift_iso-58']['vals'][0]
        avg_is2_d = self.results[iso]['scaler_2']['avg_shift_iso-58']['d_stat'][0]
        avg_is2_dsys = self.results[iso]['scaler_2']['avg_shift_iso-58']['d_syst'][0]
        plt.plot([-1, lensc2], [avg_is2, avg_is2], c='r')
        plt.fill([-1, lensc2, lensc2, -1],
                 [avg_is2 - avg_is2_d, avg_is2 - avg_is2_d, avg_is2 + avg_is2_d, avg_is2 + avg_is2_d], 'r', alpha=0.2)
        plt.fill([-1, lensc2, lensc2, -1],  # systematic error
                 [avg_is2 - avg_is2_d - avg_is2_dsys, avg_is2 - avg_is2_d - avg_is2_dsys,
                  avg_is2 + avg_is2_d + avg_is2_dsys, avg_is2 + avg_is2_d + avg_is2_dsys], 'r',
                 alpha=0.05)
        # plt.plot(range(len(ni56_isoShift_yData)), ni56_isoShift_yData, '-o', label='preferred')
        # plt.plot(range(len(ni56_isoShift_yData)), ni56_isoShift_alt_yData, 'r-o', label='alternative')
        plt.xticks(range(lensc0), self.results[iso]['file_numbers'], rotation=-30)
        plt.axis('tight')
        # plt.axis([-0.5, lensc0 - 0.5, -580, -470])
        plt.title('Isotope Shift Ni {}-58 for all runs'.format(iso[:2]))
        plt.xlabel('Run Number')
        plt.ylabel('Isotope Shift  [MHz]')
        plt.legend(loc='lower right')
        props = dict(boxstyle='round', facecolor='white', alpha=1)
        ax.text(0.05, 0.95, results_text, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        if self.save_plots_to_file:
            filename = 'shift_iso-58_{}_sc012_overview'.format(iso[:2])
            plt.savefig(self.resultsdir + filename + '.png')
        else:
            plt.show()
        plt.close()
        plt.clf()

    ''' stacking files related: '''

    def writeXMLfromDict(self, dictionary, filename, tree_name_str):
        """
        filename must be in form name.xml
        """
        root = ET.Element(tree_name_str)
        xmlWriteDict(root, dictionary)
        xml = ET.ElementTree(root)
        xml.write(filename)

    ''' 55 Nickel related: '''

    def ni55_A_B_analysis(self, isotope, scaler='scaler_012', calibrated=False):
        if calibrated:
            isotope = '{}_cal'.format(isotope)  # make sure to use the calibrated isos

        # copy the results dict
        res_dict = self.results[isotope]

        # update the scaler
        scaler = self.update_scalers_in_db(scaler)

        # Get all coefficients from fit results
        Al = res_dict[scaler]['hfs_pars']['Al'][0]
        Al_d = res_dict[scaler]['hfs_pars']['Al'][1]
        Au = res_dict[scaler]['hfs_pars']['Au'][0]
        Au_d = res_dict[scaler]['hfs_pars']['Au'][1]
        A_rat = res_dict[scaler]['hfs_pars']['Arat'][0]
        A_rat_d = res_dict[scaler]['hfs_pars']['Arat'][1]
        A_rat_fixed = res_dict[scaler]['hfs_pars']['Arat'][2]

        Bl = res_dict[scaler]['hfs_pars']['Bl'][0]
        Bl_d = res_dict[scaler]['hfs_pars']['Bl'][1]
        Bu = res_dict[scaler]['hfs_pars']['Bu'][0]
        Bu_d = res_dict[scaler]['hfs_pars']['Bu'][1]
        B_rat = res_dict[scaler]['hfs_pars']['Brat'][0]
        B_rat_d = res_dict[scaler]['hfs_pars']['Brat'][1]
        B_rat_fixed = res_dict[scaler]['hfs_pars']['Brat'][2]

        # calculate µ and Q values
        # reference moments stored in format: (IsoMass_A, IsoSpin_I, IsoDipMom_µ, IsoDipMomErr_µerr, IsoQuadMom_Q, IsoQuadMomErr_Qerr)
        m_ref, I_ref, mu_ref, mu_ref_d, Q_ref, Q_ref_d = self.nuclear_spin_and_moments['61Ni']
        m_55, I_55, mu_55, mu_55_d, Q_55, Q_55_d = self.nuclear_spin_and_moments['55Ni']
        # reference A and B factors stored in format: (Al, Al_d, Au, Au_d, Arat, Arat_d, Bl, Bl_d, Bu, Bu_d, Brat, Brat_d)
        Al_ref, Al_d_ref, Au_ref, Au_d_ref, Arat_ref, Arat_d_ref, Bl_ref, Bl_d_ref, Bu_ref, Bu_d_ref, Brat_ref, Brat_d_ref = self.reference_A_B_vals['61Ni']

        # magnetic dipole moment
        mu_55 = mu_ref * Al/Al_ref * I_55/I_ref
        mu_55_d = np.sqrt((mu_ref_d * Al/Al_ref*I_55/I_ref)**2 + (Al_d * mu_ref/Al_ref*I_55/I_ref)**2 + (Al_d_ref * mu_ref*Al/Al_ref**2*I_55/I_ref)**2)
        # electric quadrupole moment
        Q_55 = Q_ref * Bl/Bl_ref
        Q_55_d = np.sqrt((Q_ref_d*Bl/Bl_ref)**2 + (Bl_d*Q_ref/Bl_ref)**2 + (Bl_d_ref*Bl*Q_ref/Bl_ref**2)**2)
        logging.info('\nspectroscopic factors: Al={0:.0f}({1:.0f}), Au={2:.0f}({3:.0f}), Arat={4:.3f}({5:.0f}),'
                     ' Bl={6:.0f}({7:.0f}), Bu={8:.0f}({9:.0f}), Brat={10:.3f}({11:.0f})'
                     .format(Al, Al_d, Au, Au_d, A_rat, A_rat_d*1000, Bl, Bl_d, Bu, Bu_d, B_rat, B_rat_d*1000))
        logging.info('\nmu55 = {0:.3f}({1:.0f}), Q55 = {2:.3f}({3:.0f})'
                     .format(mu_55, mu_55_d*1000, Q_55, Q_55_d*1000))

        # write to results dict
        moments_dict = {'mu': {'vals': [mu_55],
                               'd_stat': [0],
                               'd_syst': [mu_55_d]},
                        'Q': {'vals': [Q_55],
                              'd_stat': [0],
                              'd_syst': [Q_55_d]}
                        }
        self.results[isotope][scaler]['moments'] = moments_dict

    ''' King Fit Related '''

    def perform_king_fit_analysis(self):
        # Define which isotopes to use
        isotopes = ['55Ni', '56Ni', '58Ni', '59Ni', '60Ni', '61Ni', '62Ni', '64Ni']
        delta_lit_radii = self.delta_lit_radii_60
        reference_run = self.run

        king = KingFitter(self.db, showing=True, litvals=delta_lit_radii, plot_y_mhz=False, font_size=18, ref_run=reference_run)
        king.kingFit(alpha=0, findBestAlpha=False, run=reference_run, find_slope_with_statistical_error=False)
        king.calcChargeRadii(isotopes=isotopes, run=reference_run, plot_evens_seperate=False, dash_missing_data=True)

        king.kingFit(alpha=361, findBestAlpha=True, run=reference_run)
        radii_alpha = king.calcChargeRadii(isotopes=isotopes, run=reference_run, plot_evens_seperate=False, dash_missing_data=True)
        print('radii with alpha', radii_alpha)

    def compare_king_pars(self):
        fig, ax = plt.subplots(1)

        x_list = []
        # calculate mass-scaled delta rms charge radii
        for iso, radii in self.delta_lit_radii_60.items():
            delta_rms_radius = radii[0]
            delta_rms_radius_d = radii[1]
            m_iso, m_iso_d = self.get_iso_property_from_db('''SELECT mass, mass_d FROM Isotopes WHERE iso = ?''',
                                                           (iso[:4],))
            m_ref, m_ref_d = self.get_iso_property_from_db('''SELECT mass, mass_d FROM Isotopes WHERE iso = ?''',
                                                           (self.ref_iso[:4],))
            mu = (m_iso - m_ref) / (m_iso * m_ref)
            mu_d = np.sqrt(np.square(m_iso_d / m_iso ** 2) + np.square(m_ref_d / m_ref ** 2))

            # add to list of x-values
            x_list.append(delta_rms_radius/mu)
        x_arr = np.array(sorted(x_list))
        x_arr = np.arange(270, 580, 1)

        # get the isotope shifts from our analysis
        shifts = []
        thisPoint = None
        isolist = ['55Ni', '56Ni', '58Ni', '60Ni']
        isolist.remove(self.ref_iso)
        for iso in isolist:
            m_iso, m_iso_d = self.get_iso_property_from_db('''SELECT mass, mass_d FROM Isotopes WHERE iso = ?''',
                                                           (iso[:4],))
            m_ref, m_ref_d = self.get_iso_property_from_db('''SELECT mass, mass_d FROM Isotopes WHERE iso = ?''',
                                                           (self.ref_iso[:4],))
            mu = (m_iso - m_ref) / (m_iso * m_ref)
            mu_d = np.sqrt(np.square(m_iso_d / m_iso ** 2) + np.square(m_ref_d / m_ref ** 2))

            iso_shift = self.results[iso]['final']['shift_iso-{}'.format(self.ref_iso[:2])]['vals'][0]
            iso_shift_d = self.results[iso]['final']['shift_iso-{}'.format(self.ref_iso[:2])]['d_stat'][0]
            iso_shift_d_syst = self.results[iso]['final']['shift_iso-{}'.format(self.ref_iso[:2])]['d_syst'][0]
            shifts.append((iso_shift/mu/1000, iso_shift_d/mu/1000, iso_shift_d_syst/mu/1000, iso))

            if iso in self.delta_lit_radii_60 and not iso == self.ref_iso:
                delta_rms = self.delta_lit_radii_60[iso]
                r = delta_rms[0]/mu
                r_d = np.sqrt((delta_rms[1]/mu)**2 + (delta_rms[0]*mu_d/mu**2)**2)
                s = iso_shift/mu/1000
                s_d = np.sqrt(((iso_shift_d+iso_shift_d_syst)/mu/1000)**2 + ((iso_shift)*mu_d/mu**2/1000)**2)
                thisPoint = (r, r_d, s, s_d)

        # add a band for each of our measured isotope shifts
        for tuples in shifts:
            # plot error band for this line
            plt.axhspan(tuples[0]-tuples[1]-tuples[2], tuples[0]+tuples[1]+tuples[2], facecolor='black', alpha=0.2)
            ax.annotate(r'$^\mathregular{{{:.0f}}}$Ni'.format(int(tuples[3][:2])), (290, tuples[0]-5))

        def _kingLine(x, k, f, a):
            return (k + f * (x))/1000  # in GHz

        def _kingLower(x, k, k_d, f, f_d, a):
            xa = _kingLine(x, k+k_d, f+f_d, a)
            xb = _kingLine(x, k+k_d, f-f_d, a)
            xc = _kingLine(x, k-k_d, f+f_d, a)
            xd = _kingLine(x, k-k_d, f-f_d, a)
            ret_arr = np.zeros(x.shape)
            for i in range(len(x)):
                ret_arr[i] = min(xa[i], xb[i], xc[i], xd[i])
            return ret_arr

        def _kingUpper(x, k, k_d, f, f_d, a):
            xa = _kingLine(x, k + k_d, f + f_d, a)
            xb = _kingLine(x, k + k_d, f - f_d, a)
            xc = _kingLine(x, k - k_d, f + f_d, a)
            xd = _kingLine(x, k - k_d, f - f_d, a)
            ret_arr = np.zeros(x.shape)
            for i in range(len(x)):
                ret_arr[i] = max(xa[i], xb[i], xc[i], xd[i])
            return ret_arr

        annotate_iso = []
        x_annotate = []
        y_annotate = []
        for src, item in self.king_literature.items():
            # get factors
            alpha = item['data']['Alpha']
            F, F_d = item['data']['F']
            Kalpha, Kalpha_d = item['data']['Kalpha']

            # get a color
            col = item['color']

            # plot line with errors:
            plt.plot(x_arr, _kingLine(x_arr-alpha, Kalpha, F, alpha), '--', c=col, lw=2)
            # plot error band for this line
            plt.fill_between(x_arr,
                             _kingLower(x_arr-alpha, Kalpha, Kalpha_d, F, F_d, alpha),
                             _kingUpper(x_arr-alpha, Kalpha, Kalpha_d, F, F_d, alpha),
                             alpha=0.4, edgecolor=col, facecolor=col)

            # plot each reference point from this source
            r_lst = []
            r_d_lst = []
            s_lst = []
            s_d_lst = []


            for iso, delta_rms in self.delta_lit_radii_60.items():
                m_iso, m_iso_d = self.get_iso_property_from_db('''SELECT mass, mass_d FROM Isotopes WHERE iso = ?''',
                                                               (iso[:4],))
                m_ref, m_ref_d = self.get_iso_property_from_db('''SELECT mass, mass_d FROM Isotopes WHERE iso = ?''',
                                                               (self.ref_iso[:4],))
                mu = (m_iso - m_ref) / (m_iso * m_ref)
                mu_d = np.sqrt(np.square(m_iso_d / m_iso ** 2) + np.square(m_ref_d / m_ref ** 2))

                if self.iso_shifts_lit[src[:-6]]['data'].get(iso, None) is not None:
                    r_lst.append(delta_rms[0] / mu)
                    r_d_lst.append(np.sqrt((delta_rms[1] / mu) ** 2 + (delta_rms[0] * mu_d / mu ** 2) ** 2))
                    src_first = ' '.join(src.split()[:2])
                    s_lst.append(self.iso_shifts_lit[src_first]['data'][iso][0] / mu / 1000)
                    s_d_lst.append(np.sqrt((self.iso_shifts_lit[src_first]['data'][iso][1] / mu / 1000) ** 2 + (
                                (self.iso_shifts_lit[src_first]['data'][iso][0]) * mu_d / mu ** 2 / 1000) ** 2))
                    if not iso in annotate_iso and 'Kauf' in src:
                        # only use Kaufmann values for the annotation:
                        annotate_iso.append(iso)
                        x_annotate.append(delta_rms[0] / mu)
                        y_annotate.append(self.iso_shifts_lit[src[:-6]]['data'][iso][0] / mu / 1000)

            plt.errorbar(r_lst, s_lst, xerr=r_d_lst, yerr=s_d_lst, fmt='o', c=col, elinewidth=1.5, label=src)

        for i, iso in enumerate(annotate_iso):
            ax.annotate(r'$^\mathregular{{{:.0f}}}$Ni'.format(int(iso[:2])), (x_annotate[i]+5, y_annotate[i]+5), color='green')

        if thisPoint is not None:
            plt.errorbar(thisPoint[0], thisPoint[2], xerr=thisPoint[1], yerr=thisPoint[3], fmt='ok', label='This Work', elinewidth=1.5)

        plt.xlabel(r'$\mu^{-1} \delta <r_c^2>$' + ' (u fm)' + r'$^2$')
        plt.ylabel(r'$\mu^{-1} \delta\nu$' + ' (u GHz)')
        plt.title('King Plot Comparison')
        plt.legend(title='Isotope Shift Measurements', numpoints=1, loc="best")
        plt.margins(0.05)
        if self.save_plots_to_file:
            plt.savefig(self.resultsdir + 'compare_kings' + '.png', bbox_inches="tight")
        else:
            plt.show()
        plt.close()
        plt.clf()

    def extract_radius_from_factors(self, iso, ref, scaler=None, isoshift=None):
        """
        Use known fieldshift and massshift parameters to calculate the difference in rms charge radii and then the
        absolute charge radii.
        Isotope Shift will be extracted from the 'Combined' results database and absolute radii are from literature.
        :param isotopes: isotopes to include e.g. ['55Ni_sum_cal', '56Ni_sum_cal', '58Ni_sum_cal', '60Ni_sum_cal',]
        :param reference: reference isotope (either 58 or 60)
        :return: delta_rms, delta_rms_d
        """
        # get the masses and calculate mu
        m_iso, m_iso_d = self.get_iso_property_from_db('''SELECT mass, mass_d FROM Isotopes WHERE iso = ?''',
                                                       (iso[:4],))
        m_ref, m_ref_d = self.get_iso_property_from_db('''SELECT mass, mass_d FROM Isotopes WHERE iso = ?''',
                                                       (ref[:4],))
        mu = (m_iso - m_ref) / (m_iso * m_ref)
        mu_d = np.sqrt(np.square(m_iso_d / m_iso ** 2) + np.square(m_ref_d / m_ref ** 2))

        # get Mass and Field Shift factors:
        if '58' in ref:
            kingFactorLit = 'Koenig 2020 58ref'
        else:
            kingFactorLit = 'Koenig 2020 60ref'
        M_alpha, M_alpha_d = self.king_literature[kingFactorLit]['data'][
            'Kalpha']  # Mhz u (lit val given in GHz u)(949000, 4000)
        F, F_d = self.king_literature[kingFactorLit]['data']['F']  # MHz/fm^2(-788, 82)
        alpha = self.king_literature[kingFactorLit]['data']['Alpha']  # u fm^2 397

        # get data and calculate radii
        delta_rms = []
        delta_rms_d = []
        if scaler is not None:
            # get per file isoshift
            files = self.results[iso]['file_names']
            if isoshift is not None:
                iso_shift = isoshift[0]
                iso_shift_d = isoshift[1]
                iso_shift_d_syst = 0
            else:
                iso_shift = self.results[iso][scaler]['shift_iso-{}'.format(ref[:2])]['vals']
                iso_shift_d = self.results[iso][scaler]['shift_iso-{}'.format(ref[:2])]['d_stat']
                iso_shift_d_syst = self.results[iso][scaler]['shift_iso-{}'.format(ref[:2])]['d_syst']
            if iso[:2] == ref[:2]:
                # this is the reference! All values zero!
                for indx, file in enumerate(files):
                    delta_rms.append(0)
                    delta_rms_d.append(0)
                avg_delta_rms = 0
                avg_delta_rms_d = 0
            else:
                for indx, file in enumerate(files):
                    # calculate radius
                    delta_rms.append(mu * ((iso_shift[indx] / mu - M_alpha) / F + alpha))
                    delta_rms_d.append(np.sqrt(np.square(mu_d * (alpha - M_alpha / F))
                                               + np.square(M_alpha_d * mu / F)
                                               + np.square(F_d * (iso_shift[indx] - mu * M_alpha) / F ** 2)
                                               + np.square((iso_shift_d[indx] + iso_shift_d_syst[indx]) / F)
                                               ))

                # get average isoshift
                avg_iso_shift = self.results[iso][scaler]['avg_shift_iso-{}'.format(ref[:2])]['vals'][0]
                avg_iso_shift_d = self.results[iso][scaler]['avg_shift_iso-{}'.format(ref[:2])]['d_stat'][0]
                avg_iso_shift_d_syst = self.results[iso][scaler]['avg_shift_iso-{}'.format(ref[:2])]['d_syst'][0]
                # calculate average radius
                avg_delta_rms = mu * ((avg_iso_shift / mu - M_alpha) / F + alpha)
                avg_delta_rms_d = np.sqrt(np.square(mu_d * (alpha - M_alpha / F))
                                          + np.square(M_alpha_d * mu / F)
                                          + np.square(F_d * (avg_iso_shift - mu * M_alpha) / F ** 2)
                                          + np.square((avg_iso_shift_d + avg_iso_shift_d_syst) / F))

        else:
            # use given shift or extract isotope shift from db where no scaler is specified.
            if isoshift is not None:
                iso_shift = isoshift[0]
                iso_shift_d = isoshift[1]
                iso_shift_d_syst = 0
            else:
                par = 'shift'
                iso_shift, iso_shift_d, iso_shift_d_syst = self.get_iso_property_from_db(
                    '''SELECT val, statErr, systErr from Combined WHERE iso = ? AND run = ? AND parname = ?''',
                    (iso, self.run, par))
            # calculate radius
            avg_delta_rms = mu * ((iso_shift / mu - M_alpha) / F + alpha)
            avg_delta_rms_d = np.sqrt(np.square(mu_d * (alpha - M_alpha / F))
                                      + np.square(M_alpha_d * mu / F)
                                      + np.square(F_d * (iso_shift - mu * M_alpha) / F ** 2)
                                      + np.square((iso_shift_d + iso_shift_d_syst) / F))
            # list of delta_rms's doesn't make much sense... Return one anyways
            delta_rms.append(avg_delta_rms)
            delta_rms_d.append(avg_delta_rms_d)

        return delta_rms, delta_rms_d, avg_delta_rms, avg_delta_rms_d

    def ni55_A_B_analysis(self, isotope, scaler='scaler_012', calibrated=False):
        if calibrated:
            isotope = '{}_cal'.format(isotope)  # make sure to use the calibrated isos

        # copy the results dict
        res_dict = self.results[isotope]

        # update the scaler
        scaler = self.update_scalers_in_db(scaler)

        # Get all coefficients from fit results
        Al = res_dict[scaler]['hfs_pars']['Al'][0]
        Al_d = res_dict[scaler]['hfs_pars']['Al'][1]
        Au = res_dict[scaler]['hfs_pars']['Au'][0]
        Au_d = res_dict[scaler]['hfs_pars']['Au'][1]
        A_rat = res_dict[scaler]['hfs_pars']['Arat'][0]
        A_rat_d = res_dict[scaler]['hfs_pars']['Arat'][1]
        A_rat_fixed = res_dict[scaler]['hfs_pars']['Arat'][2]

        Bl = res_dict[scaler]['hfs_pars']['Bl'][0]
        Bl_d = res_dict[scaler]['hfs_pars']['Bl'][1]
        Bu = res_dict[scaler]['hfs_pars']['Bu'][0]
        Bu_d = res_dict[scaler]['hfs_pars']['Bu'][1]
        B_rat = res_dict[scaler]['hfs_pars']['Brat'][0]
        B_rat_d = res_dict[scaler]['hfs_pars']['Brat'][1]
        B_rat_fixed = res_dict[scaler]['hfs_pars']['Brat'][2]

        # calculate µ and Q values
        # reference moments stored in format: (IsoMass_A, IsoSpin_I, IsoDipMom_µ, IsoDipMomErr_µerr, IsoQuadMom_Q, IsoQuadMomErr_Qerr)
        m_ref, I_ref, mu_ref, mu_ref_d, Q_ref, Q_ref_d = self.nuclear_spin_and_moments['61Ni']
        m_55, I_55, mu_55, mu_55_d, Q_55, Q_55_d = self.nuclear_spin_and_moments['55Ni']
        # reference A and B factors stored in format: (Al, Al_d, Au, Au_d, Arat, Arat_d, Bl, Bl_d, Bu, Bu_d, Brat, Brat_d)
        Al_ref, Al_d_ref, Au_ref, Au_d_ref, Arat_ref, Arat_d_ref, Bl_ref, Bl_d_ref, Bu_ref, Bu_d_ref, Brat_ref, Brat_d_ref = self.reference_A_B_vals['61Ni']

        # magnetic dipole moment
        mu_55 = mu_ref * Al/Al_ref * I_55/I_ref
        mu_55_d = np.sqrt((mu_ref_d * Al/Al_ref*I_55/I_ref)**2 + (Al_d * mu_ref/Al_ref*I_55/I_ref)**2 + (Al_d_ref * mu_ref*Al/Al_ref**2*I_55/I_ref)**2)
        # electric quadrupole moment
        Q_55 = Q_ref * Bl/Bl_ref
        Q_55_d = np.sqrt((Q_ref_d*Bl/Bl_ref)**2 + (Bl_d*Q_ref/Bl_ref)**2 + (Bl_d_ref*Bl*Q_ref/Bl_ref**2)**2)
        logging.info('\nspectroscopic factors: Al={0:.0f}({1:.0f}), Au={2:.0f}({3:.0f}), Arat={4:.3f}({5:.0f}),'
                     ' Bl={6:.0f}({7:.0f}), Bu={8:.0f}({9:.0f}), Brat={10:.3f}({11:.0f})'
                     .format(Al, Al_d, Au, Au_d, A_rat, A_rat_d*1000, Bl, Bl_d, Bu, Bu_d, B_rat, B_rat_d*1000))
        logging.info('\nmu55 = {0:.3f}({1:.0f}), Q55 = {2:.3f}({3:.0f})'
                     .format(mu_55, mu_55_d*1000, Q_55, Q_55_d*1000))

        # write to results dict
        moments_dict = {'mu': {'vals': [mu_55],
                               'd_stat': [0],
                               'd_syst': [mu_55_d]},
                        'Q': {'vals': [Q_55],
                              'd_stat': [0],
                              'd_syst': [Q_55_d]}
                        }
        self.results[isotope][scaler]['moments'] = moments_dict

    ''' Results and Plotting '''

    def calc_weighted_avg(self, values, uncertainties):
        """
        Based on 'Bevington - Data Reduction and Error Analysis for the Physical Sciences'
        :param values:
        :param uncertainties:
        :return:
        """
        x = np.asarray(values)
        n = x.__len__()
        x_d = np.asarray(uncertainties)
        # calculate weights inversely proportional to the square of the uncertainties
        if not any(x_d == 0):
            w = 1 / np.square(x_d)
        else:
            logging.warning('ZERO value in uncertainties found during weighted average calculation. '
                            'Calculating mean and standard deviation instead of weighting!')
            return x.mean(), x.std(), x.std(), x.std(), x.std() / np.sqrt(n)

        if n > 1:  # only makes sense for more than one data point. n=1 will also lead to div0 error
            # calculate weighted average and sum of weights:
            wavg = np.sum(x * w) / np.sum(w)  # (Bevington 4.17)
            # calculate the uncertainty of the weighted mean
            wavg_d = np.sqrt(1 / np.sum(w))  # (Bevington 4.19)

            # calculate weighted average variance
            wvar = np.sum(w * np.square(x - wavg)) / np.sum(w) * n / (n - 1)  # (Bevington 4.22)
            # calculate weighted standard deviations
            wstd = np.sqrt(wvar / n)  # (Bevington 4.23)

            # calculate (non weighted) standard deviations from the weighted mean
            std = np.sqrt(np.sum(np.square(x - wavg)) / (n - 1))
            # calculate the standard deviation of the average
            std_avg = std / np.sqrt(n)

        else:  # for only one value, return that value
            wavg = x[0]
            # use the single value uncertainty for all error estimates
            wavg_d = x_d[0]
            wstd = x_d[0]
            std = x_d[0]
            std_avg = x_d[0]

        return wavg, wavg_d, wstd, std, std_avg

    def make_results_dict_scaler(self,
                                 centers, centers_d_fit, centers_d_stat, center_d_syst, fitpars, rChi, hfs_pars=None):
        # calculate weighted average of center parameter
        wavg, wavg_d, wstd, std, std_avg = self.calc_weighted_avg(centers, centers_d_stat)
        if self.combined_unc == 'wavg_d':
            d_fit = wavg_d
        elif self.combined_unc == 'wstd':
            d_fit = wstd
        elif self.combined_unc == 'std_avg':
            d_fit = std_avg
        else:
            d_fit = std

        ret_dict = {'center_fits':
                        {'vals': centers,
                         'd_fit': centers_d_fit,
                         'd_stat': centers_d_stat,
                         'd_syst': center_d_syst
                         },
                    'avg_center_fits':
                        {'vals': [wavg],
                         'd_fit': [d_fit],
                         'd_stat': [d_fit],
                         'd_syst': [center_d_syst[0]]
                         },
                    'rChi':
                        {'vals': rChi
                         },
                    'all_fitpars': fitpars
                    }
        if hfs_pars is not None:
            ret_dict['hfs_pars'] = hfs_pars
        return ret_dict

    def export_results(self):
        ###################
        # export results  #
        ###################
        to_file_dict = {}
        # if there are analysis parameters stored, delete them. Will be written new.
        try:
            del to_file_dict['analysis_parameters']
        except:
            pass
        # iterate over copy of self.results (so we don't work on the original in case it's still needed)
        copydict = TiTs.deepcopy(self.results)
        for keys, vals in copydict.items():
            # xml cannot take numbers as first letter of key
            if not type(vals['file_times'][0]) == type('string'):
                vals['file_times'] = [datetime.strftime(t, '%Y-%m-%d %H:%M:%S') for t in vals['file_times']]
            to_file_dict['i' + keys] = vals
        # add analysis parameters
        to_file_dict['analysis_parameters'] = self.analysis_parameters
        results_file = self.results_name + '.xml'
        self.writeXMLfromDict(to_file_dict, os.path.join(self.resultsdir, results_file), 'BECOLA_Analysis')

    def import_results(self, results_xml, is_gate_analysis=False):
        results_name = results_xml[:-4]  # cut .xml from the end
        results_path_ext = 'results\\' + results_name + '\\' + results_name + '.xml'
        results_path = os.path.join(self.workdir, results_path_ext)
        ele = TiTs.load_xml(results_path)
        res_dict = TiTs.xml_get_dict_from_ele(ele)[1]
        # evaluate strings in dict
        res_dict = TiTs.evaluate_strings_in_dict(res_dict)
        # remove 'analysis_paramters' from dict
        del res_dict['analysis_parameters']
        # stored dict has 'i' in front of isotopes. Remove that again!
        for keys, vals in res_dict.items():
            # xml cannot take numbers as first letter of key but dicts can
            if keys[0] == 'i':
                vals['file_times'] = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t in vals['file_times']]
                res_dict[keys[1:]] = vals
                del res_dict[keys]
        if is_gate_analysis:
            TiTs.merge_extend_dicts(self.gate_analysis_res, res_dict, overwrite=True, force_overwrite=True)  # Merge dicts. Prefer new content
        else:
            TiTs.merge_extend_dicts(self.results, res_dict, overwrite=True, force_overwrite=True)  # Merge dicts. Prefer new content

    def plot_parameter_for_isos_and_scaler(self, isotopes, scaler_list, parameter,
                                           offset=None, overlay=None, unit='MHz', onlyfiterrs=False,
                                           digits=1 , factor=1, plotAvg=False, plotstyle='band', folder=None):
        """
        Make a nice plot of the center fit frequencies with statistical errors for the given isotopes
        For better readability an offset can be specified
        :param isotopes: list of str: names of isotopes as list
        :param scaler: list of int or str: scaler number as int or in string in format 'scaler_0'
        :param offset: list or True: optional. If given a list, this list must match the list of isotopes with an offset
         for each. If TRUE, offset will be extracted from results avg_shift
        :param plotstyle: str: either 'band' for a band of errors connecting the datapoints or 'classic' for just errorbars
        :return:
        """
        fig, ax = plt.subplots()
        x_type = 'file_numbers'  # alternative: 'file_numbers', 'file_times'
        scaler_nums = []
        for sc in scaler_list:
            scaler = self.update_scalers_in_db(sc)
            if '_' in scaler:
                scaler_nums.append(scaler.split('_')[1])
            else:
                scaler_nums.append(scaler)
            for i in range(len(isotopes)):
                iso = isotopes[i]
                x_ax = self.results[iso][x_type]
                if 'all_fitpars' in parameter:
                    # the 'all fitpars is organized a little different.
                    # For each file they are just stored as a dict like in db
                    # Parameter must be specified as 'all_fitpars:par' with par being the specific parameter to plot
                    fitres_list = self.results[iso][scaler]['all_fitpars']
                    parameter_plot = parameter.split(':')[1]
                    centers = factor*np.array([i[parameter_plot][0] for i in fitres_list])
                    centers_d_stat = factor*np.array([i[parameter_plot][1] for i in fitres_list])
                    centers_d_syst = factor*np.array([0 for i in fitres_list])
                    # get weighted average
                    wavg, wavg_d, fixed = self.results[iso][scaler]['avg_fitpars'][parameter_plot]
                    wavg = factor*wavg
                    wavg_d = factor*wavg_d
                    if fixed == True:
                        wavg_d = '-'
                    else:
                        wavg_d = '{:.0f}'.format(10 * wavg_d)  # times 10 for representation in brackets
                else:
                    centers = factor*np.array(self.results[iso][scaler][parameter]['vals'])
                    zero_arr = np.zeros(len(centers))  # prepare zero array with legth of centers in case no errors are given
                    if onlyfiterrs:
                        centers_d_stat = factor*np.array(self.results[iso][scaler][parameter].get('d_fit', zero_arr))
                        centers_d_syst = 0
                    else:
                        centers_d_stat = factor*np.array(self.results[iso][scaler][parameter].get('d_stat', zero_arr))
                        centers_d_syst = factor*np.array(self.results[iso][scaler][parameter].get('d_syst', zero_arr))
                    # calculate weighted average:
                    if not np.any(centers_d_stat == 0) and not np.sum(1/centers_d_stat**2) == 0:
                        d = centers.std()
                        wavg, wavg_d, wstd, std, std_avg = self.calc_weighted_avg(centers, centers_d_stat)
                        if self.combined_unc == 'wavg_d':
                            d = wavg_d
                        elif self.combined_unc == 'wstd':
                            d = wstd
                        elif self.combined_unc == 'std_avg':
                            d = std_avg
                        else:
                            d = std
                        wavg_d = '{:.0f}'.format(10**digits*d)  # times 10 for representation in brackets
                    else:  # some values don't have error, just calculate mean instead of weighted avg
                        wavg = centers.mean()
                        wavg_d = '-'
                # determine color
                if len(scaler_list) > 1:
                    # if there is more than one scaler, color determined by scaler
                    col = self.scaler_colors[scaler]
                    labelstr = 'pmt{}'.format(scaler.split('_')[-1])
                else:
                    # for only one scaler, color detremined by isotope
                    col = self.results[iso]['color']
                    labelstr = iso
                off = 0
                if offset is True:
                    avg_shift = -self.results[iso][scaler]['avg_shift_iso-58'][0]
                    off = round(avg_shift, -1)
                elif type(offset) in (list, tuple):
                    # offset might be given manually per isotope
                    off = offset[i]
                # plot center frequencies in MHz:
                if off != 0:
                    plt_label = '{0} {1:.{6:d}f}({2}){3} (offset: {4}{5})'\
                        .format(labelstr, wavg, wavg_d,  unit, off, unit, digits)
                else:
                    plt_label = '{0} {1:.{4:d}f}({2}){3}'\
                        .format(labelstr, wavg, wavg_d, unit, digits)
                if plotstyle == 'band':
                    # plot values as points
                    plt.plot(x_ax, np.array(centers) + off, '--o', color=col, label=plt_label)
                    # plot error band for statistical errors
                    plt.fill_between(x_ax,
                                     np.array(centers) + off - centers_d_stat,
                                     np.array(centers) + off + centers_d_stat,
                                     alpha=0.5, edgecolor=col, facecolor=col)
                    # plot error band for systematic errors on top of statistical errors
                    plt.fill_between(x_ax,
                                     np.array(centers) + off - centers_d_syst - centers_d_stat,
                                     np.array(centers) + off + centers_d_syst + centers_d_stat,
                                     alpha=0.2, edgecolor=col, facecolor=col)
                else:
                    # plot values as dots with statistical errorbars
                    plt.errorbar(x_ax, np.array(centers), yerr=np.array(centers_d_stat), fmt='.', color=col, label=plt_label)
                    # plot error band for systematic errors on top of statistical errors
                    # if not np.all(np.asarray(centers_d_syst) == 0):
                    #     plt.fill_between(x_ax,
                    #                      np.array(centers) + off - centers_d_syst,
                    #                      np.array(centers) + off + centers_d_syst,
                    #                      alpha=0.2, edgecolor=col, facecolor=col)
                if plotAvg:
                    avg_parameter = 'avg_{}'.format(parameter)
                    # also plot average isotope shift
                    avg_shift = self.results[iso][scaler][avg_parameter]['vals'][0]
                    avg_shift_d = self.results[iso][scaler][avg_parameter]['d_stat'][0]
                    avg_shift_d_syst = self.results[iso][scaler][avg_parameter]['d_syst'][0]
                    # plot weighted average as red line
                    plt.plot([x_ax[0], x_ax[-1]], [avg_shift, avg_shift], 'r',
                             label='{0} avg: {1:.{5:d}f}({2:.0f})[{3:.0f}]{4}'
                             .format(iso, avg_shift, 10**digits*avg_shift_d, 10**digits*avg_shift_d_syst, unit, digits))
                    # plot error of weighted average as red shaded box around that line
                    plt.fill([x_ax[0], x_ax[-1], x_ax[-1], x_ax[0]],
                             [avg_shift - avg_shift_d, avg_shift - avg_shift_d,
                              avg_shift + avg_shift_d, avg_shift + avg_shift_d], 'r',
                             alpha=0.2)
                    # plot systematic error as lighter red shaded box around that line
                    plt.fill([x_ax[0], x_ax[-1], x_ax[-1], x_ax[0]],
                             [avg_shift - avg_shift_d_syst-avg_shift_d, avg_shift - avg_shift_d_syst-avg_shift_d,
                              avg_shift + avg_shift_d_syst+avg_shift_d, avg_shift + avg_shift_d_syst+avg_shift_d], 'r',
                             alpha=0.1)
        if overlay is not None:
            plt.axhline(y=overlay, color='red')
        if x_type == 'file_times':
            plt.xlabel('date')
            days_fmt = mpdate.DateFormatter('%d.%B')
            ax.xaxis.set_major_formatter(days_fmt)
        else:
            plt.xlabel('run numbers')
        plt.xticks(rotation=45)
        plt.ylabel('{} [{}]'.format(parameter, unit))
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        plt.title('{} in {} for isotopes: {}'.format(parameter, unit, isotopes))
        plt.legend(title='Legend', bbox_to_anchor=(1.04, 0.5), loc="center left")
        plt.margins(0.05)
        if self.save_plots_to_file:
            isonums = []
            summed = ''
            for isos in isotopes:
                if 'cal' in isos:
                    isonums.append(isos[:2]+'c')
                else:
                    isonums.append(isos[:2])
                if 'sum' in isos:
                    summed = 'sum'
            parameter = parameter.replace(':', '_')  # colon is no good filename char
            if 'final' in scaler_list:
                parameter = '0_' + parameter
            filename = parameter + '_' + ''.join(isonums) + summed + '_sc' + 'a'.join(scaler_nums)
            path_add = ''
            if folder is not None:
                path_add = '{}\\'.format(folder)
            plot_folder = self.resultsdir + path_add
            if not os.path.exists(plot_folder):
                os.makedirs(plot_folder)
            plt.savefig(plot_folder + filename + '.png', bbox_inches="tight")
        else:
            plt.show()
        plt.close()
        plt.clf()

    def plot_parameter_for_isos_vs_scaler(self, isotopes, scaler_list, parameter,
                                           offset=None, overlay=None, unit='MHz', onlyfiterrs=False,
                                          digits=1, factor=1, folder=None):
        """
        Make a nice plot of the center fit frequencies with statistical errors for the given isotopes
        For better readability an offset can be specified
        :param isotopes: list of str: names of isotopes as list
        :param scaler: list of int or str: scaler number as int or in string in format 'scaler_0'
        :param offset: list or True: optional. If given a list, this list must match the list of isotopes with an offset
         for each. If TRUE, offset will be extracted from results avg_shift
        :return:
        """
        fig, ax = plt.subplots()
        x_type = 'scaler combination'
        x_ax = np.arange(len(scaler_list))
        scaler_nums = []
        for i in range(len(isotopes)):
            iso = isotopes[i]
            centers = np.zeros(len(scaler_list))
            centers_d_stat = np.zeros(len(scaler_list))
            centers_d_syst = np.zeros(len(scaler_list))
            for indx, sc in enumerate(scaler_list):
                scaler = self.update_scalers_in_db(sc)
                if '_' in scaler:
                    scaler_nums.append(scaler.split('_')[1])
                else:
                    scaler_nums.append(scaler)
                if 'all_fitpars' in parameter:
                    # the 'all fitpars is organized a little different.
                    # For each file they are just stored as a dict like in db
                    # Parameter must be specified as 'all_fitpars:par' with par being the specific parameter to plot
                    fitres_list = self.results[iso][scaler]['all_fitpars']
                    parameter_plot = parameter.split(':')[1]
                    centers[indx] = factor*np.array([i[parameter_plot][0] for i in fitres_list])[0]
                    centers_d_stat[indx] = factor*np.array([i[parameter_plot][1] for i in fitres_list])[0]
                    centers_d_syst[indx] = factor*np.array([0 for i in fitres_list])[0]
                else:
                    centers[indx] = factor*np.array(self.results[iso][scaler][parameter]['vals'])
                    if onlyfiterrs:
                        centers_d_stat[indx] = factor*np.array(self.results[iso][scaler][parameter].get('d_fit',[0]))[0]
                        centers_d_syst[indx] = 0
                    else:
                        centers_d_stat[indx] = factor*np.array(self.results[iso][scaler][parameter].get('d_stat',[0]))[0]
                        centers_d_syst[indx] = factor*np.array(self.results[iso][scaler][parameter].get('d_syst',[0]))[0]
            # calculate weighted average:
            if not np.any(centers_d_stat == 0) and not np.sum(1/centers_d_stat**2) == 0:
                d = centers.std()
                wavg, wavg_d, wstd, std, std_avg = self.calc_weighted_avg(centers, centers_d_stat)
                if self.combined_unc == 'wavg_d':
                    d = wavg_d
                elif self.combined_unc == 'wstd':
                    d = wstd
                elif self.combined_unc == 'std_avg':
                    d = std_avg
                else:
                    d = std
                wavg_d = '{:.0f}'.format(10**digits*d)  # times 10 for representation in brackets
            else:  # some values don't have error, just calculate mean instead of weighted avg
                wavg = centers.mean()
                wavg_d = '-'

            # color detremined by isotope
            col = self.results[iso]['color']
            labelstr = iso

            off = 0
            if offset is True:
                off = round(-wavg, -1)
            elif type(offset) in (list, tuple):
                # offset might be given manually per isotope
                off = offset[i]
            # plot center frequencies in MHz:
            if off != 0:
                plt_label = '{0} {1:.{6:d}f}({2}){3} (offset: {4}{5})' \
                    .format(labelstr, wavg, wavg_d, unit, off, unit, digits)
            else:
                plt_label = '{0} {1:.{4:d}f}({2}){3}' \
                    .format(labelstr, wavg, wavg_d, unit, digits)
            plt.plot(x_ax, np.array(centers) + off, '--o', color=col, label=plt_label)
            # plot error band for statistical errors
            plt.fill_between(x_ax,
                             np.array(centers) + off - centers_d_stat,
                             np.array(centers) + off + centers_d_stat,
                             alpha=0.5, edgecolor=col, facecolor=col)
            # plot error band for systematic errors on top of statistical errors
            plt.fill_between(x_ax,
                             np.array(centers) + off - centers_d_syst - centers_d_stat,
                             np.array(centers) + off + centers_d_syst + centers_d_stat,
                             alpha=0.2, edgecolor=col, facecolor=col)
        if overlay is not None:
            plt.axhline(y=overlay, color='red')

        plt.xlabel(x_type)
        plt.xticks(x_ax, [self.update_scalers_in_db(sc) for sc in scaler_list], rotation=45)
        plt.ylabel('{} [{}]'.format(parameter, unit))
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        plt.title('{} in {} for isotopes: {}'.format(parameter, unit, isotopes))
        plt.legend(title='Legend', bbox_to_anchor=(1.04, 0.5), loc="center left")
        plt.margins(0.05)
        if self.save_plots_to_file:
            isonums = []
            summed = ''
            for isos in isotopes:
                if 'cal' in isos:
                    isonums.append(isos[:2] + 'c')
                else:
                    isonums.append(isos[:2])
                if 'sum' in isos:
                    summed = 'sum'
            parameter = parameter.replace(':', '_')  # colon is no good filename char
            if 'final' in scaler_list:
                parameter = '0_' + parameter
            filename = parameter + '_' + ''.join(isonums) + summed + '_sc' + 'a'.join(scaler_nums)
            path_add = ''
            if folder is not None:
                path_add = '{}\\'.format(folder)
            plot_folder = self.resultsdir + path_add
            if not os.path.exists(plot_folder):
                os.makedirs(plot_folder)
            plt.savefig(plot_folder + filename + '.png', bbox_inches="tight")
        else:
            plt.show()
        plt.close()
        plt.clf()

    def plot_parameter_for_isos_final(self, isotopes, parameter, overlay=None, unit='MHz', onlyfiterrs=False,
                                      factor=1, plotstyle='band', folder=None):
        """
        Make a nice plot of the center fit frequencies with statistical errors for the given isotopes
        For better readability an offset can be specified
        :param isotopes: list of str: names of isotopes as list
        :param scaler: list of int or str: scaler number as int or in string in format 'scaler_0'
        :param offset: list or True: optional. If given a list, this list must match the list of isotopes with an offset
         for each. If TRUE, offset will be extracted from results avg_shift
        :param plotstyle: str: either 'band' for a band of errors connecting the datapoints or 'classic' for just errorbars
        :return:
        """
        fig, ax = plt.subplots()
        x_type = 'isotopes'  # alternative: 'file_numbers', 'file_times'

        scaler = 'final'

        x_ax = [int(iso[:2]) for iso in isotopes]

        centers = factor*np.array([self.results[iso][scaler][parameter]['vals'][0] for iso in isotopes])
        if onlyfiterrs:
            centers_d_stat = factor*np.array([self.results[iso][scaler][parameter].get('d_fit', [0])[0] for iso in isotopes])
            centers_d_syst = 0
        else:
            centers_d_stat = factor*np.array([self.results[iso][scaler][parameter].get('d_stat', [0])[0] for iso in isotopes])
            centers_d_syst = factor*np.array([self.results[iso][scaler][parameter].get('d_syst', [0])[0] for iso in isotopes])

        col = 'black'

        if plotstyle == 'band':
            # plot values as points
            plt.plot(x_ax, np.array(centers), '--o', color=col)
            # plot error band for statistical errors
            plt.fill_between(x_ax,
                             np.array(centers) - centers_d_stat,
                             np.array(centers) + centers_d_stat,
                             alpha=0.5, edgecolor=col, facecolor=col)
            # plot error band for systematic errors on top of statistical errors
            plt.fill_between(x_ax,
                             np.array(centers) - centers_d_syst - centers_d_stat,
                             np.array(centers) + centers_d_syst + centers_d_stat,
                             alpha=0.2, edgecolor=col, facecolor=col)
        else:
            # plot values as points with statistical errorbars. Dashed line in between
            plt.errorbar(x_ax, np.array(centers), yerr=np.array(centers_d_stat), fmt='--o', color=col)
            # plot error band for systematic errors on top of statistical errors
            # if not np.all(np.asarray(centers_d_syst) == 0):
            #     plt.fill_between(x_ax,
            #                      np.array(centers) + off - centers_d_syst,
            #                      np.array(centers) + off + centers_d_syst,
            #                      alpha=0.2, edgecolor=col, facecolor=col)

        if overlay is not None:
            plt.axhline(y=overlay, color='red')

        plt.xlabel('Isotope')
        plt.xticks(rotation=45)
        plt.ylabel('{} [{}]'.format(parameter, unit))
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        plt.title('{} in {} for isotopes: {}'.format(parameter, unit, isotopes))
        plt.legend(title='Legend', bbox_to_anchor=(1.04, 0.5), loc="center left")
        plt.margins(0.05)
        if self.save_plots_to_file:
            parameter = parameter.replace(':', '_')  # colon is no good filename char
            filename = '0_' + parameter + '_' + ''.join([str(i) for i in x_ax]) + '_final'
            path_add = ''
            if folder is not None:
                path_add = '{}\\'.format(folder)
            plot_folder = self.resultsdir + path_add
            if not os.path.exists(plot_folder):
                os.makedirs(plot_folder)
            plt.savefig(plot_folder + filename + '.png', bbox_inches="tight")
        else:
            plt.show()
        plt.close()
        plt.clf()

    ''' Final Plotting '''
    def make_final_plots(self):
        # centroid fit results
        isolist_cal = ['{}_cal'.format(iso) for iso in self.isotopes_single]
        shiftslist_cal = isolist_cal.copy()
        shiftslist_cal.remove('{}_cal'.format(self.ref_iso))

        final_isos = self.all_isotopes
        # isotope shifts
        self.plot_shifts_chain(final_isos, self.ref_iso, 'final', dash_missing_data=True, tip_scale=250)
        # radii
        self.plot_radii_chain(final_isos, self.ref_iso, 'final', dash_missing_data=True)

    def plot_centroids(self, isolist, scaler_list, overlay=None, unit='MHz', onlyfiterrs=False,
                                           digits=1 , plotAvg=False, plotSyst=False):
        parameter = 'center_fits'
        # make one separate plot for each isotope
        for iso in isolist:
            fig, ax1 = plt.subplots()

            x_type = 'file_times'  # alternative: 'file_numbers', 'file_times'
            x_ax = self.results[iso][x_type]
            file_numbers = self.results[iso]['file_numbers']  # for use as secondary axis

            for sc in scaler_list:
                scaler = self.update_scalers_in_db(sc)

                # get the values
                centers = np.array(self.results[iso][scaler][parameter]['vals'])
                zero_arr = np.zeros(
                    len(centers))  # prepare zero array with legth of centers in case no errors are given
                if onlyfiterrs:
                    centers_d_stat = np.array(self.results[iso][scaler][parameter].get('d_fit', zero_arr))
                    centers_d_syst = 0
                else:
                    centers_d_stat = np.array(self.results[iso][scaler][parameter].get('d_stat', zero_arr))
                    centers_d_syst = np.array(self.results[iso][scaler][parameter].get('d_syst', zero_arr))

                # calculate weighted average:
                if not np.any(centers_d_stat == 0) and not np.sum(1 / centers_d_stat ** 2) == 0:
                    d = centers.std()
                    wavg, wavg_d, wstd, std, std_avg = self.calc_weighted_avg(centers, centers_d_stat)
                    if self.combined_unc == 'wavg_d':
                        d = wavg_d
                    elif self.combined_unc == 'wstd':
                        d = wstd
                    elif self.combined_unc == 'std_avg':
                        d = std_avg
                    else:
                        d = std
                    wavg_d = '{:.0f}'.format(10 ** digits * d)  # times 10 for representation in brackets
                else:  # some values don't have error, just calculate mean instead of weighted avg
                    wavg = centers.mean()
                    wavg_d = '-'

                # determine color by scaler
                col = self.scaler_colors[scaler]
                labelstr = 'scaler {}'.format(scaler.split('_')[-1])
                if scaler == 'scaler_c012':
                    # for the combined scaler use color determined by isotope
                    col = self.results[iso]['color']
                    labelstr = 'wAvg scalers'

                plt_label = '{0} {1:.{4:d}f}({2}){3}' \
                    .format(labelstr, wavg, wavg_d, unit, digits)

                # Do the plotting
                if '012' in scaler:
                    # plot values as dotted line
                    ax1.plot(x_ax, np.array(centers), '--', color=col)
                    # plot error band for statistical errors
                    ax1.fill_between(x_ax,
                                     np.array(centers) - centers_d_stat,
                                     np.array(centers) + centers_d_stat,
                                     alpha=0.5, edgecolor=col, facecolor=col,
                                     label=plt_label)
                else:
                    # plot values as dots with statistical errorbars
                    ax1.errorbar(x_ax, np.array(centers), yerr=np.array(centers_d_stat), fmt='.', color=col,
                                 label=plt_label)

                # For the combined scaler, also plot the weighted avg over all scalers
                if scaler == 'scaler_c012' and plotAvg:
                    avg_parameter = 'avg_{}'.format(parameter)
                    # also plot average isotope shift
                    avg_shift = self.results[iso][scaler][avg_parameter]['vals'][0]
                    avg_shift_d = self.results[iso][scaler][avg_parameter]['d_stat'][0]
                    avg_shift_d_syst = self.results[iso][scaler][avg_parameter]['d_syst'][0]
                    # plot weighted average as red line
                    labelstr = 'wAvg files'
                    ax1.plot([x_ax[0], x_ax[-1]], [avg_shift, avg_shift], 'red')
                    # plot error of weighted average as red shaded box around that line
                    ax1.fill([x_ax[0], x_ax[-1], x_ax[-1], x_ax[0]],
                             [avg_shift - avg_shift_d, avg_shift - avg_shift_d,
                              avg_shift + avg_shift_d, avg_shift + avg_shift_d], 'red',
                             alpha=0.2,
                             label='{0}: {1:.{5:d}f}({2:.0f})[{3:.0f}]{4}'
                             .format(labelstr, avg_shift, 10 ** digits * avg_shift_d, 10 ** digits * avg_shift_d_syst,
                                     unit, digits))
                    if plotSyst:
                        # plot systematic error as lighter red shaded box around that line
                        ax1.fill([x_ax[0], x_ax[-1], x_ax[-1], x_ax[0]],
                                 [avg_shift - avg_shift_d_syst - avg_shift_d, avg_shift - avg_shift_d_syst - avg_shift_d,
                                  avg_shift + avg_shift_d_syst + avg_shift_d, avg_shift + avg_shift_d_syst + avg_shift_d],
                                 'red',
                                 alpha=0.1)

            # if any overlay was specified, plot as a red line
            if overlay is not None:
                plt.axhline(y=overlay, color='red')

            # work on the axes
            ax1.margins(0.05)
            if x_type == 'file_times':
                # create primary axis with the dates
                hours_fmt = mpdate.DateFormatter('%Hh')
                ax1.xaxis.set_major_formatter(hours_fmt)
                # create a days axis
                ax_day = ax1.twiny()
                ax_day.xaxis.set_ticks_position("bottom")
                ax_day.xaxis.set_label_position("bottom")
                ax_day.spines["bottom"].set_position(("axes", -0.07))  # Offset the days axis below the hours
                alldates = [datetime(2018, 4, d, 0, 0, 0) for d in range(13, 24, 1)]
                ax_day.set_xticks(alldates)  # same tick locations
                ax_day.set_xbound(ax1.get_xbound())
                days_fmt = mpdate.DateFormatter('%d.%b')
                ax_day.xaxis.set_major_formatter(days_fmt)
                ax_day.set_xlabel('date and time')
                # create a secondary axis with the run numbers
                ax_num = ax1.twiny()
                ax_num.set_xlabel('data set numbers')
                ax_num.set_xticks(x_ax)  # same tick locations
                ax_num.set_xbound(ax1.get_xbound())  # same axis range
                ax_num.set_xticklabels(file_numbers, rotation=90)
            else:
                plt.xlabel('run number')
            ax1.set_ylabel('centroid /{}'.format(unit))
            ax1.get_yaxis().get_major_formatter().set_useOffset(False)

            # create labels and title
            cal = ''
            if 'cal' in iso:
                cal = 'calibrated '
            title = ax1.set_title('Centroids in {} for all data sets of {}{}'.format(unit, cal, iso[:4]))
            title.set_y(1.2)
            fig.subplots_adjust(top=0.85)
            ax1.legend(title='Scaler', bbox_to_anchor=(1.04, 0.5), loc="center left")
            if self.save_plots_to_file:
                parameter = parameter.replace(':', '_')  # colon is no good filename char
                if 'final' in scaler_list:
                    parameter = '0_' + parameter
                if '_c' in scaler_list[0]:
                    scaler = '_c'
                else:
                    scaler = ''
                filename = parameter + '_' + iso + scaler
                plt.savefig(self.resultsdir + filename + '.png', bbox_inches="tight")
            else:
                plt.show()
            plt.close()
            plt.clf()

    def plot_shifts(self, isolist, scaler_list, overlay=None, unit='MHz', onlyfiterrs=False,
                                           digits=1 , plotAvg=False, plotSyst=False):
        parameter = 'shift_iso-{}'.format(self.ref_iso[:2])
        # make one separate plot for each isotope
        for iso in isolist:
            fig, ax1 = plt.subplots()

            x_type = 'file_times'  # alternative: 'file_numbers', 'file_times'
            x_ax = self.results[iso][x_type]
            file_numbers = self.results[iso]['file_numbers']  # for use as secondary axis

            for sc in scaler_list:
                scaler = self.update_scalers_in_db(sc)

                # get the values
                centers = np.array(self.results[iso][scaler][parameter]['vals'])
                zero_arr = np.zeros(
                    len(centers))  # prepare zero array with legth of centers in case no errors are given
                if onlyfiterrs:
                    centers_d_stat = np.array(self.results[iso][scaler][parameter].get('d_fit', zero_arr))
                    centers_d_syst = 0
                else:
                    centers_d_stat = np.array(self.results[iso][scaler][parameter].get('d_stat', zero_arr))
                    centers_d_syst = np.array(self.results[iso][scaler][parameter].get('d_syst', zero_arr))

                # calculate weighted average:
                if not np.any(centers_d_stat == 0) and not np.sum(1 / centers_d_stat ** 2) == 0:
                    d = centers.std()
                    wavg, wavg_d, wstd, std, std_avg = self.calc_weighted_avg(centers, centers_d_stat)
                    if self.combined_unc == 'wavg_d':
                        d = wavg_d
                    elif self.combined_unc == 'wstd':
                        d = wstd
                    elif self.combined_unc == 'std_avg':
                        d = std_avg
                    else:
                        d = std
                    wavg_d = '{:.0f}'.format(10 ** digits * d)  # times 10 for representation in brackets
                else:  # some values don't have error, just calculate mean instead of weighted avg
                    wavg = centers.mean()
                    wavg_d = '-'

                # determine color by scaler
                col = self.scaler_colors[scaler]
                labelstr = 'scaler {}'.format(scaler.split('_')[-1])
                if scaler == 'scaler_c012':
                    # for the combined scaler use color determined by isotope
                    col = self.results[iso]['color']
                    labelstr = 'wAvg scalers'

                plt_label = '{0} {1:.{4:d}f}({2}){3}' \
                    .format(labelstr, wavg, wavg_d, unit, digits)

                # Do the plotting
                if scaler == 'scaler_c012':
                    # plot values as points
                    ax1.plot(x_ax, np.array(centers), '--', color=col)
                    # plot error band for statistical errors
                    ax1.fill_between(x_ax,
                                     np.array(centers) - centers_d_stat,
                                     np.array(centers) + centers_d_stat,
                                     alpha=0.5, edgecolor=col, facecolor=col,
                                     label=plt_label)
                else:
                    # plot values as dots with statistical errorbars
                    ax1.errorbar(x_ax, np.array(centers), yerr=np.array(centers_d_stat), fmt='.', color=col,
                                 label=plt_label)

                # For the combined scaler, also plot the weighted avg over all scalers
                if scaler == 'scaler_c012' and plotAvg:
                    avg_parameter = 'avg_{}'.format(parameter)
                    # also plot average isotope shift
                    avg_shift = self.results[iso][scaler][avg_parameter]['vals'][0]
                    avg_shift_d = self.results[iso][scaler][avg_parameter]['d_stat'][0]
                    avg_shift_d_syst = self.results[iso][scaler][avg_parameter]['d_syst'][0]
                    # plot weighted average as red line
                    labelstr = 'wAvg files'
                    ax1.plot([x_ax[0], x_ax[-1]], [avg_shift, avg_shift], 'red')
                    # plot error of weighted average as red shaded box around that line
                    ax1.fill([x_ax[0], x_ax[-1], x_ax[-1], x_ax[0]],
                             [avg_shift - avg_shift_d, avg_shift - avg_shift_d,
                              avg_shift + avg_shift_d, avg_shift + avg_shift_d], 'red',
                             alpha=0.2,
                             label='{0}: {1:.{5:d}f}({2:.0f})[{3:.0f}]{4}'
                             .format(labelstr, avg_shift, 10 ** digits * avg_shift_d, 10 ** digits * avg_shift_d_syst,
                                     unit, digits))
                    if plotSyst:
                        # plot systematic error as lighter red shaded box around that line
                        ax1.fill([x_ax[0], x_ax[-1], x_ax[-1], x_ax[0]],
                                 [avg_shift - avg_shift_d_syst - avg_shift_d, avg_shift - avg_shift_d_syst - avg_shift_d,
                                  avg_shift + avg_shift_d_syst + avg_shift_d, avg_shift + avg_shift_d_syst + avg_shift_d],
                                 'red',
                                 alpha=0.1)

            # if any overlay was specified, plot as a red line
            if overlay is not None:
                plt.axhline(y=overlay, color='red')

            # work on the axes
            ax1.margins(0.05)
            if x_type == 'file_times':
                # create primary axis with the dates
                hours_fmt = mpdate.DateFormatter('%Hh')
                ax1.xaxis.set_major_formatter(hours_fmt)
                # create a days axis
                ax_day = ax1.twiny()
                ax_day.xaxis.set_ticks_position("bottom")
                ax_day.xaxis.set_label_position("bottom")
                ax_day.spines["bottom"].set_position(("axes", -0.07))  # Offset the days axis below the hours
                alldates = [datetime(2018, 4, d, 0, 0, 0) for d in range(13, 24, 1)]
                ax_day.set_xticks(alldates)  # same tick locations
                ax_day.set_xbound(ax1.get_xbound())
                days_fmt = mpdate.DateFormatter('%d.%b')
                ax_day.xaxis.set_major_formatter(days_fmt)
                ax_day.set_xlabel('date and time')
                # create a secondary axis with the run numbers
                ax_num = ax1.twiny()
                ax_num.set_xlabel('data set numbers')
                ax_num.set_xticks(x_ax)  # same tick locations
                ax_num.set_xbound(ax1.get_xbound())  # same axis range
                ax_num.set_xticklabels(file_numbers, rotation=90)
            else:
                plt.xlabel('run number')
            ax1.set_ylabel('isotope shift A-60 /{}'.format(unit))
            ax1.get_yaxis().get_major_formatter().set_useOffset(False)

            # create labels and title
            cal = ''
            if 'cal' in iso:
                cal = 'calibrated '
            title = ax1.set_title('Isotope shifts in {} for all data sets of {}{}'.format(unit, cal, iso[:4]))
            title.set_y(1.2)
            fig.subplots_adjust(top=0.85)
            ax1.legend(title='Scaler', bbox_to_anchor=(1.04, 0.5), loc="center left")
            if self.save_plots_to_file:
                parameter = parameter.replace(':', '_')  # colon is no good filename char
                if 'final' in scaler_list:
                    parameter = '0_' + parameter
                if '_c' in scaler_list[0]:
                    scaler = '_c'
                else:
                    scaler = ''
                filename = parameter + '_' + iso + scaler
                plt.savefig(self.resultsdir + filename + '.png', bbox_inches="tight")
            else:
                plt.show()
            plt.close()
            plt.clf()

    def plot_radii_chain(self, isolist, refiso, scaler, plot_evens_seperate=False, dash_missing_data=True, calibrated=False,
                   includelitvals=True):
        font_size = 12
        ref_key = refiso[:4]
        if scaler == 'final':
            rms_key = 'delta_rms_iso-{}'.format(refiso[:2])
        else:
            rms_key = 'avg_delta_rms_iso-{}'.format(refiso[:2])
        thisVals = {key: [self.results[key][scaler][rms_key]['vals'][0],
                           self.results[key][scaler][rms_key]['d_stat'][0]]
                     for key in isolist}
        thisVals[refiso] = [0, 0]
        col = ['r', 'b', 'k', 'g']

        data_dict = {'This Work': {'data': thisVals, 'color': 'red'}}
        src_list = []

        # get the literature values
        if includelitvals:
            for src, vals in self.delta_rms_lit.items():
                col = vals['color']
                litvals = TiTs.deepcopy(vals['data'])
                ref_val = litvals[ref_key]
                for iso, vals in litvals.items():
                    litvals[iso] = (vals[0] - ref_val[0], vals[1])
                data_dict[src] = {'data': litvals, 'color': col}
                src_list.append(src)

        # sort sources for appearance
        src_sorted = tuple(['This Work'] + sorted(src_list))

        # start plotting
        for src in src_sorted:
            col = data_dict[src]['color']
            data = data_dict[src]['data']
            keyVals = sorted(data)
            x = []
            y = []
            yerr = []
            for i in keyVals:
                x.append(int(''.join(filter(str.isdigit, i))))
                y.append(data[i][0])
                yerr.append(data[i][1])

            plt.xticks(rotation=0)
            ax = plt.gca()
            ax.set_ylabel(r'$\delta$ < r' + r'$^2$ > (fm $^2$) ', fontsize=font_size)
            ax.set_xlabel('A', fontsize=font_size)
            if plot_evens_seperate:
                x_odd = [each for each in x if each % 2 != 0]
                y_odd = [each for i, each in enumerate(y) if x[i] % 2 != 0]
                y_odd_err = [each for i, each in enumerate(yerr) if x[i] % 2 != 0]
                x_even = [each for each in x if each % 2 == 0]
                y_even = [each for i, each in enumerate(y) if x[i] % 2 == 0]
                y_even_err = [each for i, each in enumerate(yerr) if x[i] % 2 == 0]

                plt.errorbar(x_even, y_even, y_even_err, fmt='o', color=col, label='even', linestyle='-')
                plt.errorbar(x_odd, y_odd, y_odd_err, fmt='^', color=col, label='odd', linestyle='--')
                # plt.legend(loc=2)
            elif dash_missing_data:
                # if some isotopes are missing, this dashes the line at these isotopes
                # has no effect when plot_evens_separate is True
                split_x_list = []
                for k, g in groupby(enumerate(x), lambda a: a[0] - a[1]):
                    split_x_list.append(list(map(itemgetter(1), g)))
                i = 0
                label_created = False
                for each in split_x_list:
                    y_vals = y[i:i + len(each)]
                    yerr_vals = yerr[i:i + len(each)]
                    if not label_created:  # only create label once
                        plt.errorbar(each, y_vals, yerr_vals, fmt='o', color=col, linestyle='-', label=src)
                        label_created=True
                    else:
                        plt.errorbar(each, y_vals, yerr_vals, fmt='o', color=col, linestyle='-')
                    # plot dashed lines between missing values
                    if len(x) > i + len(each):  # might be last value
                        x_gap = [x[i + len(each) - 1], x[i + len(each)]]
                        y_gap = [y[i + len(each) - 1], y[i + len(each)]]
                        plt.plot(x_gap, y_gap, c=col, linestyle='--')
                    i = i + len(each)
            else:
                plt.errorbar(x, y, yerr, fmt='o', color=col, linestyle='-')
        ax.set_xmargin(0.05)
        plt.legend(loc='lower right')
        plt.margins(0.1)
        plt.gcf().set_facecolor('w')
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.tight_layout(True)
        if self.save_plots_to_file:
            isonums = []
            summed = ''
            for isos in isolist:
                if 'cal' in isos:
                    isonums.append(isos[:2] + 'c')
                else:
                    isonums.append(isos[:2])
                if 'sum' in isos:
                    summed = 'sum'
            parameter = 'charge_radii_{}ref'.format(ref_key)
            if scaler == 'final':
                parameter = '0_' + parameter
            filename = parameter + '_' + ''.join(isonums) + summed + '_sc' + scaler
            plt.savefig(self.resultsdir + filename + '.png', bbox_inches="tight")
        else:
            plt.show()
        plt.close()
        plt.clf()

    def plot_shifts_chain(self, isolist, refiso, scaler, plot_evens_seperate=False, dash_missing_data=True, calibrated=False,
                   includelitvals=True, tip_scale=0, relative=False):
        font_size = 12
        ref_key = refiso[:4]
        if scaler == 'final':
            rms_key = 'shift_iso-{}'.format(refiso[:2])
        else:
            rms_key = 'avg_shift_iso-{}'.format(refiso[:2])
        thisVals = {key: [self.results[key][scaler][rms_key]['vals'][0],
                          self.results[key][scaler][rms_key]['d_stat'][0]
                          + self.results[key][scaler][rms_key]['d_syst'][0]]
                    for key in isolist}
        thisVals[refiso] = [0, 0]

        data_dict = {'This Work': {'data': thisVals, 'color': 'red'}}
        src_list = []

        # get the literature values
        if includelitvals:
            for src, vals in self.iso_shifts_lit.items():
                col = vals['color']
                litvals = TiTs.deepcopy(vals['data'])
                ref_val = litvals[ref_key]
                for iso, vals in litvals.items():
                    litvals[iso] = (vals[0]-ref_val[0], vals[1])
                data_dict[src] = {'data': litvals, 'color': col}
                src_list.append(src)

        # sort sources for appearance
        src_sorted = tuple(['This Work'] + sorted(src_list))

        # start plotting
        for src in src_sorted:
            col = data_dict[src]['color']
            data = data_dict[src]['data']
            keyVals = sorted(data)
            x = []
            y = []
            yerr = []
            for num, i in enumerate(keyVals):
                isoint = int(''.join(filter(str.isdigit, i)))
                if relative:
                    # only really makes sense for consecutive values. Implemented it anyways.
                    if i != keyVals[-1]:
                        next = keyVals[num+1]
                        nextint = int(''.join(filter(str.isdigit, next)))
                        x.append(isoint)
                        y.append((data[i][0] - data[next][0])/(nextint-isoint))
                        yerr.append(np.sqrt(data[i][1]**2 + data[next][1]**2)/(nextint-isoint))
                else:
                    x.append(isoint)
                    y.append(data[i][0] + (60-isoint)*tip_scale)
                    yerr.append(data[i][1])

            plt.xticks(rotation=0)
            ax = plt.gca()
            if tip_scale != 0:
                tipped = r'$ - {}\cdot(60-$A$)$'.format(tip_scale)
            else:
                tipped = ''
            ax.set_ylabel(r'$\delta$ $\nu$' + tipped + ' (MHz)', fontsize=font_size)
            if relative:
                ax.set_ylabel(r'$\delta\nu^{60,A}-\delta\nu^{60,A+1}$' + tipped + ' (MHz)', fontsize=font_size)
            ax.set_xlabel('A', fontsize=font_size)
            if plot_evens_seperate:
                x_odd = [each for each in x if each % 2 != 0]
                y_odd = [each for i, each in enumerate(y) if x[i] % 2 != 0]
                y_odd_err = [each for i, each in enumerate(yerr) if x[i] % 2 != 0]
                x_even = [each for each in x if each % 2 == 0]
                y_even = [each for i, each in enumerate(y) if x[i] % 2 == 0]
                y_even_err = [each for i, each in enumerate(yerr) if x[i] % 2 == 0]

                plt.errorbar(x_even, y_even, y_even_err, fmt='o', color=col, label='even', linestyle='-')
                plt.errorbar(x_odd, y_odd, y_odd_err, fmt='^', color=col, label='odd', linestyle='--')
                # plt.legend(loc=2)
            elif dash_missing_data:
                # if some isotopes are missing, this dashes the line at these isotopes
                # has no effect when plot_evens_separate is True
                split_x_list = []
                for k, g in groupby(enumerate(x), lambda a: a[0] - a[1]):
                    split_x_list.append(list(map(itemgetter(1), g)))
                i = 0
                label_created = False
                for each in split_x_list:
                    y_vals = y[i:i + len(each)]
                    yerr_vals = yerr[i:i + len(each)]
                    if not label_created:  # only create label once
                        plt.errorbar(each, y_vals, yerr_vals, fmt='o', color=col, linestyle='-', label=src)
                        label_created=True
                    else:
                        plt.errorbar(each, y_vals, yerr_vals, fmt='o', color=col, linestyle='-')
                    # plot dashed lines between missing values
                    if len(x) > i + len(each):  # might be last value
                        x_gap = [x[i + len(each) - 1], x[i + len(each)]]
                        y_gap = [y[i + len(each) - 1], y[i + len(each)]]
                        plt.plot(x_gap, y_gap, c=col, linestyle='--')
                    i = i + len(each)
            else:
                plt.errorbar(x, y, yerr, fmt='o', color=col, linestyle='-')
        ax.set_xmargin(0.05)
        plt.legend(loc='lower right')
        plt.margins(0.1)
        plt.gcf().set_facecolor('w')
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.tight_layout(True)
        if self.save_plots_to_file:
            isonums = []
            summed = ''
            for isos in isolist:
                if 'cal' in isos:
                    isonums.append(isos[:2] + 'c')
                else:
                    isonums.append(isos[:2])
                if 'sum' in isos:
                    summed = 'sum'
            parameter = 'iso_shifts_{}ref'.format(ref_key)
            if tip_scale != 0:
                parameter += '_tipped'
            if relative:
                parameter += 'relative'
            if scaler == 'final':
                parameter = '0_' + parameter
            filename = parameter + '_' + ''.join(isonums) + summed + '_sc' + scaler
            plt.savefig(self.resultsdir + filename + '.png', bbox_inches="tight")
        else:
            plt.show()
        plt.close()
        plt.clf()

    def plot_results_table(self, ref_iso):
        '''
        | isotope || Resonance | Iso-Shift | delta_RMS |
        |         || n | c | s | n | c | s | n | c | s |
        ------------------------------------------------
        |   60Ni  ||   |   |   |   |   |   |   |   |   |
        '''
        header_labels = ('Resonance [MHz]', 'Isotope Shift (iso-{}) [MHz]'.format(ref_iso), 'delta rms [fm]')

        header_labels = ('avg_center_fits', 'avg_shift_iso-{}'.format(ref_iso[:2]), 'avg_delta_rms_iso-{}'.format(ref_iso[:2]))
        prop_digits = (1, 1, 3)
        column_labels = ['raw', 'cal', 'sum', 'raw', 'cal', 'sum', 'raw', 'cal', 'sum']
        row_labels = ['60Ni', '58Ni', '56Ni', '55Ni']
        row_colors = [self.isotope_colors[int(i[:2])] for i in row_labels]

        def get_data_per_row(iso):
            row_data = []
            for p_num, property in enumerate(header_labels):
                for vari in ('', '_cal', '_sum_cal'):
                    try:
                        scaler = 'scaler_c012'
                        if 'sum' in vari:
                            # for 55 Nickel the combined scalers is more reliable. Use for all sum
                            scaler = 'scaler_012'
                        data_dict = self.results['{}{}'.format(iso, vari)][scaler][property]
                        row_data.append('{:.{}f}({:.0f})[{:.0f}]'.format(data_dict['vals'][0], prop_digits[p_num],
                                                                         data_dict['d_stat'][0]*10**prop_digits[p_num],
                                                                         data_dict['d_syst'][0]*10**prop_digits[p_num]))
                    except:
                        row_data.append('-')
            return row_data

        tableprint = ''

        # add header to output
        for col in range(10):
            if (col-1) % 3 == 0:
                tableprint += '{}\t'.format(header_labels[int((col-1)/3)])
            else:
                tableprint += '\t'
        # add column headers to output
        tableprint += '\n\t'
        for col in column_labels:
            tableprint += '{}\t'.format(col)
        # add data per isotope to output
        tableprint += '\n'
        for iso in row_labels:
            tableprint += '{}\t'.format(iso)
            for data in get_data_per_row(iso):
                tableprint += '{}\t'.format(data)
            tableprint += '\n'

        print(tableprint)

        with open(self.resultsdir + '0_results_table.txt', 'a+') as rt:  # open summary file in append mode and create if it doesn't exist
            rt.write(tableprint)


if __name__ == '__main__':

    analysis = NiAnalysis()
    # fitting the first time for extraction of calibration and comparison of uncalibrated data

    # final plots
    analysis.get_final_results()
    # analysis.export_results()
