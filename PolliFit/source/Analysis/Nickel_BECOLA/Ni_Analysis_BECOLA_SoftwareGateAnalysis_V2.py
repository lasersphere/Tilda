"""
Created on 2020-05-18

@author: fsommer

Module Description:
Analysis of the Nickel Data from BECOLA taken on 13.04.-23.04.2018.
Special script to take a closer look at the systematic influence of software gates on the fit results
"""

import ast
import os
import sqlite3
from datetime import datetime
import re
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mpdate

from mpl_toolkits.mplot3d.axes3d import get_test_data
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from scipy.optimize import curve_fit

import BatchFit
import Physics
import Tools
import TildaTools as TiTs
from lxml import etree as ET
from XmlOperations import xmlWriteDict
from Measurement.XMLImporter import XMLImporter
from KingFitter import KingFitter
from Analysis.Nickel_BECOLA.FileConversion import DictToXML

class NiAnalysis_softwGates():
    def __init__(self):
        logging.getLogger().setLevel(logging.INFO)
        # Name this analysis run
        self.run_name = 'Ni_Analysis_SoftwGate_Int0Method'

        """
        ############################ Folders and Database !##########################################################
        Specify where files and db are located, and where results will be saved!
        """
        # working directory:
        # get user folder to access ownCloud
        user_home_folder = os.path.expanduser("~")
        ownCould_path = 'ownCloud\\User\\Felix\\Measurements\\Nickel_online_Becola\\Analysis\\tof_analysis'
                        #'Nickel54_online_Becola20\\Analysis\\XML_Data'
        self.workdir = os.path.join(user_home_folder, ownCould_path)
        # data folder
        self.datafolder = os.path.join(self.workdir, 'Bunched')
        # results folder
        analysis_start_time = datetime.now()
        self.results_name = self.run_name + '_' + analysis_start_time.strftime("%Y-%m-%d_%H-%M")
        results_path_ext = 'results\\' + self.results_name + '\\'
        self.resultsdir = os.path.join(self.workdir, results_path_ext)
        os.makedirs(self.resultsdir)
        # database
        self.db = os.path.join(self.workdir, 'Ni_Becola_tof.sqlite')
        Tools.add_missing_columns(self.db)


        """
        ############################Analysis Parameters!##########################################################
        Specify how you want to run this Analysis!
        """
        # fit from scratch or use FitRes db?
        self.do_the_fitting = True  # if False, an .xml file has to be specified in the next variable!
        load_results_from = 'Ni54_onlineAnalysis_2020-07-12_17-45.xml'  # load fit results from this file
        self.get_gate_analysis = False  # get information from gate analysis (and use for uncertainties)
        load_gate_analysis_from = 'SoftwareGateAnalysis_2020-06-17_13-13_narrow90p-3sig_AsymmetricVoigt.xml'

        # Isotopes
        self.isotopes_single = ['56Ni', '58Ni', '60Ni', '62Ni', '64Ni']  # data is good enough for single file fitting
        self.isotopes_summed = ['54Ni', '55Ni']  # data must be summed in order to be fittable
        self.all_isotopes = self.isotopes_single + self.isotopes_summed

        # line parameters
        self.run = 'VoigtAsy'  # lineshape from runs and a new lines
        self.initial_par_guess = {'sigma': (34.0, [10, 40]), 'gamma': (12.0, [0, 30]),
                                  'asy': (3.9, True),  # in case VoigtAsy is used
                                  'dispersive': (-0.04, False),  # in case FanoVoigt is used
                                  'centerAsym': (-6.2, [-10, -1]), 'nPeaksAsym': (1, True), 'IntAsym': (0.052, [0, 0.5])
                                  # 'centerAsym': (-6.2, True), 'nPeaksAsym': (1, True), 'IntAsym': (0.052, True)
                                  # in case AsymmetricVoigt is used
                                  }

        # list of scaler combinations to fit:
        self.scaler_combinations = [[0], [1], [2], [0, 1, 2]]

        # plot options
        self.save_plots_to_file = True  # if False plots will be displayed during the run for closer inspection
        self.isotope_colors = {60: 'b', 58: 'k', 56: 'g', 55: 'c', 54: 'm', 62: 'purple', 64: 'orange'}
        self.scaler_colors = {'scaler_0': 'navy', 'scaler_1': 'maroon', 'scaler_2': 'orangered',
                              'scaler_012': 'fuchsia', 'scaler_12': 'yellow',
                              'scaler_c012': 'magenta', 'scaler_c0': 'purple', 'scaler_c1': 'grey', 'scaler_c2': 'orange'}

        # determine time gates TODO: DC runs should be identified
        self.is_dc_data = False
        self.tof_width_sigma = 2  # how many sigma to use around tof? (1: 68.3% of data, 2: 95.4%, 3: 99.7%)
        self.summed_time_gates = {}

        # acceleration set voltage (Buncher potential), negative
        self.accVolt_set = 29847  # omit voltage sign, assumed to be negative TODO: Should be from files

        # Determine calibration parameters
        self.ref_iso = '60Ni'
        self.calibration_method = 'absolute60'  # can be 'absolute58', 'absolute60' 'absolute' or 'None'
        self.use_handassigned = False  # use hand-assigned calibrations? If false will interpolate on time axis
        self.accVolt_corrected = (self.accVolt_set, 0)  # Used later for calibration. Might be used her to predefine calib? (abs_volt, err)

        # Kingfit options
        self.KingFactorLit = 'Koenig 2020 60ref'  # which king fit factors to use? kaufm60, koenig60,koenig58

        # Uncertainy Options
        self.combined_unc = 'std'  # 'std': most conservative, 'wavg_d': error of the weighted, 'wstd': weighted std

        """
        ############################ Other presets!##########################################################
        Not so important stuff
        """
        # define calibration tuples:
        # do voltage calibration with these calibration pairs.
        self.calib_tuples = [(6191, 6192), (6207, 6208), (6224, 6225), (6232, 6233), (6242, 6243),
                             (6253, 6254), (6258, 6259), (6269, 6270), (6284, 6285), (6294, 6295), (6301, 6302),
                             (6310, 6311), (6323, 6324), (6340, 6342), (6362, 6363), (6395, 6396),
                             (6418, 6419), (6462, 6463), (6467, 6466), (6501, 6502)]
        # assign 56 runs to calibration tuples. Format: (56file, (58reference, 60reference))
        self.files56_handassigned_to_calibs = [(6202, (6191, 6192)), (6203, (6191, 6192)), (6204, (6191, 6192)),
                                               (6211, (6207, 6208)), (6213, (6207, 6208)), (6214, (6207, 6208)),
                                               (6238, (6242, 6243)), (6239, (6242, 6243)), (6240, (6242, 6243)),
                                               (6251, (6253, 6254)), (6252, (6253, 6254))]

        self.analysis_parameters = {'reference_isotope': self.ref_iso,
                                    'run': self.run,
                                    'first_fit': 'from scratch' if self.do_the_fitting else load_results_from,
                                    'calibration': self.calibration_method,
                                    'use_handassigned': self.use_handassigned,
                                    'initial_par_guess': self.initial_par_guess,
                                    'gate_parameters': {'midtof': 'from summed files',
                                                        'gatewidth': '{} times sigma from summed files'.format(self.tof_width_sigma),
                                                        'delay': 'from summed files',
                                                        'gate_std_from': load_gate_analysis_from},
                                    'KingFactors': self.KingFactorLit
                                    }

        self.init_uncertainties_exp_physics()
        self.init_stuff()

        # import results
        if self.get_gate_analysis:
            # import the results from a previous run gate analysis to get a per-file estimate on those errors
            self.import_results(load_gate_analysis_from)
        if not self.do_the_fitting:
            # import the results from a previous run as sepcified in load_results_from
            self.import_results(load_results_from)

    def init_uncertainties_exp_physics(self):
        """
        ### Uncertainties ###
        All uncertainties that we can quantify and might want to respect
        """
        self.accVolt_set_d = 10  # V. uncertainty of scan volt. Estimated by Miller for Calcium meas.
        self.wavemeter_wsu30_mhz_d = 3*2  # MHz. Kristians wavemeter paper. Factor 2 because of frequency doubling.
        self.matsuada_volts_d = 0.03  # V. ~standard dev after rebinning
        self.lineshape_d_syst = 1.0  # MHz. Offset between VoigtAsym and AsymmetricVoigt
        self.bunch_structure_d = 0.2  # MHz. Slope difference between 58&56 VoigtAsy allfix: 20kHz/bin, +-5bin width --> 200kHz
        self.heliumneon_drift = 5  # MHz. Max drift according to datasheet of the SIOS China2 HeNe. 1h stability ~1MHz. TODO: how is the influence?
        self.laserionoverlap_anglemrad_d = 1  # mrad. ideally angle should be 0. Max possible deviation is ~1mrad
        self.laserionoverlap_MHz_d = (self.accVolt_set -  # should turn out to around ~200kHz
                                      np.sqrt(self.accVolt_set ** 2 / (
                                                  1 + (self.laserionoverlap_anglemrad_d / 1000) ** 2))) * 15

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
        # The timegates primarily depend on the mass of the isotopes!
        for iso in self.all_isotopes:  # write timegates to db
            m = self.masses[iso][0]/1000000
            midtof = int(m * 4.6592 + 267.7437)/100  # formula determined imperically for Nickel data
            delaylist = [0, 0.194, 0.267]  # also determined impirically for Nickel @ BECOLA
            sigma = 0.1
            self.update_gates_in_db(iso, midtof, 2 * self.tof_width_sigma * sigma, delaylist)

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
        self.restframe_trans_freq = {'58Ni': (850343678.0, 2.0), '60Ni': (850344183.0, 2.0)}

        ''' literature value IS 60-58'''
        iso_shifts_kaufm = {  # PRL 124, 132502, 2020
            '58Ni': (-509.1, 2.5 + 4.2),
            '59Ni': (-214.3, 2.5+2.0),  # From Simons Nickel_shift_results.txt(30.04.20), not published yet
            '60Ni': (0, 0),
            '61Ni': (280.8, 2.7 + 2.0),
            '62Ni': (503.9, 2.5 + 3.9),
            '63Ni': (784.9,	2.5	+ 5.0),  # From Simons Nickel_shift_results.txt(30.04.20), not published yet
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
            '64Ni': (Physics.freqFromWavenumber(0.01691+0.01701),  # 60-62 and 62-64 combined.
                     Physics.freqFromWavenumber(np.sqrt(0.00012**2+0.00026**2)))}  # Quadr. error prop
        iso_shifts_koenig = {  # private com. excel sheet mid 2020
            '58Ni': (self.restframe_trans_freq['58Ni'][0] - self.restframe_trans_freq['60Ni'][0],
                     np.sqrt(self.restframe_trans_freq['58Ni'][1]**2 + self.restframe_trans_freq['60Ni'][1]**2)),
            '60Ni': (0, 0),
            '62Ni': (502.87, 3.43),
            '64Ni': (1026.14, 3.79)}

        self.iso_shifts_lit = {'Kaufmann 2020 (incl.unbup.!)': {'data': iso_shifts_kaufm, 'color': 'green'},  # (incl.unbup.!)
                               'Steudel 1980': {'data': iso_shifts_steudel, 'color': 'black'},
                               'Koenig 2020': {'data': iso_shifts_koenig, 'color': 'blue'}}


        ''' literature Mass Shift and Field Shift constants '''
        self.king_literature = {'Kaufmann 2020 60ref': {'data': {'Alpha': 396, 'F': (-769, 60), 'Kalpha': (948000, 3000)},
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
        delta_rms_koenig = {'58Ni': (-0.275, 0.0082),  # private com. excel sheet mid 2020
                             '60Ni': (0, 0),
                             '62Ni': (0.2226, 0.0059),
                             '64Ni': (0.3642, 0.0095)}

        self.delta_rms_lit = {'Kaufmann 2020 (incl.unbup.!)': {'data': delta_rms_kaufm, 'color': 'green'},  # (incl.unbup.!)
                              'Steudel 1980': {'data': delta_rms_steudel, 'color': 'black'},
                              'Koenig 2020': {'data': delta_rms_koenig, 'color': 'blue'}}

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

        print(
            'iso\t<r^2>^{1/2}_{0µe}\t\Delta<r^2>^{1/2}_{0µe}\t<r^2>^{1/2}_{0µe}(A-A_{60})\t\Delta <r^2>^{1/2}_{0µe}(A-A_{60})')
        for iso, radi in sorted(lit_radii.items()):
            dif = self.delta_lit_radii_60.get(iso, (0, 0))
            print('%s\t%.3f\t%.3f\t%.5f\t%.5f' % (iso, radi[0], radi[1], dif[0], dif[1]))

        # note down laser frequencies:   TODO: Should be from files! Basically only used for sumfiles
        self.laser_freqs = {'54Ni': 2*425624179,
                            '58Ni': 2*425618884,
                            '60Ni': 2*425611628,
                            '62Ni': 2*425604733,
                            '64Ni': 2*425598175
                            }

    def init_stuff(self):
        """
        Initialization stuff
        """

        # extract line to use and insert restframe transition frequency
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute(
            '''SELECT lineVar FROM Runs WHERE run = ? ''', (self.run,))
        lineVar = cur.fetchall()
        self.line = lineVar[0][0]
        cur.execute('''SELECT * FROM Lines WHERE lineVar = ? ''', (self.line,))  # get original line to copy from
        copy_line = cur.fetchall()
        copy_line_list = list(copy_line[0])
        copy_line_list[3] = self.restframe_trans_freq[self.ref_iso][0]
        line_new = tuple(copy_line_list)
        cur.execute('''INSERT OR REPLACE INTO Lines VALUES (?,?,?,?,?,?,?,?,?)''', line_new)
        con.commit()
        con.close()

        # calculate differential doppler shifts
        self.diff_dopplers = {key: Physics.diffDoppler(self.restframe_trans_freq[self.ref_iso][0],
                                                       self.accVolt_set,
                                                       self.masses[key][0] / 1e6)
                              for key in self.masses.keys()}
        # adjust center fit estimations to accVoltage
        # self.adjust_center_ests_db()

        # time reference
        self.ref_datetime = datetime.strptime('2018-04-13_13:08:55', '%Y-%m-%d_%H:%M:%S')  # run 6191, first 58 we use

        # create results dictionary:
        self.results = {}

        # Set the scaler variable to a defined starting point:
        self.update_scalers_in_db('012')

    ''' analysis '''

    def int0_method(self):
        """
        Kristian pointed out, that the time-projected data is probably not very reliable for determining the midTOF.
        Thats true especially for the offline data, since there all stable isotopes are inside the bunch and can produce
        beam induced background. A better option is to vary the mid-tof with a very narrow gate and look at which
        setting the peak Intensity Int0 is highest (thus Int0-Method)
        :return:
        """
        # define which files to run the analysis on:
        # file_list = ['BECOLA_10165.xml']  #, 'BECOLA_10166.xml', 'BECOLA_10182.xml', 'BECOLA_10169.xml', 'BECOLA_10170.xml']
        file_list, file_nos, file_times = self.pick_files_from_db_by_type_and_num('%%')


        for file in file_list:
            # get information on this file from db:
            fname, fpath, fdate, ftype, faccV, flaser = self.get_file_info_from_db(file)
            fno = int(re.split('[_.]', fname)[1])
            fiso = ftype[:4]


            # do an initial fitting with standard gates (set in init)
            self.update_scalers_in_db('scaler_012')
            ref_center_MHz, ref_center_MHz_fiterrs, ref_fitpars = self.fit_files([file], plotname='original')
            volt, d, d_syst = self.centerFreq_to_absVoltage(fiso, ref_center_MHz[0], ref_center_MHz_fiterrs[0], 0, file=file)

            self.results[file] = {}
            self.results[file]['File'] = file
            self.results[file]['color'] = 'k'
            self.results[file]['Isotope'] = int(fiso[:2])
            self.results[file]['Resonance(V)'] = {'vals': [volt], 'd_fit': [d]}

            # fit with narrow gates and some shifting
            Int0_gatewidth = 0.06
            variation = 0.40
            samples = int(2*variation/Int0_gatewidth+1)
            m = self.masses[fiso][0] / 1000000
            mid0 = (m * 4.6377 + 269.3738)/100  # formula determined imperically for Nickel data
            midtof_variation_arr = np.linspace(mid0-variation, mid0+variation, samples)
            delay_orig = [0, 0.194, 0.267]
            midtof_variation_arr_per_scaler = {'scaler_0': midtof_variation_arr+delay_orig[0],
                                               'scaler_1': midtof_variation_arr+delay_orig[1],
                                               'scaler_2': midtof_variation_arr+delay_orig[2]}


            Int0_results = {'scaler_0': {'Int0': {'vals': [], 'd_fit': []},
                                         'center_fits': {'vals': [], 'd_fit': []},
                                         'tproj_midtof': {'vals': [], 'd_fit': []},
                                         'tproj_sigma': {'vals': [], 'd_fit': []}},
                            'scaler_1': {'Int0': {'vals': [], 'd_fit': []},
                                         'center_fits': {'vals': [], 'd_fit': []},
                                         'tproj_midtof': {'vals': [], 'd_fit': []},
                                         'tproj_sigma': {'vals': [], 'd_fit': []}},
                            'scaler_2': {'Int0': {'vals': [], 'd_fit': []},
                                         'center_fits': {'vals': [], 'd_fit': []},
                                         'tproj_midtof': {'vals': [], 'd_fit': []},
                                         'tproj_sigma': {'vals': [], 'd_fit': []}
                                         }}


            for sc in Int0_results.keys():
                scaler = self.update_scalers_in_db(sc)
                num = scaler.split('_')[-1]
                self.update_gates_in_db(ftype, delaylist=[0, 0, 0], gatewidth=Int0_gatewidth)
                for midtof in midtof_variation_arr_per_scaler[scaler]:
                    # write new midtof to isotope db
                    self.update_gates_in_db(ftype, midtof=midtof)
                    center_MHz, center_MHz_fiterrs, fitpars = self.fit_files([file], plotname=str(midtof))
                    Int0, Int0_d, Int0_fix = fitpars[0]['Int0']
                    Int0_results[sc]['Int0']['vals'].append(Int0)
                    Int0_results[sc]['Int0']['d_fit'].append(Int0_d)
                    Int0_results[sc]['center_fits']['vals'].append(center_MHz[0])
                    Int0_results[sc]['center_fits']['d_fit'].append(center_MHz_fiterrs[0])
                # find  mid-tof via time projection
                self.update_gates_in_db(ftype, midtof=mid0, gatewidth=variation, delaylist=delay_orig)
                midtof_us, midtof_d_us, sigma, sigma_d = self.find_midtof_for_file(file, num)
                Int0_results[sc]['tproj_midtof']['vals'].append(midtof_us)
                Int0_results[sc]['tproj_midtof']['d_fit'].append(midtof_d_us)
                Int0_results[sc]['tproj_sigma']['vals'].append(sigma)
                Int0_results[sc]['tproj_sigma']['d_fit'].append(sigma_d)
            self.update_gates_in_db(ftype, delaylist=delay_orig)

            TiTs.merge_extend_dicts(self.results[file], Int0_results)
            self.results[file]['variation'] = midtof_variation_arr_per_scaler
            scaler_results = self.fit_and_plot_Int0([file], ['scaler_0', 'scaler_1', 'scaler_2'], 'Int0')
            TiTs.merge_extend_dicts(self.results[file], scaler_results)
            # self.plot_parameter_for_isos_and_scaler([file], ['scaler_0', 'scaler_1', 'scaler_2'],
            #                                         'center_fits', onlyfiterrs=True, plotstyle='classic', x_type='variation')

        self.plot_results_table(file_list)

    def fit_and_plot_Int0(self, files, scaler_list, parameter, unit='cts' , factor=1, folder=None):
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
        scaler_nums = []
        scaler_results = {}
        plotrange_y = [0, 0]
        for sc in scaler_list:
            scaler = self.update_scalers_in_db(sc)
            # determine color
            col = self.scaler_colors[scaler]

            if '_' in scaler:
                scaler_nums.append(scaler.split('_')[1])
            else:
                scaler_nums.append(scaler)
            for i in range(len(files)):
                f = files[i]
                x_ax = self.results[f]['variation'][scaler]

                # also plot a line for the t-projection midtof:
                tof_x0 = self.results[f][scaler]['tproj_midtof']['vals'][0]
                tof_x0_d = self.results[f][scaler]['tproj_midtof']['d_fit'][0]
                tof_sigma = self.results[f][scaler]['tproj_sigma']['vals'][0]
                tof_sigma_d = self.results[f][scaler]['tproj_sigma']['d_fit'][0]
                tproj_label = '{}: Proj: center: {:.2f}({:.0f}), sigma:{:.2f}({:.0f})' \
                    .format(scaler, tof_x0, 100*tof_x0_d, tof_sigma, 100 * tof_sigma_d)
                # plt.axvline(x=tof_x0, color=col, label=tproj_label)

                centers = factor*np.array(self.results[f][scaler][parameter]['vals'])
                zero_arr = np.zeros(len(centers))  # prepare zero array with legth of centers in case no errors are given
                centers_d_stat = factor*np.array(self.results[f][scaler][parameter].get('d_fit', zero_arr))

                if 2*centers.max() > plotrange_y[1]:
                    plotrange_y[1] = 2*centers.max()

                # plot center frequencies in MHz:
                # plot values as dots with statistical errorbars
                plt.errorbar(x_ax, np.array(centers), yerr=np.array(centers_d_stat), fmt='.', color=col)
                # plt.plot(x_ax, np.array(centers), '.', color=col)

                # fit
                try:
                    popt, perr = self.fit_shape(centers, x_ax, err_axis=centers_d_stat)
                    print(popt, perr)
                except:
                    popt = 1,1,1,1
                    perr = 1,1,1,1
                a, sigma, x0, o = popt
                a_d, sigma_d, x0_d, o_d = perr
                # A, alpha, x0, o, b = popt
                # A_d, alpha_d, x0_d, o_d, b_d = perr
                # sigma, sigma_d = self.exp_power_distr_variance(alpha, b, alpha_d, b_d)

                # attach fit results to scaler results
                scaler_results[scaler] = {'midtof': {'vals': [x0], 'd_fit': [x0_d]},
                                          'sigma': {'vals': [sigma], 'd_fit': [sigma_d]}}

                # plot fitresults
                x = np.arange(x_ax[0], x_ax[-1], 0.01)
                fit_label = '{}: Int0: center: {:.2f}({:.0f}), sigma:{:.2f}({:.0f})'\
                    .format(scaler, x0, 100*x0_d, sigma, 100*sigma_d)
                plt.plot(x, self.fitfunc(x, *popt), label=fit_label, color=col)
                # plt.plot(x, self.exp_power_distr(x, *popt), label=fit_label, color=col)

        plt.xlabel('midtof')
        plt.xticks(rotation=45)
        plt.ylabel('{} [{}]'.format(parameter, unit))
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        plt.ylim(plotrange_y[0], plotrange_y[1])
        plt.title('{} in {} for file: {}'.format(parameter, unit, files))
        plt.legend(title='Legend', bbox_to_anchor=(1.04, 0.5), loc="center left")
        plt.margins(0.05)
        if self.save_plots_to_file:
            parameter = parameter.replace(':', '_')  # colon is no good filename char
            filename = parameter + '_' + '_'.join(files) + '_sc' + 'a'.join(scaler_nums)
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

        return scaler_results





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

    def get_file_info_from_db(self, filename):
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute(
            '''SELECT file, filePath, date, type, accVolt, laserFreq FROM Files WHERE file = ? ''', (filename,))
        file_info = cur.fetchall()[0]
        con.close()

        return file_info

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
        ref_freq_dev = 850343800 - self.restframe_trans_freq[self.ref_iso][
            0]  # stand_ests are for 580343800MHz. Ref freq might be updated
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
                if selecttuple[0] <= fileno <= selecttuple[1]:
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
            scaler_db_string = str(scalers).join(('[', ']'))
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

    def get_iso_for_file(self, file):
        """

        :param file:
        :return:
        """
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute(
            '''SELECT type FROM Files WHERE file = ?''', (file,))
        type = cur.fetchall()
        con.close()
        iso = type[0][0]
        return iso

    ''' analysis related '''
    def find_midtof_for_file(self, file, scaler):
        # load the spec data from file
        filepath = os.path.join(self.datafolder, file)
        spec = XMLImporter(path=filepath)
        tproj = spec.t_proj[0][scaler]
        timebins = np.arange(1024)/100
        # fit time-projection
        popt, perr = self.fit_shape(tproj, timebins)
        ampl, sigma, center, offset = popt
        ampl_d, sigma_d, center_d, offset_d = perr
        # if center_d > 10 or center < 500 or center > 600:
        #     # something went wrong while fitting. Maybe this is DC data? Use middle value
        #     center = 5.12
        midtof_us = center
        midtof_d_us = center_d
        return midtof_us, midtof_d_us, sigma, sigma_d

    def fit_shape(self, cts_axis, time_axis, err_axis=None):
        x = time_axis
        y = cts_axis

        # estimates:: amplitude: sigma*sqrt(2pi)*(max_y-min_y), sigma=10, center:position of max_y, offset: min_y
        start_pars = np.array([0.1 * 2.51 * (max(y) - min(y)), 0.1, x[np.argwhere(y == max(y))[0, 0]], min(y)])
        if err_axis is not None:
            popt, pcov = curve_fit(self.fitfunc, x, y, start_pars, sigma=err_axis, absolute_sigma=True)
        else:
            popt, pcov = curve_fit(self.fitfunc, x, y, start_pars)

        # # estimates:: amplitude: sigma*sqrt(2pi)*(max_y-min_y), sigma=10, center:position of max_y, offset: min_y
        # start_pars = np.array([(max(y)-min(y))*0.2*1.77, 0.2, x[np.argwhere(y == max(y))[0, 0]], min(y), 2])
        # if err_axis is not None:
        #     popt, pcov = curve_fit(self.exp_power_distr, x, y, start_pars, sigma=err_axis, absolute_sigma=True)
        # else:
        #     popt, pcov = curve_fit(self.exp_power_distr, x, y, start_pars)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr

    def fitfunc(self, t, a, s, t0, o):
        """
        fitfunction for time projection
        t: time
        t0: mid-tof
        a: cts_max
        s: sigma
        o: offset
        """
        # Gauss function
        return o + a * 1/(s*np.sqrt(2*np.pi))*np.exp(-1/2*np.power((t-t0)/s, 2))

    def exp_power_distr(self, x, A, alpha, x0, o, b):
        """
        Exponential power distribution or Generalized error distribution ('Subbotin').
        For b = 1 it resembles a laplace-, for b=2 a normal- and b=inf uniform-distribution
        Subbotin, M. T. (1923), On the law of frequency of error, Matematicheskii Sbornik, 31, 296-301.
        found here: Tim (https://stats.stackexchange.com/users/35989/tim), Is there a plateau-shaped distribution?, URL (version: 2016-03-25): https://stats.stackexchange.com/q/203634
        :param x: function value
        :param A: amplitude parameter
        :param alpha: scale parameter
        :param x0: center parameter
        :param o: offset
        :param b: distribution shape parameter
        :return:
        """
        from scipy.special import gamma
        return o + A*b/(2*alpha*gamma(1/b))*np.exp(-1/2*(abs(x-x0)/alpha)**b)

    def exp_power_distr_variance(self, alpha, b, alpha_d, b_d):
        from scipy.special import gamma, psi
        # derivative of the gamma function is: gamma'(x)=gamma(x)*psi(x), where psi is polygamma(0, x)
        var = alpha**2 * gamma(3/b)/gamma(1/b)    # =sigma^2
        var_d = np.sqrt(np.square(alpha_d**2 * gamma(3/b)/gamma(1/b))
                        + np.square(var*(psi(3/b)-psi(1/b))*b_d))
        sigma = np.sqrt(var)
        sigma_d = 1/2/np.sqrt(var)*var_d
        return sigma, sigma_d

    def fit_files(self, filelist, plotname=''):
        filearray = np.array(filelist)  # needed for batch fitting

        # define and create (if not exists) the output folder
        plot_specifier = 'plots\\' + self.scaler_name + '_' + plotname + '\\'  # e.g. scaler_012_cal
        plot_folder = os.path.join(self.resultsdir, plot_specifier)
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)

        # for softw_gates_trs from file use 'File' and from db use None.
        BatchFit.batchFit(filearray, self.db, self.run, x_as_voltage=True, softw_gates_trs=None, guess_offset=True,
                          save_to_folder=plot_folder)

        # get fitresults (center) vs run
        all_rundate = []
        all_fitpars = []
        all_center_MHz = []
        all_center_MHz_fiterrs = []  # only the fit errors, nothing else!

        # get fit results
        for files in filelist:
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            # Get corresponding isotope
            cur.execute(
                '''SELECT date, type, accVolt FROM Files WHERE file = ? ''', (files,))
            filefetch = cur.fetchall()
            file_date, file_type, file_volt = filefetch[0]
            file_date = datetime.strptime(file_date, '%Y-%m-%d %H:%M:%S')
            # Query fitresults for file and isotope combo
            cur.execute(
                '''SELECT pars FROM FitRes WHERE file = ? AND iso = ? AND run = ?''', (files, file_type, self.run))
            pars = cur.fetchall()
            con.close()
            try:
                # if the fit went wrong there might not be a value to get from the fitpars...
                parsdict = ast.literal_eval(pars[0][0])
            except Exception as e:
                # Fit went wrong!
                # replace with standard value and big error...
                parsdict = {'center': (-510, 30, False)}  # TODO: use better dummy value (take from all_Center_MHz list)
            all_fitpars.append(parsdict)
            all_center_MHz.append(parsdict['center'][0])

            # === uncertainties ===
            # fit uncertainty
            fit_d = parsdict['center'][1]
            all_center_MHz_fiterrs.append(fit_d)

        return all_center_MHz, all_center_MHz_fiterrs, all_fitpars

    def plot_and_fit_3darray(self, data_arr, file):
            # Plot 3D data of dac voltage deviation: np.array (self.nrOfScans, self.nrOfSteps)) values are: devFromSetVolt
            # extract numpy array to X, Y, Z data arrays:
            # try:
            # in case the file was bad, there might not be good data for plotting and fitting... better try.
            x = np.arange(data_arr.shape[1])
            y = np.arange(data_arr.shape[0])
            X, Y = np.meshgrid(x, y)
            Z = data_arr

            # define fit function (a simple plane)
            def plane(x, y, mx, my, coff):
                return x * mx + y * my + coff

            def _plane(M, *args):
                """
                2D function generating a 3D plane
                :param M: xdata parameter, 2-dimensional array of x and y values
                :param args: slope and offset passed to plane function
                :return: array of z-values
                """
                x, y = M  # x: steps, y: scans
                arr = np.zeros(x.shape)
                arr += plane(x, y, *args)
                return arr

            # Define start parameters [mx, my, offset]
            p0 = [0, 0, Z[0, 0]]
            # make 1-D data (necessary to use curve_fit)
            xdata = np.vstack((X.ravel(), Y.ravel()))
            # fit
            popt, pcov = curve_fit(_plane, xdata, Z.ravel(), p0)

            # calculate average and maximum deviation after correction
            fit_plane = plane(X, Y, *popt)
            standard_dev = np.sqrt(np.square((fit_plane-Z)).mean())
            max_dev = (fit_plane-Z).max()

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, plane(X, Y, *popt), rstride=1, cstride=1)
            # ax.plot_surface(X, Y, np.where(plane(X, Y, *popt)<Z, plane(X, Y, *popt), np.nan), rstride=1, cstride=1)
            surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')
            # ax.plot_surface(X, Y, np.where(plane(X, Y, *popt)>=Z, plane(X, Y, *popt), np.nan), rstride=1, cstride=1, alpha=0.7)
            fig.colorbar(surf)
            plt.xlabel('gateWidth')
            plt.ylabel('midTof')
            if self.save_plots_to_file:
                filename = '3drep_{}'.format(file)
                plt.savefig(self.resultsdir + filename + '.png', bbox_inches="tight")
            else:
                plt.show()
            plt.close()
            plt.clf()

            plt.imshow(data_arr, cmap='hot', interpolation='nearest')
            if popt[2] == 1:
                # error while fitting
                plt.title(file + ': Deviation from dac set voltage.\n'
                                                 '!error while fitting!')
            else:
                plt.title(file + ': Deviation from dac set voltage.\n'
                                                 'offset: {0:.4f}V\n'
                                                 'step_slope: {1:.4f}V\n'
                                                 'scan_slope: {2:.4f}V\n'
                                                 'standard deviation after correction: {3:.4f}V\n'
                                                 'maximum deviation after correction: {4:.4f}V.'
                          .format(popt[2], popt[0] * 1000, popt[1], standard_dev, max_dev))
            plt.xlabel('step number')
            plt.ylabel('scan number')
            plt.colorbar()
            if self.save_plots_to_file:
                filename = '3drep_b_{}'.format(file)
                plt.savefig(self.resultsdir + filename + '.png', bbox_inches="tight")
            else:
                plt.show()
            plt.close()
            plt.clf()


            # except Warning as w:
            #     self.fit_success = w
            # except Exception as e:
            #     self.standard_v_dev = -1
            #     self.volt_correct = (1, 1)
            #     self.fit_success = e

    def centerFreq_to_absVoltage(self, isostring, deltanu, nu_d, nu_dsyst, laserfreq=None, file=None):
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
        if file is not None:
            cur.execute(
                '''SELECT laserFreq FROM Files WHERE file = ? ''', (file,))
            db_laserfreq = cur.fetchall()
        else:
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

    ''' visualization '''

    def plot_parameter_for_isos_and_scaler(self, isotopes, scaler_list, parameter,
                                           offset=None, overlay=None, unit='MHz', onlyfiterrs=False,
                                           digits=1 , factor=1, plotAvg=False, plotstyle='band', folder=None, x_type='file_numbers'):
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
        scaler_nums = []
        for sc in scaler_list:
            scaler = self.update_scalers_in_db(sc)
            if '_' in scaler:
                scaler_nums.append(scaler.split('_')[1])
            else:
                scaler_nums.append(scaler)
            for i in range(len(isotopes)):
                iso = isotopes[i]
                x_ax = self.results[iso][x_type][scaler]
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
                        wavg, wavg_d, wstd, std = self.calc_weighted_avg(centers, centers_d_stat)
                        if self.combined_unc == 'wavg_d':
                            d = wavg_d
                        elif self.combined_unc == 'wstd':
                            d = wstd
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

    ''' results related '''

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
            w = 1/np.square(x_d)
        else:
            logging.warning('ZERO value in uncertainties found during weighted average calculation. '
                            'Calculating mean and standard deviation instead of weighting!')
            return x.mean(), x.std(), x.std(), x.std()

        if n > 1:  # only makes sense for more than one data point. n=1 will also lead to div0 error
            # calculate weighted average and sum of weights:
            wavg = np.sum(x*w)/np.sum(w)  # (Bevington 4.17)
            # calculate the uncertainty of the weighted mean
            wavg_d = np.sqrt(1/np.sum(w))  # (Bevington 4.19)

            # calculate weighted average variance
            wvar = np.sum(w*np.square(x-wavg))/np.sum(w) * n/(n-1)  # (Bevington 4.22)
            # calculate weighted standard deviations
            wstd = np.sqrt(wvar/n)  # (Bevington 4.23)

            # calculate (non weighted) standard deviations from the weighted mean
            std = np.sqrt(np.sum(np.square(x-wavg))/(n-1))
        else:  # for only one value, return that value
            wavg = x[0]
            # use the single value uncertainty for all error estimates
            wavg_d = x_d[0]
            wstd = x_d[0]
            std = x_d[0]

        return wavg, wavg_d, wstd, std

    def writeXMLfromDict(self, dictionary, filename, tree_name_str):
        """
        filename must be in form name.xml
        """
        root = ET.Element(tree_name_str)
        xmlWriteDict(root, dictionary)
        xml = ET.ElementTree(root)
        xml.write(filename)

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
        for keys, vals in TiTs.deepcopy(self.results).items():
            # xml cannot take numbers as first letter of key
            if vals.get('file_times') is not None:
                vals['file_times'] = [datetime.strftime(t, '%Y-%m-%d %H:%M:%S') for t in vals['file_times']]
            to_file_dict['i' + keys] = vals
        # add analysis parameters
        to_file_dict['analysis_parameters'] = self.analysis_parameters
        results_file = self.results_name + '.xml'
        self.writeXMLfromDict(to_file_dict, os.path.join(self.resultsdir, results_file), 'BECOLA_Analysis')

    def import_results(self, results_file):
        results_name = results_file[:-4]  # cut .xml from the end
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
        return res_dict

    def write_gates_to_file(self, file, mid_tof, gate_width):
        # prepare new software gates
        voltage_gates = [-np.inf, np.inf]
        del_list = self.delaylist_orig  # scaler delay list
        softw_gates = []
        for each_del in del_list:
            softw_gates.append(
                [voltage_gates[0], voltage_gates[1],
                 mid_tof + each_del - 0.5 * gate_width,
                 mid_tof + each_del + 0.5 * gate_width]
            )
        print(softw_gates)
        # load the spec data from file and already insert new software gates
        filepath = os.path.join(self.datafolder, file)
        title, xml_dict = DictToXML.readDictFromXML(filepath)
        xml_dict['tracks']['track0']['header']['softwGates'] = softw_gates

        # write back to xml file
        DictToXML.writeXMLfromDict(xml_dict, filepath, title)

    def plot_results_table(self, filelist):
        '''
        '''
        header_labels = ('File', 'Isotope', 'Resonance(V)', 'midtof sc0', 'midtof sc1', 'midtof sc2', 'sigma sc0', 'sigma sc1', 'sigma sc2', 'tproj_midtof sc0', 'tproj_midtof sc1', 'tproj_midtof sc2')

        def get_data_per_row(file):
            row_data = []

            for p_num, property in enumerate(header_labels):
                try:
                    prop = self.results[file].get(property, None)
                    if prop is not None:
                        if type(prop) == type({}):
                            row_data.append(prop['vals'][0])
                            row_data.append(prop['d_fit'][0])
                        else:
                            row_data.append(prop)
                    else:
                        property, sc = property.split(' ')
                        scaler = self.update_scalers_in_db(int(sc[-1]))
                        prop = self.results[file][scaler].get(property, None)
                        if type(prop) == type({}):
                            row_data.append(100*prop['vals'][0])
                            row_data.append(100*prop['d_fit'][0])
                        else:
                            row_data.append(prop)

                except Exception as e:
                    row_data.append('-')

            return row_data

        tableprint = ''

        # add header to output
        for num, col in enumerate(header_labels):
            tableprint += '{}\t'.format(col)
        # add column headers to output
        tableprint += '\n'
        for file in filelist:
            for data in get_data_per_row(file):
                tableprint += '{}\t'.format(data)
            tableprint += '\n'

        print(tableprint)

        with open(self.resultsdir + '0_results_table.txt', 'a+') as rt:  # open summary file in append mode and create if it doesn't exist
            rt.write(tableprint)


if __name__ == '__main__':
    analysis = NiAnalysis_softwGates()
    analysis.int0_method()
    analysis.export_results()


