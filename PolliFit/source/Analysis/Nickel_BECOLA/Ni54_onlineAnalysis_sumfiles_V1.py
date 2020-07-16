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
        self.run_name = 'Ni54_onlineAnalysis'

        """
        ############################ Folders and Database !##########################################################
        Specify where files and db are located, and where results will be saved!
        """
        # working directory:
        # get user folder to access ownCloud
        user_home_folder = os.path.expanduser("~")
        ownCould_path = 'ownCloud\\User\\Felix\\Measurements\\Nickel54_online_Becola20\\Analysis\\XML_Data'
        self.workdir = os.path.join(user_home_folder, ownCould_path)
        # data folder
        self.datafolder = os.path.join(self.workdir, 'Sums')
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
        self.do_the_fitting = True  # if False, an .xml file has to be specified in the next variable!
        load_results_from = 'Ni54_onlineAnalysis_2020-07-12_17-45.xml'  # load fit results from this file
        self.get_gate_analysis = False  # get information from gate analysis (and use for uncertainties)
        load_gate_analysis_from = 'SoftwareGateAnalysis_2020-06-17_13-13_narrow90p-3sig_AsymmetricVoigt.xml'

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

        # determine time gates
        self.tof_mid = {'54Ni': 5.200, '55Ni': 5.237, '56Ni': 5.276, '58Ni': 5.383, '60Ni': 5.408}  # mid-tof for each isotope (from fitting)
        self.tof_delay = [0, 0.195, 0.265]
        self.tof_sigma = 0.098  # 1 sigma of the tof-peaks from fitting, avg over all scalers 56,58,60 Ni
        self.tof_width_sigma = 2  # how many sigma to use around tof? (1: 68.3% of data, 2: 95.4%, 3: 99.7%)

        # acceleration set voltage (Buncher potential), negative
        self.accVolt_set = 29847  # omit voltage sign, assumed to be negative

        # Determine calibration parameters
        self.ref_iso = '60Ni'
        self.calibration_method = 'absolute60'  # can be 'absolute58', 'absolute60' 'absolute' or 'None'
        self.use_handassigned = False  # use hand-assigned calibrations? If false will interpolate on time axis
        self.accVolt_corrected = (self.accVolt_set, 0)  # Used later for calibration. Might be used her to predefine calib? (abs_volt, err)

        # Kingfit options
        self.KingFactorLit = 'Koenig 2020 60ref'  # which king fit factors to use? kaufm60, koenig60,koenig58

        # Uncertainy Options
        self.combined_unc = 'std'  # 'std': most conservative, 'wavg_d': error of the weighted, 'wstd': weighted std

        # plot options
        self.save_plots_to_file = True  # if False plots will be displayed during the run for closer inspection
        self.isotope_colors = {58: 'k', 60: 'b', 56: 'g', 54: 'm'}
        self.scaler_colors = {'scaler_0': 'navy', 'scaler_1': 'maroon', 'scaler_2': 'orangered',
                              'scaler_012': 'fuchsia', 'scaler_12': 'yellow',
                              'scaler_c012': 'magenta', 'scaler_c0': 'purple', 'scaler_c1': 'grey', 'scaler_c2': 'orange'}

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
                                    'gate_parameters': {'midtof': str(self.tof_mid),
                                                        'gatewidth': 2*self.tof_width_sigma*self.tof_sigma,
                                                        'delay': self.tof_delay,
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
            '60Ni': (59930785.3, 0.4)
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

        # note down laser frequencies:
        self.laser_freqs = {'54Ni': 2*425624179,
                            '58Ni': 2*425608874,
                            '60Ni': 2*425601785
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

        # remove bad files from db
        con = sqlite3.connect(self.db)  # connect to db
        cur = con.cursor()
        with open(os.path.join(self.datafolder, 'badruns.txt'), 'r') as bad:  # all runs to be excluded from analysis
            for line in bad:  # each line contains just the run number of a bad run. Reasons see separate excel.
                splt = line.split()  # get only the file-number in a list with len=1, removes the linebreak
                filename = 'BECOLA_{}.xml'.format(splt[0])  # create filename from filenumber
                cur.execute(
                    '''DELETE FROM Files WHERE file = ? ''', (filename,))  # delete row of specified file
                con.commit()  # commit changes to db
        cur.execute(
            '''DELETE FROM Files WHERE file LIKE ? ''', ('Sum%',))  # Also delete sum files from last run!
        con.commit()  # commit changes to db
        con.close()  # close db connection

        # set gates in db
        for iso, midtof in self.tof_mid.items():
            self.update_gates_in_db(iso, midtof, 2*self.tof_width_sigma*self.tof_sigma, self.tof_delay)

    ''' analysis '''
    def fitting_initial_separate(self):
        if self.do_the_fitting:
            for sc in self.scaler_combinations:
                # write the scaler to db for usage
                scaler = self.update_scalers_in_db(sc)
                # reset initial parameter guess:
                self.write_to_db_lines(self.run,
                                       sigma=self.initial_par_guess['sigma'],
                                       gamma=self.initial_par_guess['gamma'],
                                       asy=self.initial_par_guess['asy'],
                                       dispersive=self.initial_par_guess['dispersive'],
                                       centerAsym=self.initial_par_guess['centerAsym'],
                                       IntAsym=self.initial_par_guess['IntAsym'],
                                       nPeaksAsym=self.initial_par_guess['nPeaksAsym'])
                # Do a fit of the original data for all even isotopes without any calibration applied.
                for iso in ['58Ni', '60Ni']:
                    filelist, runNo, center_freqs, center_fit_errs, center_freqs_d, center_freqs_d_syst, start_times, \
                    fitpars, rChi = \
                        self.chooseAndFitRuns(iso, reset=(self.accVolt_set, iso))
                    # include results to self.results dict
                    isodict = {iso: {'file_names': filelist,
                                     'file_numbers': runNo,
                                     'file_times': start_times,
                                     scaler:
                                         self.make_results_dict_scaler(center_freqs, center_fit_errs, center_freqs_d,
                                                                       center_freqs_d_syst, fitpars, rChi),
                                     'color': self.isotope_colors[int(iso[:2])]
                                     }
                               }
                    TiTs.merge_extend_dicts(self.results, isodict, overwrite=True, force_overwrite=True)
            # for 55 Nickel just get the basic file info. Can't fit the single files:
            self.choose_runs_write_basics_to_results('54Ni')
            # export the results of initial fitting.
            self.export_results()

    def combine_single_scaler_centers(self, isolist, calibrated=False):
        """ use single scaler results and combine per file """
        sc_prefix = 'scaler_'
        if calibrated:
            if not 'sum' in isolist[0]:
                # For the single-files we use the scalers separate but with combined calibration
                sc_prefix += 'c'
            isolist = ['{}_cal'.format(i) for i in isolist]
        for iso in isolist:
            if 'sum' in iso:
                # for summed isos the accV dict for c012 scaler has not been created yet.
                accV_dict = self.results[iso]['scaler_012']['acc_volts']
                isodict = {iso: {'color': self.isotope_colors[int(iso[:2])],
                                 'scaler_c012': {'acc_volts': accV_dict}
                                 }}
                TiTs.merge_extend_dicts(self.results, isodict, overwrite=True, force_overwrite=True)
            # load fit results for all scalers from db
            iso_r = self.results[iso]
            file_names = iso_r['file_names']
            # create list for combined scaler centers and stat errs
            centers_combined = []
            centers_combined_d_fit = []
            centers_combined_d_stat = []
            centers_combined_d_syst = []
            for indx in range(len(file_names)):
                # load all scaler results with fit errors. Only these are important for the combination.
                val_arr = np.array([iso_r['{}{}'.format(sc_prefix, sc)]['center_fits']['vals'][indx]
                                    for sc in range(3)])
                err_arr = np.array([iso_r['{}{}'.format(sc_prefix, sc)]['center_fits']['d_fit'][indx]
                                    for sc in range(3)])

                # calculate weighted average and various error estimates
                wavg, wavg_d, wstd, std = self.calc_weighted_avg(val_arr, err_arr)
                if self.combined_unc == 'wavg_d':
                    d_fit = wavg_d
                elif self.combined_unc == 'wstd':
                    d_fit = wstd
                else:
                    d_fit = std

                # Append to combined results list. For error use the larger of standard deviation or weighted avg errors
                centers_combined.append(wavg)
                centers_combined_d_fit.append(d_fit)

                # statistic uncertainties
                # also kind of a fit ucnertainty but I'll add it to stat since it's determined with other fits
                if self.get_gate_analysis and not 'sum' in iso:
                    gatewidth_std = self.results[iso[:4]]['scaler_012']['bunchwidth_std_all']['vals'][indx]
                else:
                    gatewidth_std = self.bunch_structure_d
                # statistic uncertainty
                if calibrated:  # ion energy has been calibrated. Calibration may have a statistic error
                    d_ion_energy_stat = self.diff_dopplers[iso[:4]] * \
                                        self.results[iso]['scaler_c012']['acc_volts']['d_stat'][indx]
                else:
                    d_ion_energy_stat = 0
                d_stat = np.sqrt(d_fit ** 2 + gatewidth_std ** 2 + d_ion_energy_stat ** 2)
                centers_combined_d_stat.append(d_stat)
                # systematic uncertainties
                d_syst = iso_r['{}{}'.format(sc_prefix, 1)]['center_fits']['d_syst'][indx]  # same for all scalers
                centers_combined_d_syst.append(d_syst)

            # calculate weighted avg of center fit and various error estimates
            wavg, wavg_d, wstd, std = self.calc_weighted_avg(centers_combined, centers_combined_d_stat)
            if self.combined_unc == 'wavg_d':
                d_fit = wavg_d
            elif self.combined_unc == 'wstd':
                d_fit = wstd
            else:
                d_fit = std

            combined_dict = {'scaler_c012': {'center_fits': {'vals': centers_combined,
                                                             'd_fit': centers_combined_d_fit,
                                                             'd_stat': centers_combined_d_stat,
                                                             'd_syst': centers_combined_d_syst},
                                             'avg_center_fits': {'vals': [wavg],
                                                                 'd_fit': [d_fit],
                                                                 'd_stat': [d_fit],
                                                                 'd_syst': [centers_combined_d_syst[0]]},
                                             }}
            TiTs.merge_extend_dicts(self.results[iso], combined_dict, overwrite=True, force_overwrite=True)

            self.plot_parameter_for_isos_and_scaler([iso], ['scaler_c012'], 'center_fits', plotstyle='classic', plotAvg=True)

    def combine_single_scaler_results(self, parameter, isolist, calibrated=False):
        """ use single scaler results and combine per file """
        sc_prefix = 'scaler_'

        if calibrated:
            if not 'sum' in isolist[0]:
                # For the single-files we use the scalers separate but with combined calibration
                sc_prefix += 'c'
            isolist = ['{}_cal'.format(i) for i in isolist]


        for iso in isolist:
            if 'sum' in iso:
                # for summed isos the accV dict for c012 scaler has not been created yet.
                accV_dict = self.results[iso]['scaler_012']['acc_volts']
                isodict = {iso: {'color': self.isotope_colors[int(iso[:2])],
                                 'scaler_c012': {'acc_volts': accV_dict}
                                 }}
                TiTs.merge_extend_dicts(self.results, isodict, overwrite=True, force_overwrite=True)

            # load fit results for all scalers from db
            iso_r = self.results[iso]
            file_names = iso_r['file_names']
            # create list for combined scaler centers and stat errs
            centers_combined = []
            centers_combined_d_fit = []
            centers_combined_d_stat = []
            centers_combined_d_syst = []
            for indx in range(len(file_names)):
                # load all scaler results with fit errors. Only these are important for the combination.
                val_arr = np.array([iso_r['{}{}'.format(sc_prefix, sc)][parameter]['vals'][indx]
                                    for sc in range(3)])
                err_arr = np.array([iso_r['{}{}'.format(sc_prefix, sc)][parameter]['d_fit'][indx]
                                    for sc in range(3)])
                stat_arr = np.array([iso_r['{}{}'.format(sc_prefix, sc)][parameter]['d_stat'][indx]
                                    for sc in range(3)])
                stat_without_fit = np.sqrt(stat_arr**2 - err_arr**2)

                # calculate weighted average and various error estimates
                wavg, wavg_d, wstd, std = self.calc_weighted_avg(val_arr, err_arr)
                if self.combined_unc == 'wavg_d':
                    d_fit = wavg_d
                elif self.combined_unc == 'wstd':
                    d_fit = wstd
                else:
                    d_fit = std  # most conservative error

                # Append to combined results list. For error use the larger of standard deviation or weighted avg errors
                centers_combined.append(wavg)
                centers_combined_d_fit.append(d_fit)

                ''' uncertainties '''
                # statistic uncertainties
                stat_add = np.average(stat_without_fit)  # statistics except the fit. Should be same for all sc
                centers_combined_d_stat.append(np.sqrt(d_fit**2+stat_add**2))
                # systematic uncertainties
                d_syst = iso_r['{}{}'.format(sc_prefix, 1)][parameter]['d_syst'][indx]  # same for all scalers
                centers_combined_d_syst.append(d_syst)

            # calculate weighted avg of center fit and various error estimates
            wavg, wavg_d, wstd, std = self.calc_weighted_avg(centers_combined, centers_combined_d_stat)

            if self.combined_unc == 'wavg_d':
                d_fit = wavg_d
            elif self.combined_unc == 'wstd':
                d_fit = wstd
            else:
                d_fit = std  # (typically) most conservative estimate

            combined_dict = {'scaler_c012': {parameter: {'vals': centers_combined,
                                                             'd_fit': centers_combined_d_fit,
                                                             'd_stat': centers_combined_d_stat,
                                                             'd_syst': centers_combined_d_syst},
                                             'avg_{}'.format(parameter): {'vals': [wavg],
                                                                 'd_fit': [d_fit],
                                                                 'd_stat': [d_fit],
                                                                 'd_syst': [centers_combined_d_syst[0]]},
                                             }}
            TiTs.merge_extend_dicts(self.results[iso], combined_dict, overwrite=True, force_overwrite=True)

            self.plot_parameter_for_isos_and_scaler([iso], ['scaler_c012'], parameter, plotstyle='classic', plotAvg=True)

    def plot_results_of_fit(self, calibrated=False):
        add_sc = []
        isolist = ['58Ni', '60Ni']
        if calibrated:
            add_sc = ['scaler_c0', 'scaler_c1', 'scaler_c2']
            isolist = ['{}_cal'.format(i) for i in isolist]

        # plot iso-results for each scaler combination
        for sc in self.scaler_combinations+['scaler_c012']+add_sc:
            # write the scaler to db for usage
            scaler = self.update_scalers_in_db(sc)
            # plot results of first fit
            self.plot_parameter_for_isos_and_scaler(isolist, [scaler], 'center_fits', offset=[450, 0, -450], folder='fit_res')
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

    def ion_energy_calibration(self):
        """
        Separated from separate_runs_analysis on 11.05.2020.
        Calibration will be done for each scaler and written to results db.
        :return:
        """
        isolist = ['54Ni', '58Ni', '60Ni']
        for sc in self.scaler_combinations+['scaler_c012']:
            logging.info('\n'
                         '## ion energy calibration started for scaler {}'
                         .format(sc))
            # write the scaler to db for usage
            scaler = self.update_scalers_in_db(sc)

            # get the scan voltage from the center fits, that will be useful here
            self.all_centerFreq_to_scanVolt(isolist, [scaler])

            # calculate isotope shift and calibrate voltage
            ''' use interpolation. '''
            #  Since we effectively calibrate the buncher potential and we expect that to drift.
            # Meaning we have no influence on it so there's no justification for assigning values to each other...
            if 'absolute' in self.calibration_method:
                # use both the 58 and 60 Nickel absolute transition frequencies
                mean_offset_58 = self.getVoltDeviationFromAbsoluteTransFreq('58Ni')
                mean_offset_60 = self.getVoltDeviationFromAbsoluteTransFreq('60Ni')
                mean_offset = ((mean_offset_58[0] + mean_offset_60[0]) / 2, (mean_offset_58[1] + mean_offset_60[1]) / 2)
                for iso in isolist:
                    # '58' or '60' in self.calibration_method will lead to only this isotope being used for calibration
                    self.calibVoltageFunct(iso, scaler, useOnly=self.calibration_method)
            else:  # No Calibration
                # for other calibration methods see mid2020 script
                self.accVolt_corrected = (self.accVolt_set, self.accVolt_set_d)  # no large scale correction
                self.getVoltDeviationToResults('58Ni', allNull=True)
                self.getVoltDeviationToResults('60Ni', allNull=True)
                for iso in isolist:
                    self.calibVoltageFunct(iso, scaler)

    def fitting_calibrated_separate(self):
        """
        Separated from separate_runs_analysis on 12.05.2020.
        Repeat the fitting for all files and scalers with calibrations applied and write to results dict.
        :return:
        """
        if self.do_the_fitting:
            # fitting with calibrations for each iso/scaler from first round
            for sc in self.scaler_combinations:
                # write the scaler to db for usage
                scaler = self.update_scalers_in_db(sc)
                # Do the fitting for each isotope with calibrations applied
                for iso in ['58Ni_cal', '60Ni_cal', '56Ni_cal']:
                    # create new isotopes with calibrated voltage applied in db (already exist in self.results)
                    self.write_voltcal_to_db(iso, scaler)
                    # Do a second set of fits for all 56, 58 & 60 runs with calibration applied.
                    filelist, runNo, center_freqs, center_fit_errs, center_freqs_d, center_freqs_d_syst, start_times, fitpars, rChi = \
                        self.chooseAndFitRuns(iso)
                    isodict = {iso:
                                   {'file_names': filelist,
                                    'file_numbers': runNo,
                                    'file_times': start_times,
                                    scaler:
                                                self.make_results_dict_scaler(center_freqs, center_fit_errs, center_freqs_d,
                                                                              center_freqs_d_syst, fitpars, rChi),
                                    'color': self.isotope_colors[int(iso[:2])]
                                    }
                               }
                    TiTs.merge_extend_dicts(self.results, isodict, overwrite=True, force_overwrite=True)

            # separate fitting for calculated combination 'scaler_c012'
            for iso in ['58Ni_cal', '60Ni_cal', '56Ni_cal']:
                # create new isotopes with calibrated voltage applied in db (already exist in self.results)
                self.write_voltcal_to_db(iso, 'scaler_c012')
                for sc in ['scaler_c0', 'scaler_c1', 'scaler_c2']:  # new scaler names to separate from 0, 1, 2 with own cal
                    # write the scaler to db for usage
                    scaler = self.update_scalers_in_db(sc)
                    # Copy the used acc Volts from 'scaler_c012' to individual scalers. Might be needed later...
                    isodict = {iso: {'color': self.isotope_colors[int(iso[:2])],
                                     scaler: {'acc_volts': self.results[iso]['scaler_c012']['acc_volts']}}}
                    TiTs.merge_extend_dicts(self.results, isodict, overwrite=True, force_overwrite=True)
                    # Do a second set of fits for all 56, 58 & 60 runs with calibration applied.
                    filelist, runNo, center_freqs, center_fit_errs, center_freqs_d, center_freqs_d_syst, start_times, fitpars, rChi = \
                        self.chooseAndFitRuns(iso)
                    # created the isodict before, now add the results
                    scaler_dict = self.make_results_dict_scaler(center_freqs, center_fit_errs, center_freqs_d,
                                                                      center_freqs_d_syst, fitpars, rChi)
                    TiTs.merge_extend_dicts(self.results[iso][scaler], scaler_dict, overwrite=True, force_overwrite=True)

                self.plot_parameter_for_isos_and_scaler([iso], ['scaler_c0', 'scaler_c1', 'scaler_c2'], 'center_fits',
                                                        onlyfiterrs=True)

            self.export_results()

    def extract_isoshifts_from_fitres(self, isolist, refiso, calibrated=False):
        """
        isotope shift extraction.
        :return:
        """
        if calibrated:
            isolist = ['{}_cal'.format(i) for i in isolist]  #make sure to use the calibrated isos when calibration = True
        # calculate isotope shift and calibrate voltage
        for iso in isolist:
            for sc in self.results[iso].keys():
                if 'scaler' in sc:  # is a scaler key
                    logging.info('\n'
                                 '## extracting isotope shifts for scaler {}'.format(sc))
                    # write the scaler to db for usage
                    scaler = self.update_scalers_in_db(sc)
                    # interpolate between calibration tuples.
                    for iso in isolist:
                        self.extract_iso_shift_interp(iso, refiso, scaler, calibrated)
                        # if not scaler == 'scaler_c012':
                        self.plot_parameter_for_isos_and_scaler([iso], [scaler], 'shift_iso-{}'.format(refiso[:2]),
                                                                    plotstyle='classic', plotAvg=True, folder='combined')

    def create_and_fit_stacked_runs(self, calibration_per_file=False):
        #########################
        # stacked runs analysis #
        #########################
        '''
        All valid runs of one isotope are stacked/rebinned to a new single file.
        Calibrations and analysis are done based on stacked files and combined scalers.
        This enables us to get results out of 55 Nickel data.
        '''
        # switch to voigt profile here...
        # self.run = 'CEC_AsymVoigt'

        # combine runs to new 3-scaler files.
        self.ni_analysis_combined_files = []
        self.create_stacked_files(calibration_per_file)

        # update database to use all three scalers for analysis
        self.update_scalers_in_db('0,1,2')  # scalers to be used for combined analysis

        # do a batchfit of the newly created files
        isolist = ['58Ni_sum', '60Ni_sum', '54Ni_sum']
        isolist = ['{}_cal'.format(i) for i in isolist]

        if self.do_the_fitting:
            for iso in isolist:
                for sc in self.scaler_combinations:
                    scaler = self.update_scalers_in_db(sc)
                    # Create the isotope in results db already:
                    accV_dict = self.results['{}_cal'.format(iso[:4])]['scaler_012']['avg_acc_volts']
                    isodict = {iso: {'color': self.isotope_colors[int(iso[:2])],
                                     scaler: {'acc_volts': {'vals': accV_dict['vals'],
                                                            'd_stat': accV_dict['d_stat'],
                                                            'd_syst': accV_dict['d_syst']
                                                            }
                                              }}}
                    TiTs.merge_extend_dicts(self.results, isodict, overwrite=True, force_overwrite=True)
                    # Do the fitting
                    filelist, runNo, center_freqs, center_fit_errs, center_freqs_d, center_freqs_d_syst, start_times, fitpars, rChi = \
                        self.chooseAndFitRuns(iso)
                    if '55' in iso:
                        # extract additional hfs fit parameters
                        al = self.param_from_fitres_db(filelist[0], iso, self.run, 'Al')
                        au = self.param_from_fitres_db(filelist[0], iso, self.run, 'Au')
                        bl = self.param_from_fitres_db(filelist[0], iso, self.run, 'Bl')
                        bu = self.param_from_fitres_db(filelist[0], iso, self.run, 'Bu')
                        if au[2]:  # A ratio fixed
                            a_rat = au
                            au = (al[0]*au[0], al[1]*au[0], True)
                        else:  # A ratio free --> calculate
                            a_rat = (au[0]/al[0],
                                     np.sqrt(np.square(au[1] / al[0]) + np.square(au[0] * al[1] / al[0]**2)),
                                     False)
                        if bu[2]:  # B ratio fixed
                            b_rat = bu
                            bu = (bl[0]*bu[0], bl[1]*bu[0], True)
                        else:  # B ratio free --> calculate
                            b_rat = (bu[0]/bl[0],
                                     np.sqrt(np.square(bu[1] / bl[0]) + np.square(bu[0] * bl[1] / bl[0]**2)),
                                     False)
                        hfs_dict = {'Al': al, 'Au': au, 'Arat': a_rat, 'Bl': bl, 'Bu': bu, 'Brat': b_rat}
                    else:
                        hfs_dict = None
                    isodict = {iso:
                                   {'file_names': filelist,
                                    'file_numbers': runNo,
                                    'file_times': start_times,
                                    scaler:
                                        self.make_results_dict_scaler(center_freqs, center_fit_errs, center_freqs_d,
                                                                      center_freqs_d_syst, fitpars, rChi,
                                                                      hfs_pars=hfs_dict),
                                    'color': self.isotope_colors[int(iso[:2])]
                                    }
                               }
                    TiTs.merge_extend_dicts(self.results, isodict, overwrite=True, force_overwrite=True)
            self.export_results()

    def calculate_charge_radii(self, isolist, refiso, calibrated=False):
        """
        Charge radii extraction.
        :return:
        """
        if calibrated:
            isolist = ['{}_cal'.format(i) for i in isolist]  # make sure to use the calibrated isos

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
        all_isos = ['55Ni', '56Ni', '58Ni', '60Ni']

        # get final values
        for iso in all_isos:
            iso_use = '{}_cal'.format(iso)  # if possible, the single file data should be used!
            scaler = 'scaler_c012'  # Best approach is to fit each scaler separate and then combine.
            description_str = 'BECOLA2018. Files and scalers fitted separate, calibrated'
            if '55' in iso:
                iso_use = '55Ni_sum_cal'  # for 55Ni, single file data is unuseable. Only summing up yields a spectrum
                scaler = 'scaler_012'  # We need all the data we can get here, so we sum all scalers before fitting
                description_str = 'BECOLA2018. Files summed, scalers fitted combined, calibrated'

            iso_dict = self.results[iso_use][scaler]
            final_vals_iso = {'center_fits':
                                      {'vals': iso_dict['avg_center_fits']['vals'],
                                       'd_stat': iso_dict['avg_center_fits']['d_stat'],
                                       'd_syst': iso_dict['avg_center_fits']['d_syst']},
                              'shift_iso-{}'.format(self.ref_iso[:2]):
                                      {'vals': iso_dict['avg_shift_iso-{}'.format(self.ref_iso[:2])]['vals'],
                                       'd_stat': iso_dict['avg_shift_iso-{}'.format(self.ref_iso[:2])]['d_stat'],
                                       'd_syst': iso_dict['avg_shift_iso-{}'.format(self.ref_iso[:2])]['d_syst']},
                              'delta_rms_iso-{}'.format(self.ref_iso[:2]):
                                      {'vals': iso_dict['avg_delta_rms_iso-{}'.format(self.ref_iso[:2])]['vals'],
                                       'd_stat': iso_dict['avg_delta_rms_iso-{}'.format(self.ref_iso[:2])]['d_stat'],
                                       'd_syst': iso_dict['avg_delta_rms_iso-{}'.format(self.ref_iso[:2])]['d_syst']},
                              }

            self.results[iso]['final'] = final_vals_iso

        # TODO: also get final A, B values and calculate µ, Q
        self.plot_results_table(ref_iso=self.ref_iso)

        # self.export_results()

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

    def update_gates_in_db(self, iso, midtof, gatewidth, delaylist):
        '''
        Write all parameters relevant for the software gate position into the database
        :param midtof: float: center of software gate in µs
        :param gatewidth: float: width of software gate in µs
        :param delaylist: list of floats: list of delays in midtof for scalers 0,1,2
        :return:
        '''
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''UPDATE Runs SET softwGateWidth = ? WHERE run = ?''', (gatewidth, self.run))
        cur.execute('''UPDATE Runs SET softwGateDelayList = ? WHERE run = ?''', (str(delaylist), self.run))
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

    def choose_runs_write_basics_to_results(self, iso, selecttuple=None):
        """
        Just find all files of iso in db and write basic info to db. Optional: Only in range of selecttuple
        :param iso: str: isotope to pick
        :param selecttuple: tuple: (lowest, highest)  filenumbers to include
        :return:
        """
        ###################
        db_like = iso+'%'
        # select files
        filelist, filenums, filedates = self.pick_files_from_db_by_type_and_num(db_like, selecttuple)
        self.results[iso] = {'file_numbers': filenums,
                                 'file_names': filelist,
                                 'file_times': filedates,
                                 'color': self.isotope_colors[int(db_like[:2])]
                                 }

    def chooseAndFitRuns(self, iso, reset=None):
        '''

        :param iso: str: example '58Ni%'
        :param reset: str: if a string is given, all files type will be reset to this string
        :return: filelist
        '''
        db_like = iso+'%'
        # select files
        filelist, runNos, filedates = self.pick_files_from_db_by_type_and_num(db_like)
        filearray = np.array(filelist)  # needed for batch fitting
        # do reset if necessary
        if reset:
            self.reset(db_like, reset)

        fixlist = []
        while filearray.__len__() > 0:
            par_guess_copy = self.initial_par_guess.copy()
            for par, guess in par_guess_copy.items():
                if par in fixlist:
                    par_guess_copy[par] = (guess[0], True)  # fix to guess value
            self.write_to_db_lines(self.run,
                                   sigma=par_guess_copy['sigma'],
                                   gamma=par_guess_copy['gamma'],
                                   asy=par_guess_copy['asy'],
                                   dispersive=par_guess_copy['dispersive'],
                                   centerAsym=par_guess_copy['centerAsym'],
                                   IntAsym=par_guess_copy['IntAsym'],
                                   nPeaksAsym=par_guess_copy['nPeaksAsym'])

            # do the batchfit
            if self.do_the_fitting:
                # define and create (if not exists) the output folder
                plot_specifier = 'plots\\' + self.scaler_name + '_' + iso[4:] + '\\'  # e.g. scaler_012_cal
                plot_folder = os.path.join(self.resultsdir, plot_specifier)
                if not os.path.exists(plot_folder):
                    os.makedirs(plot_folder)
                # for softw_gates_trs from file use 'File' and from db use None.
                BatchFit.batchFit(filearray, self.db, self.run, x_as_voltage=True, softw_gates_trs=None, guess_offset=True,save_to_folder=plot_folder)
            filearray = []  # all files fitted. May be filled with files that need a second fit.
            # get fitresults (center) vs run for 58
            all_rundate = []
            all_center_MHz = []
            all_center_MHz_fiterrs = []  # only the fit errors, nothing else!
            all_center_MHz_d = []  # real statistic uncertainties
            all_center_MHz_d_syst = []
            all_rChi = []
            all_fitpars = []
            # get fit results
            for indx, files in enumerate(filelist):
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
                # Query rChi from fitRes
                cur.execute(
                    '''SELECT rChi FROM FitRes WHERE file = ? AND iso = ? AND run = ?''', (files, file_type, self.run))
                rChi = cur.fetchall()
                con.close()
                try:
                    # if the fit went wrong there might not be a value to get from the fitpars...
                    parsdict = ast.literal_eval(pars[0][0])
                except Exception as e:
                    # Fit went wrong!
                    # replace with standard value and big error...
                    parsdict = None
                    filearray.append(files)
                all_fitpars.append(parsdict)
                for par, valstuple in parsdict.items():
                    if 'sy' in par:  # only asymmetry parameters for now
                        if type(valstuple[2]) == list:  # range specified
                            if valstuple[0]-valstuple[1] < valstuple[2][0] or valstuple[0]+valstuple[1] > valstuple[2][1]:
                                # value plus error outside bounds. Probably not a good fit result. Try fixing this par!
                                filearray.append(files)  # add file to list of 'fit again'
                                fixlist.append(par)

                all_rundate.append(file_date)
                all_center_MHz.append(parsdict['center'][0])
                all_rChi.append(rChi[0][0])

                # === uncertainties ===

                # == statistic uncertainites (changing on a file-to-file basis):
                # fit uncertainty
                d_fit = parsdict['center'][1]
                all_center_MHz_fiterrs.append(d_fit)
                # also kind of a fit ucnertainty but I'll add it to stat since it's determined with other fits
                if self.get_gate_analysis and not 'sum' in iso:
                    gatewidth_std = self.results[iso[:4]]['scaler_012']['bunchwidth_std_all']['vals'][indx]
                else:
                    gatewidth_std = self.bunch_structure_d
                # statistic uncertainty
                if 'cal' in iso:  # ion energy has been calibrated. Calibration may have a statistic error
                    d_ion_energy_stat = self.diff_dopplers[iso[:4]] * \
                                        self.results[iso][self.scaler_name]['acc_volts']['d_stat'][indx]
                else:
                    d_ion_energy_stat = 0
                d_stat = np.sqrt(d_fit**2 + gatewidth_std**2 + d_ion_energy_stat**2)
                all_center_MHz_d.append(d_stat)

                # == systematic uncertainties (same for all files):
                if 'cal' in iso:  # ion energy has been calibrated. Uncertainties from calibration
                    d_ion_energy_syst = self.diff_dopplers[iso[:4]] * \
                                   (self.results[iso][self.scaler_name]['acc_volts']['d_syst'][indx]
                                    + self.matsuada_volts_d)  # TODO: Should matsuada be added in quadrature here?
                else:  # not calibrated. Uncertainty from buncher potential
                    d_ion_energy_syst = self.diff_dopplers[iso[:4]]*(self.accVolt_set_d + self.matsuada_volts_d)  # not statistic
                d_laser_syst = np.sqrt(self.wavemeter_wsu30_mhz_d**2 + self.heliumneon_drift**2)
                d_alignment = self.laserionoverlap_MHz_d
                d_fitting_syst = self.lineshape_d_syst  # self.bunch_structure_d replaced by gate analysis statistic
                # combine all above quadratically
                d_syst = np.sqrt(d_ion_energy_syst**2 + d_laser_syst**2 + d_alignment**2 + d_fitting_syst**2)
                all_center_MHz_d_syst.append(d_syst)
            filearray = np.array(filearray)

        return filelist, runNos, all_center_MHz, all_center_MHz_fiterrs, all_center_MHz_d, all_center_MHz_d_syst, all_rundate, all_fitpars, all_rChi

    def centerFreq_to_absVoltage(self, isostring, deltanu, nu_d, nu_dsyst):
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
        nuL = db_laserfreq[0][0]
        nuoff = self.restframe_trans_freq[self.ref_iso][0]

        velo = Physics.invRelDoppler(nuL, nuoff+deltanu)
        ener = Physics.relEnergy(velo, m*Physics.u)
        volt = ener/Physics.qe  # convert energy to voltage

        diffdopp = Physics.diffDoppler(nuoff + deltanu, volt, m)
        d = nu_d/diffdopp
        d_syst = nu_dsyst/diffdopp
        return volt, d, d_syst

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

    def get_weighted_avg_linepars(self, iso_list, scaler_list):
        """
        Extract a weighted average of all line parameters to fix them later
        :param iso_list:
        :param scaler_list:
        :return:
        """
        for isostring in iso_list:
            for sc in scaler_list:
                scaler = self.update_scalers_in_db(sc)
                fitres_list = self.results[isostring][scaler]['all_fitpars']
                fitres_avg = {}  # empty dict for storing avgs
                for par, vals in fitres_list[0].items():  # just take the first file to get to the pars. Same for all files...
                    par_vals = np.array([i[par][0] for i in fitres_list])
                    par_errs = np.array([i[par][1] for i in fitres_list])
                    par_fixed = vals[2]

                    if not par_fixed:  # averaging only makes sense for non-fixed parameters
                        weights = 1 / par_errs ** 2
                        pars_avg = np.sum(weights * par_vals) / np.sum(weights)
                        pars_avg_d = np.sqrt(1 / np.sum(weights))
                    elif type(par_fixed) == list:  # bound parameters. Take care of values close to bounds
                        low_bound, up_bound = par_fixed
                        hit_bounds_indices = []
                        for indx, val in enumerate(par_vals):
                            if val-low_bound < (up_bound-low_bound)/1000 or up_bound-val < (up_bound-low_bound)/1000:
                                hit_bounds_indices.append(indx)
                        if len(hit_bounds_indices) < len(par_vals):  # not all values hit the bounds
                            # delete the ones that hit the bounds; only use others for average
                            par_vals = np.delete(par_vals, hit_bounds_indices)
                            par_errs = np.delete(par_errs, hit_bounds_indices)
                            # now do the weighted average
                            weights = 1 / par_errs ** 2
                            pars_avg = np.sum(weights * par_vals) / np.sum(weights)
                            pars_avg_d = np.sqrt(1 / np.sum(weights))
                        else:  # all values hit the bounds
                            weights = 1 / par_errs ** 2
                            pars_avg = np.sum(weights * par_vals) / np.sum(weights)
                            pars_avg_d = np.sqrt(1 / np.sum(weights))
                    else:
                        pars_avg = vals[0]
                        pars_avg_d = vals[1]

                    fitres_avg[par] = (pars_avg, pars_avg_d, par_fixed)
                self.results[isostring][scaler]['avg_fitpars'] = fitres_avg

    def getVoltDeviationFromAbsoluteTransFreq(self, iso):
        #  Relative calibration frequency is the isotope shift. If iso=ref_iso then that should be 0 MHz
        rel_cal_freq = self.restframe_trans_freq[iso[:4]][0] - self.restframe_trans_freq[self.ref_iso][0]

        # Get center dac data from previous fit results
        file_times = self.results[iso]['file_times']
        center_freq_dict = self.results[iso][self.scaler_name]['center_fits']
        rel_center = np.array(center_freq_dict['vals'])
        rel_center_d = np.array(center_freq_dict['d_stat'])
        rel_center_d_sys = np.array(center_freq_dict['d_syst'])  # This includes the uncertainty we want to reduce!!!
        rel_center_d_sys = np.sqrt(np.square(rel_center_d_sys)
                                   - np.square(self.accVolt_set_d * self.diff_dopplers[iso[:4]])
                                   )  # remove buncher potential uncertainty since we calibrate that now.

        volt_dev = -(rel_center - rel_cal_freq)/self.diff_dopplers[iso[:4]] #(neg) acc.voltages are stored as positive, so we need the -
        volt_dev_d = rel_center_d/self.diff_dopplers[iso[:4]]
        # systematics from absolute center uncertainty:
        volt_dev_d_sys = np.sqrt(np.square(rel_center_d_sys) + np.square(self.restframe_trans_freq[iso[:4]][1])
                                 )/self.diff_dopplers[iso[:4]]
        # correction dict
        volt_dev_dict = {'vals': volt_dev.tolist(),
                         'd_stat': volt_dev_d.tolist(),
                         'd_syst': volt_dev_d_sys.tolist()}
        self.results[iso][self.scaler_name]['voltage_deviation'] = volt_dev_dict

        return (volt_dev.mean(), volt_dev_d.mean()+volt_dev_d_sys.mean())

    def getVoltDeviationToResults(self, iso, offset=None, allNull=False):
        """
        Since the absolute transition frequency must be a constant, we can use it to correct some experimental parameters.
        The measured value depends on laser freq and absolute acceleration voltage with the laser frequency contributing
        directly (1MHz -> 1MHz) while the voltages contributed through diff dooplers (1V -> 15MHz). Since we think, that
        the laser is under control on a 1MHz scale, larger deviations can be attributed to the acceleration voltage.
        :param iso:
        :param offset: tuple or None: (offset, offset_err) to be added to all results
        :param allNull: bool: option to set all entries to zero, basically skipping this function.
        :return:
        """
        # Get center dac data from previous fit results
        file_times = self.results[iso]['file_times']
        center_dac_dict = self.results[iso][self.scaler_name]['center_scanvolt']
        scanvolt = np.array(center_dac_dict['vals'])
        scanvolt_d = np.array(center_dac_dict['d_stat'])
        scanvolt_d_sys = np.array(center_dac_dict['d_syst'])

        # deviation from mean value
        scanvolt_dev = -(scanvolt - scanvolt.mean())  # scanvoltages are neg but acc.volt ist stored positive. Switch sign
        scanvolt_dev_d = scanvolt_d  # errors propagate unaltered
        scanvolt_dev_d_sys = scanvolt_d_sys

        # if there is an offset, add it to all values
        if offset is not None:
            scanvolt_dev += offset[0]
            scanvolt_dev_d_sys += offset[1]

        # if this function is not inteded to give results:
        if allNull:
            scanvolt_dev *= 0
            scanvolt_dev_d *= 0
            scanvolt_dev_d_sys *= 0

        # correction dict
        volt_dev_dict = {'vals': scanvolt_dev.tolist(),
                            'd_stat': scanvolt_dev_d.tolist(),
                            'd_syst': scanvolt_dev_d_sys.tolist()}
        self.results[iso][self.scaler_name]['voltage_deviation'] = volt_dev_dict

    def calibVoltageFunct(self, isotope, scaler, useOnly=None, userefscaler=False):
        """
        return a voltage calibration based on a datetime object using the offset from literature IS and the deviations
        from constant transition frequency.
        :param datetime_obj:
        :return:
        """
        scaler = self.update_scalers_in_db(scaler)
        if userefscaler:
            ref_scaler = userefscaler
        else:
            ref_scaler = scaler

        # get the global voltage offset from isotope shift correction
        volt_offset, volt_offset_d = self.accVolt_corrected

        # get info like run times and voltages from database
        # For 58Ni reference
        ni58_numbers = self.results['58Ni']['file_numbers']
        dates_58 = self.results['58Ni']['file_times']
        volt58_dev = self.results['58Ni'][scaler]['voltage_deviation']['vals']
        volt58_dev_d = self.results['58Ni'][scaler]['voltage_deviation']['d_stat']
        volt58_dev_d_syst = self.results['58Ni'][scaler]['voltage_deviation']['d_syst']
        # For 60Ni reference
        ni60_numbers = self.results['60Ni']['file_numbers']
        dates_60 = self.results['60Ni']['file_times']
        volt60_dev = self.results['60Ni'][scaler]['voltage_deviation']['vals']
        volt60_dev_d = self.results['60Ni'][scaler]['voltage_deviation']['d_stat']
        volt60_dev_d_syst = self.results['60Ni'][scaler]['voltage_deviation']['d_syst']
        # For the isotope to be calibrated
        iso_times = self.results[isotope]['file_times']
        iso_names = self.results[isotope]['file_names']
        iso_numbers = self.results[isotope]['file_numbers']
        iso_color = self.results[isotope]['color']

        # make floats (seconds relative to reference-time) out of the dates
        iso_dates = list((t - self.ref_datetime).total_seconds() for t in iso_times)
        ref_58_dates = list((t - self.ref_datetime).total_seconds() for t in dates_58)
        ref_60_dates = list((t - self.ref_datetime).total_seconds() for t in dates_60)

        # use np.interp to assign voltage deviations to the requested run.
        interpolation_58 = np.interp(iso_dates, ref_58_dates, [volt_offset-v for v in volt58_dev])
        interpolation_58_d = np.interp(iso_dates, ref_58_dates, volt58_dev_d)
        interpolation_58_d_syst = np.interp(iso_dates, ref_58_dates, volt58_dev_d_syst)

        # use np.interp to assign voltage deviations to the requested run.
        interpolation_60 = np.interp(iso_dates, ref_60_dates, [volt_offset - v for v in volt60_dev])
        interpolation_60_d = np.interp(iso_dates, ref_60_dates, volt60_dev_d)
        interpolation_60_d_syst = np.interp(iso_dates, ref_60_dates, volt60_dev_d_syst)

        # calculate the final voltage value combining the interpolated deviation with the overall offset
        if '58' in useOnly:
            # use only the 58 Nickel interpolation
            voltcorrect = interpolation_58
            voltcorrect_d = interpolation_58_d  # the overall correction is completely systematic (same for all files)
            voltcorrect_d_syst = interpolation_58_d_syst
        elif '60' in useOnly:
            # use only the 60 Nickel interpolation
            voltcorrect = interpolation_60
            voltcorrect_d = interpolation_60_d  # the overall correction is completely systematic (same for all files)
            voltcorrect_d_syst = interpolation_60_d_syst
        else:  # standard case
            # combine 58 and 60 Nickel references
            voltcorrect = (interpolation_58 + interpolation_60)/2
            sqare_err = 1/2*np.sqrt(interpolation_58_d**2 + interpolation_60_d**2)  # the overall correction is completely systematic (same for all files)
            voltcorrect_d = np.array([max(abs(interpolation_58[i]-voltcorrect[i]), sqare_err[i])
                                      for i in range(len(voltcorrect))])
            voltcorrect_d_syst = (interpolation_58_d_syst + interpolation_60_d_syst)/2  # should be the same anyways

        # calculate weighted avg of center fit and various error estimates
        wavg, wavg_d, wstd, std = self.calc_weighted_avg(voltcorrect, voltcorrect_d)
        if self.combined_unc == 'wavg_d':
            d_fit = wavg_d
        elif self.combined_unc == 'wstd':
            d_fit = wstd
        else:
            d_fit = std

        # write calibration voltages back into database
        isotope_cal = '{}_cal'.format(isotope)
        voltdict = {isotope_cal: {scaler: {'acc_volts': {'vals': voltcorrect.tolist(),
                                                         'd_stat': voltcorrect_d.tolist(),
                                                         'd_syst': voltcorrect_d_syst.tolist()},
                                           'avg_acc_volts': {'vals': [wavg],
                                                             'd_fit': [d_fit],
                                                             'd_stat': [d_fit],
                                                             'd_syst': [voltcorrect_d_syst[0]]}
                                           },
                                  'file_names': iso_names,
                                  'file_numbers': iso_numbers,
                                  'file_times': iso_times,
                                  'color': iso_color
                                  },
                    isotope: {scaler: {'acc_volts': {'vals': voltcorrect.tolist(),
                                                         'd_stat': voltcorrect_d.tolist(),
                                                         'd_syst': voltcorrect_d_syst.tolist()},
                                           'avg_acc_volts': {'vals': [wavg],
                                                             'd_fit': [d_fit],
                                                             'd_stat': [d_fit],
                                                             'd_syst': [voltcorrect_d_syst[0]]}
                                           }
                    }}
        TiTs.merge_extend_dicts(self.results, voltdict)

        plot_all = True

        if plot_all or scaler == 'scaler_c012':
            # make a quick plot of references and calibrated voltages
            fig, ax = plt.subplots()

            ref_58_timedeltas = np.array([timedelta(seconds=s) for s in ref_58_dates])
            ref_58_dates = np.array(self.ref_datetime + ref_58_timedeltas)  # convert back to datetime

            ref_60_timedeltas = np.array([timedelta(seconds=s) for s in ref_60_dates])
            ref_60_dates = np.array(self.ref_datetime + ref_60_timedeltas)  # convert back to datetime

            iso_timedeltas = np.array([timedelta(seconds=s) for s in iso_dates])
            iso_dates = np.array(self.ref_datetime + iso_timedeltas)  # convert back to datetime

            ax.plot(ref_58_dates, volt_offset - np.array(volt58_dev), '--', color='black', label='58Ni reference')
            ax.plot(ref_60_dates, volt_offset - np.array(volt60_dev), '--', color='blue', label='60Ni reference')
            # plot error band for statistical errors
            ax.fill_between(ref_58_dates,
                             volt_offset - np.array(volt58_dev) - volt58_dev_d,
                             volt_offset - np.array(volt58_dev) + volt58_dev_d,
                             alpha=0.5, edgecolor='black', facecolor='black')
            ax.fill_between(ref_60_dates,
                            volt_offset - np.array(volt60_dev) - volt60_dev_d,
                            volt_offset - np.array(volt60_dev) + volt60_dev_d,
                            alpha=0.5, edgecolor='blue', facecolor='blue')
            # plot error band for systematic errors on top of statistical errors
            ax.fill_between(ref_58_dates,
                             volt_offset - np.array(volt58_dev) - volt58_dev_d_syst - volt58_dev_d,
                             volt_offset - np.array(volt58_dev) + volt58_dev_d_syst + volt58_dev_d,
                             alpha=0.2, edgecolor='black', facecolor='black')
            ax.fill_between(ref_60_dates,
                            volt_offset - np.array(volt60_dev) - volt60_dev_d_syst - volt60_dev_d,
                            volt_offset - np.array(volt60_dev) + volt60_dev_d_syst + volt60_dev_d,
                            alpha=0.2, edgecolor='blue', facecolor='blue')
            # and finally plot the interpolated voltages
            ax.errorbar(iso_dates, voltcorrect, yerr=voltcorrect_d+voltcorrect_d_syst, marker='s', linestyle='None',
                        color=iso_color, label='{} interpolated'.format(isotope))
            # make x-axis dates
            plt.xlabel('date')
            days_fmt = mpdate.DateFormatter('%d.%B')
            ax.xaxis.set_major_formatter(days_fmt)
            plt.legend(loc='best')
            plt.xticks(rotation=45)  # rotate date labels 45 deg for better readability
            plt.margins(0.05)
            if self.save_plots_to_file:
                filename = 'voltInterp_' + isotope[:2] + '_sc' + scaler.split('_')[1]
                folder = 'calibration\\'
                plot_folder = self.resultsdir + folder
                if not os.path.exists(plot_folder):
                    os.makedirs(plot_folder)
                plt.savefig(plot_folder + filename + '.png', bbox_inches="tight")
            else:
                plt.show()
            plt.close()
            plt.clf()

    def write_voltcal_to_db(self, iso_cal, scaler):
        """

        :param iso:
        :return:
        """
        iso_orig = iso_cal[:-4]  # e.g. 58Ni_cal --> 58Ni
        file_names = self.results[iso_cal]['file_names']
        file_nums = self.results[iso_cal]['file_numbers']
        calvolt = self.results[iso_cal][scaler]['acc_volts']['vals']
        # Write calibrations to XML database
        print('Updating self.db with new voltages now...')
        for file in file_names:
            ind = file_names.index(file)
            new_voltage = calvolt[ind]
            fileno = file_nums[ind]

            # Update 'Files' in self.db
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            cur.execute('''UPDATE Files SET accVolt = ?, type = ? WHERE file = ? ''',
                        (new_voltage, '{}_{}'.format(iso_cal, fileno), file))
            con.commit()
            con.close()

            # Create new isotopes in self.db
            self.create_new_isotope_in_db(iso_orig, '{}_{}'.format(iso_cal, fileno), new_voltage)
        print('...self.db update completed!')

    def fit_time_projections(self, cts_axis, time_axis):
        x = time_axis
        y = cts_axis
        # estimates:: amplitude: sigma*sqrt(2pi)*(max_y-min_y), sigma=10, center:position of max_y, offset: min_y
        start_pars = np.array(
            [10 * 2.51 * (max(cts_axis) - min(cts_axis)), 10, np.argwhere(cts_axis == max(cts_axis))[0, 0],
             min(cts_axis)])
        popt, pcov = curve_fit(self.fitfunc, x, y, start_pars)
        ampl, sigma, center, offset = popt
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
        return o + a * 1 / (s * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * np.power((t - t0) / s, 2))

    ''' isoshift related: '''

    def extract_iso_shift_interp(self, isotope, reference, scaler, calibrated=True):
        scaler = self.update_scalers_in_db(scaler)
        ref = '{}{}'.format(reference[:2], isotope[2:])

        # Get info for isotope of interest
        iso_files = self.results[isotope]['file_names']
        iso_numbers = self.results[isotope]['file_numbers']
        iso_dates = self.results[isotope]['file_times']
        iso_center = self.results[isotope][scaler]['center_fits']['vals']
        iso_center_d_fit = self.results[isotope][scaler]['center_fits']['d_fit']
        iso_center_d_stat = self.results[isotope][scaler]['center_fits']['d_stat']
        iso_center_d_syst = self.results[isotope][scaler]['center_fits']['d_syst']  # only accVolt, affects both isos
        if calibrated:
            iso_volts = self.results[isotope][scaler]['acc_volts']['vals']
            iso_volts_d_stat = self.results[isotope][scaler]['acc_volts']['d_stat']
            iso_volts_d_syst = self.results[isotope][scaler]['acc_volts']['d_syst']
        else:
            iso_volts = np.full((len(iso_numbers)), self.accVolt_set)
            iso_volts_d_stat = np.full((len(iso_numbers)), 0)
            iso_volts_d_syst = np.full((len(iso_numbers)), self.accVolt_set_d)

        # Get info for 58 Nickel:
        ref_dates = self.results[ref]['file_times']
        ref_numbers = self.results[ref]['file_numbers']
        ref_center = self.results[ref][scaler]['center_fits']['vals']
        ref_center_d_fit = self.results[ref][scaler]['center_fits']['d_fit']
        ref_center_d_stat = self.results[ref][scaler]['center_fits']['d_stat']
        ref_center_d_syst = self.results[ref][scaler]['center_fits']['d_syst']  # only accVolt, affects both isos

        # isotope shift in interpolation is not trivial, since the combination of 2 runs is not clear anymore.
        # I'll do an interpolation here as well...
        # make floats (seconds relative to reference-time) out of the dates
        zero_time = self.ref_datetime
        ref_dates = list((t - zero_time).total_seconds() for t in ref_dates)
        iso_dates = list((t - zero_time).total_seconds() for t in iso_dates)
        # use np.interp to get a matching Nickel 58 center frequency for each Nickel 56 run.
        ref_center_interp = np.interp(iso_dates, ref_dates, ref_center)
        ref_center_d_fit_interp = np.interp(iso_dates, ref_dates, ref_center_d_fit)
        ref_center_d_stat_interp = np.interp(iso_dates, ref_dates, ref_center_d_stat)
        ref_center_d_syst_interp = np.interp(iso_dates, ref_dates, ref_center_d_syst)

        # calculate isotope shifts now:
        iso_shifts = np.array(iso_center) - np.array(ref_center_interp)

        ''' UNCERTAINTIES '''
        # calculate delta diff dopplers. Needed for voltage uncertainties.
        delta_diff_doppler = self.diff_dopplers[isotope[:4]] - self.diff_dopplers[reference[:4]]

        # fit errors only:
        iso_shifts_d_fit = np.sqrt(np.array(iso_center_d_fit) ** 2 + np.array(ref_center_d_fit_interp) ** 2)
        # statistical errors:
        if self.get_gate_analysis and not 'sum' in isotope:
            # get bunch structure statistic uncertainty
            iso_bunch_std = self.results[isotope[:4]]['scaler_012']['bunchwidth_std_all']['vals']
            ref_bunch_std = self.results[reference[:4]]['scaler_012']['bunchwidth_std_all']['vals']
            ref_bunch_std_interp = np.interp(iso_dates, ref_dates, ref_bunch_std)
            bunch_unc = np.sqrt(np.array(iso_bunch_std)**2 + np.array(ref_bunch_std_interp)**2)
        else:
            bunch_unc = self.bunch_structure_d
        ion_energy_d_stat = np.array(iso_volts_d_stat) * delta_diff_doppler  # calibrated voltage may have statistic unc
        # combined statistic:
        iso_shifts_d_stat = np.sqrt(iso_shifts_d_fit**2 + ion_energy_d_stat**2 + bunch_unc**2)

        # systematic errors:
        ion_energy_d_syst = (np.array(iso_volts_d_syst) + self.matsuada_volts_d) * delta_diff_doppler  # calibration for iso is interpolated from ref, so only that value needs to go in here
        laser_freq_d_syst = self.wavemeter_wsu30_mhz_d  # Laser is changed between isotopes.  TODO: HeNe drift?
        alignment_d_syst = self.laserionoverlap_MHz_d  # Can't guarantee the alignment is still the same...

        iso_shifts_d_syst = np.sqrt(ion_energy_d_syst**2
                                    + laser_freq_d_syst**2
                                    + alignment_d_syst**2)

        # calculate an average value using weighted avg
        iso_shift_avg, wavg_d, wstd, std = self.calc_weighted_avg(iso_shifts, iso_center_d_stat)
        if self.combined_unc == 'wavg_d':
            iso_shift_avg_d = wavg_d
        elif self.combined_unc == 'wstd':
            iso_shift_avg_d = wstd
        else:
            iso_shift_avg_d = std  # most conservative error
        iso_shift_avg_d_syst = sum(iso_shifts_d_syst)/len(iso_shifts_d_syst)  # should all be the same anyways

        if isotope[:2] == reference[:2]:
            # This is the reference. Set all values to 0(0)[0]
            iso_shifts = np.zeros(len(iso_shifts))
            iso_shifts_d_fit = np.zeros(len(iso_shifts))
            iso_shifts_d_stat = np.zeros(len(iso_shifts_d_stat))
            iso_shifts_d_syst = np.zeros(len(iso_shifts_d_syst))
            iso_shift_avg = 0
            iso_shift_avg_d = 0
            iso_shift_st_dev = 0
            iso_shift_avg_d_syst = 0

        # write isoshift to results dict
        shift_dict = {isotope: {scaler: {'shift_iso-{}'.format(reference[:2]): {'vals': iso_shifts.tolist(),
                                                                                'd_fit': iso_shifts_d_fit.tolist(),
                                                                                 'd_stat': iso_shifts_d_stat.tolist(),
                                                                                 'd_syst': iso_shifts_d_syst.tolist()},
                                         'avg_shift_iso-{}'.format(reference[:2]): {'vals': [iso_shift_avg],
                                                                                    'd_stat': [iso_shift_avg_d],
                                                                                    'd_syst': [iso_shift_avg_d_syst]}
                                         }}}
        TiTs.merge_extend_dicts(self.results, shift_dict, overwrite=True, force_overwrite=True)

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

    def create_stacked_files(self, calibration_per_file=False):

        # we always want to use the calibrated isotopes as a starting point:
        c = ''  # since we stack on a scan-volt basis, it does not really matter here

        # stack nickel 58 runs to new file Sum58_9999.xml. Only use calibration runs
        ni58_files, ni58_filenos, ni58_filetimes = self.pick_files_from_db_by_type_and_num('%58Ni%')
        self.stack_runs('58Ni{}'.format(c), ni58_files, (1300, 1400), binsize=1, bake_in_calib=calibration_per_file)
        # stack nickel 60 runs to new file Sum60_9999.xml. Only use calibration runs
        ni60_files, ni60_filenos, ni60_filetimes = self.pick_files_from_db_by_type_and_num('%60Ni%')
        self.stack_runs('60Ni{}'.format(c), ni60_files, (1300, 1400), binsize=1, bake_in_calib=calibration_per_file)
        # select and stack nickel 54 runs to new file Sum54_9999.xml
        ni54_files, ni54_filenos, ni54_filetimes = self.pick_files_from_db_by_type_and_num('%54Ni%', selecttuple=(10008, 10151))  # 6315 , selecttuple=(10008, 10151)
        self.stack_runs('54Ni{}'.format(c), ni54_files, (1300, 1400), binsize=1, bake_in_calib=calibration_per_file)

    def stack_runs(self, isotope, files, volttuple, binsize, bake_in_calib=False):
        ##############
        # stack runs #
        ##############
        # sum all the isotope runs
        # self.time_proj_res_per_scaler = self.stack_time_projections(isotope, files)
        self.addfiles_trs(isotope, files, volttuple, binsize, bake_in_calib)

    def stack_time_projections(self, isotope, filelist):
        zeroarr_sc = np.zeros(1024)  # array for one scaler
        zeroarr = np.array([zeroarr_sc.copy(), zeroarr_sc.copy(), zeroarr_sc.copy()])
        timebins = np.arange(1024)
        for files in filelist:
            filepath = os.path.join(self.datafolder, files)
            # load the spec data from file
            spec = XMLImporter(path=filepath)
            for sc_no in range(3):
                for track in range(spec.nrTracks):
                    # sum time projections for each scaler
                    zeroarr[sc_no] += spec.t_proj[track][sc_no]
        logging.info('------- time projection fit results: --------')
        timeproj_res = {'scaler_0':{}, 'scaler_1':{}, 'scaler_2':{}}
        for sc_no in range(3):
            # fit time-projection
            popt, perr = self.fit_time_projections(zeroarr[sc_no], timebins)
            ampl, sigma, center, offset = popt
            ampl_d, sigma_d, center_d, offset_d = perr
            # create plot for this scaler
            plt.plot(timebins, zeroarr[sc_no], '.', label='scaler_{}\n'
                                                         'midTof: {:.2f}({:.0f})\n'
                                                         'sigma: {:.2f}({:.0f})'.format(sc_no,
                                                                                        center, 100*center_d,
                                                                                        sigma, 100*sigma_d))
            plt.plot(timebins, self.fitfunc(timebins, ampl, sigma, center, offset), '-')
            logging.info('Scaler_{}: amplitude: {}, sigma: {}, center: {}({}), offset:{}'
                         .format(sc_no, ampl, sigma, center, center_d, offset))
            timeproj_res['scaler_{}'.format(sc_no)] = {'sigma': sigma, 'center': center}

        plotting = True

        if plotting:
            plt.title('Stacked time projections for isotope: {}'.format(isotope))
            plt.legend(title='Scalers', bbox_to_anchor=(1.04, 0.5), loc="center left")
            plt.margins(0.05)
            if self.save_plots_to_file:
                filename = 'timeproject_files' + str(filelist[0]) + 'to' + str(filelist[-1])
                plot_folder = self.resultsdir + 'summed\\'
                if not os.path.exists(plot_folder):
                    os.makedirs(plot_folder)
                plt.savefig(plot_folder + filename + '_sc0a1a2.png', bbox_inches="tight")
            else:
                plt.show()
            plt.close()
            plt.clf()
        return timeproj_res

    def addfiles(self, iso, filelist, voltrange, binsize, bake_in_calib=False):
        """
        Load all files from list and rebin them into voltrange with binsize
        :param iso:
        :param filelist:
        :param voltrange:
        :param binsize:
        :return:
        """
        # create arrays for rebinning the data
        volt_arr = np.arange(start=voltrange[0], stop=voltrange[1], step=binsize)  # array of the voltage steps
        zeroarr = np.zeros(len(volt_arr))  # zero array with the same dimension as volt_arr to use as dummy
        cts_sum = [zeroarr.copy(), zeroarr.copy(), zeroarr.copy()]  # Array for the on-beam counts per bin. One per pmt
        avgbg_sum = [zeroarr.copy(), zeroarr.copy(), zeroarr.copy()]  # Array to keep track of the background (off-beam)
        real_volt_arr = zeroarr.copy()  # Array to keep track what the real avg voltage per step is
        nOfScans_arr = zeroarr.copy()  # Array to keep track how many scans we have on each step

        # voltage calibrations could be used to adapt the scan-voltage per file
        if bake_in_calib:
            # import voltage calibrations from combined scaler results on a per-file basis
            volt_corrections = self.results[iso]['scaler_012']['acc_volts']
            filenames = self.results[iso]['file_names']
            buncher_potential = self.accVolt_set  # calibrations on a per-file level. Global accVolt is unchanged
        else:
            # do not correct voltages. Use global correction instead!
            volt_corrections = None
            buncher_potential = self.accVolt_set

        # extract data from each file and sort into binning
        for files in filelist:
            # create filepath for XMLImporter
            filepath = os.path.join(self.datafolder, files)
            # get gates from stacked time projection
            sc0_res = self.time_proj_res_per_scaler['scaler_0']
            sc1_res = self.time_proj_res_per_scaler['scaler_1']
            sc2_res = self.time_proj_res_per_scaler['scaler_2']
            if '54' in iso:
                sc0_res = {'center': 517, 'sigma': 3}
                sc1_res = {'center': 537, 'sigma': 3}
                sc2_res = {'center': 544, 'sigma': 3}
            sig_mult = self.tof_width_sigma  # how many sigma to include left and right of midtof (1: 68.3% of data, 2: 95.4%, 3: 99.7%)
            spec = XMLImporter(path=filepath,  # import data from XML, gated on the timpeak for each scaler
                               softw_gates=[[-350, 0,
                                             sc0_res['center']/100 - sc0_res['sigma']/100*sig_mult,
                                             sc0_res['center']/100 + sc0_res['sigma']/100*sig_mult],
                                            [-350, 0,
                                             sc1_res['center']/100 - sc1_res['sigma']/100*sig_mult,
                                             sc1_res['center']/100 + sc1_res['sigma']/100*sig_mult],
                                            [-350, 0,
                                             sc2_res['center']/100 - sc2_res['sigma']/100*sig_mult,
                                             sc2_res['center']/100 + sc2_res['sigma']/100*sig_mult]])
            offst = 5  # background offset from midTof in multiple of sigma
            background = XMLImporter(path=filepath,  # sample spec of the same width, clearly separated from the timepeaks
                                     softw_gates=[[-350, 0,
                                                   (sc0_res['center']-offst*sc0_res['sigma'])/100 - sc0_res['sigma']/100*sig_mult,
                                                   (sc0_res['center']-offst*sc0_res['sigma'])/100 + sc0_res['sigma']/100*sig_mult],
                                                  [-350, 0,
                                                   (sc1_res['center']-offst*sc1_res['sigma'])/100 - sc1_res['sigma']/100*sig_mult,
                                                   (sc1_res['center']-offst*sc1_res['sigma'])/100 + sc1_res['sigma']/100*sig_mult],
                                                  [-350, 0,
                                                   (sc2_res['center']-offst*sc2_res['sigma'])/100 - sc2_res['sigma']/100*sig_mult,
                                                   (sc2_res['center']-offst*sc2_res['sigma'])/100 + sc2_res['sigma']/100*sig_mult]])
            for track in range(spec.nrTracks):
                # check the stepsize of the data and return a warning if it's bigger than the binsize
                stepsize = spec.stepSize[track]  # for track 0
                nOfSteps = spec.getNrSteps(track)  # for track 0
                nOfScans = spec.nrScans[track]  # for track 0
                if stepsize > 1.1*binsize:
                    logging.warning('Stepsize of file {} larger than specified binsize ({}>{})!'
                                    .format(files, stepsize, binsize))
                # get volt (x) data, cts (y) data and errs
                voltage_x = spec.x[track]
                bg_sum_totalcts = [sum(background.cts[track][0]), sum(background.cts[track][1]), sum(background.cts[track][2])]
                for sc_sums in bg_sum_totalcts:
                    if sc_sums == 0: sc_sums = 1  # cannot have zero values

                for scaler in range(3):
                    # apply ion energy correction from calibration
                    volt_cor = 0
                    if volt_corrections is not None:
                        # get voltage correction for this file
                        file_index = filenames.index(files)
                        volt_cor = volt_corrections['vals'][file_index]-self.accVolt_set
                    # bin data
                    for step, volt in enumerate(voltage_x):
                        volt_c = volt-volt_cor
                        bg_avg = bg_sum_totalcts[scaler] / nOfSteps
                        if voltrange[0] <= volt_c <= voltrange[1]:  # only use if inside desired range
                            voltind = (np.abs(volt_arr - volt_c)).argmin()  # find closest index in voltage array
                            # add data to the arrays
                            cts_sum[scaler][voltind] += spec.cts[track][scaler][step]  # no normalization here

                            # keep track of nOfScans and real voltage
                            if scaler == 0:
                                if real_volt_arr[voltind] == 0:
                                    real_volt_arr[voltind] += volt_c
                                else:
                                    # else do a weighted average with the nOfScans as weights
                                    # TODO: use bg_avg as weights instead of nOfScans?! Because thats what we use for the normalization as well
                                    real_volt_arr[voltind] = (real_volt_arr[voltind] / avgbg_sum[scaler][voltind] ** 2
                                                              + volt_c / bg_avg ** 2) \
                                                             / (1 / avgbg_sum[scaler][voltind] ** 2 + 1 / bg_avg ** 2)
                                    # real_volt_arr[voltind] = (real_volt_arr[voltind]/nOfScans_arr[voltind]**2
                                    #                           + volt_c/nOfScans**2)\
                                    #                          / (1/nOfScans_arr[voltind]**2 + 1/nOfScans**2)
                            avgbg_sum[scaler][voltind] += bg_avg  # for keeping track of a total scale
                            nOfScans_arr[voltind] += nOfScans

        # calculate uncertainty for on- and off-beam arrays as sqrt(n)
        cts_err = [np.sqrt(cts_sum[0]), np.sqrt(cts_sum[1]), np.sqrt(cts_sum[2])]
        avgbg_err = [np.sqrt(avgbg_sum[0]), np.sqrt(avgbg_sum[1]), np.sqrt(avgbg_sum[2])]

        # make sure there are no zero values in the off-beam array to avoid division-by-zero error.
        include_indx = nOfScans_arr != 0  # only include values where at least one scan brought data
        # Atttention! We must not remove values from the middle of the array! This would screw up the pollifit voltage.
        check_lst = np.split(include_indx, np.where(np.diff(include_indx) == 1)[0] + 1)  # Difference True->False = 1
        # the check list can't contain more than 3 arrays. These must be [False, True, False]. 2 or 1 array is always ok
        if len(check_lst) > 3 or (len(check_lst) == 3 and check_lst[1][0] is False):
            # Seems like we have found non-include values in the middle of our data. This must not be!
            include_indx = np.full(len(nOfScans_arr), True, dtype=bool)  # Instead, we now include all values.
            # This will crash the analysis with div0 error soon...that's better than screwing up our data
            cts_sum = 1/0  # well actually for clarity we can crash it right here!
            # Hint: if there is supposed to be gaps in the data, start creating different tracks for the xml here!

        # also delete first and last of the remaining values, these can be a little skewed...
        for scaler in range(3):
            cts_sum[scaler] = cts_sum[scaler][include_indx][1:-1]
            cts_err[scaler] = cts_err[scaler][include_indx][1:-1]
            avgbg_sum[scaler] = avgbg_sum[scaler][include_indx][1:-1]
            avgbg_err[scaler] = avgbg_err[scaler][include_indx][1:-1]
        volt_arr = volt_arr[include_indx][1:-1]
        real_volt_arr = real_volt_arr[include_indx][1:-1]
        nOfScans_arr = nOfScans_arr[include_indx][1:-1]

        # fit a line to the real voltage
        def _line(x, m, b):
            return m * x + b
        # start parameters
        p0 = [binsize, real_volt_arr[0]]
        # do the fitting
        popt, pcov = curve_fit(_line, np.arange(len(real_volt_arr)), real_volt_arr, p0)
        perr = np.sqrt(np.diag(pcov))  # TODO: use this somewhere?

        # extract real start volt and stepsize from the calibration:
        real_start_v = popt[1]
        real_stepsize_v = popt[0]

        plotting = True

        if plotting:
            # plot on-beam and off-beam array before any normalization:
            plt.errorbar(volt_arr, np.sum(np.array(cts_sum), axis=0),
                         yerr=np.sum(np.array(cts_err), axis=0), fmt='.',
                         label='on-beam')
            plt.errorbar(volt_arr, np.sum(np.array(avgbg_sum), axis=0),
                         yerr=np.sum(np.array(avgbg_err), axis=0), fmt='.',
                         label='off-beam')
            if self.save_plots_to_file:
                filename = 'added_' + str(iso) + '_files' + str(filelist[0]) + 'to' + str(filelist[-1])
                plot_folder = self.resultsdir + 'summed\\'
                if not os.path.exists(plot_folder):
                    os.makedirs(plot_folder)
                plt.savefig(plot_folder + filename + '_sc0a1a2.png', bbox_inches="tight")
            else:
                plt.show()
            plt.close()
            plt.clf()

        # Normalize cts_arr and create a final uncertainty array
        total_scale_factor = 0  # height of the heighest peak for isotope intensity
        for scaler in range(3):
            scale_factor = avgbg_sum[scaler].mean()  # pollifit needs to work with integers
            cts_sum[scaler] = (cts_sum[scaler]/avgbg_sum[scaler] * scale_factor).astype(int)
            cts_err[scaler] = (np.sqrt(np.square(cts_err[scaler]/np.square(avgbg_sum[scaler])) +
                                      np.square(cts_sum[scaler]/np.square(avgbg_sum[scaler])*avgbg_err[scaler])) * scale_factor).astype(int)
            total_scale_factor += (cts_sum[scaler].max() - cts_sum[scaler].min())  # we also need to scale the intensity of the isotope

        if plotting:
            # plot the summed up data for each scaler
            for sc in range(3):
                plt.errorbar(volt_arr, cts_sum[sc], yerr=cts_err[sc], fmt='.', label='scaler_{}'.format(sc))
            if self.save_plots_to_file:
                filename = 'added_' + str(iso) + '_files' + str(filelist[0]) + 'to' + str(filelist[-1])
                plot_folder = self.resultsdir + 'summed\\'
                if not os.path.exists(plot_folder):
                    os.makedirs(plot_folder)
                plt.savefig(plot_folder + filename + '.png', bbox_inches="tight")
            else:
                plt.show()
            plt.close()
            plt.clf()

            # plot the summed up data for all scalers combined:
            plt.errorbar(volt_arr, np.sum(np.array(cts_sum), axis=0),
                         yerr=np.sum(np.array(cts_err), axis=0), fmt='.k', label='all_scalers')
            if self.save_plots_to_file:
                filename = 'added_' + str(iso) + '_files' + str(filelist[0]) + 'to' + str(filelist[-1])
                plot_folder = self.resultsdir + 'summed\\'
                if not os.path.exists(plot_folder):
                    os.makedirs(plot_folder)
                plt.savefig(plot_folder + filename + '_sc012.png', bbox_inches="tight")
            else:
                plt.show()
            plt.close()
            plt.clf()

        self.make_sumXML_file(iso, real_start_v, real_stepsize_v, len(cts_sum[0]), np.array(cts_sum), np.array(cts_err),
                              peakHeight=total_scale_factor, accV=buncher_potential)

    def addfiles_trs(self, iso, filelist, voltrange, binsize, bake_in_calib=False):
        """
        Load all files from list and rebin them into voltrange with binsize
        :param iso:
        :param filelist:
        :param voltrange:
        :param binsize:
        :return:
        """
        # create arrays for rebinning the data
        nOfTracks = 1  # Just put everything in one track. The other is just artificial anyways (backwards scan)
        nOfScalers = 3
        nOfBins = 1024
        cts_trs_array = np.zeros((nOfTracks, nOfScalers, (voltrange[1]-voltrange[0])/binsize, nOfBins))  # nrOfTracks, nrOfScalers, nrOfSteps, nrOfTimeBins
        bg_trs_array = cts_trs_array.copy()

        volt_arr = np.arange(start=voltrange[0], stop=voltrange[1], step=binsize)  # array of the voltage steps
        zeroarr = np.zeros(len(volt_arr))  # zero array with the same dimension as volt_arr to use as dummy
        real_volt_arr = zeroarr.copy()  # Array to keep track what the real avg voltage per step is
        nOfScans_arr = zeroarr.copy()  # Array to keep track how many scans we have on each step

        # voltage calibrations could be used to adapt the scan-voltage per file
        if bake_in_calib:
            # import voltage calibrations from combined scaler results on a per-file basis
            volt_corrections = self.results[iso]['scaler_012']['acc_volts']
            filenames = self.results[iso]['file_names']
            buncher_potential = self.accVolt_set  # calibrations on a per-file level. Global accVolt is unchanged
        else:
            # do not correct voltages. Use global correction instead!
            volt_corrections = None
            buncher_potential = self.accVolt_set

        # extract data from each file and sort into binning
        for files in filelist:
            # create filepath for XMLImporter
            filepath = os.path.join(self.datafolder, files)
            spec = XMLImporter(path=filepath)

            # trs_data = spec.time_res_zf  # time resolved list of pmt events in form of indices, zf is for zero free,
            trs_data = spec.time_res  # time resolved matrices. Probably much more efficient here than zf. (tracks, scaler, step, bin)
            #  list contains numpy arrays with structure: ('sc', 'step', 'time', 'cts')
            #  indices in list correspond to track indices

            for track, trackdata in enumerate(trs_data):
                # check the stepsize of the data and return a warning if it's bigger than the binsize
                stepsize = spec.stepSize[track]  # for track 0
                nOfSteps = spec.getNrSteps(track)  # for track 0
                nOfScans = spec.nrScans[track]  # for track 0
                if stepsize > 1.1*binsize:
                    logging.warning('Stepsize of file {} larger than specified binsize ({}>{})!'
                                    .format(files, stepsize, binsize))
                # get volt (x) data, cts (y) data and errs
                voltage_x = spec.x[track]

                for scaler, scalerdata in enumerate(trackdata):
                    # take an off-beam background sample
                    bgrange = int(nOfBins / 3)  # where should the background sample be taken? 0-x
                    sumcts_offbeam = trackdata[scaler, :, :bgrange].sum()
                    norm_factor = sumcts_offbeam / bgrange / nOfSteps

                    for step, stepdata in enumerate(scalerdata):
                        volt = voltage_x[step]
                        # apply ion energy correction from calibration
                        volt_cor = 0
                        if volt_corrections is not None:
                            # get voltage correction for this file
                            file_index = filenames.index(files)
                            volt_cor = volt_corrections['vals'][file_index] - self.accVolt_set
                        volt_c = volt - volt_cor
                        if voltrange[0] <= volt_c <= voltrange[1]:  # only use if inside desired range
                            voltind = (np.abs(volt_arr - volt_c)).argmin()  # find closest index in voltage array
                            # add data to the arrays
                            cts_trs_array[0, scaler, voltind, :] += stepdata
                            bg_trs_array[0, scaler, voltind, :] += np.full(stepdata.shape, norm_factor)
                            # keep track of nOfScans and real voltage
                            if scaler == 0:
                                if real_volt_arr[voltind] == 0:
                                    real_volt_arr[voltind] += volt_c
                                else:
                                    # else do a weighted average with the nOfScans as weights
                                    real_volt_arr[voltind] = (real_volt_arr[voltind] / nOfScans_arr[voltind] ** 2
                                                              + volt_c / nOfScans ** 2) \
                                                             / (1 / nOfScans_arr[voltind] ** 2 + 1 / nOfScans ** 2)
                            nOfScans_arr[voltind] += nOfScans


                # for event in trackdata:
                #     sc, step, time, cts = event
                #     volt = voltage_x[step]
                #
                #     # apply ion energy correction from calibration
                #     volt_cor = 0
                #     if volt_corrections is not None:
                #         # get voltage correction for this file
                #         file_index = filenames.index(files)
                #         volt_cor = volt_corrections['vals'][file_index] - self.accVolt_set
                #     volt_c = volt - volt_cor
                #     if voltrange[0] <= volt_c <= voltrange[1]:  # only use if inside desired range
                #         voltind = (np.abs(volt_arr - volt_c)).argmin()  # find closest index in voltage array
                #         # add data to the arrays
                #         cts_trs_array[0, sc, voltind, time] += cts
                #         # keep track of nOfScans and real voltage
                #         if sc == 0:
                #             if real_volt_arr[voltind] == 0:
                #                 real_volt_arr[voltind] += volt_c
                #             else:
                #                 # else do a weighted average with the nOfScans as weights
                #                 real_volt_arr[voltind] = (real_volt_arr[voltind]/nOfScans_arr[voltind]**2
                #                                           + volt_c/nOfScans**2)\
                #                                          / (1/nOfScans_arr[voltind]**2 + 1/nOfScans**2)
                #         nOfScans_arr[voltind] += nOfScans

        # make sure there are no zero values in the off-beam array to avoid division-by-zero error.
        include_indx = nOfScans_arr != 0  # only include values where at least one scan brought data
        # # Attention! We must not remove values from the middle of the array! This would screw up the pollifit voltage.
        # check_lst = np.split(include_indx, np.where(np.diff(include_indx) == 1)[0] + 1)  # Difference True->False = 1
        # # the check list can't contain more than 3 arrays. These must be [False, True, False]. 2 or 1 array is always ok
        # if len(check_lst) > 3 or (len(check_lst) == 3 and check_lst[1][0] is False):
        #     # Seems like we have found non-include values in the middle of our data. This must not be!
        #     include_indx = np.full(len(nOfScans_arr), True, dtype=bool)  # Instead, we now include all values.
        #     # This will crash the analysis with div0 error soon...that's better than screwing up our data
        #     cts_sum = 1/0  # well actually for clarity we can crash it right here!
        #     # Hint: if there is supposed to be gaps in the data, start creating different tracks for the xml here!

        # also delete first and last of the remaining values, these can be a little skewed...
        cts_trs_array = cts_trs_array[:, :, include_indx, :]
        bg_trs_array = bg_trs_array[:, :, include_indx, :]
        cts_trs_array = cts_trs_array*bg_trs_array.mean()/bg_trs_array
        # for scaler in range(3):
        #     cts_sum[scaler] = cts_sum[scaler][include_indx][1:-1]
        volt_arr = volt_arr[include_indx]
        real_volt_arr = real_volt_arr[include_indx]
        nOfScans_arr = nOfScans_arr[include_indx]

        # fit a line to the real voltage
        def _line(x, m, b):
            return m * x + b
        # start parameters
        p0 = [binsize, real_volt_arr[0]]
        # do the fitting
        popt, pcov = curve_fit(_line, np.arange(len(real_volt_arr)), real_volt_arr, p0)
        perr = np.sqrt(np.diag(pcov))  # TODO: use this somewhere?

        # extract real start volt and stepsize from the calibration:
        real_start_v = popt[1]
        real_stepsize_v = popt[0]

        # go back to xml style data:
        track_data_list = []
        for track in range(nOfTracks):
            data_tuples_list = []
            for index, cts in np.ndenumerate(cts_trs_array[track]):
                # Loop over data array and extract index + cts and combine them to a tuple.
                data_point_tuple = index + (int(cts),)
                data_tuples_list.append(data_point_tuple)  # append tuple to list
            dt = [('sc', 'u2'), ('step', 'u4'), ('time', 'u4'), ('cts', 'u4')]  # data type for npy array
            track_data_list.append(
                np.array(data_tuples_list, dtype=dt))  # convert list to npy array with given data format
        data_tuples_arr = np.stack(track_data_list)
        # Create voltage projection
        track_data_list = []
        for track in range(nOfTracks):
            voltage_projections = []
            for scaler in range(nOfScalers):
                proj_list = cts_trs_array[track].sum(axis=2)[scaler].tolist()
                int_proj = [int(i) for i in proj_list]  # Must be an array of integers!
                voltage_projections.append(int_proj)
            track_data_list.append(np.array(voltage_projections))  # make array of arrays from list of arrays.
        voltage_projection_arr = np.stack(track_data_list)

        ###################################
        # Prepare dicts for writing to XML #
        ###################################
        file_creation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        header_dict = {'type': 'trs',
                       'isotope': iso,
                       'isotopeStartTime': file_creation_time,
                       'accVolt': self.accVolt_set,
                       'laserFreq': Physics.wavenumber(self.laser_freqs[iso])/2,
                       'nOfTracks': nOfTracks,
                       'version': 99.0}

        tracks_dict = {}
        for tracknum in range(nOfTracks):
            trackname = 'track{}'.format(tracknum)
            # track times:
            track_start_time_str = file_creation_time
            track_end_time_str = file_creation_time
            # info for track in header
            track_dict_header = {'trigger': {},  # Need a trigger dict!
                                 'activePmtList': list(range(cts_trs_array.shape[1])),  # Must be in form [0,1,2]
                                 'colDirTrue': True,
                                 'dacStartRegister18Bit': 0,
                                 'dacStartVoltage': real_start_v,
                                 'dacStepSize18Bit': None,  # old format xml importer checks whether val or None
                                 'dacStepsizeVoltage': real_stepsize_v,
                                 'dacStopRegister18Bit': cts_trs_array.shape[2] - 1,  # not real but should do the trick
                                 'dacStopVoltage': float(real_start_v) + (
                                             float(real_stepsize_v) * int(len(volt_arr) - 1)),
                                 # nOfSteps-1 bc startVolt is the first step
                                 'invertScan': False,
                                 'nOfBins': nOfBins,
                                 'nOfBunches': 1,  # dummy val
                                 # at BECOLA this corresponds to number of Sequences (Seqs in excel)
                                 'nOfCompletedSteps': float(sum(nOfScans_arr)),
                                 'nOfScans': int(nOfScans_arr.mean()),
                                 'nOfSteps': cts_trs_array.shape[2],
                                 'postAccOffsetVolt': 0,
                                 'postAccOffsetVoltControl': 0,
                                 'SoftBinWidth_us': 1024,  # shrink later!
                                 'softwGates': [volt_arr[0], volt_arr[-1], 5.12, 10.24],
                                 # For each Scaler: [DAC_Start_Volt, DAC_Stop_Volt, scaler_delay, softw_Gate_width]
                                 'workingTime': [track_start_time_str, track_end_time_str],
                                 'waitAfterReset1us': 0,  # looks like I need those for the importer
                                 'waitForKepco1us': 0  # looks like I need this too
                                 }
            track_dict_data = {
                'scalerArray_explanation': 'time resolved data. List of tuples, each tuple consists of: (scaler_number, line_voltage_step_number, time_stamp, number_of_counts), datatype: np.int32',
                'scalerArray': data_tuples_arr[tracknum]}
            track_dict_projections = {
                'voltage_projection_explanation': 'voltage_projection of the time resolved data. List of Lists, each list represents the counts of one scaler as listed in activePmtList.Dimensions are: (len(activePmtList), nOfSteps), datatype: np.int32',
                'voltage_projection': voltage_projection_arr[tracknum]}
            tracks_dict[trackname] = {'header': track_dict_header,
                                      'data': track_dict_data,
                                      'projections': track_dict_projections}

        # Combine to xml_dict
        xml_dict = {'header': header_dict,
                         'tracks': tracks_dict
                         }

        ################
        # Write to XML #
        ################
        # if not self.excel_extraction_failed:  # actually that is not a big problem in this newer Version...
        iso = iso[:4]
        bakein = ''
        type = '{}_sum_cal'.format(iso)  # we will always use a calibrated sum file
        if buncher_potential == self.accVolt_set:
            # calibrations baked in to scan voltage
            bakein = 'c'
            type = '{}_sum_cal'.format(iso)
        xml_name = 'Sum{}{}_9999.xml'.format(iso, bakein)
        xml_filepath = os.path.join(self.datafolder, xml_name)
        self.writeXMLfromDict(xml_dict, xml_filepath, 'BecolaData')
        self.ni_analysis_combined_files.append(xml_name)

        # add file to database
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''INSERT OR IGNORE INTO Files (file, filePath, date, type) VALUES (?, ?, ?, ?)''',
                    (xml_name, os.path.relpath(xml_filepath, self.workdir), file_creation_time, type))
        con.commit()
        cur.execute(
            '''UPDATE Files SET offset = ?, accVolt = ?,  laserFreq = ?, laserFreq_d = ?, colDirTrue = ?, 
            voltDivRatio = ?, lineMult = ?, lineOffset = ?, errDateInS = ? WHERE file = ? ''',
            (str(nOfTracks*[0]), buncher_potential, self.laser_freqs[iso], 0, True, str({'accVolt': 1.0, 'offset': 1.0}), 1, 0, 1,
             xml_name))
        con.commit()
        # create new isotope
        cur.execute('''SELECT * FROM Isotopes WHERE iso = ? ''', (iso,))  # get original isotope to copy from
        mother_isopars = cur.fetchall()
        isopars_lst = list(mother_isopars[0])  # change into list to replace some values
        isopars_lst[0] = type
        # if isopars_lst[3] != 0:
        #     # spin different from zero, several sidepeaks, adjust scaling!
        #     peakHeight = peakHeight / 10
        # bg_estimate = sum(cts_list[:, -1])
        # isopars_lst[11] = int(peakHeight) / bg_estimate * 1000  # change intensity scaling
        new_isopars = tuple(isopars_lst)
        cur.execute('''INSERT OR REPLACE INTO Isotopes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                    new_isopars)
        con.commit()
        con.close()



    def make_sumXML_file(self, isotope, startVolt, stepSizeVolt, nOfSteps, cts_list, err_list=None, peakHeight=1, accV=29850):
        ####################################
        # Prepare dicts for writing to XML #
        ####################################
        iso = isotope[:4]
        bakein = ''
        type = '{}_sum_cal'.format(iso)  # we will always use a calibrated sum file
        if accV == self.accVolt_set:
            # calibrations baked in to scan voltage
            bakein = 'c'
            type = '{}_sum_cal'.format(iso)
        file_creation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        header_dict = {'type': 'cs',
                       'isotope': iso,
                       'isotopeStartTime': file_creation_time,
                       'accVolt': accV,
                       'laserFreq': Physics.wavenumber(self.laser_freqs[iso])/2,
                       'nOfTracks': 1,
                       'version': 99.0}

        track_dict_header = {'trigger': {},  # Need a trigger dict!
                             'activePmtList': [0, 1, 2],  # Must be in form [0,1,2]
                             'colDirTrue': True,
                             'dacStartRegister18Bit': 0,
                             'dacStartVoltage': startVolt,
                             'dacStepSize18Bit': None,  # old format xml importer checks whether val or None
                             'dacStepsizeVoltage': stepSizeVolt,
                             'dacStopRegister18Bit': nOfSteps - 1,  # not real but should do the trick
                             'dacStopVoltage': float(startVolt) + (
                                         float(stepSizeVolt) * int(nOfSteps - 1)),
                             # nOfSteps-1 bc startVolt is the first step
                             'invertScan': False,
                             'nOfCompletedSteps': float(int(nOfSteps)),
                             'nOfScans': 1,
                             'nOfSteps': nOfSteps,
                             'postAccOffsetVolt': 0,
                             'postAccOffsetVoltControl': 0,
                             'softwGates': [],
                             # For each Scaler: [DAC_Start_Volt, DAC_Stop_Volt, scaler_delay, softw_Gate_width]
                             'workingTime': [file_creation_time, file_creation_time],
                             'waitAfterReset1us': 0,  # looks like I need those for the importer
                             'waitForKepco1us': 0  # looks like I need this too
                             }
        track_dict_data = {
            'scalerArray_explanation': 'continously acquired data. List of Lists, each list represents the counts of '
                                       'one scaler as listed in activePmtList.Dimensions are: (len(activePmtList), '
                                       'nOfSteps), datatype: np.int32',
            'scalerArray': cts_list}
        if err_list is not None:
            track_dict_data['errorArray'] = err_list
            track_dict_data['errorArray_explanation'] = 'Optional: Non-standard errors. If this was not present, ' \
                                                        'np.sqrt() would be used for errors during XML import. ' \
                                                        'List of lists, each list represents the errors of one scaler '\
                                                        'as listed in activePmtList.Dimensions  are: ' \
                                                        '(len(activePmtList), nOfSteps), datatype: np.int32'

        # Combine to xml_dict
        xml_dict = {'header': header_dict,
                         'tracks': {'track0': {'header': track_dict_header,
                                               'data': track_dict_data
                                               }
                                    }
                         }

        ################
        # Write to XML #
        ################
        xml_name = 'Sum{}{}_9999.xml'.format(iso, bakein)
        xml_filepath = os.path.join(self.datafolder, xml_name)
        self.writeXMLfromDict(xml_dict, xml_filepath, 'BecolaData')
        self.ni_analysis_combined_files.append(xml_name)
        # add file to database
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''INSERT OR IGNORE INTO Files (file, filePath, date, type) VALUES (?, ?, ?, ?)''',
                    (xml_name, os.path.relpath(xml_filepath, self.workdir), file_creation_time, type))
        con.commit()
        cur.execute(
            '''UPDATE Files SET offset = ?, accVolt = ?,  laserFreq = ?, laserFreq_d = ?, colDirTrue = ?, 
            voltDivRatio = ?, lineMult = ?, lineOffset = ?, errDateInS = ? WHERE file = ? ''',
            ('[0]', accV, self.laser_freqs[iso], 0, True, str({'accVolt': 1.0, 'offset': 1.0}), 1, 0, 1,
             xml_name))
        con.commit()
        # create new isotope
        cur.execute('''SELECT * FROM Isotopes WHERE iso = ? ''', (iso,))  # get original isotope to copy from
        mother_isopars = cur.fetchall()
        isopars_lst = list(mother_isopars[0])  # change into list to replace some values
        isopars_lst[0] = type
        if isopars_lst[3] != 0:
            # spin different from zero, several sidepeaks, adjust scaling!
            peakHeight = peakHeight/10
        bg_estimate = sum(cts_list[:, -1])
        isopars_lst[11] = int(peakHeight)/bg_estimate*1000  # change intensity scaling
        new_isopars = tuple(isopars_lst)
        cur.execute('''INSERT OR REPLACE INTO Isotopes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                    new_isopars)
        con.commit()
        con.close()

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

    def extract_radius_from_factors(self, iso, ref, scaler=None):
        """
        Use known fieldshift and massshift parameters to calculate the difference in rms charge radii and then the
        absolute charge radii.
        Isotope Shift will be extracted from the 'Combined' results database and absolute radii are from literature.
        :param isotopes: isotopes to include e.g. ['55Ni_sum_cal', '56Ni_sum_cal', '58Ni_sum_cal', '60Ni_sum_cal',]
        :param reference: reference isotope (either 58 or 60)
        :return: delta_rms, delta_rms_d
        """
        # get the masses and calculate mu
        m_iso, m_iso_d = self.get_iso_property_from_db('''SELECT mass, mass_d FROM Isotopes WHERE iso = ?''', (iso[:4],))
        m_ref, m_ref_d = self.get_iso_property_from_db('''SELECT mass, mass_d FROM Isotopes WHERE iso = ?''', (ref[:4],))
        mu = (m_iso-m_ref)/(m_iso*m_ref)
        mu_d = np.sqrt(np.square(m_iso_d/m_iso**2)+np.square(m_ref_d/m_ref**2))

        # get Mass and Field Shift factors:
        if '58' in ref:
            kingFactorLit = 'Koenig 2020 58ref'
        else:
            kingFactorLit = 'Koenig 2020 60ref'
        M_alpha, M_alpha_d = self.king_literature[kingFactorLit]['data']['Kalpha']  # Mhz u (lit val given in GHz u)(949000, 4000)
        F, F_d = self.king_literature[kingFactorLit]['data']['F']  # MHz/fm^2(-788, 82)
        alpha = self.king_literature[kingFactorLit]['data']['Alpha']  # u fm^2 397

        # get data and calculate radii
        delta_rms = []
        delta_rms_d = []
        if scaler is not None:
            # get per file isoshift
            files = self.results[iso]['file_names']
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
            # extract isotope shift from db where no scaler is specified.
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

    def make_results_dict_scaler(self,
                                 centers, centers_d_fit, centers_d_stat, center_d_syst, fitpars, rChi, hfs_pars=None):
        # calculate weighted average of center parameter
        wavg, wavg_d, wstd, std = self.calc_weighted_avg(centers, centers_d_stat)
        if self.combined_unc == 'wavg_d':
            d_fit = wavg_d
        elif self.combined_unc == 'wstd':
            d_fit = wstd
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
            vals['file_times'] = [datetime.strftime(t, '%Y-%m-%d %H:%M:%S') for t in vals['file_times']]
            to_file_dict['i' + keys] = vals
        # add analysis parameters
        to_file_dict['analysis_parameters'] = self.analysis_parameters
        results_file = self.results_name + '.xml'
        self.writeXMLfromDict(to_file_dict, os.path.join(self.resultsdir, results_file), 'BECOLA_Analysis')

    def import_results(self, results_xml):
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
        self.plot_centroids(['56Ni_cal', '58Ni_cal', '60Ni_cal'],
                            ['scaler_c0', 'scaler_c1', 'scaler_c2', 'scaler_c012'],
                            overlay=None, unit='MHz', onlyfiterrs=False, digits=1, plotAvg=True)

        self.plot_shifts(['56Ni_cal', '58Ni_cal'],
                            ['scaler_c0', 'scaler_c1', 'scaler_c2', 'scaler_c012'],
                            overlay=None, unit='MHz', onlyfiterrs=False, digits=1, plotAvg=True)

        self.plot_centroids(['56Ni_cal', '58Ni_cal', '60Ni_cal'],
                            ['scaler_0', 'scaler_1', 'scaler_2', 'scaler_012'],
                            overlay=None, unit='MHz', onlyfiterrs=False, digits=1, plotAvg=True)

        self.plot_shifts(['56Ni_cal', '58Ni_cal'],
                         ['scaler_0', 'scaler_1', 'scaler_2', 'scaler_012'],
                         overlay=None, unit='MHz', onlyfiterrs=False, digits=1, plotAvg=True)

        final_isos = ['55Ni', '56Ni', '58Ni', '60Ni']
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
                    wavg, wavg_d, wstd, std = self.calc_weighted_avg(centers, centers_d_stat)
                    if self.combined_unc == 'wavg_d':
                        d = wavg_d
                    elif self.combined_unc == 'wstd':
                        d = wstd
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
                    wavg, wavg_d, wstd, std = self.calc_weighted_avg(centers, centers_d_stat)
                    if self.combined_unc == 'wavg_d':
                        d = wavg_d
                    elif self.combined_unc == 'wstd':
                        d = wstd
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
        if calibrated:
            isolist = ['{}_cal'.format(i) for i in isolist]  # make sure to use the calibrated isos
        font_size = 12
        ref_key = refiso[:4]
        if scaler == 'final':
            rms_key = 'delta_rms_iso-{}'.format(refiso[:2])
        else:
            rms_key = 'avg_delta_rms_iso-{}'.format(refiso[:2])
        thisVals = {key: [self.results[key][scaler][rms_key]['vals'][0],
                           self.results[key][scaler][rms_key]['d_syst'][0]]
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
        if calibrated:
            isolist = ['{}_cal'.format(i) for i in isolist]  # make sure to use the calibrated isos
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
    analysis.fitting_initial_separate()
    analysis.combine_single_scaler_centers(['58Ni', '60Ni'])
    analysis.plot_results_of_fit(calibrated=False)
    analysis.ion_energy_calibration()
    # stacked run analysis for inclusion of nickel 54
    analysis.create_and_fit_stacked_runs(calibration_per_file=True)
    analysis.export_results()
