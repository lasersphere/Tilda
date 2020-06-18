"""
Created on 2018-12-19

@author: fsommer

Module Description:  Analysis of the Nickel Data from BECOLA taken on 13.04.-23.04.2018
"""

import ast
import os
import sqlite3
from datetime import datetime, timedelta
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
        self.run_name = 'mid2020'

        # Set working directory and database
        ''' working directory: '''
        # get user folder to access ownCloud
        user_home_folder = os.path.expanduser("~")
        # self.workdir = 'C:\\DEVEL\\Analysis\\Ni_Analysis\\XML_Data' # old working directory
        ownCould_path = 'ownCloud\\User\\Felix\\Measurements\\Nickel_online_Becola\\Analysis\\XML_Data'
        self.workdir = os.path.join(user_home_folder, ownCould_path)
        ''' data folder '''
        self.datafolder = os.path.join(self.workdir, 'SumsRebinned')
        ''' results folder'''
        analysis_start_time = datetime.now()
        self.results_name = self.run_name + '_' + analysis_start_time.strftime("%Y-%m-%d_%H-%M")
        results_path_ext = 'results\\' + self.results_name + '\\'
        self.resultsdir = os.path.join(self.workdir, results_path_ext)
        os.makedirs(self.resultsdir)
        ''' database '''
        self.db = os.path.join(self.workdir, 'Ni_Becola.sqlite')
        Tools.add_missing_columns(self.db)
        logging.info('\n'
                     '########## BECOLA Nickel Analysis Started! ####################################################\n'
                     '## database is: {0}\n'
                     '## data folder: {1}\n'
                     .format(self.db, self.datafolder))

        """
        ############################Analysis Parameters!##########################################################
        Specify how you want to run this Analysis!
        """
        # Select runs; Format: ['run58', 'run60', 'run56']
        # to use a different lineshape you must create a new run under runs and a new linevar under lines and link the two.
        self.run = 'AsymmetricVoigt'

        # fit from scratch or use FitRes db?
        self.do_the_fitting = False  # if False, an .xml file has to be specified in the next variable!
        load_results_from = 'mid2020_2020-06-18_14-06.xml'  # load fit results from this file
        self.get_gate_analysis = True
        load_gate_analysis_from = 'SoftwareGateAnalysis_2020-06-17_13-13_narrow90p-3sig_AsymmetricVoigt.xml'
        # print results to results folder? Also show plots?
        self.save_plots_to_file = True  # if False plots will be displayed during the run for closer inspection
        # list of scaler combinations to fit:
        self.scaler_combinations = [[0], [1], [2], [0, 1, 2]]
        # determine time gates
        self.tof_mid = {'55Ni': 5.237, '56Ni': 5.276, '58Ni': 5.383, '60Ni': 5.408}  # mid-tof for each isotope (from fitting)
        self.tof_delay = [0, 0.195, 0.265]
        self.tof_sigma = 0.098  # 1 sigma of the tof-peaks from fitting, avg over all scalers 56,58,60 Ni
        self.tof_width_sigma = 2  # how many sigma to use around tof? (1: 68.3% of data, 2: 95.4%, 3: 99.7%)
        # acceleration set voltage (Buncher potential), negative
        self.accVolt_set = 29850  # omit voltage sign, assumed to be negative
        self.calibration_method = 'absolute'  # can be 'absolute', 'relative' 'combined', 'isoshift' or 'None'
        self.use_handassigned = False  # use hand-assigned calibrations? If false will interpolate on time axis
        self.KingFactorLit = 'koenig60'  # which king fit factors to use? kaufm60, koenig60,koenig58
        self.accVolt_corrected = (self.accVolt_set, 0)  # Used later for calibration. Might be used her to predefine calib? (abs_volt, err)
        self.initial_par_guess = {'sigma': (34.5, [0, 50]), 'gamma': (11.3, [0, 40]),
                                  'asy': (3.9, False),  # in case VoigtAsy is used
                                  'dispersive': (-0.04, False),  # in case FanoVoigt is used
                                  'centerAsym': (-6.2, True), 'nPeaksAsym': (1, True), 'IntAsym': (0.052, True)
                                  # in case AsymmetricVoigt is used
                                  }
        self.isotope_colors = {58: 'black', 60: 'blue', 56: 'green', 55: 'purple'}
        self.scaler_colors = {'scaler_0': 'blue', 'scaler_1': 'green', 'scaler_2': 'red',
                              'scaler_012': 'black', 'scaler_12': 'yellow',
                              'scaler_c012': 'pink', 'scaler_c0': 'purple', 'scaler_c1': 'grey', 'scaler_c2': 'orange'}

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

        self.analysis_parameters = {'run': self.run,
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
        self.matsuada_volts_d = 0.03  # V. ~standard dev after rebinning TODO: calc real avg over std dev?
        self.lineshape_d_syst = 1.0  # MHz. Offset between VoigtAsym and AsymmetricVoigt TODO: Can we say which is better?
        self.bunch_structure_d = 0.2  # MHz. Slope difference between 58&56 VoigtAsy allfix: 20kHz/bin, +-5bin width --> 200kHz TODO: check whether this is good standard value (also for summed)
        self.heliumneon_drift = 5  # TODO: does that influence our measurements? (1 MHz in Kristians calibration)
        self.laserionoverlap_anglemrad_d = 1  # mrad. ideally angle should be 0. Max possible deviation is ~1mrad TODO: check
        self.laserionoverlap_MHz_d = (self.accVolt_set -  # TODO: This formular should be doublechecked...
                                      np.sqrt(self.accVolt_set ** 2 / (
                                                  1 + (self.laserionoverlap_anglemrad_d / 1000) ** 2))) * 15

        ''' Masses '''
        # # Reference:   'The Ame2016 atomic mass evaluation: (II). Tables, graphs and references'
        # #               Chinese Physics C Vol.41, No.3 (2017) 030003
        # #               Meng Wang, G. Audi, F.G. Kondev, W.J. Huang, S. Naimi, Xing Xu
        self.masses = {
            '55Ni': (54951330.0, 0.8),
            '56Ni': (55942127.9, 0.5),
            '57Ni': (56939791.5, 0.6),
            '58Ni': (57935341.8, 0.4),
            '59Ni': (58934345.6, 0.4),
            '60Ni': (59930785.3, 0.4)
        }
        # # Write masses to self.db:
        # con = sqlite3.connect(self.db)
        # cur = con.cursor()
        # for iso, mass_tupl in masses.items():
        #     cur.execute('''UPDATE Isotopes SET mass = ?, mass_d = ? WHERE iso = ? ''',
        #                 (mass_tupl[0] * 10 ** -6, mass_tupl[1] * 10 ** -6, iso))
        # con.commit()
        # con.close()

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
        # Reference: ??
        # NIST: observed wavelength air 352.454nm corresponds to 850586060MHz
        # upper lvl 28569.203cm-1; lower lvl 204.787cm-1
        # resulting wavenumber 28364.416cm-1 corresponds to 850343800MHz
        # KURUCZ database: 352.4535nm, 850344000MHz, 28364.424cm-1
        # Some value I used in the excel sheet: 850347590MHz Don't remember where that came from...
        # Kristians col/acol value 2020: 850343673(7) MHz. Still preliminary
        self.restframe_trans_freq = 850343673.0
        self.restframe_trans_freq_d = 7.0

        ''' literature value IS 60-58'''
        # Reference: ??
        # isotope shift of Nickel-60 with respect to Nickel-58 (=fNi60-fNi58)
        # Collaps 2017: 508.2(4)[76] MHz (Simon's PHD thesis)
        # Collaps 2016: 510.6(6)[95]MHz (Simon's PHD thesis)
        # Collaps published: 509.1(25)[42] (PRL 124, 132502, 2020)
        # Steudel 1980: 0.01694(9) cm-1 corresponds to 507.8(27) MHz
        self.literature_IS60vs58 = 509.1
        self.literature_IS60vs58_d_stat = 2.5
        self.literature_IS60vs58_d_syst = 4.2

        ''' literature Mass Shift and Field Shift constants '''
        king_literature = {'kaufm60': {'Alpha': 396, 'F': (-769, 60), 'Kalpha': (948000, 3000)},  # Kaufmann.2020 10.1103/PhysRevLett.124.132502
                           'koenig60': {'Alpha': 388, 'F': (-761.87, 89.22), 'Kalpha': (953881, 4717)},  # König.2020 private com
                           'koenig58': {'Alpha': 419, 'F': (-745.27, 96.79), 'Kalpha': (930263, 3009)}  # König.2020 private com
                           }

        self.literature_massshift = king_literature[self.KingFactorLit]['Kalpha']  # Mhz u (lit val given in GHz u)(949000, 4000)
        self.literature_fieldshift = king_literature[self.KingFactorLit]['F']  # MHz/fm^2(-788, 82)
        self.literature_alpha = king_literature[self.KingFactorLit]['Alpha']  # u fm^2 397

        ''' literature radii '''
        delta_rms_kaufm = {'58Ni': (-0.275, 0.007),
                           '60Ni': (0, 0),
                           '61Ni': (0.083, 0.005),
                           '62Ni': (0.223, 0.005),
                           '64Ni': (0.368, 0.009),
                           '68Ni': (0.620, 0.021)}
        delta_rms_steudel = {'58Ni': (-0.218, 0.040),  # 58-60
                             '60Ni': (0, 0),  # Shifts are given in pairs. Most ref to 60.
                             '61Ni': (0.065, 0.017),  # 60-61
                             '62Ni': (0.170, 0.035),  # 60-62
                             '64Ni': (0.280, 0.041)}  # 60-62 and 62-64 combined. Quadratic error prop
        delta_rms_koenig = {'58Ni': (-0.275, 0.0082),  # private com. excel sheet mid 2020
                             '60Ni': (0, 0),
                             '62Ni': (0.2226, 0.0059),
                             '64Ni': (0.3642, 0.0095)}

        self.delta_rms_lit = {'Kaufmann 2020': delta_rms_kaufm,
                              'Steudel 1980': delta_rms_steudel,
                              'Koenig 2020': delta_rms_koenig}

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
        self.laser_freqs = {'55Ni': 851264686.7203143,  # 14197.61675,
                            '56Ni': 851253864.2125804,  # 14197.38625,
                            '58Ni': 851238644.9486578,  # 14197.13242,
                            '60Ni': 851224124.8007469  # 14196.89025
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
        copy_line_list[3] = self.restframe_trans_freq
        line_new = tuple(copy_line_list)
        cur.execute('''INSERT OR REPLACE INTO Lines VALUES (?,?,?,?,?,?,?,?,?)''', line_new)
        con.commit()
        con.close()
        # TODO: include uncertainty; write to Lines db

        # calculate differential doppler shifts
        # TODO: Why calculate here with volt+600??
        self.diff_dopplers = {
        key: Physics.diffDoppler(self.restframe_trans_freq, self.accVolt_set, self.masses[key][0] / 1e6)
        for key in self.masses.keys()}
        self.diff_dopplers2 = {
            key: Physics.diffDoppler(self.restframe_trans_freq, self.accVolt_set + 1660, self.masses[key][0] / 1e6)
            for key in self.masses.keys()}
        # adjust center fit estimations to accVoltage
        self.adjust_center_ests_db()

        # time reference
        self.ref_datetime = datetime.strptime('2018-04-13_13:08:55', '%Y-%m-%d_%H:%M:%S')  # run 6191, first 58 we use

        # create results dictionary:
        '''
        self.results['isotope']
                        ['file_names']
                        ['file_numbers']
                        ['file_times']
                        ['scaler_no']
                            ['center_fits']
                                ['vals']
                                ['d_stat']
                                ['d_syst']
                            ['shifts_iso-58']
                                ['vals']
                                ['d_stat']
                                ['d_syst']
                            ['avg_shift_iso-58']
                                ['vals']
                                ['d_stat']
                                ['d_syst']
                            ['acc_volts']
                                ['vals']
                                ['d_stat']
                                ['d_syst']
                            ['hfs_pars']
                                ['Al']
                                ['Au']
                                ['Bl']
                                ['Bu']
                        ['color']
        '''
        self.results = {}

        # current scaler variable:
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
                for iso in ['58Ni', '60Ni', '56Ni']:
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
            self.choose_runs_write_basics_to_results('55Ni')
            # export the results of initial fitting.
            self.export_results()

    def combine_single_scaler_centers(self, isolist, calibrated=False):
        """ use single scaler results and combine per file """
        sc_prefix = 'scaler_'
        if calibrated:
            if not 'sum' in isolist[0]:  # TODO: ugly workaround. Should find a good convention on the c use
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

                # calculate weighted average
                weights = 1 / err_arr ** 2
                wavg, sumw = np.average(val_arr, weights=weights, returned=True)
                werr = np.sqrt(1 / sumw)
                # calculate standard deviation of values
                std = np.std(val_arr)

                # Append to combined results list. For error use the larger of standard deviation or weighted avg errors
                centers_combined.append(wavg)
                d_fit = max(werr, std)
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

            # calculate weighted avg of center fit
            weights = 1 / np.array(centers_combined_d_stat) ** 2
            wavg, sumw = np.average(np.array(centers_combined), weights=weights, returned=True)
            wavg_d = np.sqrt(1 / sumw)
            # calculate standard deviation of values
            st_dev = np.array(centers_combined).std()
            combined_dict = {'scaler_c012': {'center_fits': {'vals': centers_combined,
                                                             'd_fit': centers_combined_d_fit,
                                                             'd_stat': centers_combined_d_stat,
                                                             'd_syst': centers_combined_d_syst},
                                             'avg_center_fits': {'vals': [wavg],
                                                                 'd_fit': [wavg_d],
                                                                 'd_stat': [max(wavg_d, st_dev)],
                                                                 'd_syst': [centers_combined_d_syst[0]]},
                                             }}
            TiTs.merge_extend_dicts(self.results[iso], combined_dict, overwrite=True, force_overwrite=True)

        self.plot_parameter_for_isos_and_scaler([isolist[1]], ['scaler_c012', 'scaler_012'], 'center_fits')

    def plot_results_of_fit(self, calibrated=False):
        add_sc = []
        isolist = ['56Ni', '58Ni', '60Ni']
        if calibrated:
            add_sc = ['scaler_c0', 'scaler_c1', 'scaler_c2']
            isolist = ['{}_cal'.format(i) for i in isolist]
        # plot iso-results for each scaler combination
        for sc in self.scaler_combinations+['scaler_c012']+add_sc:
            # write the scaler to db for usage
            scaler = self.update_scalers_in_db(sc)
            # plot results of first fit
            self.plot_parameter_for_isos_and_scaler(isolist, [scaler], 'center_fits', offset=[450, 0, -450])
            self.all_centerFreq_to_scanVolt(isolist, [scaler])
            self.plot_parameter_for_isos_and_scaler(isolist, [scaler], 'center_scanvolt', unit='V')
            if scaler != 'scaler_c012':  # fitpars don't make sense for the calculated combined scaler
                self.plot_parameter_for_isos_and_scaler(isolist, [scaler], 'rChi')
                self.get_weighted_avg_linepars(isolist, [scaler])
                self.plot_parameter_for_isos_and_scaler(isolist, [scaler], 'all_fitpars:center', unit='')
                for par, vals in self.initial_par_guess.items():
                    used, fixed = self.check_par_in_lineshape(par)
                    if used and not vals[1]==True:  # only plot when used and not fixed
                        self.plot_parameter_for_isos_and_scaler(isolist, [scaler], 'all_fitpars:{}'.format(par), unit='')

        # plot all scaler-results for each isotope
        for iso in isolist:
            self.plot_parameter_for_isos_and_scaler([iso], self.scaler_combinations+['scaler_c012']+add_sc,
                                                    'center_fits', onlyfiterrs=True)
            self.plot_parameter_for_isos_and_scaler([iso], self.scaler_combinations+add_sc, 'all_fitpars:center')
            for par, vals in self.initial_par_guess.items():
                used, fixed = self.check_par_in_lineshape(par)
                if used and not vals[1]==True:  # only plot when used and not fixed
                    self.plot_parameter_for_isos_and_scaler([iso], self.scaler_combinations+add_sc, 'all_fitpars:{}'.format(par), unit='')

    def ion_energy_calibration(self):
        """
        Separated from separate_runs_analysis on 11.05.2020.
        Calibration will be done for each scaler and written to results db.
        :return:
        """
        for sc in self.scaler_combinations+['scaler_c012']:
            logging.info('\n'
                         '## ion energy calibration started for scaler {}'
                         .format(sc))
            # write the scaler to db for usage
            scaler = self.update_scalers_in_db(sc)

            # calculate isotope shift and calibrate voltage
            if self.use_handassigned:
                # TODO: calibration must be updated here
                self.calibrateVoltage(self.calib_tuples)
                self.assign_calibration_voltage_handassigned('56Ni', '58Ni', scaler, self.files56_handassigned_to_calibs)
            else:
                # interpolate between calibration tuples.
                if self.calibration_method == 'isoshift':  # calibrate vs literature isotope shift
                    self.accVolt_corrected = self.calibVoltageOffset()  # TODO: This is new, clean up around it...
                    self.getVoltDeviationToResults('58Ni', allNull=True)
                    self.getVoltDeviationToResults('60Ni', allNull=True)
                    self.getVoltDeviationToResults('56Ni', allNull=True)  # never actually used, just for fun
                    for iso in ['55Ni', '56Ni', '58Ni', '60Ni']:
                        self.calibVoltageFunct(iso, scaler, use58only=True)
                elif self.calibration_method == 'combined':  # calibrate vs literature isotope shift + adj. offsets
                    self.accVolt_corrected = self.calibVoltageOffset()  # TODO: This is new, clean up around it...
                    self.getVoltDeviationToResults('58Ni')
                    self.getVoltDeviationToResults('60Ni')
                    self.getVoltDeviationToResults('56Ni')  # never actually used, just for fun
                    for iso in ['55Ni', '56Ni', '58Ni', '60Ni']:
                        self.calibVoltageFunct(iso, scaler, use58only=False)
                elif self.calibration_method == 'relative':  # calibrate vs relative transition frequency
                    self.accVolt_corrected = (self.accVolt_set, self.accVolt_set_d)  # no large scale correction
                    self.getVoltDeviationToResults('58Ni')
                    self.getVoltDeviationToResults('60Ni')
                    self.getVoltDeviationToResults('56Ni')  # never actually used, just for fun
                    for iso in ['55Ni', '56Ni', '58Ni', '60Ni']:
                        self.calibVoltageFunct(iso, scaler, use58only=False)
                elif self.calibration_method == 'absolute':  # calibrate vs absolute transition frequency
                    mean_offset_58 = self.getVoltDeviationFromAbsoluteTransFreq('58Ni')
                    self.getVoltDeviationToResults('60Ni', offset=mean_offset_58)
                    self.getVoltDeviationToResults('56Ni', offset=mean_offset_58)  # never actually used, just for fun
                    for iso in ['55Ni', '56Ni', '58Ni', '60Ni']:
                        self.calibVoltageFunct(iso, scaler, use58only=True)
                elif self.calibration_method == 'None':  # No Calibration
                    self.accVolt_corrected = (self.accVolt_set, self.accVolt_set_d)  # no large scale correction
                    self.getVoltDeviationToResults('58Ni', allNull=True)
                    self.getVoltDeviationToResults('60Ni', allNull=True)
                    self.getVoltDeviationToResults('56Ni', allNull=True)  # never actually used, just for fun
                    for iso in ['55Ni', '56Ni', '58Ni', '60Ni']:
                        self.calibVoltageFunct(iso, scaler, use58only=False)
                self.plot_parameter_for_isos_and_scaler(['56Ni', '58Ni', '60Ni'], [scaler], 'voltage_deviation',
                                                        unit='V')
                self.plot_parameter_for_isos_and_scaler(['58Ni_cal'], [scaler], 'acc_volts', unit='V')

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
                    # TODO: I used avg fitpars here before. Don't think I wanna do this after all...
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

            self.export_results()

    def extract_isoshifts_from_fitres(self, isolist, refiso, calibrated=False):
        """
        isotope shift extraction.
        :return:
        """
        if calibrated:
            isolist = ['{}_cal'.format(i) for i in isolist]  #make sure to use the calibrated isos when calibration = True
        # TODO: rework uncertainties!!
        # calculate isotope shift and calibrate voltage
        for iso in isolist:
            for sc in self.results[iso].keys():
                if 'scaler' in sc:  # is a scaler key
                    logging.info('\n'
                                 '## extracting isotope shifts for scaler {}'.format(sc))
                    # write the scaler to db for usage
                    scaler = self.update_scalers_in_db(sc)
                    if self.use_handassigned:
                            self.extract_iso_shift_handassigned(iso, refiso, scaler, self.files56_handassigned_to_calibs, calibrated)
                            self.plot_parameter_for_isos_and_scaler([iso], [scaler], 'shifts_iso-{}'.format(refiso[:2]))
                    else:
                        # interpolate between calibration tuples.
                        for iso in isolist:
                            self.extract_iso_shift_interp(iso, refiso, scaler, calibrated)
                            self.plot_parameter_for_isos_and_scaler([iso], [scaler], 'shifts_iso-{}'.format(refiso[:2]))

        # self.plot_parameter_for_isos_vs_scaler(isolist, self.scaler_combinations + ['scaler_c012'],
        #                                        'avg_shift_iso-{}'.format(refiso[:2]), offset=True)

    def create_and_fit_stacked_runs(self, calibrated=False):
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
        self.ni55analysis_combined_files = []
        self.create_stacked_files(calibrated)

        # update database to use all three scalers for analysis
        self.update_scalers_in_db('0,1,2')  # scalers to be used for combined analysis

        # do a batchfit of the newly created files
        isolist = ['58Ni_sum', '60Ni_sum', '56Ni_sum', '55Ni_sum']
        if calibrated:
            isolist = ['{}_cal'.format(i) for i in isolist]

        if self.do_the_fitting:
            for iso in isolist:
                for sc in self.scaler_combinations:
                    scaler = self.update_scalers_in_db(sc)
                    if calibrated:
                        # Create the isotope in results db already:
                        accV_dict = self.results['{}_cal'.format(iso[:4])]['scaler_c012']['acc_volts']
                        isodict = {iso: {'color': self.isotope_colors[int(iso[:2])],
                                         scaler: {'acc_volts': {'vals': [np.array(accV_dict['vals']).mean()],
                                                                'd_stat': [np.array(accV_dict['d_stat']).mean()],
                                                                'd_syst': [np.array(accV_dict['d_syst']).mean()]
                                                                }
                                                  }}}
                        TiTs.merge_extend_dicts(self.results, isodict, overwrite=True, force_overwrite=True)
                    # Do a first set of fits for all 58 & 60 runs without any calibration applied.
                    filelist, runNo, center_freqs, center_fit_errs, center_freqs_d, center_freqs_d_syst, start_times, fitpars, rChi = \
                        self.chooseAndFitRuns(iso)
                    # TODO: copy plot of fit from data to results folder
                    if '55' in iso:
                        # extract additional hfs fit parameters
                        al = self.param_from_fitres_db(filelist[0], iso, self.run, 'Al')
                        au = self.param_from_fitres_db(filelist[0], iso, self.run, 'Au')
                        bl = self.param_from_fitres_db(filelist[0], iso, self.run, 'Bl')
                        bu = self.param_from_fitres_db(filelist[0], iso, self.run, 'Bu')
                        hfs_dict = {'Al': al, 'Au': au, 'Bl': bl, 'Bu': bu}
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

        # plot results of first fit
        self.plot_parameter_for_isos_vs_scaler(isolist, self.scaler_combinations, 'center_fits', offset=True)



        # # write final result to database
        # self.write_shift_to_combined_db('60Ni_sum_cal', self.run,
        #                                 (isoShift60, isoShift60_d, isoShift60_d_syst),
        #                                 'BECOLA 2018; 3 scalers combined; calibrated; ni60 runs summed to file: {}'
        #                                 .format(self.ni55analysis_combined_files[2]))
        # self.write_shift_to_combined_db('56Ni_sum_cal', self.run,
        #                                 (isoShift56, isoShift56_d, isoShift56_d_syst),
        #                                 'BECOLA 2018; 3 scalers combined; calibrated; ni56 runs summed to file: {}'
        #                                 .format(self.ni55analysis_combined_files[2]))
        # self.write_shift_to_combined_db('55Ni_sum_cal', self.run,
        #                                 (isoShift55, isoShift55_d, isoShift55_d_syst),
        #                                 'BECOLA 2018; 3 scalers combined; calibrated; ni56 runs summed to file: {}'
        #                                 .format(self.ni55analysis_combined_files[3]))
        #
        # self.plot_sum_results()
        #
        # # Extract A and B factors and do stuff:
        # self.ni55_A_B_analysis('55Ni_sum_cal')
        #
        # dr55, dr55_d = self.extract_radius_from_factors('55Ni_sum_cal', '58Ni_sum_cal')
        # print('Ni55 deltar_rms: {}+-{}'.format(dr55, dr55_d))
        # dr56, dr56_d = self.extract_radius_from_factors('56Ni_sum_cal', '58Ni_sum_cal')
        # print('Ni56 deltar_rms: {}+-{}'.format(dr56, dr56_d))
        # dr60, dr60_d = self.extract_radius_from_factors('60Ni_sum_cal', '58Ni_sum_cal')
        # print('Ni60 deltar_rms: {}+-{}'.format(dr60, dr60_d))

        ####################
        # Do the king plot #
        ####################
        # TODO: Probably need to automate reference run here? But need to adjust value in Collaps results as well...
        # TODO: Also: Do I want 56/58Ni or 56/58Ni_sum_cal
        # refrun = self.run
        # self.perform_king_fit_analysis(self.delta_lit_radii_58,
        #                                isotopes=['55Ni_sum_cal', '56Ni_sum_cal', '56Ni', '58Ni', '59Ni', '60Ni', '61Ni',
        #                                          '62Ni', '64Ni'],
        #                                reference_run=refrun)

        # ###################
        # # export results  #
        # ###################
        # to_file_dict = {}
        # for keys, vals in self.results.items():
        #     # xml cannot take numbers as first letter of key
        #     to_file_dict['i' + keys] = vals
        # # add analysis parameters
        # to_file_dict['parameters'] = {'line_profiles': self.run,
        #                               'calib_assignment': 'handassigned' if self.use_handassigned else 'interpolated'}
        # dateandtime = datetime.now()
        # filename = 'results\\' + self.run_name + '_' + dateandtime.strftime("%Y-%m-%d_%H-%M") + '.xml'
        # self.writeXMLfromDict(to_file_dict, os.path.join(self.workdir, filename), 'BECOLA_Analysis')

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
        center_new = center_old + (self.accVolt_set - acc_volt) * Physics.diffDoppler(self.restframe_trans_freq, acc_volt, mass)
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
        stand_ests = {'55Ni': -1174, '56Ni': -712, '58Ni': -225, '60Ni': 293}  # values that worked fine for 29850V
        ref_freq_dev = 850343800 - self.restframe_trans_freq  # stand_ests are for 580343800MHz. Ref freq might be updated
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
        # do the batchfit
        if self.do_the_fitting:
            # define and create (if not exists) the output folder
            plot_specifier = 'plots\\' + self.scaler_name + '_' + iso[4:] + '\\'  # e.g. scaler_012_cal
            plot_folder = os.path.join(self.resultsdir, plot_specifier)
            if not os.path.exists(plot_folder):
                os.makedirs(plot_folder)
            # for softw_gates_trs from file use 'File' and from db use None.
            BatchFit.batchFit(filearray, self.db, self.run, x_as_voltage=True, softw_gates_trs=None, guess_offset=True,
                              save_to_folder=plot_folder)
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
                parsdict = {'center': (-510, 30, False)}  # TODO: use better dummy value (take from all_Center_MHz list)
            all_fitpars.append(parsdict)
            all_rundate.append(file_date)
            all_center_MHz.append(parsdict['center'][0])
            all_rChi.append(rChi[0][0])

            # === uncertainties ===
            # TODO: Renew this whole section from scratch!

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
        nuoff = self.restframe_trans_freq

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
                center_freq = self.results[isostring][scaler]['center_fits']['vals']
                # TODO: Check uncertainties here. Most come from calculation of freq and are not in dacVolts!!
                center_freq_d_fit = self.results[isostring][scaler]['center_fits']['d_fit']
                #center_freq_dsyst = self.results[isostring][scaler]['center_fits']['d_syst']

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

    def assign_calibration_voltage_handassigned(self, isotope, ref_isotope, scaler, assigned):
        '''
        :return:
        '''
        scaler = self.update_scalers_in_db(scaler)
        # get reference run times and voltages from database
        ref_numbers = self.results[ref_isotope]['file_numbers']
        ref_volts = self.results[ref_isotope][scaler]['acc_volts']['vals']
        ref_volts_d = self.results[ref_isotope][scaler]['acc_volts']['d_stat']
        ref_volts_d_syst = self.results[ref_isotope][scaler]['acc_volts']['d_syst']

        # get run times for isotope to be calibrated from database
        iso_numbers = self.results[isotope]['file_numbers']
        iso_names = self.results[isotope]['file_names']
        iso_volts = [29850] * len(iso_numbers)
        iso_volts_d = [29850] * len(iso_numbers)
        iso_volts_d_syst = [29850] * len(iso_numbers)

        for tuples in assigned:
            iso_run_no = tuples[0]
            ref_run_no = tuples[1][0]

            ref_indx = ref_numbers.index(ref_run_no)
            volt = ref_volts[ref_indx]
            volt_d = ref_volts_d[ref_indx]
            volt_d_syst = ref_volts_d_syst[ref_indx]

            iso_indx = iso_numbers.index(iso_run_no)
            file = iso_names[iso_indx]
            iso_volts[iso_indx] = volt
            iso_volts_d[iso_indx] = volt_d
            iso_volts_d_syst[iso_indx] = volt_d_syst

            # Update 'Files' in self.db
            isotope_cal = '{}_cal'.format(isotope)
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            cur.execute('''UPDATE Files SET accVolt = ?, type = ? WHERE file = ? ''',
                        (volt, '{}_{}'.format(isotope_cal, ref_run_no), file))
            con.commit()
            con.close()

            # Create new isotopes in self.db
            self.create_new_isotope_in_db(isotope, '{}_{}'.format(isotope_cal, ref_run_no), volt)

        # write calibration voltages back into database
        isotope_cal = '{}_cal'.format(isotope)
        voltdict = {isotope_cal: {scaler: {'acc_volts': {'vals': iso_volts,
                                                         'd_stat': iso_volts_d,
                                                         'd_syst': iso_volts_d_syst
                                                         }
                                           }
                                  }
                    }
        TiTs.merge_extend_dicts(self.results, voltdict)

    def calibrateVoltage(self, calibration_tuples):
        '''

        :param calibration_tuples:
        :return: calib_tuples_with_isoshift_and_calibrationvoltage:
                contains a tuple for each calibration point with entries: (58ref, 60ref, isoshift, isoshift_d, calVolt, calVoltStatErr, calVoltSystErr)
        '''
        #######################
        # Calibration process #
        #######################
        scaler = self.scaler_name

        # Calibration sets of 58/60Ni
        calib_tuples = calibration_tuples
        calib_tuples_with_isoshift = []

        # copy the results dicts and remove all non-calibration values
        isos = ['58Ni', '60Ni']
        dicts = [{}, {}]
        for j in range(2):
            new_results_dict = self.results[isos[j]]
            indexlist = [new_results_dict['file_numbers'].index(i[j]) for i in calib_tuples]
            for keys, vals in new_results_dict.items():
                if type(vals) is list:
                    new_results_dict[keys] = [vals[i] for i in indexlist]
            for keys, vals in new_results_dict[scaler].items():
                for keys2, vals2 in vals.items():
                    if type(vals2) is list:
                        vals[keys2] = [vals2[i] for i in indexlist]
            dicts[j] = new_results_dict
        new58_dict = dicts[0]
        new60_dict = dicts[1]

        # Calculate Isotope shift for all calibration tuples and add to list.
        for tuples in calib_tuples:
            # Get 58Nickel center fit parameter in MHz
            run58 = tuples[0]
            indx_58 = new58_dict['file_numbers'].index(run58)
            run58file = 'BECOLA_'+str(run58)+'.xml'
            center58 = self.param_from_fitres_db(run58file, '58Ni', self.run, 'center')
            center58_d_syst = new58_dict[scaler]['center_fits']['d_syst'][indx_58]

            # Get 60Nickel center fit parameter in MHz
            run60 = tuples[1]
            indx_60 = new60_dict['file_numbers'].index(run60)
            run60file = 'BECOLA_' + str(run60) + '.xml'
            center60 = self.param_from_fitres_db(run60file, '60Ni', self.run, 'center')
            center60_d_syst = new60_dict[scaler]['center_fits']['d_syst'][indx_60]

            # Calculate isotope shift of 60Ni with respect to 58Ni for this calibration point
            isoShift = center60[0]-center58[0]
            isoShift_d = np.sqrt(center60[1]**2+center58[1]**2)  # statistical uncertainty from fits
            isoShift_d_syst = np.sqrt(center58_d_syst ** 2 + center60_d_syst **2)
            tuple_with_isoshift = tuples + (isoShift, isoShift_d, isoShift_d_syst)
            calib_tuples_with_isoshift.append(tuple_with_isoshift)

        # write isotope shifts for nickel 60 to dict
        ni60_shifts = [i[2] for i in calib_tuples_with_isoshift]
        ni60_shifts_d = [i[3] for i in calib_tuples_with_isoshift]
        ni60_shifts_d_syst = [i[4] for i in calib_tuples_with_isoshift]
        new60_dict[scaler]['shifts_iso-58']['vals'] = ni60_shifts
        new60_dict[scaler]['shifts_iso-58']['d_stat'] = ni60_shifts_d
        new60_dict[scaler]['shifts_iso-58']['d_syst'] = ni60_shifts_d_syst
        # TODO: this is no weighted avg. Maybe remove...
        new60_dict[scaler]['avg_shift_iso-58']['vals'] = [sum(ni60_shifts)/len(ni60_shifts)]
        # write the new dicts to self.results
        TiTs.merge_extend_dicts(self.results, {'58Ni_ref': new58_dict}, overwrite=True, force_overwrite=True)
        TiTs.merge_extend_dicts(self.results, {'60Ni_ref': new60_dict}, overwrite=True, force_overwrite=True)

        # plot isotope shift for all calibration points (can be removed later on):
        self.plot_parameter_for_isos_and_scaler(['60Ni'], [scaler], 'shifts_iso-58')

        # Calculate resonance DAC Voltage from the 'center' positions
        calib_tuples_with_isoshift_and_calibrationvoltage = []
        average_calib_voltage = []
        # Do the voltage calibration for each tuple.
        for tuples in calib_tuples_with_isoshift:
            # get filenames
            run58, run60, isoShift, isoShift_d, isoShift_d_syst = tuples
            run58file = 'BECOLA_' + str(run58) + '.xml'
            run60file = 'BECOLA_' + str(run60) + '.xml'
            calib_point_dict = {run58file: {},
                                run60file: {}}

            # calculate centerDAC and get some usefull info
            for files, dicts in calib_point_dict.items():  # only 2 elements to iterate: 58 and 60
                con = sqlite3.connect(self.db)
                cur = con.cursor()
                # get laser frequency and accelVolt
                cur.execute(
                    '''SELECT type, accVolt, laserFreq, colDirTrue FROM Files WHERE file = ? ''', (files,))
                iso, accVolt, laserFreq, colDirTrue = cur.fetchall()[0]
                # Query fitresults for file and isotope combo
                if files is run58file:
                    # check whether it's a 58 or 60 file to choose the correct run
                    run = self.run
                else:
                    run = self.run
                cur.execute(
                    '''SELECT pars FROM FitRes WHERE file = ? AND iso = ? AND run = ?''', (files, iso, run))
                pars = cur.fetchall()
                # get mass
                cur.execute(
                    '''SELECT mass FROM Isotopes WHERE iso = ? ''', (iso,))
                isoMass = cur.fetchall()[0][0]
                con.close()

                # write to dict
                dicts['filename'] = files
                dicts['iso'] = iso
                dicts['accVolt'] = float(accVolt)
                dicts['laserFreq'] = float(laserFreq)
                dicts['colDirTrue'] = colDirTrue
                parsDict = ast.literal_eval(pars[0][0])
                center = parsDict['center'][0]
                dicts['center'] = float(center)
                dicts['isoMass'] = float(isoMass)

                # calculate resonance frequency
                dicts['resonanceFreq'] = self.restframe_trans_freq + dicts['center']
                # calculate relative velocity
                relVelocity = Physics.invRelDoppler(dicts['laserFreq'], dicts['resonanceFreq'])
                # calculate relativistic energy of the beam particles at resonance freq and thereby resonance Voltage
                centerE = Physics.relEnergy(relVelocity, dicts['isoMass'] * Physics.u)/Physics.qe
                # get DAC resonance voltage
                centerDAC = centerE - dicts['accVolt']  # TODO: can we get an error (statistical) here?

                dicts['centerDAC'] = centerDAC

            # alternative calibration process
            accVolt = calib_point_dict[run58file]['accVolt']  # should be the same for 58 and 60
            # Calculate differential Doppler shift for 58 and 60 nickel
            # TODO: uncertainties on this? Well, the voltage is quite uncertain, but the effect should be minimal
            diff_Doppler_58 = Physics.diffDoppler(self.restframe_trans_freq, accVolt, 58)
            diff_Doppler_60 = Physics.diffDoppler(self.restframe_trans_freq, accVolt, 60)
            # calculate measured isotope shift
            # TODO: uncertainties? Statistical uncertainty of the resonance frequency fit.
            # I think it is better to go through centerDAC values in order to incorporate wavemeter uncertainties instead of using 'resonanceFreq' directly.
            # calculate velocity for 58 and 60
            velo58sign = -1 if calib_point_dict[run58file]['colDirTrue'] else 1
            velo58 = velo58sign * Physics.relVelocity((accVolt + calib_point_dict[run58file]['centerDAC'])*Physics.qe,
                                         calib_point_dict[run58file]['isoMass']*Physics.u)
            velo60sign = -1 if calib_point_dict[run60file]['colDirTrue'] else 1
            velo60 = velo60sign * Physics.relVelocity((accVolt + calib_point_dict[run60file]['centerDAC']) * Physics.qe,
                                         calib_point_dict[run60file]['isoMass']*Physics.u)
            # calculate resonance frequency for 58 and 60
            # TODO: The uncertainty here comes from the wavemeter and is systematic. Check that this is correct.
            f_reso58 = Physics.relDoppler(calib_point_dict[run58file]['laserFreq'], velo58)
            f_reso58_d = Physics.relDoppler(self.wavemeter_wsu30_mhz_d, velo58)  # will be very close to wavemeter_d
            f_reso60 = Physics.relDoppler(calib_point_dict[run60file]['laserFreq'], velo60)
            f_reso60_d = Physics.relDoppler(self.wavemeter_wsu30_mhz_d, velo60)  # will be very close to wavemeter_d
            # isotope shift from calibration tuple:
            # calculate isotope shift
            isoShift = f_reso60 - f_reso58
            isoShift_d_stat = isoShift_d  # TODO: This should be the statistical part of the uncertainty, coming from the center fit uncertainties
            # isoShift_d_syst = np.sqrt(np.square(f_reso58_d) + np.square(f_reso60_d))  # TODO: This is a systematic uncertainty from the wavemeter uncertainty

            # calculate calibration Voltage
            calibrated_voltage = (isoShift-self.literature_IS60vs58)/(diff_Doppler_60-diff_Doppler_58)+accVolt
            # TODO: Uncertainties are now split into systematic and statistic. Use accordingly!
            calibrated_voltage_d_stat = np.sqrt(np.square(self.literature_IS60vs58_d_stat / (diff_Doppler_60 - diff_Doppler_58)) +
                                           np.square(isoShift_d_stat / (diff_Doppler_60 - diff_Doppler_58)))
            # TODO: For now I'm working with the statistical uncertainty only. But I need to add up the systematics as well.
            calibrated_voltage_d_syst = np.sqrt(np.square(self.literature_IS60vs58_d_syst /(diff_Doppler_60-diff_Doppler_58)) +
                                           np.square(isoShift_d_syst / (diff_Doppler_60 - diff_Doppler_58)))

            # create a new tuple with (58ref, 60ref, isoshift, calVolt, calVoltErr)
            tuple_withcalibvolt = tuples + (calibrated_voltage, calibrated_voltage_d_stat, calibrated_voltage_d_syst)
            # contains a tuple for each calibration point with entries: (58ref, 60ref, isoshift, isoshift_d, isoshift_d_syst, calVolt, calVoltStatErr, calVoltSystErr)
            calib_tuples_with_isoshift_and_calibrationvoltage.append(tuple_withcalibvolt)

            average_calib_voltage.append(calibrated_voltage)
            #print(calibrated_voltage)

            # display calibration graph
            #plt.plot(voltage_list, IS_perVolt_list)
            #plt.scatter(calibrated_voltage, m * calibrated_voltage + b)
            #plt.title('Voltage Calibration for Calibration Tuple [Ni58:{}/Ni60:{}]'.format(run58, run60))
            #plt.xlabel('voltage [V]')
            #plt.ylabel('isotope shift [MHz]')
            #plt.show()

        self.ni56_average_calib_voltage = sum(average_calib_voltage[:7])/len(calib_tuples[:7])  # TODO: lets see how this does to the 56 runs


        acc_volt_dict = {'acc_volts': {'vals': [i[5] for i in calib_tuples_with_isoshift_and_calibrationvoltage],
                                      'd_stat': [i[6] for i in calib_tuples_with_isoshift_and_calibrationvoltage],
                                      'd_syst': [i[7] for i in calib_tuples_with_isoshift_and_calibrationvoltage]
                                      }
                         }
        calibrations_dict = {'58Ni_ref': {scaler: acc_volt_dict},
                             '60Ni_ref': {scaler: acc_volt_dict}
                             }
        # write to results dict
        TiTs.merge_extend_dicts(self.results, calibrations_dict, overwrite=True, force_overwrite=True)

        # plot calibration voltages:
        setvolt = 29850
        self.plot_parameter_for_isos_and_scaler(['58Ni_ref'], [scaler], 'acc_volts', overlay=setvolt, unit='V')


        # Write calibrations to XML database
        print('Updating self.db with new voltages now...')
        for entries in calib_tuples_with_isoshift_and_calibrationvoltage:
            calibration_name = str(entries[0]) + 'w' +  str(entries[1])
            file58 = 'BECOLA_' + str(entries[0]) + '.xml'
            file58_newType = '58Ni_cal' + calibration_name
            file60 = 'BECOLA_' + str(entries[1]) + '.xml'
            file60_newType = '60Ni_cal' + calibration_name
            new_voltage = entries[5]

            # Update 'Files' in self.db
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            cur.execute('''UPDATE Files SET accVolt = ?, type = ? WHERE file = ? ''', (new_voltage, file58_newType, file58))
            cur.execute('''UPDATE Files SET accVolt = ?, type = ? WHERE file = ? ''', (new_voltage, file60_newType, file60))
            con.commit()
            con.close()

            # Create new isotopes in self.db
            self.create_new_isotope_in_db('58Ni', file58_newType, new_voltage)
            self.create_new_isotope_in_db('60Ni', file60_newType, new_voltage)
        print('...self.db update completed!')

        return calib_tuples_with_isoshift_and_calibrationvoltage

    def calibVoltageOffset(self):
        """
        Do an offset calibration for the acceleration voltage using the literature isotope shift.
        This will return ONE calibration voltage
        :return:
        """
        # Get shift data from previous fit results
        file_times = self.results['60Ni']['file_times']
        isoshift_dict = self.results['60Ni'][self.scaler_name]['shifts_iso-58']
        shift = isoshift_dict['vals']
        shift_d = isoshift_dict['d_stat']
        shift_d_sys = 0.0  #isoshift_dict['d_syst']  # due to voltage uncertainty: the thing we want to correct for! Set 0

        # make floats (seconds relative to reference-time) out of the dates
        time_floats = list((t - self.ref_datetime).total_seconds() for t in file_times)

        # compare to literature isoshift
        lit_shift_dev = np.array(shift) - self.literature_IS60vs58
        lit_shift_dev_d = np.array(shift_d)  # only these errors change on a file-to-file basis!
        lit_shift_dev_d_sys = np.sqrt(np.array(shift_d_sys).mean()**2 +  # should be constant, taking mean anyway
                                      (self.literature_IS60vs58_d_stat + self.literature_IS60vs58_d_syst)**2)
        lit_shift_dev_d_abs = lit_shift_dev_d + lit_shift_dev_d_sys  # TODO: Is this the correct way to take the absolute err?

        # fit a straight line to the deviation
        # NOTE: One could also include a slope, but with the uncertainties, the error of the slope turns out bigger than the value.
        def _offset(x, b):
            return b
        b_opt = curve_fit(_offset, time_floats, lit_shift_dev, p0=lit_shift_dev[0],
                          sigma=lit_shift_dev_d, absolute_sigma=True)
        b_err = np.sqrt(np.diag(b_opt[1]))

        #TODO: might want to bring down the literature IS error across this fit. Will be too small else!!
        accVolt_corrected = b_opt[0][0] / (self.diff_dopplers['60Ni']-self.diff_dopplers['58Ni'])+self.accVolt_set
        accVolt_corrected_d = b_err[0] / (self.diff_dopplers['60Ni']-self.diff_dopplers['58Ni'])
        accVolt_corrected_d_sys = lit_shift_dev_d_sys.mean() / (self.diff_dopplers['60Ni']-self.diff_dopplers['58Ni'])
        accVolt_corrected_d_total = np.sqrt(accVolt_corrected_d**2+accVolt_corrected_d_sys**2)

        return (accVolt_corrected, accVolt_corrected_d_total)

    def getVoltDeviationFromAbsoluteTransFreq(self, iso):
        # Get center dac data from previous fit results
        file_times = self.results[iso]['file_times']
        center_freq_dict = self.results[iso][self.scaler_name]['center_fits']
        rel_center = np.array(center_freq_dict['vals'])
        rel_center_d = np.array(center_freq_dict['d_stat'])
        rel_center_d_sys = np.array(center_freq_dict['d_syst'])  # This includes the uncertainty we want to reduce!!!
        rel_center_d_sys = np.sqrt(np.square(rel_center_d_sys)
                                    - np.square(self.accVolt_set_d * self.diff_dopplers[iso[:4]])
                                    )  # remove buncher potential uncertainty since we calibrate that now.

        volt_dev = -rel_center/self.diff_dopplers[iso[:4]]  #(neg) acc.voltages are stored as positive, so we need the -
        volt_dev_d = rel_center_d/self.diff_dopplers[iso[:4]]
        # systematics from absolute center uncertainty:
        volt_dev_d_sys = np.sqrt(np.square(rel_center_d_sys) + np.square(self.restframe_trans_freq_d)
                                 )/self.diff_dopplers[iso[:4]]
        # correction dict
        volt_dev_dict = {'vals': volt_dev.tolist(),
                         'd_stat': volt_dev_d.tolist(),
                         'd_syst': volt_dev_d_sys.tolist()}
        self.results[iso][self.scaler_name]['voltage_deviation'] = volt_dev_dict

        # TODO: Maybe weighted average instead of mean here?
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

        # TODO: Only fit uncertainties in scanvolt_d!
        #  Must include the other uncertainties(wavemeter, ...) here.

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

    def calibVoltageFunct(self, isotope, scaler, use58only=False, userefscaler=False):
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

        # get reference run times and voltages from database
        ni58_numbers = self.results['58Ni']['file_numbers']
        dates_58 = self.results['58Ni']['file_times']
        volt58_dev = self.results['58Ni'][scaler]['voltage_deviation']['vals']
        volt58_dev_d = self.results['58Ni'][scaler]['voltage_deviation']['d_stat']
        volt58_dev_d_syst = self.results['58Ni'][scaler]['voltage_deviation']['d_syst']
        ni60_numbers = self.results['60Ni']['file_numbers']
        dates_60 = self.results['60Ni']['file_times']
        volt60_dev = self.results['60Ni'][scaler]['voltage_deviation']['vals']
        volt60_dev_d = self.results['60Ni'][scaler]['voltage_deviation']['d_stat']
        volt60_dev_d_syst = self.results['60Ni'][scaler]['voltage_deviation']['d_syst']
        ref_dates = []
        ref_volt_cor = []  # remember: A positive entry means negative deviation, so we can add for correction!
        ref_volt_cor_d = []
        ref_volt_cor_d_syst = []
        for tuples in self.calib_tuples:
            #TODO: Check how I deal with the calibration tuple assignment in other places of the code!
            run_no_58 = tuples[0]
            run_no_60 = tuples[1]
            ind58 = ni58_numbers.index(run_no_58)
            ind60 = ni60_numbers.index(run_no_60)
            # Calculate means and errors
            if use58only:
                ref_time = dates_58[ind58]
                ref_v = volt58_dev[ind58]
                ref_v_d = volt58_dev_d[ind58]
                ref_v_d_syst = volt58_dev_d_syst[ind58]
            else:
                ref_time = dates_58[ind58]+(dates_60[ind60]-dates_58[ind58])/2
                ref_v = (volt58_dev[ind58]+volt60_dev[ind60])/2
                ref_v_d = np.sqrt(np.square(volt58_dev_d[ind58]/2) + np.square(volt60_dev_d[ind60]/2))
                ref_v_d_syst = np.sqrt(np.square(volt58_dev_d_syst[ind58]/2) + np.square(volt60_dev_d_syst[ind60]/2))
            # Append to list
            ref_dates.append(ref_time)
            ref_volt_cor.append(volt_offset - ref_v)
            ref_volt_cor_d.append(ref_v_d)
            ref_volt_cor_d_syst.append(np.sqrt(ref_v_d_syst**2+volt_offset_d**2))
        iso_times = self.results[isotope]['file_times']
        iso_names = self.results[isotope]['file_names']
        iso_numbers = self.results[isotope]['file_numbers']
        iso_color = self.results[isotope]['color']

        # make floats (seconds relative to reference-time) out of the dates
        iso_dates = list((t - self.ref_datetime).total_seconds() for t in iso_times)
        ref_dates = list((t - self.ref_datetime).total_seconds() for t in ref_dates)

        # use np.interp to assign voltage deviations to the requested run.
        interpolation = np.interp(iso_dates, ref_dates, ref_volt_cor)
        interpolation_d = np.interp(iso_dates, ref_dates, ref_volt_cor_d)
        interpolation_d_syst = np.interp(iso_dates, ref_dates, ref_volt_cor_d_syst)

        # calculate the final voltage value combining the interpolated deviation with the overall offset
        voltcorrect = interpolation
        voltcorrect_d = interpolation_d  # the overall correction is completely systematic (same for all files)
        voltcorrect_d_syst = interpolation_d_syst

        # calculate weighted avg of center fit
        weights = 1 / np.array(voltcorrect_d) ** 2
        wavg, sumw = np.average(np.array(voltcorrect), weights=weights, returned=True)
        wavg_d = np.sqrt(1 / sumw)
        # calculate standard deviation of values
        st_dev = np.array(voltcorrect).std()

        # write calibration voltages back into database
        isotope_cal = '{}_cal'.format(isotope)
        voltdict = {isotope_cal: {scaler: {'acc_volts': {'vals': voltcorrect.tolist(),
                                                         'd_stat': voltcorrect_d.tolist(),
                                                         'd_syst': voltcorrect_d_syst.tolist()},
                                           'avg_acc_volts': {'vals': [wavg],
                                                             'd_stat': [max(wavg_d, st_dev)],
                                                             'd_syst': [voltcorrect_d_syst[0]]}
                                           },
                                  'file_names': iso_names,
                                  'file_numbers': iso_numbers,
                                  'file_times': iso_times,
                                  'color': iso_color
                                  }
                    }
        TiTs.merge_extend_dicts(self.results, voltdict)

        # make a quick plot of references and calibrated voltages
        fig, ax = plt.subplots()
        ref_timedeltas = np.array([timedelta(seconds=s) for s in ref_dates])
        ref_dates = np.array(self.ref_datetime + ref_timedeltas)  # convert back to datetime
        iso_timedeltas = np.array([timedelta(seconds=s) for s in iso_dates])
        iso_dates = np.array(self.ref_datetime + iso_timedeltas)  # convert back to datetime
        ax.plot(ref_dates, np.array(ref_volt_cor), '--o', color='black', label='accVolt_corrected')
        # plot error band for statistical errors
        ax.fill_between(ref_dates,
                         np.array(ref_volt_cor) - ref_volt_cor_d,
                         np.array(ref_volt_cor) + ref_volt_cor_d,
                         alpha=0.5, edgecolor='black', facecolor='black')
        # plot error band for systematic errors on top of statistical errors
        ax.fill_between(ref_dates,
                         np.array(ref_volt_cor) - ref_volt_cor_d_syst - ref_volt_cor_d,
                         np.array(ref_volt_cor) + ref_volt_cor_d_syst + ref_volt_cor_d,
                         alpha=0.2, edgecolor='black', facecolor='black')
        # and finally plot the interpolated voltages
        ax.errorbar(iso_dates, voltcorrect, yerr=voltcorrect_d+voltcorrect_d_syst, marker='s', linestyle='None',
                    color=iso_color, label='{} interpolated voltage'.format(isotope))
        # make x-axis dates
        plt.xlabel('date')
        days_fmt = mpdate.DateFormatter('%d.%B')
        ax.xaxis.set_major_formatter(days_fmt)
        plt.legend(loc='best')
        plt.xticks(rotation=45)  # rotate date labels 45 deg for better readability
        plt.margins(0.05)
        if self.save_plots_to_file:
            filename = 'voltInterp_' + isotope[:2] + '_sc' + scaler.split('_')[1]
            plt.savefig(self.resultsdir + filename + '.png')
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
        if calibrated and not 'sum' in isotope:
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
        laser_freq_d_syst = self.wavemeter_wsu30_mhz_d  # relative deviation of the two laser freqs to each other. HeNE drift should cancel for isotope shifts TODO: check!
        alignment_d_syst = self.laserionoverlap_MHz_d  # TODO: Could be eliminated if there are no changes to ion or laser optics between ref & iso

        iso_shifts_d_syst = np.sqrt(ion_energy_d_syst**2
                                    + laser_freq_d_syst**2
                                    + alignment_d_syst**2)

        # calculate an average value using weighted avg
        weights = 1/iso_shifts_d_stat**2
        iso_shift_avg = np.sum(weights*iso_shifts)/np.sum(weights)
        iso_shift_avg_d = np.sqrt(1/np.sum(weights))
        iso_shift_avg_d_syst = sum(iso_shifts_d_syst)/len(iso_shifts_d_syst)  # should all be the same anyways
        # also get the standard deviation. Use this if larger!
        iso_shift_st_dev = np.std(iso_shifts)

        # write isoshift to results dict
        shift_dict = {isotope: {scaler: {'shifts_iso-{}'.format(reference[:2]): {'vals': iso_shifts.tolist(),
                                                                                 'd_stat': iso_shifts_d_stat.tolist(),
                                                                                 'd_syst': iso_shifts_d_syst.tolist()},
                                         'avg_shift_iso-{}'.format(reference[:2]): {'vals': [iso_shift_avg],
                                                                                    'd_stat': [max(iso_shift_avg_d, iso_shift_st_dev)],
                                                                                    'd_syst': [iso_shift_avg_d_syst]}
                                         }}}
        TiTs.merge_extend_dicts(self.results, shift_dict, overwrite=True, force_overwrite=True)

    def extract_iso_shift_handassigned(self, isotope, reference, scaler, assigned, calibrated=False):
        # TODO: lots to do here! Since I changed reference to be a variable, I also need to reflect that in picking from tuples!
        scaler = self.update_scalers_in_db(scaler)
        ref = '{}{}'.format(reference[:2], isotope[2:])

        # Get info for isotope of interest
        iso_files = self.results[isotope]['file_names']
        iso_numbers = self.results[isotope]['file_numbers']
        iso_dates = self.results[isotope]['file_times']
        iso_center = self.results[isotope][scaler]['center_fits']['vals']
        iso_center_d = self.results[isotope][scaler]['center_fits']['d_stat']
        iso_center_d_syst = self.results[isotope][scaler]['center_fits']['d_syst']
        if calibrated:
            iso_volts = self.results[isotope][scaler]['acc_volts']['vals']
            iso_volts_d = self.results[isotope][scaler]['acc_volts']['d_stat']
            iso_volts_d_syst = self.results[isotope][scaler]['acc_volts']['d_syst']
        else:
            iso_volts = np.full((len(assigned)), self.accVolt_set)
            iso_volts_d = np.full((len(assigned)), 0)
            iso_volts_d_syst = np.full((len(assigned)), 0)

        # Get info for 58 Nickel:
        ref_dates = self.results[ref]['file_times']
        ref_numbers = self.results[ref]['file_numbers']
        ref_center = self.results[ref][scaler]['center_fits']['vals']
        ref_center_d = self.results[ref][scaler]['center_fits']['d_stat']
        ref_center_d_syst = self.results[ref][scaler]['center_fits']['d_syst']

        # create new lists that match iso_center lists in length and contain the corresponding 58 centers
        ref_center_assigned = []
        ref_center_d_assigned = []
        ref_center_d_syst_assigned = []
        iso_center_assigned = []
        iso_center_d_assigned = []
        iso_center_d_syst_assigned = []

        for tuples in assigned:
            if '56' in isotope:
                iso_run_no = tuples[0]
            else:  # only makes sense if it is 60 now
                iso_run_no = tuples[1][1]
            ref_run_no = tuples[1][0]

            iso_indx = iso_numbers.index(iso_run_no)
            ref_indx = ref_numbers.index(ref_run_no)

            iso_center_assigned.append(iso_center[iso_indx])
            iso_center_d_assigned.append(iso_center_d[iso_indx])
            iso_center_d_syst_assigned.append(iso_center_d_syst[iso_indx])

            ref_center_assigned.append(ref_center[ref_indx])
            ref_center_d_assigned.append(ref_center_d[ref_indx])
            ref_center_d_syst_assigned.append(ref_center_d_syst[ref_indx])

        diff_Doppler_ref = Physics.diffDoppler(self.restframe_trans_freq, np.array(iso_volts), int(reference[:2]))
        diff_Doppler_iso = Physics.diffDoppler(self.restframe_trans_freq, np.array(iso_volts), int(isotope[:2]))
        delta_diff_doppler = diff_Doppler_iso - diff_Doppler_ref

        # calculate isotope shifts now:
        iso_shifts = np.array(iso_center_d_assigned) - np.array(ref_center_assigned)
        iso_shifts_d = np.sqrt(np.array(iso_center_d_assigned) ** 2 + np.array(ref_center_d_assigned) ** 2)
        iso_shifts_d = np.sqrt(iso_shifts_d ** 2 + (delta_diff_doppler * iso_volts_d) ** 2)
        # TODO: still need systematic errors
        iso_shifts_d_syst = np.sqrt(2 * self.wavemeter_wsu30_mhz_d ** 2)
        # use differential doppler shift with cal voltage uncertainty to get MHz uncertainty:
        # TODO: do I need the systematic uncertainties of center fit positions here as well?
        iso_shifts_d_syst = np.sqrt((delta_diff_doppler * iso_volts_d_syst) ** 2 + iso_shifts_d_syst ** 2)

        # calculate an average value using weighted avg
        weights = 1 / iso_shifts_d ** 2
        iso_shift_avg = np.sum(weights * iso_shifts) / np.sum(weights)
        iso_shift_avg_d = np.sqrt(1 / np.sum(weights))
        # TODO: how to assign systematic uncertainties here? I think I should not weight them...
        iso_shift_avg_d_syst = sum(iso_shifts_d_syst) / len(iso_shifts_d_syst)
        #weights_syst = 1/iso_shifts_d_syst ** 2
        #iso_shift_avg_d_syst = np.sqrt(1 / np.sum(weights_syst))

        # write isoshift to results dict
        shift_dict = {isotope: {scaler: {'shifts_iso-{}'.format(reference[:2]): {'vals': iso_shifts.tolist(),
                                                                                 'd_stat': iso_shifts_d.tolist(),
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
        plt.errorbar(range(lensc0), self.results[iso]['scaler_0']['shifts_iso-58']['vals'], c='b',
                     yerr=self.results[iso]['scaler_0']['shifts_iso-58']['d_stat'],
                     label='scaler 0')
        lensc1 = len(self.results[iso]['file_numbers'])
        plt.errorbar(range(lensc1), self.results[iso]['scaler_1']['shifts_iso-58']['vals'], c='g',
                     yerr=self.results[iso]['scaler_1']['shifts_iso-58']['d_stat'],
                     label='scaler 1')
        lensc2 = len(self.results[iso]['file_numbers'])
        plt.errorbar(range(lensc2), self.results[iso]['scaler_2']['shifts_iso-58']['vals'], c='r',
                     yerr=self.results[iso]['scaler_2']['shifts_iso-58']['d_stat'],
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
            filename = 'shifts_iso-58_{}_sc012_overview'.format(iso[:2])
            plt.savefig(self.resultsdir + filename + '.png')
        else:
            plt.show()
        plt.close()
        plt.clf()

    ''' stacking files related: '''

    def create_stacked_files(self, calibrated=False):
        restriction = [6315, 6502]
        if calibrated:
            c = '_cal'
        else:
            c = ''
        # stack nickel 58 runs to new file Sum58_9999.xml. Only use calibration runs
        ni58_files, ni58_filenos, ni58_filetimes = self.pick_files_from_db_by_type_and_num('%58Ni%', selecttuple=[0, 6502])
        self.stack_runs('58Ni{}'.format(c), ni58_files, (-44, 14), binsize=1)
        # stack nickel 60 runs to new file Sum60_9999.xml. Only use calibration runs
        ni60_files, ni60_filenos, ni60_filetimes = self.pick_files_from_db_by_type_and_num('%60Ni%', selecttuple=[0, 6502])
        self.stack_runs('60Ni{}'.format(c), ni60_files, (-44, 14), binsize=1)
        # stack nickel 56 runs to new file Sum56_9999.xml
        ni56_files, ni56_filenos, ni56_filetimes = self.pick_files_from_db_by_type_and_num('%56Ni%', selecttuple=[0, 6502])
        self.stack_runs('56Ni{}'.format(c), ni56_files, (-34, 14), binsize=1)
        # select and stack nickel 55 runs to new file Sum55_9999.xml
        ni55_files, ni55_filenos, ni55_filetimes = self.pick_files_from_db_by_type_and_num('%55Ni%', selecttuple=restriction)
        self.stack_runs('55Ni{}'.format(c), ni55_files, (-263, -30), binsize=3)

    def stack_runs(self, isotope, files, volttuple, binsize):
        ##############
        # stack runs #
        ##############
        # sum all the isotope runs
        self.time_proj_res_per_scaler = self.stack_time_projections(isotope, files)
        self.addfiles(isotope, files, volttuple, binsize)

    def stack_time_projections(self, isotope, filelist):
        zeroarr_sc = np.zeros(1024)  # array for one scaler
        zeroarr = np.array([zeroarr_sc.copy(), zeroarr_sc.copy(), zeroarr_sc.copy()])
        timebins = np.arange(1024)
        for files in filelist:
            filepath = os.path.join(self.datafolder, files)
            # load the spec data from file
            spec = XMLImporter(path=filepath)
            for sc_no in range(3):
                # sum time projections for each scaler
                zeroarr[sc_no] += spec.t_proj[0][sc_no]
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
        plt.title('Stacked time projections for isotope: {}'.format(isotope))
        plt.legend(title='Scalers', bbox_to_anchor=(1.04, 0.5), loc="center left")
        plt.margins(0.05)
        if self.save_plots_to_file:
            filename = 'timeproject_files' + str(filelist[0]) + 'to' + str(filelist[-1])
            plt.savefig(self.resultsdir + filename + '.png', bbox_inches="tight")
        else:
            plt.show()
        plt.close()
        plt.clf()
        return timeproj_res

    def addfiles(self, iso, filelist, voltrange, binsize):
        """
        Load all files from list and rebin them into voltrange with binsize
        :param iso:
        :param filelist:
        :param voltrange:
        :param binsize:
        :return:
        """
        # Define the scan voltage range
        scanrange = voltrange[1]-voltrange[0]  # volts scanning up from startvolt

        # create arrays for rebinning the data
        volt_arr = np.arange(start=voltrange[0], stop=voltrange[1], step=binsize)  # array of the voltage steps
        zeroarr = np.zeros(len(volt_arr))  # should contain all the 55 scans so roughly -350 to +100
        cts_sum = [zeroarr.copy(), zeroarr.copy(), zeroarr.copy()]  # array for the absolute counts per bin. Same dimension as voltage of course
        err_sum = [zeroarr.copy(), zeroarr.copy(), zeroarr.copy()]  # we can add the errors from files or calc them new
        avgbg_sum = [zeroarr.copy(), zeroarr.copy(), zeroarr.copy()]  # array to keep track of the backgrounds

        # import voltage calibrations from combined scaler results on a per-file basis
        if '_cal' in iso:
            volt_corrections = self.results[iso]['scaler_c012']['acc_volts']
            filenames = self.results[iso]['file_names']
        else:
            volt_corrections = None

        # extract data from each file
        for files in filelist:
            # create filepath for XMLImporter
            filepath = os.path.join(self.datafolder, files)
            # get gates from stacked time projection
            sc0_res = self.time_proj_res_per_scaler['scaler_0']
            sc1_res = self.time_proj_res_per_scaler['scaler_1']
            sc2_res = self.time_proj_res_per_scaler['scaler_2']
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
            offst = 4  # background offset from midTof in multiple of sigma
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
            # check the stepsize of the data and return a warning if it's bigger than the binsize
            stepsize = spec.stepSize[0]
            nOfSteps = spec.getNrSteps(0)
            if stepsize > 1.1*binsize:
                logging.warning('Stepsize of file {} larger than specified binsize ({}>{})!'
                                .format(files, stepsize, binsize))
            # get volt (x) data, cts (y) data and errs
            voltage_x = spec.x[0]
            bg_sum_totalcts = [sum(background.cts[0][0]), sum(background.cts[0][1]), sum(background.cts[0][2])]
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
                    if voltrange[0] <= volt_c <= voltrange[1]:  # only use if inside desired range
                        voltind = (np.abs(volt_arr - volt_c)).argmin()  # find closest index in voltage array
                        cts_sum[scaler][voltind] += spec.cts[0][scaler][step]  # no normalization here
                        err_sum[scaler][voltind] += spec.err[0][scaler][step]  # just adding, since not same measurement TODO: probably not correct! Should be quadratic and then we can again just use the sqrt of cts
                        avgbg_sum[scaler][voltind] += bg_sum_totalcts[scaler] / nOfSteps

                # plt.plot(voltage_x, scaler_sum_cts[scaler], drawstyle='steps', label=filenumber)
        # plt.show()
        sumerr = [np.sqrt(cts_sum[0]), np.sqrt(cts_sum[1]), np.sqrt(cts_sum[2])]
        # prepare cts_arr for transfer to xml file
        zero_ind = np.where(
            avgbg_sum[0] == 0)  # find zero-values. Attention! These should only be at start and end, not middle
        for scaler in range(3):
            avgbg_sum[scaler] = np.delete(avgbg_sum[scaler], zero_ind)
            cts_sum[scaler] = np.delete(cts_sum[scaler], zero_ind)
            err_sum[scaler] = np.delete(err_sum[scaler], zero_ind)
            sumerr[scaler] = np.delete(sumerr[scaler], zero_ind)
        volt_arr = np.delete(volt_arr, zero_ind)

        # plt.errorbar(volt_arr, cts_sum[0], yerr=sumerr[0], fmt='.')
        # plt.show()
        plt.errorbar(volt_arr, cts_sum[0] / avgbg_sum[0], yerr=sumerr[0] / avgbg_sum[0], fmt='.', label='scaler_0')
        plt.errorbar(volt_arr, cts_sum[1] / avgbg_sum[1], yerr=sumerr[1] / avgbg_sum[1], fmt='.', label='scaler_1')
        plt.errorbar(volt_arr, cts_sum[2] / avgbg_sum[2], yerr=sumerr[2] / avgbg_sum[2], fmt='.', label='scaler_2')
        if self.save_plots_to_file:
            filename = 'added_' + str(iso) + '_files' + str(filelist[0]) + 'to' + str(filelist[-1])
            plt.savefig(self.resultsdir + filename + '.png', bbox_inches="tight")
        else:
            plt.show()
        plt.close()
        plt.clf()

        # prepare cts_arr for transfer to xml file
        total_scale_factor = 0  # This might need to be adjusted when changing the number of scalers involved.
        # replaced scale factor by avgbg_sum.mean()
        for scaler in range(3):
            scale_factor = avgbg_sum[scaler].mean()
            cts_sum[scaler] = (cts_sum[scaler] / avgbg_sum[scaler] * scale_factor).astype(int)
            sumerr[scaler] = (sumerr[scaler] / avgbg_sum[scaler] * scale_factor).astype(int)
            total_scale_factor += (cts_sum[scaler].max() - cts_sum[scaler].min())  # we also need to scale the intensity of the isotope

        self.make_sumXML_file(iso, volt_arr[0], binsize, len(cts_sum[0]), np.array(cts_sum), np.array(sumerr), peakHeight=total_scale_factor)

    def make_sumXML_file(self, isotope, startVolt, stepSizeVolt, nOfSteps, cts_list, err_list=None, peakHeight=1):
        ####################################
        # Prepare dicts for writing to XML #
        ####################################
        iso = isotope[:4]
        if '_cal' in isotope:
            type = '{}_sum_cal'.format(iso)
        else:
            type = '{}_sum'.format(iso)
        file_creation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        header_dict = {'type': 'cs',
                       'isotope': iso,
                       'isotopeStartTime': file_creation_time,
                       'accVolt': 29850,
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
        xml_name = 'Sum{}_9999.xml'.format(iso)
        xml_filepath = os.path.join(self.datafolder, xml_name)
        self.writeXMLfromDict(xml_dict, xml_filepath, 'BecolaData')
        self.ni55analysis_combined_files.append(xml_name)
        # add file to database
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''INSERT OR IGNORE INTO Files (file, filePath, date, type) VALUES (?, ?, ?, ?)''',
                    (xml_name, os.path.relpath(xml_filepath, self.workdir), file_creation_time, type))
        con.commit()
        cur.execute(
            '''UPDATE Files SET offset = ?, accVolt = ?,  laserFreq = ?, laserFreq_d = ?, colDirTrue = ?, 
            voltDivRatio = ?, lineMult = ?, lineOffset = ?, errDateInS = ? WHERE file = ? ''',
            ('[0]', 29850, self.laser_freqs[iso], 0, True, str({'accVolt': 1.0, 'offset': 1.0}), 1, 0, 1,
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
        isopars_lst[11] = int(peakHeight)/bg_estimate*1000 # change intensity scaling
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

    def plot_sum_results(self):
        # Plot Isotope shift 56-58 for all scalers to compare.
        iso_list = ['55Ni_sum_cal', '56Ni_sum_cal', '58Ni_sum_cal', '60Ni_sum_cal']
        scaler_list = ['scaler_0', 'scaler_1', 'scaler_2', 'scaler_012']

        for scaler in scaler_list:
            shift_list = []
            d_stat_list = []
            for iso in iso_list:
                iso_col = self.results[iso]['color']
                shift = self.results[iso][scaler]['shifts_iso-58']['vals']
                shift_d = self.results[iso][scaler]['shifts_iso-58']['d_stat']
                shift_d_syst = self.results[iso][scaler]['shifts_iso-58']['d_syst']
                plt.errorbar([int(iso[:2])], shift, c=iso_col,
                             yerr=self.results[iso][scaler]['shifts_iso-58']['d_stat'], fmt=' o'
                             #, label='{0}: {1:.1f}({2:.0f})[{3:.0f}]MHz'.format(iso, shift[0], 10 * shift_d[0], 10 * shift_d_syst[0])
                             )
                # Plot weighted average and errorband for all scalers
                shift_list.append(shift[0])
                d_stat_list.append(shift_d[0])

            plt.plot([55, 56, 58, 60], shift_list, '--', label = scaler)

        #plt.xticks(range(4), ['55', '56', '58', ], rotation=-30)
        plt.axis([54.5, 60.5, -1000, 600])
        plt.title('Isotope Shift Ni 56-58 for all runs')
        plt.xlabel('Isotope A')
        plt.ylabel('Isotope Shift  [MHz]')
        plt.legend(loc='lower right')
        if self.save_plots_to_file:
            filename = 'shifts_iso-58_summedAll_overview'
            plt.savefig(self.resultsdir + filename + '.png')
        else:
            plt.show()
        plt.close()
        plt.clf()

    def ni55_A_B_analysis(self, isotope):
        res_dict = self.results[isotope]
        # Get all coefficients from fit results
        Al = res_dict['scaler_012']['hfs_pars']['Al'][0]
        Al_d = res_dict['scaler_012']['hfs_pars']['Al'][1]
        A_rat_fixed = res_dict['scaler_012']['hfs_pars']['Au'][2]
        if A_rat_fixed:
            A_rat = res_dict['scaler_012']['hfs_pars']['Au'][0]
            A_rat_d = 0
            Au = Al * A_rat
            Au_d = Al_d * A_rat
        else:
            Au = res_dict['scaler_012']['hfs_pars']['Au'][0]
            Au_d = res_dict['scaler_012']['hfs_pars']['Au'][1]
            A_rat = Au/Al
            A_rat_d = np.sqrt(np.square(Au_d/Al)+np.square(Au*Al_d/Al/Al))
        Bl = res_dict['scaler_012']['hfs_pars']['Bl'][0]
        Bl_d = res_dict['scaler_012']['hfs_pars']['Bl'][1]
        B_rat_fixed = res_dict['scaler_012']['hfs_pars']['Bu'][2]
        if B_rat_fixed:
            B_rat = res_dict['scaler_012']['hfs_pars']['Bu'][0]
            B_rat_d = 0
            Bu = Bl * B_rat
            Bu_d = Bl_d * B_rat
        else:
            Bu = res_dict['scaler_012']['hfs_pars']['Bu'][0]
            Bu_d = res_dict['scaler_012']['hfs_pars']['Bu'][1]
            B_rat = Bu/Bl
            B_rat_d = np.sqrt(np.square(Bu_d/Bl)+np.square(Bu*Bl_d/Bl/Bl))

        # calculate µ and Q values
        # reference moments stored in format: (IsoMass_A, IsoSpin_I, IsoDipMom_µ, IsoDipMomErr_µerr, IsoQuadMom_Q, IsoQuadMomErr_Qerr)
        m_ref, I_ref, mu_ref, mu_ref_d, Q_ref, Q_ref_d = self.nuclear_spin_and_moments['61Ni']
        m_55, I_55, mu_55, mu_55_d, Q_55, Q_55_d = self.nuclear_spin_and_moments['55Ni']
        # reference A and B factors stored in format: (Al, Al_d, Au, Au_d, Arat, Arat_d, Bl, Bl_d, Bu, Bu_d, Brat, Brat_d)
        Al_ref, Al_d_ref, Au_ref, Au_d_ref, Arat_ref, Arat_d_ref, Bl_ref, Bl_d_ref, Bu_ref, Bu_d_ref, Brat_ref, Brat_d_ref = self.reference_A_B_vals['61Ni']
        # differential hyperfine anomaly (assuming the lit_val for 55 is correct, we can extract the hyperfine anomaly)
        hfs_anom_55_61 = - Al / Al_ref * I_55 / I_ref * mu_ref/mu_55 + 1
        hfs_anom_55_61_d = np.sqrt((Al_d / Al_ref*I_55/I_ref*mu_ref/mu_55)**2 +
                                   (Al_d_ref * Al/Al_ref**2 * I_55/I_ref*mu_ref/mu_55)**2 +
                                   (mu_ref_d * Al / Al_ref * I_55 / I_ref /mu_55)**2 +
                                   (mu_55_d * Al / Al_ref * I_55 / I_ref * mu_ref/mu_55**2)**2)
        # magnetic dipole moment
        mu_55 = mu_ref * Al/Al_ref * I_55/I_ref
        mu_55_d = np.sqrt((mu_ref_d * Al/Al_ref*I_55/I_ref)**2 + (Al_d * mu_ref/Al_ref*I_55/I_ref)**2 + (Al_d_ref * mu_ref*Al/Al_ref**2*I_55/I_ref)**2)
        # electric quadrupole moment
        Q_55 = Q_ref * Bl/Bl_ref
        Q_55_d = np.sqrt((Q_ref_d*Bl/Bl_ref)**2 + (Bl_d*Q_ref/Bl_ref)**2 + (Bl_d_ref*Bl*Q_ref/Bl_ref**2)**2)
        logging.info('\nspectroscopic factors: Al={0:.0f}({1:.0f}), Au={2:.0f}({3:.0f}), Arat={4:.3f}({5:.0f}),'
                     ' Bl={6:.0f}({7:.0f}), Bu={8:.0f}({9:.0f}), Brat={10:.3f}({11:.0f})'
                     .format(Al, Al_d, Au, Au_d, A_rat, A_rat_d*1000, Bl, Bl_d, Bu, Bu_d, B_rat, B_rat_d*1000))
        logging.info('\nmu55 = {0:.3f}({1:.0f}), Q55 = {2:.3f}({3:.0f}), hfs_anomaly: {4:.4f}({5:.0f})'
                     .format(mu_55, mu_55_d*1000, Q_55, Q_55_d*1000, hfs_anom_55_61, hfs_anom_55_61_d*10000))

    def ni55_A_B_analysis_all(self, isotope):
        # TODO: comment difference between this and above
        res_dict = self.results[isotope]
        Al = []
        Al_d = []
        A_rat_fixed = False
        Au = []
        Au_d = []
        Bl = []
        Bl_d = []
        B_rat_fixed = False
        Bu = []
        Bu_d = []
        for key, val in res_dict.items():
            if 'scaler' in key:
                # found scaler, use for analysis
                Al.append(val['hfs_pars']['Al'][0])
                Al_d.append(val['hfs_pars']['Al'][1])
                A_rat_fixed = val['hfs_pars']['Au'][2]
                Au.append(val['hfs_pars']['Au'][0])
                Au_d.append(val['hfs_pars']['Au'][1])
                Bl.append(val['hfs_pars']['Bl'][0])
                Bl_d.append(val['hfs_pars']['Bl'][1])
                B_rat_fixed = val['hfs_pars']['Bu'][2]
                Bu.append(val['hfs_pars']['Bu'][0])
                Bu_d.append(val['hfs_pars']['Bu'][1])
        Al_avg = np.sum(np.array(Al)*1/np.square(np.array(Al_d)))/np.sum(1/np.square(np.array(Al_d)))
        Al_avg_d = np.sqrt(np.sum(1/np.square(np.array(Al_d))))
        if A_rat_fixed:
            a_rat = Au[0]
            Au_avg = a_rat * Al_avg
            Au_avg_d = a_rat * Al_avg_d
            Au = a_rat * np.array(Al)
            Au_d = a_rat * np.array(Al_d)
        else:
            Au_avg = np.sum(np.array(Au) * 1 / np.square(np.array(Au_d))) / np.sum(1 / np.square(np.array(Au_d)))
            Au_avg_d = np.sqrt(np.sum(1 / np.square(np.array(Au_d))))
        Bl_avg = np.sum(np.array(Bl) * 1 / np.square(np.array(Bl_d))) / np.sum(1 / np.square(np.array(Bl_d)))
        Bl_avg_d = np.sqrt(np.sum(1 / np.square(np.array(Bl_d))))
        if B_rat_fixed:
            b_rat = Bu[0]
            Bu_avg = b_rat * Bl_avg
            Bu_avg_d = b_rat * Bl_avg_d
            Bu = b_rat * np.array(Bl)
            Bu_d = b_rat * np.array(Bl_d)
        else:
            Bu_avg = np.sum(np.array(Bu) * 1 / np.square(np.array(Bu_d))) / np.sum(1 / np.square(np.array(Bu_d)))
            Bu_avg_d = np.sqrt(np.sum(1 / np.square(np.array(Bu_d))))

        # plot results A lower
        plt.title('A lower')
        plt.errorbar(range(len(Al)), Al, yerr=Al_d)
        plt.plot([-1, len(Al)], [Al_avg, Al_avg], c='b')
        plt.fill([-1, len(Al), len(Al), -1],  # statistical error
                 [Al_avg - Al_avg_d, Al_avg - Al_avg_d, Al_avg + Al_avg_d, Al_avg + Al_avg_d], 'b', alpha=0.2)
        plt.show()
        # plot results A upper
        plt.title('A upper')
        plt.errorbar(range(len(Au)), Au, yerr=Au_d, c='r')
        plt.plot([-1, len(Au)], [Au_avg, Au_avg], c='r')
        plt.fill([-1, len(Au), len(Au), -1],  # statistical error
                 [Au_avg - Au_avg_d, Au_avg - Au_avg_d, Au_avg + Au_avg_d, Au_avg + Au_avg_d], 'r', alpha=0.2)
        plt.show()
        # plot results B lower
        plt.title('B lower')
        plt.errorbar(range(len(Bl)), Bl, yerr=Bl_d)
        plt.plot([-1, len(Bl)], [Bl_avg, Bl_avg], c='b')
        plt.fill([-1, len(Bl), len(Al), -1],  # statistical error
                 [Bl_avg - Bl_avg_d, Bl_avg - Bl_avg_d, Bl_avg + Bl_avg_d, Bl_avg + Bl_avg_d], 'b', alpha=0.2)
        plt.show()
        # plot results B upper
        plt.title('B upper')
        plt.errorbar(range(len(Bu)), Bu, yerr=Bu_d, c='r')
        plt.plot([-1, len(Bu)], [Bu_avg, Bu_avg], c='r')
        plt.fill([-1, len(Bu), len(Bu), -1],  # statistical error
                 [Bu_avg - Bu_avg_d, Bu_avg - Bu_avg_d, Bu_avg + Bu_avg_d, Bu_avg + Bu_avg_d], 'r', alpha=0.2)
        plt.show()


    ''' King Fit Related '''

    def perform_king_fit_analysis(self, delta_lit_radii, isotopes, reference_run):
        # raise (Exception('stopping before king fit'))
        # delta_lit_radii.pop('62_Ni')  # just to see which point is what
        king = KingFitter(self.db, showing=True, litvals=delta_lit_radii, plot_y_mhz=False, font_size=18, ref_run=reference_run)
        # run = 'narrow_gate_asym'
        # isotopes = sorted(delta_lit_radii.keys())
        # print(isotopes)
        # king.kingFit(alpha=0, findBestAlpha=False, run=run, find_slope_with_statistical_error=True)
        king.kingFit(alpha=0, findBestAlpha=False, run=reference_run, find_slope_with_statistical_error=False)
        king.calcChargeRadii(isotopes=isotopes, run=reference_run, plot_evens_seperate=False, dash_missing_data=True)

        # king.kingFit(alpha=378, findBestAlpha=True, run=run, find_slope_with_statistical_error=True)
        king.kingFit(alpha=361, findBestAlpha=True, run=reference_run)
        radii_alpha = king.calcChargeRadii(isotopes=isotopes, run=reference_run, plot_evens_seperate=False, dash_missing_data=True)
        print('radii with alpha', radii_alpha)
        # king.calcChargeRadii(isotopes=isotopes, run=run)

        # TODO: Add to results

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
        M_alpha, M_alpha_d = self.literature_massshift
        F, F_d = self.literature_fieldshift
        alpha = self.literature_alpha

        # get data and calculate radii
        delta_rms = []
        delta_rms_d = []
        if scaler is not None:
            # get per file isoshift
            files = self.results[iso]['file_names']
            iso_shift = self.results[iso][scaler]['shifts_iso-{}'.format(ref[:2])]['vals']
            iso_shift_d = self.results[iso][scaler]['shifts_iso-{}'.format(ref[:2])]['d_stat']
            iso_shift_d_syst = self.results[iso][scaler]['shifts_iso-{}'.format(ref[:2])]['d_syst']
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

    def make_results_dict_scaler(self,
                                 centers, centers_d_fit, centers_d_stat, center_d_syst, fitpars, rChi, hfs_pars=None):
        # calculate weighted average of center parameter
        weights = 1 / np.array(centers_d_stat) ** 2
        wavg, sumw = np.average(np.array(centers), weights=weights, returned=True)
        wavg_d = np.sqrt(1 / sumw)
        # also calculate std deviation
        st_dev = np.array(centers).std()

        ret_dict = {'center_fits':
                        {'vals': centers,
                         'd_fit': centers_d_fit,
                         'd_stat': centers_d_stat,
                         'd_syst': center_d_syst
                         },
                    'avg_center_fits':
                        {'vals': [wavg],
                         'd_fit': [wavg_d],
                         'd_stat': [max(wavg_d, st_dev)],
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
                                           digits=1 , factor=1, shiftavg=True, plotstyle='band'):
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
            scaler_nums.append(scaler.split('_')[1])
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
                        weights = 1/centers_d_stat**2
                        wavg, sumw = np.average(centers, weights=weights, returned=True)
                        wavg_d = '{:.0f}'.format(10**digits*np.sqrt(1 / sumw))  # times 10 for representation in brackets
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
                if 'shifts_iso-' in parameter and shiftavg:
                    avg_parameter = 'avg_shift_iso-{}'.format(parameter[-2:])
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
            filename = parameter + '_' + ''.join(isonums) + summed + '_sc' + 'a'.join(scaler_nums)
            plt.savefig(self.resultsdir + filename + '.png', bbox_inches="tight")
        else:
            plt.show()
        plt.close()
        plt.clf()

    def plot_parameter_for_isos_vs_scaler(self, isotopes, scaler_list, parameter,
                                           offset=None, overlay=None, unit='MHz', onlyfiterrs=False,
                                          digits=1, factor=1, shiftavg=True):
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
                scaler_nums.append(scaler.split('_')[1])
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
                weights = 1/centers_d_stat**2
                wavg, sumw = np.average(centers, weights=weights, returned=True)
                wavg_d = '{:.0f}'.format(10**digits*np.sqrt(1 / sumw))  # times 10 for representation in brackets
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
            filename = parameter + '_' + ''.join(isonums) + summed + '_sc' + 'a'.join(scaler_nums)
            plt.savefig(self.resultsdir + filename + '.png', bbox_inches="tight")
        else:
            plt.show()
        plt.close()
        plt.clf()

    def plot_radii(self, isolist, refiso, scaler, plot_evens_seperate=False, dash_missing_data=True, calibrated=False,
                   includelitvals=True):
        if calibrated:
            isolist = ['{}_cal'.format(i) for i in isolist]  # make sure to use the calibrated isos
        font_size = 12
        ref_key = refiso[:4]
        rms_key = 'avg_delta_rms_iso-{}'.format(refiso[:2])
        thisVals = {key: [self.results[key][scaler][rms_key]['vals'][0],
                           self.results[key][scaler][rms_key]['d_syst'][0]]
                     for key in isolist}
        thisVals[refiso] = [0, 0]
        col = ['r', 'b', 'k', 'g']

        dictlist = [thisVals]
        srclist = ['This Work']
        if includelitvals:
            for src, isos in self.delta_rms_lit.items():
                litvals = TiTs.deepcopy(isos)
                ref_val = self.delta_rms_lit[src][ref_key]
                srclist.append(src)
                for iso, vals in litvals.items():
                    litvals[iso] = (vals[0]-ref_val[0], vals[1])
                dictlist.append(litvals)
            # TODO: How to deal with different refisos here?
        for num, finalVals in enumerate(dictlist):
            keyVals = sorted(finalVals)
            x = []
            y = []
            yerr = []
            print('iso\t $\delta$ <r$^2$>[fm$^2$]')
            for i in keyVals:
                x.append(int(''.join(filter(str.isdigit, i))))
                y.append(finalVals[i][0])
                yerr.append(finalVals[i][1])
                print('%s\t%.3f(%.0f)\t%.1f' % (i, finalVals[i][0], finalVals[i][1] * 1000,
                                                finalVals[i][1] / max(abs(finalVals[i][0]), 0.000000000000001) * 100))

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

                plt.errorbar(x_even, y_even, y_even_err, fmt='{}o'.format(col[num]), label='even', linestyle='-')
                plt.errorbar(x_odd, y_odd, y_odd_err, fmt='{}^'.format(col[num]), label='odd', linestyle='--')
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
                        plt.errorbar(each, y_vals, yerr_vals, fmt='{}o'.format(col[num]), linestyle='-', label=srclist[num])
                        label_created=True
                    else:
                        plt.errorbar(each, y_vals, yerr_vals, fmt='{}o'.format(col[num]), linestyle='-')
                    # plot dashed lines between missing values
                    if len(x) > i + len(each):  # might be last value
                        x_gap = [x[i + len(each) - 1], x[i + len(each)]]
                        y_gap = [y[i + len(each) - 1], y[i + len(each)]]
                        plt.plot(x_gap, y_gap, c='{}'.format(col[num]), linestyle='--')
                    i = i + len(each)
            else:
                plt.errorbar(x, y, yerr, fmt='{}o'.format(col[num]), linestyle='-')
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
                        data_dict = self.results['{}{}'.format(iso, vari)]['scaler_c012'][property]
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
    analysis.combine_single_scaler_centers(['56Ni', '58Ni', '60Ni'])
    analysis.plot_results_of_fit()
    analysis.extract_isoshifts_from_fitres(['56Ni', '60Ni'], refiso='58Ni')
    analysis.extract_isoshifts_from_fitres(['56Ni', '58Ni'], refiso='60Ni')
    analysis.calculate_charge_radii(['56Ni', '60Ni'], refiso='58Ni')
    analysis.calculate_charge_radii(['56Ni', '58Ni'], refiso='60Ni')
    analysis.plot_radii(['56Ni', '58Ni'], refiso='60Ni', scaler='scaler_c012')
    analysis.ion_energy_calibration()
    # fitting with calibrated ion energies for final result extraction
    analysis.fitting_calibrated_separate()
    analysis.combine_single_scaler_centers(['56Ni', '58Ni', '60Ni'], calibrated=True)
    analysis.plot_results_of_fit(calibrated=True)
    analysis.extract_isoshifts_from_fitres(['56Ni', '60Ni'], refiso='58Ni', calibrated=True)
    analysis.extract_isoshifts_from_fitres(['56Ni', '58Ni'], refiso='60Ni', calibrated=True)
    analysis.calculate_charge_radii(['56Ni', '60Ni'], refiso='58Ni', calibrated=True)
    analysis.calculate_charge_radii(['56Ni', '58Ni'], refiso='60Ni', calibrated=True)
    # stacked run analysis for inclusion of nickel 55
    analysis.create_and_fit_stacked_runs(calibrated=True)
    analysis.combine_single_scaler_centers(['55Ni_sum', '56Ni_sum', '58Ni_sum', '60Ni_sum'], calibrated=True)
    analysis.extract_isoshifts_from_fitres(['55Ni_sum', '56Ni_sum', '60Ni_sum'], refiso='58Ni_sum', calibrated=True)
    analysis.extract_isoshifts_from_fitres(['55Ni_sum', '56Ni_sum', '58Ni_sum'], refiso='60Ni_sum', calibrated=True)
    analysis.calculate_charge_radii(['55Ni_sum', '56Ni_sum', '60Ni_sum'], refiso='58Ni_sum', calibrated=True)
    analysis.calculate_charge_radii(['55Ni_sum', '56Ni_sum', '58Ni_sum'], refiso='60Ni_sum', calibrated=True)
    analysis.plot_radii(['55Ni_sum', '56Ni_sum', '60Ni_sum'], refiso='58Ni_sum', scaler='scaler_012', calibrated=True)
    analysis.plot_radii(['55Ni_sum', '56Ni_sum', '58Ni_sum'], refiso='60Ni_sum', scaler='scaler_012', calibrated=True)
    analysis.export_results()
    analysis.plot_results_table(ref_iso='60Ni')
