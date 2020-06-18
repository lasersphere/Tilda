"""
Created on 2020-06-02

@author: fsommer

Module Description:
Analysis of the Nickel Data from BECOLA taken on 13.04.-23.04.2018.
Special script to compare results with different lineshapes.
Results must be generated with main analysis skript before and then imported here from results.xml
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

class NiAnalysis_lineshapes():
    def __init__(self):
        logging.getLogger().setLevel(logging.INFO)
        # Name this analysis run
        self.run_name = 'LineshapeAnalysis'

        # Set working directory and database
        ''' working directory: '''
        # get user folder to access ownCloud
        user_home_folder = os.path.expanduser("~")
        # self.workdir = 'C:\\DEVEL\\Analysis\\Ni_Analysis\\XML_Data' # old working directory
        ownCould_path = 'ownCloud\\User\\Felix\\Measurements\\Nickel_online_Becola\\Analysis\\XML_Data'
        # ownCould_path = 'ownCloud\\User\\Felix\\Measurements\\Nickel_offline_Becola20\\XML_Data'  # offline 2020
        self.workdir = os.path.join(user_home_folder, ownCould_path)
        ''' data folder '''
        self.datafolder = os.path.join(self.workdir, 'SumsRebinned')
        ''' results folder'''
        analysis_start_time = datetime.now()
        self.results_name = self.run_name + '_' + analysis_start_time.strftime("%Y-%m-%d_%H-%M")
        results_path_ext = 'results\\' + self.results_name + '\\'
        self.resultsdir = os.path.join(self.workdir, results_path_ext)
        os.mkdir(self.resultsdir)
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
        self.run = 'Voigt'
        self.timebin_size = 4.8  # length of timegate in 10ns (4.8 = 48ns)

        # fit from scratch or use FitRes db?
        self.do_the_fitting = True  # if False, a .xml file has to be specified in the next variable!
        self.load_results_from = 'SoftwareGateAnalysis_2020-05-28_16-48.xml'  # load fit results from this file
        # print results to results folder? Also show plots?
        self.save_plots_to_file = True  # if False plots will be displayed during the run for closer inspection
        # acceleration set voltage (Buncher potential), negative
        self.accVolt_set = 29850  # omit voltage sign, assumed to be negative
        self.calibration_method = 'absolute'  # can be 'absolute', 'relative' 'combined', 'isoshift' or 'None'
        self.use_handassigned = False  # use hand-assigned calibrations? If false will interpolate on time axis
        self.accVolt_corrected = (self.accVolt_set, 0)  # Used later for calibration. Might be used her to predefine calib? (abs_volt, err)
        self.initial_par_guess = {'sigma': (31.4, False), 'gamma': (18.4, False),
                                  'asy': (3.9, True),  # in case VoigtAsy is used
                                  'dispersive': (-0.04, False),  # in case FanoVoigt is used
                                  'centerAsym': (-6.4, True), 'nPeaksAsym': (3, True), 'IntAsym': (0.163, False)
                                  # in case AsymmetricVoigt is used
                                  }
        self.isotope_colors = {58: 'black', 60: 'blue', 56: 'green', 55: 'purple'}
        self.scaler_colors = {'scaler_0': 'blue', 'scaler_1': 'green', 'scaler_2': 'red',
                              'scaler_012': 'black', 'scaler_c012': 'pink',
                              'Voigt': 'blue',  # for this script only!
                              'VoigtAsy': 'green',
                              'AsymmetricVoigt': 'red',
                              'FanoVoigt': 'purple'
                              }

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
                                    'first_fit': 'from scratch' if self.do_the_fitting else self.load_results_from,
                                    'calibration': self.calibration_method,
                                    'use_handassigned': self.use_handassigned,
                                    'initial_par_guess': self.initial_par_guess
                                    }

        self.init_uncertainties_exp_physics()
        self.init_stuff()

    def init_uncertainties_exp_physics(self):
        """
                ### Uncertainties ###
                All uncertainties that we can quantify and might want to respect
                """
        self.accVolt_set_d = 10  # V. uncertainty of scan volt. Estimated by Miller for Calcium meas.
        self.wavemeter_wsu30_mhz_d = 3 * 2  # MHz. Kristians wavemeter paper. Factor 2 because of frequency doubling.
        self.matsuada_volts_d = 0.03  # V. ~standard dev after rebinning TODO: calc real avg over std dev?
        self.lineshape_d_syst = 0.7  # MHz. TODO: conservative estimate. Double check
        self.bunch_structure_d = 0.2  # MHz. Slope difference between 58&56 VoigtAsy allfix: 20kHz/bin, +-5bin width --> 200kHz TODO: check whether this is good standard value (also for summed)
        self.heliumneon_drift = 0  # TODO: does that influence our measurements? (1 MHz in Kristians calibration)
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
        # TODO: Add literature factors from COLLAPS to skip Kingplot
        self.literature_massshift = (948000, 3000)  # Mhz u (lit val given in GHz u)(949000, 4000)
        self.literature_fieldshift = (-769, 60)  # MHz/fm^2(-788, 82)
        self.literature_alpha = 396  # u fm^2 397

        ''' literature radii '''
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
        # self.adjust_center_ests_db()

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
                                ['val']
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
        # cur.execute(
        #     '''DELETE FROM Files WHERE file LIKE ? ''', ('Sum%',))  # Also delete sum files from last run!
        con.commit()  # commit changes to db
        con.close()  # close db connection

    ''' analysis '''

    def lineshape_analysis(self):
        """
        Load Results for Voigt, VoigtAsy and AsymmetricVoigt Lineshapes
        :return:
        """
        # reset isotope type and acc voltage in db
        self.lineshape_res = {'Voigt': {},
                              'VoigtAsy': {},
                              'AsymmetricVoigt': {},
                              'FanoVoigt': {}
                              }
        # set results files
        self.lineshape_res['Voigt']['xml_file'] = '2020reworked_2020-06-03_17-14_allfix_Voigt.xml'
        self.lineshape_res['VoigtAsy']['xml_file'] = '2020reworked_2020-06-03_18-37_asy_VoigtAsy.xml'
        self.lineshape_res['AsymmetricVoigt']['xml_file'] = '2020reworked_2020-06-03_18-02_allfix_AsymmetricVoigt.xml'
        self.lineshape_res['FanoVoigt']['xml_file'] = '2020reworked_2020-06-03_17-54_allfix_FanoVoigt.xml'
        # import results
        for shape, dict in self.lineshape_res.items():
            res_file = dict['xml_file']
            self.results = self.import_results(res_file)
            TiTs.merge_extend_dicts(dict, self.results)

        # self.results will be re-organized for this analysis with 'scaler'-keys replaced by 'lineshape'
        # this should allow to re-use the established functions for plotting etc...
        for iso in ['56Ni', '58Ni', '60Ni']:
            for lineshape, dict in self.lineshape_res.items():
                self.results[iso][lineshape] = {}
            self.compare_center_freqs(iso, 'center_fits')
            self.compare_center_freqs(iso, 'rChi')
        for iso in ['56Ni', '60Ni']:
            self.compare_center_freqs(iso, 'shifts_iso-58')

    ''' db related '''
    def update_scalers_in_db(self, scalers):
        '''
        Update the scaler parameter for all runs in the runs database
        :param scalers: int or str: either an int (0,1,2) if a single scaler is used or a string '0,1,2' for all
        :return: scaler name string
        '''
        if 'scaler_' in str(scalers):
            scaler_db_string = ','.join(list(scalers.split('_')[-1])).join(('[', ']'))
            scaler_db_string.replace('c', '')  # remove c if scaler_c012
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

    def adjust_center_ests_db(self):
        """
        Write new center fit estimations for the standard isotopes into the db. use self.accVolt_set as reference
        :return:
        """
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        stand_ests = {'55Ni': -900, '56Ni': -500, '58Ni': -0, '60Ni': 500}  # values that worked fine for 29850V
        ref_freq_dev = 850343673 - self.restframe_trans_freq  # stand_ests are for 580343800MHz. Ref freq might be updated
        for iso, mass_tupl in stand_ests.items():
            cur.execute('''UPDATE Isotopes SET center = ? WHERE iso = ? ''',
                        (stand_ests[iso] + ref_freq_dev + (29850 - self.accVolt_set) * self.diff_dopplers[iso], iso))
        con.commit()
        con.close()

    ''' analysis related '''
    def compare_center_freqs(self, iso, prop):
        for lineshape, dict in self.lineshape_res.items():
            self.results[iso][lineshape][prop] = {'vals': [], 'd_fit': []}
            self.results[iso][lineshape]['{}_dev'.format(prop)] = {'vals': [], 'd_fit': []}
        wavg = []
        wavg_d = []
        for indx, file in enumerate(self.lineshape_res['Voigt'][iso]['file_numbers']):
            # get center ests for all lineshapes
            centers = []
            center_fit_d = []
            for lineshape, dict in self.lineshape_res.items():
                c = dict[iso][self.scaler_name][prop]['vals'][indx]
                d_arr = dict[iso][self.scaler_name][prop].get('d_fit', None)
                if d_arr is not None:
                    d = d_arr[indx]
                else:
                    # try to get the stat errors instead
                    d_arr = dict[iso][self.scaler_name][prop].get('d_stat', None)
                    if d_arr is not None:
                        d = d_arr[indx]
                    else:
                        d = 0
                centers = np.append(centers, c)
                center_fit_d = np.append(center_fit_d, d)
            # try to get a weighted average
            if not np.any(center_fit_d == 0) and not np.sum(1 / center_fit_d ** 2) == 0:
                weights =1/np.square(center_fit_d)
                wavg.append(np.average(centers, weights=weights))
                wavg_d.append(np.sqrt(1/np.sum(1/np.square(center_fit_d))))
            else:  # some values don't have error, just calculate mean instead of weighted avg
                wavg.append(centers.mean())
                wavg_d.append(0)
            # write deviation from avg in results
            for lineshape, dict in self.lineshape_res.items():
                c = dict[iso][self.scaler_name][prop]['vals'][indx]
                d_arr = dict[iso][self.scaler_name][prop].get('d_fit', None)
                if d_arr is not None:
                    d = d_arr[indx]
                else:
                    # try to get the stat errors instead
                    d_arr = dict[iso][self.scaler_name][prop].get('d_stat', None)
                    if d_arr is not None:
                        d = d_arr[indx]
                    else:
                        d = 0
                self.results[iso][lineshape][prop]['vals'].append(c)
                self.results[iso][lineshape][prop]['d_fit'].append(d)
                self.results[iso][lineshape]['{}_dev'.format(prop)]['vals'].append(c-wavg[-1])
                self.results[iso][lineshape]['{}_dev'.format(prop)]['d_fit'].append(d)
        self.plot_parameter_for_isos_and_scaler([iso], self.lineshape_res.keys(), prop, onlyfiterrs=True, shiftavg=False)
        self.plot_parameter_for_isos_and_scaler([iso], self.lineshape_res.keys(), '{}_dev'.format(prop), onlyfiterrs=True, overlay=0)

    def sum_file_comparison(self):
        """ the summed up data of all files might be a good candidate to compare the different lineshapes """
        self.adjust_center_ests_db()

        runlist = ['Voigt', 'VoigtAsy', 'FanoVoigt', 'AsymmetricVoigt']
        self.initial_par_guess = {'sigma': (31.4, False), 'gamma': (18.4, False),
                                  'asy': (3.9, False),  # in case VoigtAsy is used
                                  'dispersive': (-0.04, False),  # in case FanoVoigt is used
                                  'centerAsym': (-6.4, True), 'nPeaksAsym': (1, True), 'IntAsym': (0.163, False)
                                  # in case AsymmetricVoigt is used
                                  }
        self.analysis_parameters = {'run': runlist,
                                    'first_fit': 'from scratch' if self.do_the_fitting else self.load_results_from,
                                    'calibration': self.calibration_method,
                                    'use_handassigned': self.use_handassigned,
                                    'initial_par_guess': self.initial_par_guess
                                    }

        sumfiles = ['Sum55Ni_9999.xml', 'Sum56Ni_9999.xml', 'Sum58Ni_9999.xml', 'Sum60Ni_9999.xml']

        for runs in runlist:
            self.write_to_db_lines(runs,
                                   sigma=self.initial_par_guess['sigma'],
                                   gamma=self.initial_par_guess['gamma'],
                                   asy=self.initial_par_guess['asy'],
                                   dispersive=self.initial_par_guess['dispersive'],
                                   centerAsym=self.initial_par_guess['centerAsym'],
                                   IntAsym=self.initial_par_guess['IntAsym'],
                                   nPeaksAsym=self.initial_par_guess['nPeaksAsym'])

            isolist, center_MHz, center_MHz_fiterrs, center_MHz_d, center_MHz_d_syst, fitpars, rChi = \
                self.fitRunsFromList(runs, sumfiles)

            for indx, iso in enumerate(isolist):
                iso_dict = {iso: {runs: {'center_fits': {'vals': [center_MHz[indx]],
                                                            'd_fit': [center_MHz_fiterrs[indx]],
                                                            'd_stat': [center_MHz_d[indx]],
                                                            'd_syst': [center_MHz_d_syst[indx]]},
                                         'all_fitpars': [fitpars[indx]],
                                         'rChi': {'vals': [rChi[indx]]}},
                                  'color': self.isotope_colors[int(iso[:2])]}}
                TiTs.merge_extend_dicts(self.results, iso_dict)

        # for iso, isodict in self.results.items():
        #     voigtAsy_rChi = isodict['VoigtAsy']['rChi']['vals'][0]
        #     for runs, rundicts in isodict.items():
        #         if type(rundicts) is dict:
        #             self.results[iso][runs]['rChi']['vals'] = [rundicts['rChi']['vals'][0]/voigtAsy_rChi]

        for keys, dicts in self.results.items():
            self.plot_parameter_for_isos_vs_scaler([keys], runlist, 'center_fits', onlyfiterrs=True, stddev=True)
        self.plot_parameter_for_isos_vs_scaler([k for k in self.results.keys()], runlist, 'rChi', onlyfiterrs=True, overlay=1)
        self.plot_parameter_for_isos_vs_scaler([k for k in self.results.keys()], runlist, 'all_fitpars:sigma', onlyfiterrs=True)
        self.plot_parameter_for_isos_vs_scaler([k for k in self.results.keys()], runlist, 'all_fitpars:gamma', onlyfiterrs=True)
        self.plot_parameter_for_isos_vs_scaler([k for k in self.results.keys()], ['AsymmetricVoigt'], 'all_fitpars:centerAsym', onlyfiterrs=True)
        self.plot_parameter_for_isos_vs_scaler([k for k in self.results.keys()], ['AsymmetricVoigt'], 'all_fitpars:IntAsym', onlyfiterrs=True, digits=3)
        self.plot_parameter_for_isos_vs_scaler([k for k in self.results.keys()], ['VoigtAsy'], 'all_fitpars:asy', onlyfiterrs=True)
        # self.plot_parameter_for_isos_vs_scaler([k for k in self.results.keys()], ['AsymmetricVoigt'], 'rChi', onlyfiterrs=True, overlay=1)

    def fitRunsFromList(self, lineshape, filelist):
            '''

            '''
            # select files
            filearray = np.array(filelist)  # needed for batch fitting

            # do the batchfit
            if self.do_the_fitting:
                # for softw_gates_trs from file use 'File' and from db use None.
                BatchFit.batchFit(filearray, self.db, lineshape, x_as_voltage=True, softw_gates_trs=None,
                                  guess_offset=True, save_file_as='.png')
            # get fitresults (center) vs run for 58
            all_rundate = []
            all_iso = []
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
                    '''SELECT pars FROM FitRes WHERE file = ? AND iso = ? AND run = ?''',
                    (files, file_type, lineshape))
                pars = cur.fetchall()
                # Query rChi from fitRes
                cur.execute(
                    '''SELECT rChi FROM FitRes WHERE file = ? AND iso = ? AND run = ?''',
                    (files, file_type, lineshape))
                rChi = cur.fetchall()
                con.close()
                iso = file_type
                all_iso.append(iso)
                try:
                    # if the fit went wrong there might not be a value to get from the fitpars...
                    parsdict = ast.literal_eval(pars[0][0])
                except Exception as e:
                    # Fit went wrong!
                    # replace with standard value and big error...
                    parsdict = {
                        'center': (-510, 30, False)}  # TODO: use better dummy value (take from all_Center_MHz list)
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
                gatewidth_std = self.bunch_structure_d
                # statistic uncertainty
                d_ion_energy_stat = 0
                d_stat = np.sqrt(d_fit ** 2 + gatewidth_std ** 2 + d_ion_energy_stat ** 2)
                all_center_MHz_d.append(d_stat)

                # == systematic uncertainties (same for all files):
                d_ion_energy_syst = self.diff_dopplers[iso[:4]] * (  # TODO: remove buncher uncertainty here?
                            self.accVolt_set_d + self.matsuada_volts_d)  # not statistic
                d_laser_syst = np.sqrt(self.wavemeter_wsu30_mhz_d ** 2 + self.heliumneon_drift ** 2)
                d_alignment = self.laserionoverlap_MHz_d
                d_fitting_syst = self.lineshape_d_syst  # self.bunch_structure_d replaced by gate analysis statistic
                # combine all above quadratically
                d_syst = np.sqrt(
                    d_ion_energy_syst ** 2 + d_laser_syst ** 2 + d_alignment ** 2 + d_fitting_syst ** 2)
                all_center_MHz_d_syst.append(d_syst)

            return all_iso, all_center_MHz, all_center_MHz_fiterrs, all_center_MHz_d, all_center_MHz_d_syst, all_fitpars, all_rChi

    ''' visualization '''
    def plot_parameter_for_isos_and_scaler(self, isotopes, scaler_list, parameter,
                                           offset=None, overlay=None, unit='MHz', onlyfiterrs=False, factor=1, shiftavg=True):
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
        x_type = 'file_numbers'  # alternative: 'file_numbers', 'file_times'
        scaler_nums = []
        for sc in scaler_list:
            # scaler = self.update_scalers_in_db(sc)
            # scaler_nums.append(scaler.split('_')[1])
            scaler = sc  # changed only for this script
            scaler_nums = scaler
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
                    wavg, wavg_d, fixed = factor*np.array(self.results[iso][scaler]['avg_fitpars'][parameter_plot])
                    if fixed == True:
                        wavg_d = '-'
                    else:
                        wavg_d = '{:.0f}'.format(10 * wavg_d)  # times 10 for representation in brackets
                else:
                    centers = factor*np.array(self.results[iso][scaler][parameter]['vals'])
                    if onlyfiterrs:
                        centers_d_stat = factor*np.array(self.results[iso][scaler][parameter]['d_fit'])
                        centers_d_syst = 0
                    else:
                        centers_d_stat = factor*np.array(self.results[iso][scaler][parameter]['d_stat'])
                        centers_d_syst = factor*np.array(self.results[iso][scaler][parameter]['d_syst'])
                    # calculate weighted average:
                    if not np.any(centers_d_stat == 0) and not np.sum(1/centers_d_stat**2) == 0:
                        weights = 1/centers_d_stat**2
                        wavg, sumw = np.average(centers, weights=weights, returned=True)
                        wavg_d = '{:.0f}'.format(10*np.sqrt(1 / sumw))  # times 10 for representation in brackets
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
                    avg_shift = self.results[iso][scaler]['avg_shift_iso-58']
                    off = round(avg_shift, -1)
                elif type(offset) in (list, tuple):
                    # offset might be given manually per isotope
                    off = offset[i]
                # plot center frequencies in MHz:
                if off != 0:
                    plt_label = '{0} {1:.1f}({2}){3} (offset: {4}{5})'\
                        .format(labelstr, wavg, wavg_d,  unit, off, unit)
                else:
                    plt_label = '{0} {1:.1f}({2}){3}'\
                        .format(labelstr, wavg, wavg_d, unit)
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
                if parameter == 'shifts_iso-58' and shiftavg:
                    # also plot average isotope shift
                    avg_shift = self.results[iso][scaler]['avg_shift_iso-58']['val']
                    avg_shift_d = self.results[iso][scaler]['avg_shift_iso-58']['d_stat']
                    avg_shift_d_syst = self.results[iso][scaler]['avg_shift_iso-58']['d_syst']
                    # plot weighted average as red line
                    plt.plot([x_ax[0], x_ax[-1]], [avg_shift, avg_shift], 'r',
                             label='{0} avg: {1:.1f}({2:.0f})[{3:.0f}]{4}'
                             .format(iso, avg_shift, 10*avg_shift_d, 10*avg_shift_d_syst, unit))
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
        plt.title('{} in {} for isotopes: {}'.format(parameter, unit, isotopes))
        plt.ylabel('{} [{}]'.format(parameter, unit))
        plt.legend(loc='best')
        if x_type == 'file_times':
            plt.xlabel('date')
            days_fmt = mpdate.DateFormatter('%d.%B')
            ax.xaxis.set_major_formatter(days_fmt)
        else:
            plt.xlabel('run numbers')
        plt.xticks(rotation=45)
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        plt.margins(0.05)
        if self.save_plots_to_file:
            isonums = []
            for isos in isotopes:
                if 'cal' in isos:
                    isonums.append(isos[:2]+'c')
                else:
                    isonums.append(isos[:2])
            parameter = parameter.replace(':', '_')  # colon is no good filename char
            filename = parameter + '_' + ''.join(isonums) + '_sc' + 'a'.join(scaler_nums)
            plt.savefig(self.resultsdir + filename + '.png', bbox_inches="tight")
        else:
            plt.show()
        plt.close()
        plt.clf()

    def plot_parameter_for_isos_vs_scaler(self, isotopes, scaler_list, parameter,
                                           offset=None, overlay=None, unit='MHz', onlyfiterrs=False,
                                          digits=1, factor=1, stddev=False):
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
                scaler = sc
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
                    centers[indx] = factor*np.array(self.results[iso][scaler][parameter]['vals'][0])
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
            if stddev:
                plt_label += '\nstd: {}'.format(np.array(centers).std())
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
        plt.xticks(x_ax, scaler_list, rotation=45)
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

    ''' results related '''

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
            if vals.get('file_times', None) is not None:
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


if __name__ == '__main__':
    analysis = NiAnalysis_lineshapes()
    analysis.sum_file_comparison()
    analysis.export_results()