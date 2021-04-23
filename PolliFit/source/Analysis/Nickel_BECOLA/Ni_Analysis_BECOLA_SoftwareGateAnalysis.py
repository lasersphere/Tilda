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

# This import registers the 3D projection, but is otherwise unused. Don't remove it!
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from scipy.optimize import curve_fit
from scipy import special

import BatchFit
import Physics
import Tools
import TildaTools as TiTs
from lxml import etree as ET
from XmlOperations import xmlWriteDict
from Measurement.XMLImporter import XMLImporter
from Analysis.Nickel_BECOLA.FileConversion import DictToXML

class NiAnalysis_softwGates():
    def __init__(self):
        logging.getLogger().setLevel(logging.INFO)
        # Name this analysis run
        self.run_name = 'SoftwareGateAnalysis'

        # Set working directory and database
        ''' working directory: '''
        # get user folder to access ownCloud
        user_home_folder = os.path.expanduser("~")
        # self.workdir = 'C:\\DEVEL\\Analysis\\Ni_Analysis\\XML_Data' # old working directory
        ownCould_path = 'ownCloud\\User\\Felix\\Measurements\\Nickel_online_Becola\\Analysis\\XML_Data'  # online 2018 data
        # ownCould_path = 'C:\\Users\\somme\\ownCloud\\User\\Felix\\Measurements\\Nickel_online_Becola\\Analysis\\tof_analysis'  # offline 2020
        # ownCould_path = 'ownCloud\\User\\Felix\\Measurements\\Nickel54_online_Becola20\\Analysis\\XML_Data'  # online 2020
        # ownCould_path = 'ownCloud\\User\\Felix\\Measurements\\Nickel54_postbeamtime_Becola20\\Analysis\\bunched'  # post beamtime 2020
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
        # Isotopes
        self.isotopes_stable = ['58Ni', '60Ni', '61Ni', '62Ni', '64Ni']

        # Select runs; Format: ['run58', 'run60', 'run56']
        # to use a different lineshape you must create a new run under runs and a new linevar under lines and link the two.
        self.run = 'AsymmetricVoigt'
        self.tof_mid = {'54Ni': 5.19, '55Ni': 5.23, '56Ni': 5.28, '58Ni': 5.36, '60Ni': 5.47, '62Ni': 5.59, '64Ni': 5.68}  # mid-tof for each isotope (from fitting) 38, 47
        self.tof_mid = {'54Ni': 5.21, '55Ni': 5.23, '56Ni': 5.28, '58Ni': 5.36, '60Ni': 5.47, '62Ni': 5.57, '64Ni': 5.66}  # mid-tof for each isotope (from fitting) 38, 47
        self.tof_delay = {'54Ni': [0, 0.184, 0.254], '55Ni': [0, 0.186, 0.257], '56Ni': [0, 0.189, 0.260], '58Ni': [0, 0.194, 0.267],
                          '60Ni': [0, 0.199, 0.273], '62Ni': [0, 0.204, 0.279], '64Ni': [0, 0.209, 0.285]}
        self.tof_sigma = {'54Ni': 0.037, '55Ni': 0.04, '56Ni': 0.10, '58Ni': 0.089, '60Ni': 0.064, '62Ni': 0.053, '64Ni': 0.049}  # 1 sigma of the tof-peaks from fitting, avg over all scalers 56,58,60 Ni
        self.tof_sigma = {'54Ni': 0.034, '55Ni': 0.04, '56Ni': 0.10, '58Ni': 0.076, '60Ni': 0.074, '62Ni': 0.076,
                          '64Ni': 0.064}  # Analysis values
        self.tof_width_sigma = 3  # how many sigma to use around tof? (1: 68.3% of data, 2: 95.4%, 3: 99.7%)

        self.timebin_size = 4.8  # length of timegate in 10ns (4.8 = 48ns)

        # fit from scratch or use FitRes db?
        self.do_the_fitting = True  # if False, a .xml file has to be specified in the next variable!
        self.load_results_from = 'SoftwareGateAnalysis_2020-08-27_13-30_allUpdated.xml'  # load fit results from this file
        # print results to results folder? Also show plots?
        self.save_plots_to_file = True  # if False plots will be displayed during the run for closer inspection
        # acceleration set voltage (Buncher potential), negative
        self.accVolt_set = 29850  # omit voltage sign, assumed to be negative
        self.calibration_method = 'absolute'  # can be 'absolute', 'relative' 'combined', 'isoshift' or 'None'
        self.use_handassigned = False  # use hand-assigned calibrations? If false will interpolate on time axis
        self.accVolt_corrected = (self.accVolt_set, 0)  # Used later for calibration. Might be used her to predefine calib? (abs_volt, err)
        self.initial_par_guess = {'sigma': (34, True), 'gamma': (12, True),  #'sigma': (31.7, True), 'gamma': (18.4, True), for VoigtAsy
                                  'asy': (3.9, True),  # in case VoigtAsy is used
                                  'dispersive': (-0.04, False),  # in case FanoVoigt is used
                                  'centerAsym': (-6, True), 'nPeaksAsym': (1, True), 'IntAsym': (0.057, True)
                                  # in case AsymmetricVoigt is used
                                  }
        self.isotope_colors = {60: 'b', 58: 'k', 56: 'g', 55: 'c', 54: 'm', 62: 'purple', 64: 'orange'}
        self.scaler_colors = {'scaler_0': 'blue', 'scaler_1': 'green', 'scaler_2': 'red',
                              'scaler_012': 'black', 'scaler_c012': 'pink'}

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
        self.lineshape_d_syst = 1.0  # MHz. Offset between VoigtAsym and AsymmetricVoigt TODO: Can we say which is better?
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
        # calculate differential doppler shifts
        # TODO: Why calculate here with volt+600??
        self.diff_dopplers = {
            key: Physics.diffDoppler(self.restframe_trans_freq, self.accVolt_set, self.masses[key][0] / 1e6)
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

    def softw_gate_analysis(self, plot=True):
        """
        First let's take a look on how the midTof value influences the center fit paramter.
        The softwGateWidth will be constant and the midTof will be varied slightly
        :return:
        """
        self.write_to_db_lines(self.run,
                               sigma=self.initial_par_guess['sigma'],
                               gamma=self.initial_par_guess['gamma'],
                               asy=self.initial_par_guess['asy'],
                               dispersive=self.initial_par_guess['dispersive'],
                               centerAsym=self.initial_par_guess['centerAsym'],
                               IntAsym=self.initial_par_guess['IntAsym'],
                               nPeaksAsym=self.initial_par_guess['nPeaksAsym'])
        # reset isotope type and acc voltage in db
        iso_list = ['55Ni', '60Ni', '58Ni', '56Ni']  # ['55Ni_sum_cal', '56Ni', '58Ni', '60Ni'] '64Ni', '62Ni', '54Ni',

        # use scaler 0 for now. Probably doesn't make a difference
        scaler_list = ['scaler_012'] #['scaler_0', 'scaler_1', 'scaler_2', 'scaler_012']  # try to put combined scalers last!

        for iso in iso_list:
            pickiso = iso+'%'
            ''' define manual file selection '''
            # self.reset(pickiso+'%', [self.accVolt_set, pickiso])  # This resets the isotope and accVolt in db!
            # filenums = [9313]  #[9295, 9299, 9303, 9305, 9310]  #[9275, 9281, 9283, 9285]
            # filelist = ['BECOLA_{}.xml'.format(num) for num in filenums]
            # filedates = []

            ''' define automatic file selection '''
            select = None
            if '55' in pickiso:
                select = (9999, 9999)  # only sumfile
            filelist, filenums, filedates = self.pick_files_from_db_by_type_and_num(pickiso, selecttuple=select)

            self.results[iso] = {'file_numbers': filenums,
                                 'file_names': filelist,
                                 'file_times': filedates,
                                 'color': self.isotope_colors[int(iso[:2])]
                                 }

            ''' define variation parameters '''
            midtof_variation = (-5, +5, 11)  # (relative midtof variation in bins, number of variations inside width) (-5, +5, 11)
            midtof_variation_arr = np.linspace(*midtof_variation)

            gatewidth_variation_arr = np.logspace(2.6, 6, 35, base=2)  #2, 6, 41 || 3, 6, 31 || 3, 10, 8 for 8-1024 bins # 5, 5.9, 7 for 33-59 / 90-100%
            scale = 34/(6-2.6)

            close_indx = (np.abs(gatewidth_variation_arr - 200*self.tof_width_sigma*self.tof_sigma[iso])).argmin()  # index that is closest to analysis gatewidth
            # possibly reduce fitting width by adding [x:] to gatewidth, midtof and midtof_err
            fit_start = 0

            for scaler in scaler_list:
                sc_name = self.update_scalers_in_db(scaler)
                try:
                    sc_num = ast.literal_eval(sc_name.split('_')[-1])
                    single_sc = True
                except:
                    single_sc = False
                    sc_num = sc_name.split('_')[-1]
                    if sc_num == '012':
                        try:
                            # try to get the new recommended scaler delays from the previous analysis
                            sc0_mid = self.results[iso]['scaler_0']['recommended_mid_SNR']['vals'][0]
                            sc1_mid = self.results[iso]['scaler_1']['recommended_mid_SNR']['vals'][0]
                            sc2_mid = self.results[iso]['scaler_2']['recommended_mid_SNR']['vals'][0]
                            new_delaylist = [0, (sc1_mid-sc0_mid)/100, (sc2_mid-sc0_mid)/100]
                            self.tof_delay[iso] = new_delaylist
                            self.results[iso]['recommended_delay_SNR'] = new_delaylist
                        except:
                            # probably no full analysis done before.
                            print('ATTENTION: Could not adapt delay list to new values!')
                self.results[iso][sc_name] = {'full_data': {'xax': np.log(gatewidth_variation_arr) / np.log(2)}}

                # delaylist = self.tof_delay[iso[:4]]  # get delay per scaler from init

                popt_res = []  # popt results per file
                perr_res = []  # popt results per file
                mid_TOF_perFile = []  # midtof results per file
                mid_d_TOF_perFile = []  # midtof results per file
                sigma_TOF_perFile = []  # tof sigma results per file
                sigma_d_TOF_perFile = []  # tof sigma uncertainty per file
                center_fit_res = []  # center fit results per file for midTof=0, width=45bin
                center_err_res = []  # center fit results per file
                midtofzero_std = []  # standard deviations for midtof=0 values over gatewidth variation
                midtofzero_gate1to3sig_std = []  # std for midtof=0 from gate = +-1Sigma to +-3Sigma
                midtofall_std = []  # standard deviations over all midtof and gatewidth variations
                Al_midtofzero_gate1to3sig_std = []  # Alo std for midtof=0 from gate = +-1Sigma to +-3Sigma
                Au_midtofzero_gate1to3sig_std = []  # Aup std for midtof=0 from gate = +-1Sigma to +-3Sigma

                SNR_max_perFile = []  # maximum SNR over all mid/width variations
                SNR_fitPars_perFile = []  # full fit results
                SNR_fitErrs_perFile = []
                mid_SNR_perFile = []  # ideal mid-tof per file according to SNR analysis
                mid_d_SNR_perFile = []
                sigma_SNR_perFile = []  # ideal gatewidth per file according to SNR analysis
                sigma_d_SNR_perFile = []

                if self.do_the_fitting:

                    for indx, file in enumerate(filelist):
                        res_array = np.zeros((midtof_variation_arr.shape[0], gatewidth_variation_arr.shape[0]))
                        res_d_array = np.zeros((midtof_variation_arr.shape[0], gatewidth_variation_arr.shape[0]))
                        SNR_array = np.zeros((midtof_variation_arr.shape[0], gatewidth_variation_arr.shape[0]))
                        SNR_d_array = np.zeros((midtof_variation_arr.shape[0], gatewidth_variation_arr.shape[0]))
                        # Also get A parameters for Nickel 55
                        Al_array = np.zeros((midtof_variation_arr.shape[0], gatewidth_variation_arr.shape[0]))
                        Al_d_array = np.zeros((midtof_variation_arr.shape[0], gatewidth_variation_arr.shape[0]))
                        Au_array = np.zeros((midtof_variation_arr.shape[0], gatewidth_variation_arr.shape[0]))
                        Au_d_array = np.zeros((midtof_variation_arr.shape[0], gatewidth_variation_arr.shape[0]))

                        # f_iso = self.get_iso_for_file(file)[:4]  # TODO: This seems to be right even though I thought it should be the real file iso...

                        # Do tof fitting to find mid-tof for this file. Doesn't shift scalers when multiple are used!
                        midtof_fit, midtof_err, sigma, sigma_err = self.find_midtof_for_file(file, scaler=sc_name)

                        mid_TOF_perFile.append(midtof_fit)
                        mid_d_TOF_perFile.append(midtof_err)
                        sigma_TOF_perFile.append(sigma)
                        sigma_d_TOF_perFile.append(sigma_err)

                        # go back to original scaler
                        sc_name = self.update_scalers_in_db(sc_name)

                        for i, midtof in enumerate(midtof_variation_arr):
                            for j, gatewidth in enumerate(gatewidth_variation_arr):
                                # set new gate parameters
                                self.update_gates_in_db(iso, midtof/100, gatewidth/100, self.tof_delay[iso[:4]])
                                # do the fitting
                                all_center_MHz, all_center_MHz_fiterrs, all_fitpars = self.fit_files([file])
                                if all_center_MHz_fiterrs[0] > 5:  # fit error unexpectedly large
                                    print('ERROR: width: ' + str(gatewidth) + ', mid: ' + str(midtof))
                                # store fit results
                                res_array[i, j] = all_center_MHz[0]
                                res_d_array[i, j] = all_center_MHz_fiterrs[0]
                                Int0 = all_fitpars[0]['Int0'][0]
                                Int0_d = all_fitpars[0]['Int0'][1]
                                bg = all_fitpars[0]['offset'][0]
                                bg_d = all_fitpars[0]['offset'][1]
                                # A parameters of Nickel 55 are also interesting
                                if '55' in iso:
                                    Al_array[i, j] = all_fitpars[0]['Al'][0]
                                    Al_d_array[i, j] = all_fitpars[0]['Al'][1]
                                    Au_array[i, j] = all_fitpars[0]['Au'][0]
                                    Au_d_array[i, j] = all_fitpars[0]['Au'][1]
                                # calculate and store SNR and its uncertainty
                                SNR_array[i, j] = Int0 / np.sqrt(bg)
                                SNR_d_array[i, j] = np.sqrt(np.square(Int0_d / bg) + np.square(Int0 / bg ** 2 * bg_d))
                                if midtof == 0:
                                    if j == close_indx:  # value most likely used in analysis
                                        # This is the set value. Note it down.
                                        center_fit_res.append(all_center_MHz[0])  # center fit results per file for midTof=0, width=45bin
                                        center_err_res.append(all_center_MHz_fiterrs[0])

                        ''' store full data for midtof and SNR'''
                        self.results[iso][sc_name]['full_data'][file] = {'centroid': {},
                                                                         'SNR': {}}
                        for i, tof in enumerate(midtof_variation_arr):
                            # store center fit results in dependence of tof and gate (relative to probable analysis value)
                            self.results[iso][sc_name]['full_data'][file]['centroid']['tof_{}'.format(tof)] = {
                                'vals': (res_array[i, :] - center_fit_res[-1]).tolist(),
                                'd_stat': (res_d_array[i, :]).tolist()}
                            # store SNR in dependence of tof and gate
                            self.results[iso][sc_name]['full_data'][file]['SNR']['tof_{}'.format(tof)] = {
                                'vals': (SNR_array[i, :]).tolist(),
                                'd_stat': (SNR_d_array[i, :]).tolist()}

                        ''' Fit a line to the mid-tof=0 centroids '''
                        # weighted average over all midtof variations:
                        for indx, d in np.ndenumerate(res_d_array):
                            if d == 0:
                                # something went wrong inm the fit. 0 can not be true.
                                res_d_array[indx] = 100
                        midTof_wavg = np.average(res_array, axis=0, weights=1/np.square(res_d_array))
                        midTof_wavg_err = np.sqrt(1/np.sum(1/np.square(res_d_array), axis=0))
                        # fit a line to this w_avg data
                        def _line(x, m, b):
                            return m*x+b
                        p0 = [0, midTof_wavg[0]]  # initial guess
                        popt, pcov = curve_fit(_line, gatewidth_variation_arr[fit_start:], midTof_wavg[fit_start:],
                                               p0, sigma=midTof_wavg_err[fit_start:], absolute_sigma=True)
                        perr = np.sqrt(np.diag(pcov))
                        popt_res.append(popt)
                        perr_res.append(perr)

                        ''' Find best mid-tof and gatewidth from SNR '''
                        """
                        From the following model it can actually be derived, that the best gatewidth is always
                        optimal_w ~= 2.8*sigma
                        The gatewidth is no parameter in the analysis, just the sigma. 
                        If we have the correct mid-tof (x0=0) and the time-gate is symmetric around this, then the gaussian
                        reduces to: A*e^(-0.5(x/sigma)^2) 
                        and the integral of that from -w/2 to w/2 is (see WolframAlpha):
                        I = A*sigma*sqrt(2pi) * erf(w/(2sqrt(2)*sigma))
                        For the SNR we divide that by the background that also scales with the gate width w:
                        bg = w*B
                        Remember the SNR = I/sqrt(bg) and lets set B=1 and then we can write:
                        SNR = A*sigma*sqrt(2pi) * erf(w/(2*sqrt(2)*s))/sqrt(w)) 
                        and we can see that that function has a maximum in w around 2.8 sigma independent of A
                        """
                        # get the maximum SNR
                        SNR_max = np.amax(SNR_array)

                        def SNR_model(x, y, SNRmax, mid, tof_sigma, beamb):
                            """
                            Assuming the ion bunch is gauss-shaped in the time domain, the signal-to-noise ratio should
                            be defined through the integral over the gaussian and the (constant) background
                            :param x: the x-position on the plotwindow/meshgrid
                            :param y: the y-position on the plotwindow/meshgrid
                            :param SNRmax: The maximum SNR. Scales the height but doesn't influence the positon
                            :param mid: the real mid-tof correction to the assumed value
                            :param tof_sigma: the real tof-sigma. This determines most of the SNR analysis
                            :return: SNR, depending on gate choice
                            """
                            # beamb = 10
                            # give x and y a real meaning:
                            width = gatewidth_variation_arr[x]  # the x-parameter is the gatewidth defined by log array
                            tof_off = midtof_variation_arr[y]  # the y parameter is the tof-variation. We assume midtof=0
                            # Integrate the gaussian from midtof+y-x/2 to midtof+y+x/2
                            # The integral is defined through imaginary errorfunction erf
                            intensity = np.sqrt(np.pi/2)*tof_sigma * (
                                    special.erf((tof_off - mid + width / 2) / (np.sqrt(2) * tof_sigma))
                                    - special.erf((tof_off - mid - width / 2) / (np.sqrt(2) * tof_sigma)))
                            intensity = intensity / (2.10177*tof_sigma)  # scaling for int=1 at 2.8 sigma
                            background = width / (2.8*tof_sigma)  # scales linear by width. scaling for bg=1 at 2.8 sigma
                            # In reality, the background will not be linear but have a beam induced straylight component
                            if iso in self.isotopes_stable:
                                # beam bg of stable isotopes will be dominated by 58Ni
                                mid_58 = 100*self.tof_mid['58Ni'] - (100*self.tof_mid[iso] + mid)
                                sigma_58 = 100*self.tof_sigma['58Ni']
                                bg_stray = beamb * np.sqrt(np.pi / 2) * sigma_58 * (
                                        special.erf((tof_off - mid_58 + width / 2) / (np.sqrt(2) * sigma_58))
                                        - special.erf((tof_off - mid_58 - width / 2) / (np.sqrt(2) * sigma_58)))\
                                           /(2.10177*sigma_58)
                                # 60Ni contribution is next. Scaled by natural abundance ratio
                                mid_60 = 100 * self.tof_mid['60Ni'] - (100 * self.tof_mid[iso] + mid)
                                sigma_60 = 100 * self.tof_sigma['60Ni']
                                bg_stray += 26/68 * beamb * np.sqrt(np.pi / 2) * sigma_60 * (
                                        special.erf((tof_off - mid_60 + width / 2) / (np.sqrt(2) * sigma_60))
                                        - special.erf((tof_off - mid_60 - width / 2) / (np.sqrt(2) * sigma_60))) \
                                           / (2.10177 * sigma_60)
                            else:
                                # beam background for radioactive isotopes is small to negligible compared to laser bg
                                bg_stray = 0
                                beamb = 0
                                # if we wanna try anyways then the beam bg is of course only the radioactive isotope
                                # mid_bb = mid
                                # sigma_bb = tof_sigma
                                # bg_stray = beamb * np.sqrt(np.pi/2)*sigma_bb * (  # scaling of 2.7 has been extracted from time-res data
                                #         special.erf((tof_off - mid_bb + width / 2) / (np.sqrt(2) * sigma_bb))
                                #         - special.erf((tof_off - mid_bb - width / 2) / (np.sqrt(2) * sigma_bb)))
                                # bg_stray = bg_stray / (2.10177*sigma_bb)  # scaling for bg_stray = beamb at 2.8 sigma
                            background = (background + bg_stray)/(1+beamb)

                            # norm = np.sqrt(2.8*tof_sigma)/(2.10177*tof_sigma)
                            norm = 1
                            # return the signal to noise calculated from this
                            return SNRmax * norm * intensity / np.sqrt(background)

                        def _SNR_model(M, *args):
                            """
                            2D function generating a 3D plane
                            :param M: xdata parameter, 2-dimensional array of x and y values
                            :param args: slope and offset passed to plane function
                            :return: array of z-values
                            """
                            x, y = M  # x: gatewidth, y: midtof
                            arr = np.zeros(x.shape)
                            arr += SNR_model(x, y, *args)
                            return arr

                        x = np.arange(SNR_array.shape[1])  # gatewidth_variation_arr
                        y = np.arange(SNR_array.shape[0])  # midtof_variation_arr
                        X, Y = np.meshgrid(x, y)
                        Z = SNR_array

                        # initial guess and bounds
                        # make sure initial guess does not violate bounds
                        initial_guess = [max(SNR_max, 0.001), 0, np.median(gatewidth_variation_arr) / 2.8, 1]
                        # Let's assume the best SNR is inside the parameter space
                        bounds = ([0.00001, 10 * midtof_variation_arr[0], gatewidth_variation_arr[0] / 2.8, 0],
                                  [np.inf, 10 * midtof_variation_arr[-1], gatewidth_variation_arr[-1] / 2.8, np.inf])
                        # make 1-D data (necessary to use curve_fit)
                        xdata = np.vstack((X.ravel(), Y.ravel()))
                        popt_SNR, pcov_SNR = curve_fit(_SNR_model, xdata, SNR_array.ravel(), p0=initial_guess,
                                                       sigma=SNR_d_array.ravel(), absolute_sigma=True, bounds=bounds)  # TODO: absolute_sigma=True gives pretty big mid uncerty for 55Ni (5.8). Better with False?
                        perr_SNR = np.sqrt(np.diag(pcov_SNR))

                        SNR_fitPars_perFile.append(popt_SNR.tolist())
                        SNR_fitErrs_perFile.append(perr_SNR.tolist())

                        mid_SNR_perFile.append(self.tof_mid[iso]*100+popt_SNR[1])
                        mid_d_SNR_perFile.append(perr[1])
                        sigma_SNR_perFile.append(popt_SNR[2])
                        sigma_d_SNR_perFile.append(perr_SNR[2])

                        # write some more stuff down
                        SNR_max_perFile.append((SNR_max))
                        SNR_max_indx = np.where(SNR_array == SNR_max)
                        mMax = midtof_variation_arr[SNR_max_indx[0][0]]
                        # mid_SNR_perFile.append(100 * self.tof_mid[iso] + mMax)
                        gMax = gatewidth_variation_arr[SNR_max_indx[1][0]]
                        # sigma_SNR_perFile.append(gMax)

                        # calculate standard deviation for mid_tof=0 values along the gatewidth variation range
                        midtof_zero_indx = np.argwhere(midtof_variation_arr == 0)[0, 0]
                        st_dev_0 = res_array[midtof_zero_indx, :].std()
                        midtofzero_std.append(st_dev_0)
                        # calculate standard deviation over full variation range
                        st_dev_all = res_array.std()
                        midtofall_std.append(st_dev_all)
                        # calculate standard deviation only within 1sigma - 3 sigma withs
                        gatewidth1Sigma = 2*1*popt_SNR[2]
                        where1Sigma = (np.abs(gatewidth_variation_arr - gatewidth1Sigma)).argmin()  # gate indx closest to 1sigma
                        gatewidth3Sigma = 2*3*popt_SNR[2]
                        where3Sigma = (np.abs(gatewidth_variation_arr - gatewidth3Sigma)).argmin()  # gate indx closest to 3sigma
                        st_dev_1to3sigma = res_array[midtof_zero_indx, where1Sigma:where3Sigma].std()
                        midtofzero_gate1to3sig_std.append(st_dev_1to3sigma)

                        # Also calculate std for A parameters
                        Al_st_dev_1to3sigma = Al_array[midtof_zero_indx, where1Sigma:where3Sigma].std()
                        Al_midtofzero_gate1to3sig_std.append(Al_st_dev_1to3sigma)
                        Au_st_dev_1to3sigma = Al_array[midtof_zero_indx, where1Sigma:where3Sigma].std()
                        Au_midtofzero_gate1to3sig_std.append(Au_st_dev_1to3sigma)


                        if plot:
                            gatew = gatewidth_variation_arr  # we'll need this often now.

                            # plot gatewidth variation
                            fig, ax = plt.subplots()
                            for i, tof in enumerate(midtof_variation_arr):
                                if tof == 0:
                                    ax.errorbar(np.log(gatew)/np.log(2), res_array[i, :]-center_fit_res[-1], yerr=res_d_array[i, :], c='k', linewidth=2.0, label='{:.0f}bins'.format(tof))
                                else:
                                    ax.plot(np.log(gatew)/np.log(2), res_array[i, :]-center_fit_res[-1], label='{:.0f}bins'.format(tof))
                            # plot weightedavg over midtofs
                            # ax.errorbar(np.log(gatew) / np.log(2), midTof_wavg-center_fit_res[-1], yerr=midTof_wavg_err, label='w_avg',
                            #             c='k', linestyle='--', linewidth=2.0)
                            # plot fit of weightedavg over tofs. Possibly reduce plotting range
                            ax.plot(np.log(gatew[fit_start:]) / np.log(2), _line(gatew[fit_start:], *popt)-center_fit_res[-1], label='w_avg_fit',
                                    c='k', linestyle='--', linewidth=3.0)
                            # vertical line for probable analysis value
                            ax.axvline(x=np.log(2*2*popt_SNR[2])/np.log(2), color='red')
                            ax.tick_params(axis='y', labelcolor='k')
                            ax.set_ylabel('fit center [MHz] rel. to analysis value', color='k')
                            ax.set_xlabel('gatewidth [bins]')

                            # plot SNR error band
                            axSNR = ax.twinx()
                            axSNR.set_ylabel('SNR', color='red')  # we already handled the x-label with ax1
                            # Do the plotting. Errors are typically very very small
                            axSNR.plot(np.log(gatew) / np.log(2), SNR_array[round(len(midtof_variation_arr)/2), :], label='SNR', c='r', linestyle='-',
                                       linewidth=3.0)
                            axSNR.tick_params(axis='y', labelcolor='red')

                            plt.title('{}, file: {}\n'
                                      'Variation of timegate parameters around midTof=bin{:.0f}.\n'
                                      'Fitresult: center=({:.0f}({:.0f})kHz/bin)*gatewidth+({:.2f}({:.0f}))MHz'
                                      .format(iso, file, 100*self.tof_mid[iso],
                                              1000*popt[0], 1000*perr[0],
                                              popt[1], 100*perr[1]))
                            # work on the ticks
                            import matplotlib.ticker as ticker
                            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0f}'.format(2**(y))))
                            plt.margins(0.05)
                            ax.legend(title='midTof', bbox_to_anchor=(1.1, 0.5), loc="center left")
                            if self.save_plots_to_file:
                                filename = 'gatewidth_var_{}_{}_sc{}'.format(iso, file, sc_num)
                                plt.savefig(self.resultsdir + filename + '.png', bbox_inches="tight")
                            else:
                                plt.show()
                            plt.close()
                            plt.clf()

                            ''' 3-D surface plot of the SNR analysis '''
                            fig = plt.figure()
                            ax = fig.add_subplot(111, projection='3d')
                            surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')
                            fig.colorbar(surf)
                            plt.xlabel('mid-tof')
                            plt.ylabel('gate width')
                            if self.save_plots_to_file:
                                filename = 'SNR-3D_{}_{}_sc{}'.format(iso, file, sc_num)
                                plt.savefig(self.resultsdir + filename + '.png', bbox_inches="tight")
                            else:
                                plt.show()
                            plt.close()
                            plt.clf()

                            ''' 2-D imshow plot of the SNR analysis together with the fit '''
                            fig, ax = plt.subplots()
                            # print the SNR values
                            im = ax.imshow(SNR_array, cmap='hot', interpolation='nearest')
                            # print the contours
                            ax.contour(X, Y, SNR_model(X, Y, *popt_SNR), 8, colors='b')
                            xerr = [[-scale * (np.log(popt_SNR[2] * 2.8 - perr_SNR[2]) / np.log(2) - 2)
                                     + scale * (np.log(popt_SNR[2] * 2.8) / np.log(2) - 2)],
                                    [scale * (np.log(popt_SNR[2] * 2.8 + perr_SNR[2]) / np.log(2) - 2)
                                     - scale * (np.log(popt_SNR[2] * 2.8) / np.log(2) - 2)]]
                            ax.errorbar(x=scale * (np.log(popt_SNR[2] * 2.8) / np.log(2) - 2),
                                        y=popt_SNR[1] - midtof_variation_arr[0],  # -global_guess[iso]
                                        xerr=xerr, yerr=perr_SNR[1], color='b', markersize=5)
                            # set title and stuff
                            plt.title('Signal-To-Noise Analysis for run{} [{}]\n'
                                      'Maximum SNR is {:.2f} for midtof: {:.0f} and gatewidth: {:.1f}'
                                      .format(file, iso, SNR_max, mMax, gMax))
                            ax.set_xlabel('gate width')
                            ax.set_ylabel('mid-tof')
                            ax.set_xticks(np.arange(len(gatewidth_variation_arr)))
                            ax.set_yticks(np.arange(len(midtof_variation_arr)))
                            ax.set_xticklabels(['{:.1f}'.format(m) for m in gatewidth_variation_arr])
                            ax.set_yticklabels(['{:.0f}'.format(m) for m in midtof_variation_arr])
                            cbar = fig.colorbar(im, orientation='horizontal')
                            cbar.set_label('SNR')
                            plt.setp(ax.get_xticklabels(), rotation=90, ha="center", va='top', rotation_mode="default")
                            if self.save_plots_to_file:
                                filename = 'SNR-heatmap_{}_{}_sc{}'.format(iso, file, sc_num)
                                plt.savefig(self.resultsdir + filename + '.png', bbox_inches="tight")
                            else:
                                plt.show()
                            plt.close()
                            plt.clf()

                        # reset to original values
                        self.update_gates_in_db(iso, 0, 2*self.tof_sigma[iso]*self.tof_width_sigma, self.tof_delay[iso])

                    ''' Calculations for this isotope over all files '''
                    popt_res = np.array(popt_res)  # popt results per file
                    perr_res = np.array(perr_res)  # popt results per file
                    # Calculate the relation between the two sigmas determined by SNR analysis and tof-fit analysis
                    sigma_SNRvsTOF = np.array(sigma_SNR_perFile)/np.array(sigma_TOF_perFile)
                    # Calculate the best overall mid and sigma for this isotope
                    # First for SNR analysis
                    mid_wavg, mid_wavg_d, mid_wstd, mid_std, mid_std_avg = self.calc_weighted_avg(mid_SNR_perFile,
                                                                                                  mid_d_SNR_perFile)
                    sigma_wavg, sigma_wavg_d, sigma_wstd, sigma_std, sigma_std_avg = self.calc_weighted_avg(sigma_SNR_perFile,
                                                                                                       sigma_d_SNR_perFile)
                    if single_sc:
                        # Analysis was done with single scaler and the scaler delay. Adjust recommended val by this
                        mid_wavg += 100*self.tof_delay[iso[:4]][sc_num]
                    best_mid_SNR = (mid_wavg, max(mid_wavg_d, mid_wstd, mid_std, mid_std_avg))
                    best_sigma_SNR = (sigma_wavg, max(sigma_wavg_d, sigma_wstd, sigma_std, sigma_std_avg))
                    # now for fitTOF analysis
                    midt_wavg, midt_wavg_d, midt_wstd, midt_std, midt_std_avg = self.calc_weighted_avg(mid_TOF_perFile,
                                                                                                  mid_d_TOF_perFile)
                    sigmat_wavg, sigmat_wavg_d, sigmat_wstd, sigmat_std, sigmat_std_avg = self.calc_weighted_avg(sigma_TOF_perFile,
                                                                                                       sigma_d_TOF_perFile)
                    best_mid_TOF = (midt_wavg, max(midt_wavg_d, midt_wstd, midt_std, midt_std_avg))
                    best_sigma_TOF = (sigmat_wavg, max(sigmat_wavg_d, sigmat_wstd, sigmat_std, sigmat_std_avg))

                    ''' compile a dictionary of the results '''
                    results_iso_scaler_new_content = {'recommended_mid_SNR': {'vals': [best_mid_SNR[0]],
                                                                          'd_fit': [best_mid_SNR[1]]},
                                                      'recommended_sigma_SNR': {'vals': [best_sigma_SNR[0]],
                                                                          'd_fit': [best_sigma_SNR[1]]},
                                                      'recommended_mid_TOF': {'vals': [best_mid_TOF[0]],
                                                                              'd_fit': [best_mid_TOF[1]]},
                                                      'recommended_sigma_TOF': {'vals': [best_sigma_TOF[0]],
                                                                               'd_fit': [best_sigma_TOF[1]]},
                                                      'gate_analysis_m': {'vals': list(popt_res[:, 0]),
                                                                          'd_fit': list(perr_res[:, 0])},
                                                      'gate_analysis_b': {'vals': list(popt_res[:, 1]),
                                                                          'd_fit': list(perr_res[:, 1])},
                                                      'fitTOF_mid': {'vals': mid_TOF_perFile,
                                                                  'd_fit': mid_d_TOF_perFile},
                                                      'fitTOF_sigma': {'vals': sigma_TOF_perFile,
                                                                    'd_fit': sigma_d_TOF_perFile},
                                                      'center_fits': {'vals': center_fit_res,
                                                                      'd_fit': center_err_res,
                                                                      'd_stat': center_err_res,
                                                                      'd_syst': midtofzero_std},  # std dev is interesting
                                                      'bunchwidth_std_0': {'vals': midtofzero_std},
                                                      'bunchwidth_std_0_1to3Sig': {'vals': midtofzero_gate1to3sig_std},
                                                      'bunchwidth_std_all': {'vals': midtofall_std},
                                                      'Alo_bunchwidth_std_0_1to3Sig': {'vals': Al_midtofzero_gate1to3sig_std},
                                                      'Aup_bunchwidth_std_0_1to3Sig': {'vals': Au_midtofzero_gate1to3sig_std},
                                                      'maxSNR': {'vals': SNR_max_perFile},
                                                      'bestSNR_mid': {'vals': mid_SNR_perFile,
                                                                         'd_fit': mid_d_SNR_perFile},
                                                      'bestSNR_sigma': {'vals': sigma_SNR_perFile,
                                                                            'd_fit': sigma_d_SNR_perFile},
                                                      'SigmaRatio_SNRvsTOF': {'vals': sigma_SNRvsTOF},
                                                      'SNR_fit': {'vals': SNR_fitPars_perFile,
                                                                          'd_fit': SNR_fitErrs_perFile}
                                                      }
                    TiTs.merge_extend_dicts(self.results[iso][sc_name], results_iso_scaler_new_content,
                                            overwrite=True, force_overwrite=True)  # Merge dicts. Prefer new content

                    self.export_results()
                else:
                    results_import = self.import_results(self.load_results_from)
                    self.results[iso] = results_import[iso]
                    bestSNR = np.array(self.results[iso][sc_name]['bestSNR_sigma']['vals'])
                    sigma_tof = 100 * np.array(self.results[iso][sc_name]['fitTOF_sigma']['vals'])
                    ratio = bestSNR / 2 / sigma_tof
                    self.results[pickiso][sc_name]['SigmaRatio_SNRvsTOF'] = {'vals': ratio}

        ''' comparisons across all isotopes '''
        mid_SNR = []
        mid_d_SNR = []
        mid_TOF = []
        mid_d_TOF = []
        for iso in sorted(iso_list):
            mid_SNR.append(self.results[iso][sc_name]['recommended_mid_SNR']['vals'][0])
            mid_d_SNR.append(self.results[iso][sc_name]['recommended_mid_SNR']['d_fit'][0])
            mid_TOF.append(self.results[iso][sc_name]['recommended_mid_TOF']['vals'][0])
            mid_d_TOF.append(self.results[iso][sc_name]['recommended_mid_TOF']['d_fit'][0])
        # create the plot
        fig, ax = plt.subplots()
        plt.errorbar(range(iso_list.__len__()), mid_SNR, mid_d_SNR, color='g')
        plt.errorbar(range(iso_list.__len__()), mid_TOF, mid_d_TOF, color='b')
        # make x-axis nice
        ax.set_xlabel('Isotope')
        ax.set_xticks(range(iso_list.__len__()))
        ax.set_xticklabels(iso_list)
        ax.set_xlim((-0.5, iso_list.__len__()-0.5))
        if self.save_plots_to_file:
            filename = 'mid_comparison_all_isos'
            plt.savefig(self.resultsdir + filename + '.png', bbox_inches="tight")
        else:
            plt.show()
        plt.close()
        plt.clf()


        self.plot_parameter_for_isos_and_scaler(iso_list, [sc_name], 'gate_analysis_m', unit='kHz/bin', onlyfiterrs=True, overlay=0, factor=1000, digits=1)
        self.plot_parameter_for_isos_and_scaler(iso_list, [sc_name], 'gate_analysis_b', onlyfiterrs=True)
        self.plot_parameter_for_isos_and_scaler(iso_list, [sc_name], 'fitTOF_mid', unit='bins', onlyfiterrs=True)
        self.plot_parameter_for_isos_and_scaler(iso_list, [sc_name], 'fitTOF_sigma', unit='bins', onlyfiterrs=True)
        self.plot_parameter_for_isos_and_scaler(iso_list, [sc_name], 'center_fits')
        self.plot_parameter_for_isos_and_scaler(iso_list, [sc_name], 'bunchwidth_std_0', onlyfiterrs=True, digits=2)
        self.plot_parameter_for_isos_and_scaler(iso_list, [sc_name], 'bunchwidth_std_all', onlyfiterrs=True, digits=2)
        self.plot_parameter_for_isos_and_scaler(iso_list, [sc_name], 'maxSNR', onlyfiterrs=True, digits=0)
        self.plot_parameter_for_isos_and_scaler(iso_list, [sc_name], 'bestSNR_mid', onlyfiterrs=True, digits=1)
        self.plot_parameter_for_isos_and_scaler(iso_list, [sc_name], 'bestSNR_sigma', onlyfiterrs=True, digits=1)
        self.plot_parameter_for_isos_and_scaler(iso_list, [sc_name], 'SigmaRatio_SNRvsTOF', unit='', onlyfiterrs=True, digits=2)

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

    def update_gates_in_db(self, iso, midtof_var, gatewidth, delaylist):
        '''
        Write all parameters relevant for the software gate position into the database
        :param midtof_var: float: center of software gate in µs
        :param gatewidth: float: width of software gate in µs
        :param delaylist: list of floats: list of delays in midtof for scalers 0,1,2
        :return:
        '''
        midtof = self.tof_mid[iso[:4]]+midtof_var
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''UPDATE Runs SET softwGateWidth = ? WHERE run = ?''', (gatewidth, self.run))
        cur.execute('''UPDATE Runs SET softwGateDelayList = ? WHERE run = ?''', (str(delaylist), self.run))
        cur.execute('''UPDATE Isotopes SET midTof = ? WHERE iso = ?''', (midtof, iso))
        con.commit()
        con.close()

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

    ''' analysis related '''

    def find_midtof_for_file(self, file, scaler=None):
        if scaler is None:
            # load from self.scaler name
            scaler = self.scaler_name
        else:
            scaler = self.update_scalers_in_db(scaler)
        try:
            sc_num = ast.literal_eval(scaler.split('_')[-1])
        except:
            # This seems to be more than one scaler. Just use scaler 0 for the fitting bc annything else doesn't make much sense...
            sc_num = 0
        # load the spec data from file
        filepath = os.path.join(self.datafolder, file)
        spec = XMLImporter(path=filepath)

        # one of the three scalers is explicitly given.
        tproj = np.array(spec.t_proj).sum(axis=0)[sc_num]  # sum over all tracks!
        timebins = np.arange(1024)
        # fit time-projection
        ampl, sigma, center, offset, center_d, sigma_d = self.fit_time_projections(tproj, timebins)
        if center_d > 30 or center < 500 or center > 600:
            # something went wrong while fitting. Maybe this is DC data? Use middle value
            center = 512
        midtof_bins = center
        midtof_d_bins = center_d
        return midtof_bins, midtof_d_bins, sigma, sigma_d

    def fit_time_projections(self, cts_axis, time_axis):
        x = time_axis
        y = cts_axis
        # estimates:: amplitude: sigma*sqrt(2pi)*(max_y-min_y), sigma=10, center:position of max_y, offset: min_y
        bg_est = np.mean(cts_axis[0:300])
        # mid_est = np.argwhere(cts_axis == max(cts_axis))[0,0]
        mid_est = 540
        start_pars = np.array([10*2.51*(max(cts_axis)-bg_est), 10, mid_est, bg_est])
        popt, pcov = curve_fit(self.fitfunc, x, y, start_pars)
        ampl, sigma, center, offset = popt
        perr = np.sqrt(np.diag(pcov))
        # plt.errorbar(x, y, np.sqrt(y))
        # plt.plot(x, self.fitfunc(x, *popt))
        # plt.show()
        return ampl, sigma, center, offset, perr[2], perr[1]

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

    def fit_files(self, filelist):
        filearray = np.array(filelist)  # needed for batch fitting

        # do the batchfit
        BatchFit.batchFit(filearray, self.db, self.run, x_as_voltage=True, softw_gates_trs=None, save_file_as='.png', guess_offset=True)

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

    ''' visualization '''

    def plot_parameter_for_isos_and_scaler(self, isotopes, scaler_list, parameter,
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
        x_type = 'file_numbers'  # alternative: 'file_numbers', 'file_times'
        scaler_nums = []
        for sc in scaler_list:
            scaler = self.update_scalers_in_db(sc)
            scaler_nums.append(scaler.split('_')[1])
            for i in range(len(isotopes)):
                iso = isotopes[i]
                x_ax_label = self.results[iso][x_type]
                x_ax = np.arange(len(self.results[iso][x_type]))
                if 'all_fitpars' in parameter:
                    # the 'all fitpars is organized a little different.
                    # For each file they are just stored as a dict like in db
                    # Parameter must be specified as 'all_fitpars:par' with par being the specific parameter to plot
                    fitres_list = self.results[iso][scaler]['all_fitpars']
                    parameter_plot = parameter.split(':')[1]
                    centers = factor * np.array([i[parameter_plot][0] for i in fitres_list])
                    centers_d_stat = factor * np.array([i[parameter_plot][1] for i in fitres_list])
                    centers_d_syst = factor * np.array([0 for i in fitres_list])
                    # get weighted average
                    wavg, wavg_d, fixed = self.results[iso][scaler]['avg_fitpars'][parameter_plot]
                    wavg = factor * wavg
                    wavg_d = factor * wavg_d
                    if fixed == True:
                        wavg_d = '-'
                    else:
                        wavg_d = '{:.0f}'.format(10 * wavg_d)  # times 10 for representation in brackets
                else:
                    centers = factor * np.array(self.results[iso][scaler][parameter]['vals'])
                    zero_arr = np.zeros(
                        len(centers))  # prepare zero array with legth of centers in case no errors are given
                    if onlyfiterrs:
                        centers_d_stat = factor * np.array(self.results[iso][scaler][parameter].get('d_fit', zero_arr))
                        centers_d_syst = 0
                    else:
                        centers_d_stat = factor * np.array(self.results[iso][scaler][parameter].get('d_stat', zero_arr))
                        centers_d_syst = factor * np.array(self.results[iso][scaler][parameter].get('d_syst', zero_arr))
                    # calculate weighted average:
                    if not np.any(centers_d_stat == 0) and not np.sum(1 / centers_d_stat ** 2) == 0:
                        weights = 1 / centers_d_stat ** 2
                        wavg, sumw = np.average(centers, weights=weights, returned=True)
                        wavg_d = '{:.0f}'.format(
                            10 ** digits * np.sqrt(1 / sumw))  # times 10 for representation in brackets
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
                if parameter == 'shifts_iso-58' and shiftavg:
                    # also plot average isotope shift
                    avg_shift = self.results[iso][scaler]['avg_shift_iso-58']['vals'][0]
                    avg_shift_d = self.results[iso][scaler]['avg_shift_iso-58']['d_stat'][0]
                    avg_shift_d_syst = self.results[iso][scaler]['avg_shift_iso-58']['d_syst'][0]
                    # plot weighted average as red line
                    plt.plot([x_ax[0], x_ax[-1]], [avg_shift, avg_shift], 'r',
                             label='{0} avg: {1:.{5:d}f}({2:.0f})[{3:.0f}]{4}'
                             .format(iso, avg_shift, 10 ** digits * avg_shift_d, 10 ** digits * avg_shift_d_syst, unit,
                                     digits))
                    # plot error of weighted average as red shaded box around that line
                    plt.fill([x_ax[0], x_ax[-1], x_ax[-1], x_ax[0]],
                             [avg_shift - avg_shift_d, avg_shift - avg_shift_d,
                              avg_shift + avg_shift_d, avg_shift + avg_shift_d], 'r',
                             alpha=0.2)
                    # plot systematic error as lighter red shaded box around that line
                    plt.fill([x_ax[0], x_ax[-1], x_ax[-1], x_ax[0]],
                             [avg_shift - avg_shift_d_syst - avg_shift_d, avg_shift - avg_shift_d_syst - avg_shift_d,
                              avg_shift + avg_shift_d_syst + avg_shift_d, avg_shift + avg_shift_d_syst + avg_shift_d],
                             'r',
                             alpha=0.1)
        if overlay is not None:
            plt.axhline(y=overlay, color='red')
        if x_type == 'file_times':
            plt.xlabel('date')
            days_fmt = mpdate.DateFormatter('%d.%B')
            ax.xaxis.set_major_formatter(days_fmt)
        else:
            plt.xlabel('run numbers')
        plt.xticks(x_ax, x_ax_label, rotation=90)
        plt.ylabel('{} [{}]'.format(parameter, unit))
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        plt.title('{} in {} for isotopes: {}'.format(parameter, unit, isotopes))
        plt.legend(title='Legend', bbox_to_anchor=(1.04, 0.5), loc="center left")
        plt.margins(0.05)
        if self.save_plots_to_file:
            isonums = []
            for isos in isotopes:
                if 'cal' in isos:
                    isonums.append(isos[:2] + 'c')
                else:
                    isonums.append(isos[:2])
            parameter = parameter.replace(':', '_')  # colon is no good filename char
            filename = parameter + '_' + ''.join(isonums) + '_sc' + 'a'.join(scaler_nums)
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
        del_list = [0, 0.185, 0.257]  # scaler delay list. ADAPT!!
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


if __name__ == '__main__':
    analysis = NiAnalysis_softwGates()
    analysis.softw_gate_analysis()


