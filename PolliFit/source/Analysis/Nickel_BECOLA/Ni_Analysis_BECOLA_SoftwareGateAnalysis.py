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
        ownCould_path = 'ownCloud\\User\\Felix\\Measurements\\Nickel_online_Becola\\Analysis\\XML_Data'
        ownCould_path = 'ownCloud\\User\\Felix\\Measurements\\Nickel_offline_Becola20\\XML_Data'  # offline 2020
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
        self.run = 'CEC_AsymVoigt_MSU'
        self.timebin_size = 4.8  # length of timegate in 10ns (4.8 = 48ns)
        self.midtof_orig = {'55Ni': 5.238, '56Ni': 5.28, '58Ni': 5.35, '60Ni': 5.47}
        self.gatewidth_orig = 0.3
        self.delaylist_orig = [0, 0.195, 0.26]

        # fit from scratch or use FitRes db?
        self.do_the_fitting = True  # if False, a .xml file has to be specified in the next variable!
        self.load_results_from = '2020reworked_2020-05-18_12-21.xml'  # load fit results from this file
        # print results to results folder? Also show plots?
        self.save_plots_to_file = True  # if False plots will be displayed during the run for closer inspection
        # acceleration set voltage (Buncher potential), negative
        self.accVolt_set = 29850  # omit voltage sign, assumed to be negative
        self.calibration_method = 'absolute'  # can be 'absolute', 'relative' 'combined', 'isoshift' or 'None'
        self.use_handassigned = False  # use hand-assigned calibrations? If false will interpolate on time axis
        self.accVolt_corrected = (
        self.accVolt_set, 0)  # Used later for calibration. Might be used her to predefine calib? (abs_volt, err)
        self.initial_par_guess = {'sigma': (31, False), 'gamma': (20, False),
                                  'asy': (3, True),  # in case VoigtAsy is used
                                  'dispersive': (0, False)}  # in case FanoVoigt is used
        self.isotope_colors = {58: 'black', 60: 'blue', 56: 'green', 55: 'purple'}
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
        self.accVolt_set_d = 10  # uncertainty of scan volt. Estimated by Miller for Calcium meas.
        self.wavemeter_wsu30_mhz_d = 3  # Kristians wavemeter paper
        self.heliumneon_drift = 0  # TODO: does that influence our measurements? (1 MHz in Kristians calibration)
        self.matsuada_volts_d = 0.02  # Rebinning and graphical Analysis TODO: get a solid estimate for this value
        self.laserionoverlap_anglemrad_d = 1  # ideally angle should be 0. Max possible deviation is ~1mrad TODO: check
        self.laserionoverlap_MHz_d = (self.accVolt_set -  # TODO: This formular should be doublechecked...
                                      np.sqrt(self.accVolt_set ** 2 / (
                                                  1 + (self.laserionoverlap_anglemrad_d / 1000) ** 2))) * 15
        self.lineshape_d_syst = 0  # TODO: investigate
        self.bunch_structure_d = 0  # TODO: investigate

        # TODO: Clarify stat/syst uncertainty definition. For now put all the above in systematics
        self.all_syst_uncertainties = np.array([self.wavemeter_wsu30_mhz_d,
                                                self.heliumneon_drift,
                                                self.matsuada_volts_d,
                                                self.laserionoverlap_MHz_d,
                                                self.lineshape_d_syst,
                                                self.bunch_structure_d])

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
        cur.execute(
            '''DELETE FROM Files WHERE file LIKE ? ''', ('Sum%',))  # Also delete sum files from last run!
        con.commit()  # commit changes to db
        con.close()  # close db connection

    ''' analysis '''

    def softw_gate_analysis(self, plot=True):
        """
        First let's take a look on how the midTof value influences the center fit paramter.
        The softwGateWidth will be constant and the midTof will be varied slightly
        :return:
        """
        # reset isotope type and acc voltage in db
        iso_list = ['58Ni']
        for pickiso in iso_list:
            self.reset(pickiso+'%', [self.accVolt_set, pickiso])

            # use scaler 1 for now. Probably doesn't make a difference
            scaler = 0
            sc_name = self.update_scalers_in_db(scaler)

            # filenums = [6251, 6501, 6502]
            filenums = [9313]  #[9295, 9299, 9303, 9305, 9310]  #[9275, 9281, 9283, 9285]
            filelist = ['BECOLA_{}.xml'.format(num) for num in filenums]
            # filelist, filenums = self.pick_files_from_db_by_type_and_num(pickiso)
            self.results[pickiso] = {'file_numbers': filenums,
                                     'file_names': filelist,
                                     'color': self.isotope_colors[int(pickiso[:2])],
                                     sc_name: {}}
            # midtof_variation = (-0.3, +0.3, 3)  # (relative midtof variation in µs, number of variations inside width)
            # midtof_variation_arr = np.linspace(*midtof_variation)
            midtof_variation_arr = np.append(-0.01*np.logspace(2, 0, 4, base=2), np.append([0], 0.01*np.logspace(0, 2, 4, base=2)))
            # gatewidth_variation = (5, 1, 11)
            # gatewidth_variation_arr = np.linspace(*gatewidth_variation)
            gatewidth_variation_arr = 0.1*np.logspace(0, 6.5, 7, base=2)  # log spaced 0.1 to 6.4 (put 6.5 for 9.1)
            delaylist = [0, 0.19, 0.26]

            popt_res = []  # popt results per file
            perr_res = []  # popt results per file

            for indx, file in enumerate(filelist):
                res_array = np.zeros((midtof_variation_arr.shape[0], gatewidth_variation_arr.shape[0]))
                res_d_array = np.zeros((midtof_variation_arr.shape[0], gatewidth_variation_arr.shape[0]))

                iso = self.get_iso_for_file(file)
                for i, midtof in enumerate(midtof_variation_arr):
                    for j, gatewidth in enumerate(gatewidth_variation_arr):
                        self.update_gates_in_db(iso, midtof, gatewidth, delaylist)
                        all_center_MHz, all_center_MHz_fiterrs, all_fitpars = self.fit_files([file])
                        res_array[i, j] = all_center_MHz[0]
                        res_d_array[i, j] = all_center_MHz_fiterrs[0]

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
                p0 = [0, midTof_wavg[0]]
                popt, pcov = curve_fit(_line, gatewidth_variation_arr, midTof_wavg, p0, sigma=midTof_wavg_err, absolute_sigma=True)
                perr = np.sqrt(np.diag(pcov))
                popt_res.append(popt)
                perr_res.append(perr)

                # self.plot_and_fit_3darray(res_array, file)

                if plot:
                    # BECOLA timebins are not 10ns
                    midtof = self.timebin_size*midtof_variation_arr
                    gatew = self.timebin_size*gatewidth_variation_arr
                    # plot tof variation (not very readable. gatewidth variation is much better!!)
                    for j, gatewidth in enumerate(gatew):
                        plt.errorbar(midtof, res_array[:, j], yerr=res_d_array[:, j], label='{:.2f}µs'.format(gatewidth))
                    plt.title('Variation of mid-tof parameter for {} file {}\n'
                              'around midTof={:.2f}µs'.format(iso, file, self.midtof_orig[iso]))
                    plt.xlabel('mid tof [µs]')
                    plt.ylabel('fit center [MHz]')
                    plt.margins(0.05)
                    plt.legend(title='gatewidth', bbox_to_anchor=(1.04, 0.5), loc="center left")
                    if self.save_plots_to_file:
                        filename = 'midtof_var_{}_{}'.format(iso, file)
                        plt.savefig(self.resultsdir + filename + '.png', bbox_inches="tight")
                    else:
                        plt.show()
                    plt.close()
                    plt.clf()

                    # plot gatewidth variation
                    fig, ax = plt.subplots()
                    for i, tof in enumerate(midtof):
                        ax.errorbar(np.log(10*gatew)/np.log(2), res_array[i, :], yerr=res_d_array[i, :], label='{:.2f}µs'.format(tof))
                    # plot weightedavg over midtofs
                    ax.errorbar(np.log(10 * gatew) / np.log(2), midTof_wavg, yerr=midTof_wavg_err, label='w_avg',
                                c='k', linestyle='--', linewidth=2.0)
                    # plot fit of weightedavg over tofs
                    ax.plot(np.log(10 * gatew) / np.log(2), _line(gatew, *popt), label='w_avg_fit',
                            c='r', linestyle='--', linewidth=2.0)
                    plt.title('Variation of gatewidth parameter for {} file {}\n'
                              'around midTof={:.2f}µs.\n'
                              'Fitresult: center=({:.2f})*gatewidth+({:.2f})MHz'
                              .format(iso, file, self.midtof_orig[iso], *popt))
                    plt.xlabel('gatewidth [µs]')
                    plt.ylabel('fit center [MHz]')
                    # plt.xscale('log', basex=1.2)
                    # plt.xticks(np.log(np.logspace(0, 6, 7, base=2))/np.log(1.2))
                    import matplotlib.ticker as ticker
                    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.2f}'.format(0.1*2**y)))
                    plt.margins(0.05)
                    plt.legend(title='midTof', bbox_to_anchor=(1.04, 0.5), loc="center left")
                    if self.save_plots_to_file:
                        filename = 'gatewidth_var_{}_{}'.format(iso, file)
                        plt.savefig(self.resultsdir + filename + '.png', bbox_inches="tight")
                    else:
                        plt.show()

                # reset to original values
                self.update_gates_in_db(iso, 0, self.gatewidth_orig, self.delaylist_orig)

            popt_res = np.array(popt_res)  # popt results per file
            perr_res = np.array(perr_res)  # popt results per file
            self.results[pickiso][sc_name] = {'gate_analysis_m': {'vals': popt_res[:, 0],
                                                                  'd_fit':perr_res[:, 0]},
                                              'gate_analysis_b': {'vals': popt_res[:, 1],
                                                                  'd_fit': perr_res[:, 1]}
                                              }

        self.plot_parameter_for_isos_and_scaler(['56Ni', '58Ni', '60Ni'], [sc_name], 'gate_analysis_m', unit='', onlyfiterrs=True)
        self.plot_parameter_for_isos_and_scaler(['56Ni', '58Ni', '60Ni'], [sc_name], 'gate_analysis_b', onlyfiterrs=True)



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
            '''SELECT file FROM Files WHERE type LIKE ? ORDER BY date ''', (type,))
        files = cur.fetchall()
        con.close()
        # convert into np array
        filelist = [f[0] for f in files]
        ret_files = []
        ret_file_nos = []
        for file in filelist:
            fileno = int(re.split('[_.]', file)[1])
            if selecttuple is not None:
                if selecttuple[0] <= fileno <= selecttuple[1]:
                    ret_files.append(file)
                    ret_file_nos.append(fileno)
            else:
                ret_files.append(file)
                ret_file_nos.append(fileno)
        return ret_files, ret_file_nos

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
        :param midtof: float: center of software gate in µs
        :param gatewidth: float: width of software gate in µs
        :param delaylist: list of floats: list of delays in midtof for scalers 0,1,2
        :return:
        '''
        midtof = self.midtof_orig[iso]+midtof_var
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

    ''' analysis related '''

    def fit_files(self, filelist):
        filearray = np.array(filelist)  # needed for batch fitting

        # do the batchfit
        BatchFit.batchFit(filearray, self.db, self.run, x_as_voltage=True, softw_gates_trs=None, save_file_as='.png')

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

    ''' visualization '''
    def plot_parameter_for_isos_and_scaler(self, isotopes, scaler_list, parameter,
                                           offset=None, overlay=None, unit='MHz', onlyfiterrs=False):
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
                x_ax = self.results[iso][x_type]
                if 'all_fitpars' in parameter:
                    # the 'all fitpars is organized a little different.
                    # For each file they are just stored as a dict like in db
                    # Parameter must be specified as 'all_fitpars:par' with par being the specific parameter to plot
                    fitres_list = self.results[iso][scaler]['all_fitpars']
                    parameter_plot = parameter.split(':')[1]
                    centers = [i[parameter_plot][0] for i in fitres_list]
                    centers_d_stat = [i[parameter_plot][1] for i in fitres_list]
                    centers_d_syst = [0 for i in fitres_list]
                    # get weighted average
                    wavg, wavg_d, fixed = self.results[iso][scaler]['avg_fitpars'][parameter_plot]
                    if fixed == True:
                        wavg_d = '-'
                    else:
                        wavg_d = '{:.0f}'.format(10 * wavg_d)  # times 10 for representation in brackets
                else:
                    centers = self.results[iso][scaler][parameter]['vals']
                    if onlyfiterrs:
                        centers_d_stat = self.results[iso][scaler][parameter]['d_fit']
                        centers_d_syst = 0
                    else:
                        centers_d_stat = self.results[iso][scaler][parameter]['d_stat']
                        centers_d_syst = self.results[iso][scaler][parameter]['d_syst']
                    # calculate weighted average:
                    valarr = np.array(centers)
                    errarr = np.array(centers_d_stat)
                    if not np.any(errarr == 0) and not np.sum(1/errarr**2) == 0:
                        weights = 1/errarr**2
                        wavg, sumw = np.average(valarr, weights=weights, returned=True)
                        wavg_d = '{:.0f}'.format(10*np.sqrt(1 / sumw))  # times 10 for representation in brackets
                    else:  # some values don't have error, just calculate mean instead of weighted avg
                        wavg = valarr.mean()
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
                elif type(offset) is list:
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
                if parameter == 'shifts_iso-58':
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
        if overlay:
            plt.plot(x_ax, overlay, color='red')
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


if __name__ == '__main__':
    analysis = NiAnalysis_softwGates()
    analysis.softw_gate_analysis()
    analysis.export_results()
