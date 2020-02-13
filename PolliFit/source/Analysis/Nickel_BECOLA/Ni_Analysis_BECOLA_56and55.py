"""
Created on 2018-12-19

@author: fsommer

Module Description:  Analysis of the Nickel Data from BECOLA taken on 13.04.-23.04.2018
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
from scipy.optimize import curve_fit

import BatchFit
import Physics
import Tools
from lxml import etree as ET
from XmlOperations import xmlWriteDict
from Analysis.Nickel_BECOLA.ExcelWrite import ExcelWriter
from Measurement.XMLImporter import XMLImporter
from KingFitter import KingFitter

class NiAnalysis():
    def __init__(self):
        logging.getLogger().setLevel(logging.INFO)
        # Set working directory and database
        ''' working directory: '''
        # get user folder to access ownCloud
        user_home_folder = os.path.expanduser("~")
        # self.workdir = 'C:\\DEVEL\\Analysis\\Ni_Analysis\\XML_Data' # old working directory
        ownCould_path = 'ownCloud\\User\\Felix\\Measurements\\Nickel_online_Becola\\Analysis\\XML_Data'
        self.workdir = os.path.join(user_home_folder, ownCould_path)
        ''' data folder '''
        self.datafolder = os.path.join(self.workdir, 'SumsRebinned')
        ''' database '''
        self.db = os.path.join(self.workdir, 'Ni_Becola.sqlite')
        Tools.add_missing_columns(self.db)
        logging.info('\n'
                     '########## BECOLA Nickel Analysis Started! ####################################################\n'
                     '## database is: {0}\n'
                     '## data folder: {1}\n'
                     .format(self.db, self.datafolder))

        # Pick isotopes and group
        self.isotopes = ['%sNi' % i for i in range(55, 60)]
        self.isotopes.remove('57Ni')
        self.isotopes.remove('59Ni')
        '''isotope groups'''
        self.odd_isotopes = [iso for iso in self.isotopes if int(iso[:2]) % 2]
        self.even_isotopes = [iso for iso in self.isotopes if int(iso[:2]) % 2 == 0]
        self.stables = ['58Ni', '60Ni', '61Ni', '62Ni', '64Ni']

        # Name this analysis run
        self.run_name = 'AsymVoigt'

        # create excel workbook to save some results
        excel_path = 'ownCloud\\User\\Felix\\Measurements\\Nickel_online_Becola\\Analysis\\Results\\analysis_advanced_rebinned.xlsx'
        self.excelpath = os.path.join(user_home_folder, excel_path)
        self.excel = ExcelWriter(self.excelpath)
        self.excel.active_sheet = self.excel.wb.copy_worksheet(self.excel.wb['Template'])
        self.excel.active_sheet.title = self.run_name

        # Select runs; Format: ['run58', 'run60', 'run56']
        # to use a different lineshape you must create a new run under runs and a new linevar under lines and link the two.
        self.runs = ['CEC_AsymVoigt', 'CEC_AsymVoigt', 'CEC_AsymVoigt', 'CEC_AsymVoigt']
        self.excel.active_sheet['B1'] = str(self.runs)

        """ 
        ### Uncertainties ###
        All uncertainties that we can quantify and might want to respect
        """
        self.wavemeter_wsu30_mhz_d = 3  # Kristians wavemeter paper


        ''' Masses '''
        # # Reference:   'The Ame2016 atomic mass evaluation: (II). Tables, graphs and references'
        # #               Chinese Physics C Vol.41, No.3 (2017) 030003
        # #               Meng Wang, G. Audi, F.G. Kondev, W.J. Huang, S. Naimi, Xing Xu
        # masses = {
        #     '55Ni': (54951330.0, 0.8),
        #     '56Ni': (55942127.9, 0.5),
        #     '57Ni': (56939791.5, 0.6),
        #     '58Ni': (57935341.8, 0.4),
        #     '59Ni': (58934345.6, 0.4),
        #     '60Ni': (59930785.3, 0.4)
        #      }
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
        nuclear_spin_and_moments = {
            '55Ni': (55, -3/2, 0.98, 0.03, 0, 0),
            '57Ni': (57, -3/2, -0.7975, 0.0014, 0, 0)
            # even isotopes 56, 58, 60 Ni have Spin 0 and since they are all even-even nucleons also the moments are zero
        }

        ''' A and B Factors '''
        # Reference:

        ''' restframe transition frequency '''
        # Reference: ??
        # NIST: observed wavelength air 352.454nm corresponds to 850586060MHz
        # upper lvl 28569.203cm-1; lower lvl 204.787cm-1
        # resulting wavenumber 28364.416cm-1 corresponds to 850343800MHz
        # KURUCZ database: 352.4535nm, 850344000MHz, 28364.424cm-1
        # Some value I used in the excel sheet: 850347590MHz Don't remember where that came from...
        self.restframe_trans_freq = 850343800
        self.excel.active_sheet['B3'] = self.restframe_trans_freq

        ''' literature value IS 60-58'''
        # Reference: ??
        # isotope shift of Nickel-60 with respect to Nickel-58 (=fNi60-fNi58)
        # Collaps 2017: 509.074(879)[7587] MHz
        # Collaps 2016: 510.7(6)[95]MHz
        # Steudel 1980: 0.01694(9) cm-1 corresponds to 507.8(27) MHz
        self.literature_IS60vs58 = 510.7
        self.literature_IS60vs58_d_stat = 0.6
        self.literature_IS60vs58_d_syst = 9.5
        self.excel.active_sheet['B4'] = self.literature_IS60vs58

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

        baret_radii_lit = {
            '58Ni': (4.8386, np.sqrt(0.0009 ** 2 + 0.0019 ** 2)),
            '60Ni': (4.8865, np.sqrt(0.0008 ** 2 + 0.002 ** 2)),
            '61Ni': (4.9005, np.sqrt(0.001 ** 2 + 0.0017 ** 2)),
            '62Ni': (4.9242, np.sqrt(0.0009 ** 2 + 0.002 ** 2)),
            '64Ni': (4.9481, np.sqrt(0.0009 ** 2 + 0.0019 ** 2))
        }

        v2_lit = {
            '58Ni': 1.283517,
            '60Ni': 1.283944,
            '61Ni': 1.283895,
            '62Ni': 1.283845,
            '64Ni': 1.284133
        }

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
                            '60Ni': 851224124.8007469   # 14196.89025
                            }

        # safe run settings to workbook
        self.excel.wb.save(self.excelpath)


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

    def separate_runs_analysis(self):
        logging.info('\n'
                     '##########################\n'
                     '# separate runs analysis #\n'
                     '##########################')
        '''
        calibrations and analysis are done separate for each scaler.
        This is supposed to reduce the systematic shifts between scalers.
        Each file is fitted separately
        '''
        # Fit again with calibrations. This time each scaler on it's own
        self.results_per_scaler = {'scaler_0': None, 'scaler_1': None, 'scaler_2': None}
        scaler56_weighted_sum = 0
        scaler56_sum_of_weights = 0
        ni56_systerrs = 0
        for scalers in range(3):
            logging.info('\n'
                         '## Analysis started for scaler number {}'
                         .format(scalers))
            # write scaler to db
            self.update_scalers_in_db(scalers)
            # Do a first set of fits for all 58 & 60 runs without any calibration applied.
            self.filelist58, self.runNos58, self.center_freqs_58, self.center_freqs_58_d, self.start_times_58 = \
                self.chooseAndFitRuns('58Ni%', '58Ni', reset=True)
            self.filelist60, self.runNos60, self.center_freqs_60, self.center_freqs_60_d, self.start_times_60 = \
                self.chooseAndFitRuns('60Ni%', '60Ni', reset=True)
            # plot results of first fit
            self.plotCenterFrequencies58and60()
            # do voltage calibration with these calibration pairs.
            self.calib_tuples = [(6191, 6192), (6207, 6208), (6224, 6225), (6231, 6233), (6232, 6233), (6242, 6243),
                                 (6253, 6254), (6258, 6259), (6269, 6270), (6284, 6285), (6294, 6295), (6301, 6302),
                                 (6310, 6311), (6323, 6324), (6340, 6342), (6362, 6363), (6395, 6396),
                                 (6418, 6419), (6467, 6466), (6501, 6502)]
            self.calib_tuples = self.calibrateVoltage(self.calib_tuples)
            # now tuples contain (58ref, 60ref, isoshift, isoshift_d, calVolt, calVoltStatErr, calVoltSystErr)

            # re-fit 58 and 60 Nickel runs for that scaler
            self.filelist58, self.runNos58, self.center_freqs_58, self.center_freqs_58_d, self.start_times_58 = \
                self.chooseAndFitRuns('58Ni%', '58Ni', reset=False)
            self.filelist60, self.runNos60, self.center_freqs_60, self.center_freqs_60_d, self.start_times_60 = \
                self.chooseAndFitRuns('60Ni%', '60Ni', reset=False)
            self.write_second_fit_to_excel(scalers)

            # Todo: Plot calibrations and all other runs on a time axis to assign calibrations!
            # Should use self.calib_tuples to get 58 and 60 reference files
            self.assign_calibrations()

            # fit 56 nickel runs and calculate Isotop shift
            ni56_point_runNos, ni56_center, ni56_center_d, ni56_isoShift_yData, ni56_isoShift_yData_d, w_avg_56isoshift, \
            ni56_isoShift_systErr = self.do_56_Analysis(scalers)

            # add results to weighted avg
            weight56 = 1/np.square(w_avg_56isoshift[1])
            scaler56_weighted_sum += weight56 * w_avg_56isoshift[0]
            self.scaler56_sum_of_weights += weight56
            self.ni56_systerrs += 1/np.square(ni56_isoShift_systErr)
            logging.info('\n'
                         '########## Separate Runs Analysis #########################################################\n'
                         '## results for scaler_{}: shift {}MHz+-{}MHz statistic +-{}MHz systematic'.
                         format(scalers, w_avg_56isoshift[0], w_avg_56isoshift[1], ni56_isoShift_systErr))

            # write results to dict (used for plotting)
            scaler_name ='scaler_' + str(scalers)
            self.results_per_scaler[scaler_name] = {'runNumbers_58': self.runNos58,
                                                    'center_freqs_58': self.center_freqs_58,
                                                    'center_freqs_58_d': self.center_freqs_58_d,
                                                    'runNumbers_60': self.runNos60,
                                                    'center_freqs_60': self.center_freqs_60,
                                                    'center_freqs_60_d': self.center_freqs_60_d,
                                                    'runNumbers_56': ni56_point_runNos,
                                                    'center_freqs_56': ni56_center,
                                                    'center_freqs_56_d': ni56_center_d,
                                                    'isoShift_56-58': ni56_isoShift_yData,
                                                    'isoShift_56-58_d': ni56_isoShift_yData_d,
                                                    'isoShift_56-58_avg': w_avg_56isoshift[0],
                                                    'isoShift_56-58_avg_d': w_avg_56isoshift[1],
                                                    'isoShift_56-58_systErr': ni56_isoShift_systErr}
        # combine all 3 scalers to final result:
        self.ni56_isoShift_final = scaler56_weighted_sum/scaler56_sum_of_weights
        self.ni56_isoShift_final_d = np.sqrt(1/scaler56_sum_of_weights)
        self.ni56_systerrs = np.sqrt(1/ni56_systerrs)
        ni56res_message = 'Isotope shift 56-58 combined: {0:.2f}({1:.0f})[{2:.0f}]MHz'\
            .format(self.ni56_isoShift_final, 100*self.ni56_isoShift_final_d, 100*ni56_systerrs)
        print(ni56res_message)
        # write final result to database
        self.write_shift_to_combined_db('56Ni', self.runs[0],
                                        (self.ni56_isoShift_final, self.ni56_isoShift_final_d, ni56_systerrs),
                                        'BECOLA 2018; 3 scalers separate; calibrated; ni56 runs: {}'
                                        .format(self.results_per_scaler['scaler_0']['runNumbers_56']))
        self.plot_56_results(ni56res_message)

    def stacked_runs_analysis(self):
        #########################
        # stacked runs analysis #
        #########################
        '''
        All valid runs of one isotope are stacked/rebinned to a new single file.
        Calibrations and analysis are done based on stacked files and combined scalers.
        This enables us to get results out of 55 Nickel data.
        '''
        # combine runs to new 3-scaler files.
        self.ni55analysis_combined_files = []
        self.create_stacked_files()

        # update database to use all three scalers for analysis
        self.update_scalers_in_db('[0,1,2]')  # scalers to be used for combined analysis

        # do a batchfit of the newly created 58 & 60 files
        BatchFit.batchFit(self.ni55analysis_combined_files[:2], self.db, self.runs[0], x_as_voltage=True,
                          softw_gates_trs=None,
                          save_file_as='.png')
        # extract isotope shift
        center58 = self.extract_center_from_fitres_db(self.ni55analysis_combined_files[0], '58Ni_sum', self.runs[0])
        center60 = self.extract_center_from_fitres_db(self.ni55analysis_combined_files[1], '60Ni_sum', self.runs[0])
        self.isoShift60 = center60[0] - center58[0]
        self.isoShift60_d_stat = np.sqrt(center60[1] ** 2 + center58[1] ** 2)

        # new calibration:
        diff_Doppler_58 = Physics.diffDoppler(self.restframe_trans_freq, 29850, 58)
        diff_Doppler_60 = Physics.diffDoppler(self.restframe_trans_freq, 29850, 60)
        # calculate calibration Voltage
        calibrated_voltage = (self.isoShift60 - self.literature_IS60vs58) / (diff_Doppler_60 - diff_Doppler_58) + 29850
        # update voltage in db
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''UPDATE Files SET accVolt = ?, type = ? WHERE file = ? ''',
                    (calibrated_voltage, '58Ni_sum_cal', self.ni55analysis_combined_files[0]))
        cur.execute('''UPDATE Files SET accVolt = ?, type = ? WHERE file = ? ''',
                    (calibrated_voltage, '60Ni_sum_cal', self.ni55analysis_combined_files[1]))
        cur.execute('''UPDATE Files SET accVolt = ?, type = ? WHERE file = ? ''',
                    (calibrated_voltage, '56Ni_sum_cal', self.ni55analysis_combined_files[2]))
        cur.execute('''UPDATE Files SET accVolt = ?, type = ? WHERE file = ? ''',
                    (calibrated_voltage, '55Ni_sum_cal', self.ni55analysis_combined_files[3]))
        con.commit()
        con.close()

        # create new isotopes in database
        self.create_new_isotope_in_db('60Ni_sum', '60Ni_sum_cal', calibrated_voltage)
        self.create_new_isotope_in_db('58Ni_sum', '58Ni_sum_cal', calibrated_voltage)
        self.create_new_isotope_in_db('56Ni_sum', '56Ni_sum_cal', calibrated_voltage)
        self.create_new_isotope_in_db('55Ni_sum', '55Ni_sum_cal', calibrated_voltage)


        # do a batchfit of the newly created 58 & 60 files
        BatchFit.batchFit(self.ni55analysis_combined_files, self.db, self.runs[0], x_as_voltage=True,
                          softw_gates_trs=None,
                          save_file_as='.png')
        # extract isotope shift
        center60 = self.extract_center_from_fitres_db(self.ni55analysis_combined_files[1], '60Ni_sum_cal', self.runs[0])
        center58 = self.extract_center_from_fitres_db(self.ni55analysis_combined_files[0], '58Ni_sum_cal', self.runs[0])
        center56 = self.extract_center_from_fitres_db(self.ni55analysis_combined_files[2], '56Ni_sum_cal', self.runs[0])
        center55 = self.extract_center_from_fitres_db(self.ni55analysis_combined_files[3], '55Ni_sum_cal', self.runs[0])
        self.isoShift56 = center56[0] - center58[0]
        self.isoShift56_d_stat = np.sqrt(center56[1] ** 2 + center58[1] ** 2)
        self.isoShift55 = center55[0] - center58[0]
        self.isoShift55_d_stat = np.sqrt(center55[1] ** 2 + center58[1] ** 2)

        # write final result to database
        self.write_shift_to_combined_db('56Ni_sum_cal', self.runs[0],
                                        (self.isoShift56, self.isoShift56_d_stat, 0),
                                        'BECOLA 2018; 3 scalers combined; calibrated; ni56 runs summed to file: {}'
                                        .format(self.ni55analysis_combined_files[2]))
        self.write_shift_to_combined_db('55Ni_sum_cal', self.runs[0],
                                        (self.isoShift55, self.isoShift55_d_stat, 0),
                                        'BECOLA 2018; 3 scalers combined; calibrated; ni56 runs summed to file: {}'
                                        .format(self.ni55analysis_combined_files[3]))

        ####################
        # Do the king plot #
        ####################
        self.perform_king_fit_analysis(self.delta_lit_radii_58,
                                       isotopes=['55Ni_sum_cal', '56Ni_sum_cal', '56Ni', '58Ni', '59Ni', '60Ni', '61Ni',
                                                 '62Ni', '64Ni'],
                                       reference_run='CEC_AsymVoigt')

    ''' db related '''

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
        center_new = center_old + (29850 - acc_volt) * Physics.diffDoppler(self.restframe_trans_freq, acc_volt, mass)
        isopars_lst = list(copy_isopars[0])  # change into list to replace some values
        isopars_lst[0] = iso_new
        isopars_lst[4] = center_new
        isopars_new = tuple(isopars_lst)
        cur.execute('''INSERT OR REPLACE INTO Isotopes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                    isopars_new)
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
            '''SELECT file FROM Files WHERE type LIKE ? ''', (type,))
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

    def update_scalers_in_db(self, scalers):
        '''
        Update the scaler parameter for all runs in the runs database
        :param scalers: int or str: either an int (0,1,2) if a single scaler is used or a string '[0,1,2]' for all
        :return:
        '''
        if type(scalers) is int:
            scaler_string = str(scalers).join(('[',']'))
        else:
            scaler_string = scalers
        self.scalers = scaler_string
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''UPDATE Runs SET Scaler = ?''', (scaler_string,))
        con.commit()
        con.close()

    def extract_center_from_fitres_db(self, file, isostring, run):
        """
        Gets the 'center' fit parameter for a given fit result (file, isotope and run)
        :param file: str: filename
        :param isostring: str: isotope name
        :param run: str: runname
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
        center = parsdict['center']  # tuple of (center frequency, Uncertainty, Fixed)

        return center

    ''' preparation and calibration '''

    def chooseAndFitRuns(self, db_like, db_type, reset=False):
        '''

        :param db_like: str: example '58Ni%'
        :param db_type: str: example '58Ni'
        :param reset: bool: defines whether the voltage and type are to be reset before fitting
        :return: filelist
        '''

        ''' Calibration runs '''
        # Pick all 58Ni and 60Ni runs and fit.
        # Basically this should do the same as batchfitting all runs in PolliFit
        # So I will start by stealing the code from there and in a later version I might adapt the code for my analysis
        """
        :params:    fileList: ndarray of str: names of files to be analyzed e.g. ['BECOLA_123.xml' 'BECAOLA_234.xml']
                    self.db: str: path to database e.g. 'C:/DEVEL/Analysis/Ni_Analysis/XML_Data/Ni_Becola.sqlite'
                    run: str: run as specified in database e.g.: 'CEC_AsymVoigt_60'
                    x_as_voltage: bool: is unit of x-axis volts? e.g. True
                    softw_gates_trs: None
                    save_file_as: str: file format for saving results e.g. '.png'
        :return: list, (shifts, shiftErrors, shifts_weighted_mean, statErr, systErr, rChi)
        """
        ###################
        # select files
        filelist, runNos = self.pick_files_from_db_by_type_and_num(db_like)
        filearray = np.array(filelist)

        if reset:
            # Reset all calibration information so that pre-calib information can be extracted.
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            # Set type in files back to bare isotopes (56Ni, 58Ni, 60Ni)
            # Set accVolt in files back to nominal 29850
            cur.execute('''UPDATE Files SET accVolt = ?, type = ? WHERE type LIKE ? ''', (29850, db_type, db_like))
            con.commit()
            con.close()

        # see what run to use
        if '58' in db_like:
            run = 0
        elif '60' in db_like:
            run = 1
        else:
            run = 2
        # do the batchfit
        #BatchFit.batchFit(filearray, self.db, self.runs[run], x_as_voltage=True, softw_gates_trs=None, save_file_as='.png')
        # get fitresults (center) vs run for 58
        all_rundate = []
        all_center_MHz = []
        all_center_MHz_d = []
        # get fit results
        for files in filelist:
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            # Get corresponding isotope
            cur.execute(
                '''SELECT date, type FROM Files WHERE file = ? ''', (files,))
            filefetch = cur.fetchall()
            iso_type = filefetch[0][1]
            file_date = filefetch[0][0]
            # Query fitresults for file and isotope combo
            cur.execute(
                '''SELECT pars FROM FitRes WHERE file = ? AND iso = ? AND run = ?''', (files, iso_type, self.runs[run]))
            pars = cur.fetchall()
            con.close()
            try:
                # if the fit went wrong there might not be a value to get from the fitpars...
                parsdict = ast.literal_eval(pars[0][0])
            except Exception as e:
                # replace with standard value and big error...
                parsdict = {'center': (-510, 30, False)}  # TODO: use better dummy value (take from all_Center_MHz list)
            all_rundate.append(file_date)
            all_center_MHz.append(parsdict['center'][0])
            all_center_MHz_d.append(parsdict['center'][1])

        return filelist, runNos, all_center_MHz, all_center_MHz_d, all_rundate

    def plotCenterFrequencies58and60(self):
        # plot center frequency in MHz for all 58,60Ni runs:
        plt.plot(self.runNos60, np.array(self.center_freqs_60) - 510, '--o', color='red', label='60Ni - 510MHz')
        plt.fill_between(self.runNos60,
                         np.array(self.center_freqs_60) - 510 - self.center_freqs_60_d,
                         np.array(self.center_freqs_60) - 510 + self.center_freqs_60_d,
                         alpha=0.5, edgecolor='red', facecolor='red')
        plt.plot(self.runNos58, self.center_freqs_58, '--o', color='blue', label='58Ni')
        plt.fill_between(self.runNos58,
                         np.array(self.center_freqs_58) - self.center_freqs_58_d,
                         np.array(self.center_freqs_58) + self.center_freqs_58_d,
                         alpha=0.5, edgecolor='blue', facecolor='blue')
        plt.title('Center Frequency FitPar in MHz for all 58,60 Ni Runs')
        plt.xlabel('run numbers')
        plt.ylabel('center fit parameter [MHz]')
        plt.legend(loc='best')
        plt.show()

    def calibrateVoltage(self, calibration_tuples):
        '''

        :param calibration_tuples:
        :return: calib_tuples_with_isoshift_and_calibrationvoltage:
                contains a tuple for each calibration point with entries: (58ref, 60ref, isoshift, calVolt, calVoltErr)
        '''
        #######################
        # Calibration process #
        #######################

        # Calibration sets of 58/60Ni
        calib_tuples = [(6191, 6192), (6207, 6208), (6224, 6225), (6232, 6233), (6242, 6243), (6253, 6254), (6258, 6259),
                        (6269, 6270), (6284, 6285), (6294, 6295), (6301, 6302), (6310, 6311), (6313, 6312), (6323, 6324),
                        (6340, 6342), (6356, 6357), (6362, 6363), (6395, 6396), (6417, 6419), (6467, 6466), (6501, 6502)]
        calib_tuples = calibration_tuples
        calib_tuples_with_isoshift = []

        # Calculate Isotope shift for all calibration tuples and add to list.
        # also write calibration tuple data to excel
        self.excel.last_row = 9  # return to start
        for tuples in calib_tuples:
            # Get 58Nickel center fit parameter in MHz
            run58 = tuples[0]
            run58file = 'BECOLA_'+str(run58)+'.xml'
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            # Get corresponding isotope
            cur.execute(
                '''SELECT type FROM Files WHERE file = ? ''', (run58file,))
            iso_type58 = cur.fetchall()[0][0]
            # Query fitresults for file and isotope combo
            cur.execute(
                '''SELECT pars FROM FitRes WHERE file = ? AND iso = ? AND run = ?''', (run58file, iso_type58, self.runs[0]))
            pars58 = cur.fetchall()
            con.close()
            pars58dict = ast.literal_eval(pars58[0][0])
            center58 = pars58dict['center']
            if 'Asym' in self.runs[0]:
                centerAsym58 = pars58dict['centerAsym']
                IntAsym58 = pars58dict['IntAsym']

            # Get 60Nickel center fit parameter in MHz
            run60 = tuples[1]
            run60file = 'BECOLA_' + str(run60) + '.xml'
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            # Get corresponding isotope
            cur.execute(
                '''SELECT type FROM Files WHERE file = ? ''', (run60file,))
            iso_type60 = cur.fetchall()[0][0]
            # Query fitresults for file and isotope combo
            cur.execute(
                '''SELECT pars FROM FitRes WHERE file = ? AND iso = ? AND run = ?''', (run60file, iso_type60, self.runs[1]))
            pars60 = cur.fetchall()
            con.close()
            pars60dict = ast.literal_eval(pars60[0][0])
            center60 = pars60dict['center']
            if 'Asym' in self.runs[1]:
                centerAsym60 = pars60dict['centerAsym']
                IntAsym60 = pars60dict['IntAsym']

            # Calculate isotope shift of 60Ni with respect to 58Ni for this calibration point
            isoShift = center60[0]-center58[0]
            isoShift_d = np.sqrt(center60[1]**2+center58[1]**2)  # statistical uncertainty from fits
            #print('Isotope shift for calibration point with runs {} and {}: {}MHz'.format(tuples[0], tuples[1], isoShift, isoShift_d))
            tuple_with_isoshift = tuples + (isoShift, isoShift_d)
            calib_tuples_with_isoshift.append(tuple_with_isoshift)

            # write calibration tuple info to workbook
            self.excel.active_sheet.cell(row=self.excel.last_row, column=1, value=str(tuples))
            self.excel.active_sheet.cell(row=self.excel.last_row, column=2, value=run58)
            self.excel.active_sheet.cell(row=self.excel.last_row, column=3, value=run60)
            stcol = int(list(self.scalers)[1])*2+4  # column in which to write center fit positions
            self.excel.active_sheet.cell(row=self.excel.last_row, column=stcol, value=center58[0])
            self.excel.active_sheet.cell(row=self.excel.last_row, column=stcol+1, value=center60[0])
            self.excel.last_row += 1
        self.excel.wb.save(self.excelpath)

        # plot isotope shift for all calibration points (can be removed later on):
        calib_isoShift_yData = []
        calib_isoShift_yData_d = []
        calib_point_runNos = []
        for tuples in calib_tuples_with_isoshift:
            calib_isoShift_yData.append(tuples[2])
            calib_isoShift_yData_d.append(tuples[3])
            calib_point_name = str(tuples[0])+'/'+str(tuples[1])
            calib_point_runNos.append(calib_point_name)

        plt.errorbar(range(len(calib_isoShift_yData)), calib_isoShift_yData, yerr=calib_isoShift_yData_d)
        plt.xticks(range(len(calib_isoShift_yData)), calib_point_runNos, rotation=-30)
        plt.title('Isotope Shift for all Calibration Points')
        plt.xlabel('Run Numbers of Calibration Pairs')
        plt.ylabel('Isotope Shift 60-58 Ni [MHz]')
        plt.show()

        # Calculate resonance DAC Voltage from the 'center' positions
        calib_tuples_with_isoshift_and_calibrationvoltage = []
        average_calib_voltage = 0
        # Do the voltage calibration for each tuple.
        self.excel.last_row = 9  # return to start
        for tuples in calib_tuples_with_isoshift:
            # get filenames
            run58, run60, isoShift, isoShift_d = tuples
            run58file = 'BECOLA_' + str(run58) + '.xml'
            run60file = 'BECOLA_' + str(run60) + '.xml'
            calib_point_dict = {run58file: {},
                                run60file: {}}

            # calculate centerDAC and get some usefull info
            for files, dicts in calib_point_dict.items(): # only 2 elements to iterate: 58 and 60
                con = sqlite3.connect(self.db)
                cur = con.cursor()
                # get laser frequency and accelVolt
                cur.execute(
                    '''SELECT type, accVolt, laserFreq, colDirTrue FROM Files WHERE file = ? ''', (files,))
                iso, accVolt, laserFreq, colDirTrue = cur.fetchall()[0]
                # Query fitresults for file and isotope combo
                if files is run58file:
                    # check whether it's a 58 or 60 file to choose the correct run
                    run = self.runs[0]
                else:
                    run = self.runs[1]
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
                centerDAC = centerE - dicts['accVolt']

                dicts['centerDAC'] = centerDAC

            # alternative calibration process
            accVolt = calib_point_dict[run58file]['accVolt']  # should be the same for 58 and 60
            # Calculate differential Doppler shift for 58 and 60 nickel
            # TODO: uncertainties on this? Well, the voltage is quite uncertain, but the effect should be minimal
            diff_Doppler_58 = Physics.diffDoppler(self.restframe_trans_freq, accVolt, 58)
            diff_Doppler_60 = Physics.diffDoppler(self.restframe_trans_freq, accVolt, 60)
            diff_Doppler_56 = Physics.diffDoppler(self.restframe_trans_freq, accVolt, 56)
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
            f_reso58_d = Physics.relDoppler(self.wavemeter_wsu30_mhz_d, velo58)
            f_reso60 = Physics.relDoppler(calib_point_dict[run60file]['laserFreq'], velo60)
            f_reso60_d = Physics.relDoppler(self.wavemeter_wsu30_mhz_d, velo60)
            # isotope shift from calibration tuple:
            isoShift_calibtuple = isoShift  # Is the same as calculated below.
            isoShift_d_stat = isoShift_d  # TODO: This should be the statistical part of the uncertainty, coming from the center fit uncertainties
            # calculate isotope shift
            isoShift = f_reso60 - f_reso58
            isoShift_d_syst = np.sqrt(np.square(f_reso58_d) + np.square(f_reso60_d))  # TODO: This is a systematic uncertainty from the wavemeter uncertainty

            # res_58 = calib_point_dict[run58file]['resonanceFreq']
            # res_60 = calib_point_dict[run60file]['resonanceFreq']
            # IS_meas = res_60 - res_58

            # calculate calibration Voltage
            calibrated_voltage = (isoShift-self.literature_IS60vs58)/(diff_Doppler_60-diff_Doppler_58)+accVolt
            # TODO: Uncertainties are now split into systematic and statistic. Use accordingly!
            calibrated_voltage_d_stat = np.sqrt(np.square(self.literature_IS60vs58_d_stat / (diff_Doppler_60 - diff_Doppler_58)) +
                                           np.square(isoShift_d_stat / (diff_Doppler_60 - diff_Doppler_58)))
            # TODO: For now I'm working with the statistical uncertainty only. But I need to add up the systematics as well.
            calibrated_voltage_d_syst = np.sqrt(np.square(self.literature_IS60vs58_d_syst /(diff_Doppler_60-diff_Doppler_58)) +
                                           np.square(isoShift_d_syst / (diff_Doppler_60 - diff_Doppler_58)))


            # # do voltage calibration to literature IS
            # accVolt = calib_point_dict[run58file]['accVolt']  # should be the same for 58 and 60
            # voltage_list = np.arange(accVolt-100, accVolt+100)
            # IS_perVolt_list = np.zeros(0) # isotope shift for an assumed voltage from voltage list
            # IS_d_perVolt_list = np.zeros(0)  # isotope shift error for an assumed voltage from voltage list
            # for volt in voltage_list:
            #     # calculate velocity for 58 and 60
            #     velo58sign = -1 if calib_point_dict[run58file]['colDirTrue'] else 1
            #     velo58 = velo58sign * Physics.relVelocity((volt + calib_point_dict[run58file]['centerDAC'])*Physics.qe,
            #                                  calib_point_dict[run58file]['isoMass']*Physics.u)
            #     velo60sign = -1 if calib_point_dict[run60file]['colDirTrue'] else 1
            #     velo60 = velo60sign * Physics.relVelocity((volt + calib_point_dict[run60file]['centerDAC']) * Physics.qe,
            #                                  calib_point_dict[run60file]['isoMass']*Physics.u)
            #     # calculate resonance frequency for 58 and 60
            #     f_reso58 = Physics.relDoppler(calib_point_dict[run58file]['laserFreq'], velo58)
            #     f_reso58_d = Physics.relDoppler(self.wavemeter_wsu30_mhz_d, velo58)
            #     f_reso60 = Physics.relDoppler(calib_point_dict[run60file]['laserFreq'], velo60)
            #     f_reso60_d = Physics.relDoppler(self.wavemeter_wsu30_mhz_d, velo60)
            #     # calculate isotope shift
            #     isoShift = f_reso60 - f_reso58
            #     isoShift_d = np.sqrt(np.square(f_reso58_d) + np.square(f_reso60_d))
            #     # append to other values
            #     IS_perVolt_list = np.append(IS_perVolt_list, np.array(isoShift))
            #     IS_d_perVolt_list = np.append(IS_d_perVolt_list, np.array(isoShift_d))
            #
            # # calibrate voltage by fitting line to plot
            # IS_perVolt_list -= self.literature_IS60vs58
            # fitpars0 = np.array([0.0, 0.0])
            # def linfunc(x, m, b):
            #     return m*(x-29850)+b
            # popt, pcov = curve_fit(linfunc, voltage_list, IS_perVolt_list, fitpars0, sigma=IS_d_perVolt_list, absolute_sigma=True)
            # m, b = popt
            # m_d, b_d = np.sqrt(np.diag(pcov))
            # calibrated_voltage = -b/m+29850
            # # Actually, the error for b should be the one resulting from the wavemeter uncertainty. So isoShift_d.
            # # Not the fit uncertainty. Because the fit uncertainty is dependent on whether I use an offset in the voltage
            # # (to bring the b value close to zero, to the data). If there is an uncertainty in the IS from the wavemeter,
            # # it will be the same for each assumed voltage and thus shift the whole line keeping the slope.
            # b_d = IS_d_perVolt_list[0]  # these should all be same size
            # calibrated_voltage_d = np.sqrt(np.square(self.literature_IS60vs58_d/m)+np.square(b_d/m)+np.square(b/np.square(m)*m_d))
            # print(calibrated_voltage_d)

            # create a new tuple with (58ref, 60ref, isoshift, calVolt, calVoltErr)
            tuple_withcalibvolt = tuples + (calibrated_voltage, calibrated_voltage_d_stat, calibrated_voltage_d_syst)
            # contains a tuple for each calibration point with entries: (58ref, 60ref, isoshift, isoshift_d, calVolt, calVoltStatErr, calVoltSystErr)
            calib_tuples_with_isoshift_and_calibrationvoltage.append(tuple_withcalibvolt)

            average_calib_voltage += calibrated_voltage
            #print(calibrated_voltage)

            # display calibration graph
            #plt.plot(voltage_list, IS_perVolt_list)
            #plt.scatter(calibrated_voltage, m * calibrated_voltage + b)
            #plt.title('Voltage Calibration for Calibration Tuple [Ni58:{}/Ni60:{}]'.format(run58, run60))
            #plt.xlabel('voltage [V]')
            #plt.ylabel('isotope shift [MHz]')
            #plt.show()

            # write calibration voltage to workbook
            if self.scalers == '[0]':
                col = 11
            elif self.scalers == '[1]':
                col = 13
            elif self.scalers == '[2]':
                col = 15
            else:
                col = 17
            self.excel.active_sheet.cell(row=self.excel.last_row, column=col, value=calibrated_voltage)
            self.excel.active_sheet.cell(row=self.excel.last_row, column=col+1, value=calibrated_voltage_d_stat)
            self.excel.last_row += 1
        self.excel.wb.save(self.excelpath)
        average_calib_voltage = average_calib_voltage/len(calib_tuples)

        # display all voltage calibrations
        # plot isotope shift for all calibration points (can be removed later on):
        calib_voltages = []
        calib_voltages_d = []
        calib_point_runNos = []
        for tuples in calib_tuples_with_isoshift_and_calibrationvoltage:
            calib_voltages.append(tuples[4])
            calib_voltages_d.append(tuples[5])
            calib_point_name = str(tuples[0])+'/'+str(tuples[1])
            calib_point_runNos.append(calib_point_name)

        print(calib_voltages)
        print(calib_voltages_d)
        plt.errorbar(range(len(calib_voltages)), calib_voltages, yerr=calib_voltages_d)
        # plt.plot(range(len(calib_voltages)), calib_voltages, '-o')
        plt.plot(range(len(calib_voltages)), [29850]*len(calib_voltages), '-o', color='red')
        plt.ylim(bottom=29840)
        plt.xticks(range(len(calib_voltages)), calib_point_runNos, rotation=-30)
        plt.title('Calibrated Voltage for all Calibration Tuples')
        plt.xlabel('Run Numbers of Calibration Pairs')
        plt.ylabel('Voltage [V]')
        plt.show()

        # Write calibrations to XML database
        print('Updating self.db with new voltages now...')
        for entries in calib_tuples_with_isoshift_and_calibrationvoltage:
            calibration_name = str(entries[0]) + 'w' +  str(entries[1])
            file58 = 'BECOLA_' + str(entries[0]) + '.xml'
            file58_newType = '58Ni_cal' + calibration_name
            file60 = 'BECOLA_' + str(entries[1]) + '.xml'
            file60_newType = '60Ni_cal' + calibration_name
            new_voltage = entries[4]

            # Update 'Files' in self.db
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            cur.execute('''UPDATE Files SET accVolt = ?, type = ? WHERE file = ? ''', (new_voltage, file58_newType, file58))
            cur.execute('''UPDATE Files SET accVolt = ?, type = ? WHERE file = ? ''', (new_voltage, file60_newType, file60))
            con.commit()
            con.close()

            # Calculate differential Doppler shift for re-assigning center fit pars
            diff_Doppler_58 = Physics.diffDoppler(self.restframe_trans_freq, new_voltage, 58)
            diff_Doppler_60 = Physics.diffDoppler(self.restframe_trans_freq, new_voltage, 60)
            # Create new isotopes in self.db
            self.create_new_isotope_in_db('58Ni', file58_newType, new_voltage)
            self.create_new_isotope_in_db('60Ni', file60_newType, new_voltage)
        print('...self.db update completed!')

        return calib_tuples_with_isoshift_and_calibrationvoltage

    def write_second_fit_to_excel(self, scaler):
        # Calculate Isotope shift for all calibration tuples and add to list.
        # also write calibration tuple data to excel

        # determine starting column in excel depending on scaler: 12,18,24 for scalers 0,1,2
        start_col = 18+scaler*5

        self.excel.last_row = 9  # return to start
        for tuples in self.calib_tuples:
            # Get 58Nickel center fit parameter in MHz
            run58 = tuples[0]
            run58file = 'BECOLA_'+str(run58)+'.xml'
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            # Get corresponding isotope
            cur.execute(
                '''SELECT type FROM Files WHERE file = ? ''', (run58file,))
            iso_type58 = cur.fetchall()[0][0]
            # Query fitresults for file and isotope combo
            cur.execute(
                '''SELECT pars FROM FitRes WHERE file = ? AND iso = ? AND  run = ?''', (run58file, iso_type58, self.runs[0]))
            pars58 = cur.fetchall()
            con.close()
            pars58dict = ast.literal_eval(pars58[0][0])
            center58 = pars58dict['center']

            # Get 60Nickel center fit parameter in MHz
            run60 = tuples[1]
            run60file = 'BECOLA_' + str(run60) + '.xml'
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            # Get corresponding isotope
            cur.execute(
                '''SELECT type FROM Files WHERE file = ? ''', (run60file,))
            iso_type60 = cur.fetchall()[0][0]
            # Query fitresults for file and isotope combo
            cur.execute(
                '''SELECT pars FROM FitRes WHERE file = ? AND iso = ? AND run = ?''', (run60file, iso_type60, self.runs[1]))
            pars60 = cur.fetchall()
            con.close()
            pars60dict = ast.literal_eval(pars60[0][0])
            center60 = pars60dict['center']

            # Calculate isotope shift of 60Ni with respect to 58Ni for this calibration point
            isoShift = center60[0]-center58[0]

            # write calibration tuple info to workbook
            self.excel.active_sheet.cell(row=self.excel.last_row, column=start_col, value=center58[0])
            self.excel.active_sheet.cell(row=self.excel.last_row, column=start_col+1, value=center58[1])
            self.excel.active_sheet.cell(row=self.excel.last_row, column=start_col+2, value=center60[0])
            self.excel.active_sheet.cell(row=self.excel.last_row, column=start_col+3, value=center60[1])
            self.excel.active_sheet.cell(row=self.excel.last_row, column=start_col+4, value=isoShift)
            self.excel.last_row += 1
        self.excel.wb.save(self.excelpath)

    def assign_calibrations(self):
        '''
        For now this just plots all calibration pairs on a time-axis with the other files so they can be handassigned.
        :return:
        '''
        calib58_runs = []
        calib58_dates = []
        calib60_runs = []
        calib60_dates = []
        calib_volts = []
        calib_volts_d = []
        file56_runs = []
        file56_dates = []
        file55_runs = []
        file55_dates = []
        for tuples in self.calib_tuples:
            runNo58ref, runNo60ref, isoshift, isoShift_d, calVolt, calVoltStatErr, calVoltSystErr = tuples
            calib58_runs.append(runNo58ref)
            calib60_runs.append(runNo60ref)
            calib_volts.append(calVolt)
            calib_volts_d.append(calVoltStatErr)
            # get times from database
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            cur.execute(
                '''SELECT date FROM Files WHERE file LIKE ?''', ('%' + str(runNo58ref) + '%',))
            fetch = cur.fetchall()
            run58_date = fetch[0][0]
            run58_date = datetime.strptime(run58_date,'%Y-%m-%d %H:%M:%S')
            cur.execute(
                '''SELECT date FROM Files WHERE file LIKE ?''', ('%' + str(runNo60ref) + '%',))
            fetch = cur.fetchall()
            run60_date = fetch[0][0]
            run60_date = datetime.strptime(run60_date, '%Y-%m-%d %H:%M:%S')
            con.close()
            # write times to lists
            calib58_dates.append(run58_date)
            calib60_dates.append(run60_date)

        # select Ni56 files
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute(
            '''SELECT file FROM Files WHERE type LIKE '56Ni%' ''')
        files = cur.fetchall()
        con.close()
        # convert into np array
        self.filelist56 = [f[0] for f in files]
        self.runNos56 = []
        for files in self.filelist56:
            file_no = int(re.split('[_.]', files)[1])
            self.runNos56.append(file_no)
        for run56No in self.runNos56:
            file56_runs.append(run56No)
            # get times from database
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            cur.execute(
                '''SELECT date FROM Files WHERE file LIKE ?''', ('%'+ str(run56No) + '%',))
            run56_date = cur.fetchall()[0][0]
            run56_date = datetime.strptime(run56_date,  '%Y-%m-%d %H:%M:%S')
            con.close()
            # write times to lists
            file56_dates.append(run56_date)

        # select Ni55 files
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute(
            '''SELECT file FROM Files WHERE type LIKE '55Ni%' ''')
        files = cur.fetchall()
        con.close()
        # convert into np array
        self.filelist55 = [f[0] for f in files]
        self.runNos55 = []
        for files in self.filelist55:
            file_no = int(re.split('[_.]', files)[1])
            self.runNos55.append(file_no)
        for run55No in self.runNos55:
            file55_runs.append(run55No)
            # get times from database
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            cur.execute(
                '''SELECT date FROM Files WHERE file LIKE ?''', ('%' + str(run55No) + '%',))
            run55_date = cur.fetchall()[0][0]
            run55_date = datetime.strptime(run55_date, '%Y-%m-%d %H:%M:%S')
            con.close()
            # write times to lists
            file55_dates.append(run55_date)

        run55_mpdate = mpdate.date2num(file55_dates)
        run56_mpdate = mpdate.date2num(file56_dates)
        run58_mpdate = mpdate.date2num(calib58_dates)
        run60_mpdate = mpdate.date2num(calib60_dates)

        plt.plot_date(run60_mpdate, calib60_runs, 'bo')
        plt.plot_date(run58_mpdate, calib58_runs, 'co')
        plt.plot_date(run56_mpdate, file56_runs, 'ro')
        plt.plot_date(run55_mpdate, file55_runs, 'mo')
        plt.show()


        fig, ax = plt.subplots()
        plt.errorbar(calib58_dates, calib_volts, yerr=calib_volts_d)
        days_fmt = mpdate.DateFormatter('%d.%B-%H:%M')
        ax.xaxis.set_major_formatter(days_fmt)
        plt.xticks(rotation=90)
        plt.show()

    ''' Nickel 56 related functions: '''

    def do_56_Analysis(self, scaler):
        #################
        # Ni56 Analysis #
        #################

        # select Ni56 files
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute(
            '''SELECT file FROM Files WHERE type LIKE '56Ni%' ''')
        files = cur.fetchall()
        con.close()
        # convert into np array
        self.filelist56 = [f[0] for f in files]
        filearray56 = np.array(self.filelist56)

        # attach the Ni56 runs to some calibration point(s) and adjust voltage plus create new isotope with adjusted center
        files56_withReference_tuples = []  # tuples of (56file, (58reference, 60reference))
        # hand-assigned calibration runs
        files56_withReference_tuples_handassigned = [(6199, (6191, 6192)), (6202, (6191, 6192)), (6203, (6191, 6192)), (6204, (6191, 6192)),
                                                     (6211, (6224, 6225)), (6213, (6224, 6225)), (6214, (6224, 6225)),
                                                     (6238, (6242, 6243)), (6239, (6242, 6243)), (6240, (6242, 6243)),
                                                     (6251, (6253, 6254)), (6252, (6253, 6254))]
        files56_withReference_tuples_handassigned_V2 = [(6202, (6207, 6208)), (6203, (6207, 6208)), (6204, (6207, 6208)),
                                                     (6211, (6207, 6208)), (6213, (6207, 6208)), (6214, (6207, 6208)),
                                                     (6238, (6242, 6243)), (6239, (6242, 6243)), (6240, (6242, 6243)),
                                                     (6251, (6253, 6254)), (6252, (6253, 6254))]

        # attach calibration to each file
        # TODO: Make a plot where 56 runs and calibrations are plotted on a time-axis
        for files in filearray56:
            # extract file number
            file_no = int(re.split('[_.]',files)[1])
            # find nearest calibration tuple
            #nearest_calib = (0, 0)
            #for calibs in calib_tuples:
                #nearest_calib = calibs if abs(calibs[0] - file_no) < abs(nearest_calib[0]-file_no) else nearest_calib
            #files56_withReference_tuples.append((file_no, (nearest_calib[0], nearest_calib[1])))
            # navigate to 58Ni reference file in self.db
            # calib_file_58 = 'BECOLA_'+str(nearest_calib[0])+'.xml'
            calibration_tuple = ()  # for hand assigned
            for refs in files56_withReference_tuples_handassigned:  # for hand assigned
                if refs[0] == file_no:
                    calibration_tuple = (refs[1][0], refs[1][1])
            calib_file_58 = 'BECOLA_' + str(calibration_tuple[0]) + '.xml'  # for hand assigned
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            # extract voltage from calibration
            cur.execute(
                '''SELECT accVolt FROM Files WHERE file = ? ''', (calib_file_58, ))
            accVolt_calib = cur.fetchall()[0][0]
            # write new voltage for 56Ni file + create name and insert new isotope type
            # calibration_name = str(nearest_calib[0]) + 'w' + str(nearest_calib[1])
            calibration_name = str(calibration_tuple[0]) + 'w' + str(calibration_tuple[1])
            file56_newType = '56Ni_cal'+ calibration_name
            cur.execute('''UPDATE Files SET accVolt = ?, type = ? WHERE file = ? ''', (accVolt_calib, file56_newType, files))
            con.commit()
            con.close()
            # calculate center shift and create new isotope for fitting
            # Calculate differential Doppler shift for re-assigning center fit pars
            # No error analysis necessary here. Will just yield a start parameter for the fit.
            diff_Doppler_56 = Physics.diffDoppler(self.restframe_trans_freq, accVolt_calib, 56)
            # Create new isotopes in self.db
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            # create new 56 calibrated isotope
            cur.execute('''SELECT * FROM Isotopes WHERE iso = ? ''', ('56Ni',))  # get original isotope to copy from
            mother_isopars = cur.fetchall()
            center56 = mother_isopars[0][4]
            new_center56 = center56 + (29850 - accVolt_calib) * diff_Doppler_56
            isopars_lst = list(mother_isopars[0])  # change into list to replace some values
            isopars_lst[0] = file56_newType
            isopars_lst[4] = new_center56
            new_isopars = tuple(isopars_lst)
            print(new_isopars)
            cur.execute('''INSERT OR REPLACE INTO Isotopes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                        new_isopars)
            con.commit()
            con.close()

        files56_withReference_tuples = files56_withReference_tuples_handassigned

        # do the batchfit for 56Ni
        BatchFit.batchFit(filearray56, self.db, self.runs[2], x_as_voltage=True, softw_gates_trs=None, save_file_as='.png')

        # prepare workbook:
        self.excel.last_row += 1
        start_col = 5*(scaler+1)
        self.excel.active_sheet.cell(row=self.excel.last_row, column=start_col, value='Scaler '+str(scaler))
        self.excel.last_row += 1
        self.excel.active_sheet.cell(row=self.excel.last_row, column=1, value='Runs56')
        self.excel.active_sheet.cell(row=self.excel.last_row, column=2, value='Ref58File')
        self.excel.active_sheet.cell(row=self.excel.last_row, column=start_col, value='56 center fit par')
        self.excel.active_sheet.cell(row=self.excel.last_row, column=start_col+1, value='56 center fit err')
        self.excel.active_sheet.cell(row=self.excel.last_row, column=start_col+2, value='IS 56vs58')
        self.excel.active_sheet.cell(row=self.excel.last_row, column=start_col+3, value='IS 56vs58 err')
        self.excel.last_row += 1
        # calculate isotope shift between 56file and reference
        ni56_center = []
        ni56_center_d =[]
        files56_withReference_andIsoshift_tuples = []
        for files56 in filearray56:
            # extract file number
            file_no = int(re.split('[_.]', files56)[1])
            # Get 56Nickel center fit parameter in MHz
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            # Get corresponding isotope
            cur.execute(
                '''SELECT type FROM Files WHERE file = ? ''', (files56,))
            iso_type56 = cur.fetchall()[0][0]
            # Query fitresults for file and isotope combo
            cur.execute(
                '''SELECT pars FROM FitRes WHERE file = ? AND iso = ? AND run = ?''', (files56, iso_type56, self.runs[2]))
            pars56 = cur.fetchall()
            con.close()
            pars56dict = ast.literal_eval(pars56[0][0])
            center56 = pars56dict['center']  # tuple of (center frequency, Uncertainty?, Fixed?)
            # add to list
            ni56_center.append(center56[0])
            ni56_center_d.append(center56[1])  # statistical error from fitting routine


            # Get reference 58Nickel center fit parameter in MHz,
            calibration_tuple = ()
            ref58_file = ''
            calVolt = 29850
            calVoltStatErr = 1
            for refs in files56_withReference_tuples:
                if refs[0] == file_no:
                    calibration_tuple = (refs[1][0], refs[1][1])
                    ref58_file = 'BECOLA_'+ str(refs[1][0])+'.xml'
                    for caltuples in self.calib_tuples:
                        # while we're at it also get calibration voltage and uncertainty
                        if calibration_tuple[0] == caltuples[0]:
                            # caltuples contain (58ref, 60ref, isoshift, isoshift_d, calVolt, calVoltStatErr)
                            calVolt = caltuples[4]
                            calVoltStatErr = caltuples[5]
                            calVoltSystErr = caltuples[6]
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            # Get corresponding isotope
            cur.execute(
                '''SELECT type FROM Files WHERE file = ? ''', (ref58_file,))
            iso_type58 = cur.fetchall()[0][0]
            # Query fitresults for file and isotope combo
            cur.execute(
                '''SELECT pars FROM FitRes WHERE file = ? AND iso = ? AND run = ?''', (ref58_file, iso_type58, self.runs[0]))
            pars58 = cur.fetchall()
            con.close()
            print(files)
            pars58dict = ast.literal_eval(pars58[0][0])
            center58 = pars58dict['center']

            # TODO: Would it make sense to calculate back to the center DAC here, and go manually to frequency again in order to include uncertainties like voltage?
            # TODO: Alternatively: use differential doppler shift with cal voltage uncertainty to get MHz uncertainty:

            # TODO: bring calibration voltage, calibrationvolatage_error from calib_tuples_with...
            diff_Doppler_58 = Physics.diffDoppler(self.restframe_trans_freq, calVolt, 58)
            diff_Doppler_56 = Physics.diffDoppler(self.restframe_trans_freq, calVolt, 56)
            delta_diff_doppler = diff_Doppler_56 - diff_Doppler_58

            # calculate isotope shift of 56 nickel with reference to 58 nickel
            isoShift56 = center56[0] - center58[0]
            isoShift56_d_stat = np.sqrt(center56[1]**2 + center58[1]**2)
            isoShift56_d_stat = np.sqrt(isoShift56_d_stat**2 + (delta_diff_doppler*calVoltStatErr)**2)
            # TODO: This is the statistical errors only now, right? Here are the systematics:
            isoShift56_d_syst = np.sqrt(2 * self.wavemeter_wsu30_mhz_d**2)
            isoShift56_d_syst = np.sqrt((delta_diff_doppler*calVoltSystErr)**2 + isoShift56_d_syst**2)
            print('Isotope shift of Ni56 run {} with respect to Ni58 run{} is: {}+-{}MHz'.
                  format(file_no, calibration_tuple, isoShift56, isoShift56_d_stat))
            files56_withReference_andIsoshift_tuples.append((file_no, calibration_tuple, (isoShift56, isoShift56_d_stat, isoShift56_d_syst)))  #(file_no, claibration_tuple, (isoshift56, errorIS_stat, errorIS_syst))

            # write to excel workbook
            self.excel.active_sheet.cell(row=self.excel.last_row, column=1, value=file_no)
            self.excel.active_sheet.cell(row=self.excel.last_row, column=2, value=ref58_file)
            self.excel.active_sheet.cell(row=self.excel.last_row, column=start_col, value=center56[0])
            self.excel.active_sheet.cell(row=self.excel.last_row, column=start_col+1, value=center56[1])
            self.excel.active_sheet.cell(row=self.excel.last_row, column=start_col+2, value=isoShift56)
            self.excel.active_sheet.cell(row=self.excel.last_row, column=start_col+3, value=isoShift56_d_stat)
            self.excel.last_row += 1
        self.excel.wb.save(self.excelpath)


        # plot isotope shift for all 56 nickel runs (can be removed later on):
        ni56_isoShift_yData = []
        ni56_isoShift_yData_d = []
        ni56_point_runNos = []
        # tuple for weighted average calculation
        w_avg_56isoshift = (0, 0)
        for tuples in files56_withReference_andIsoshift_tuples:
            # (file_no, calibration_tuple, (isoshift56, errorIS))
            ni56_isoShift_yData.append(tuples[2][0])
            ni56_isoShift_yData_d.append(tuples[2][1])
            ni56_point_name = str(tuples[0])
            ni56_point_runNos.append(ni56_point_name)

            val = tuples[2][0]
            weight = 1/tuples[2][1]**2
            w_avg_56isoshift = (w_avg_56isoshift[0]+(weight*val), w_avg_56isoshift[1]+weight)
        # weighted average as tuple of (w_avg, w_avg_d)
        w_avg_56isoshift = (w_avg_56isoshift[0]/w_avg_56isoshift[1], np.sqrt(1/w_avg_56isoshift[1]))

        ni56_isoShift_alt_yData = -525.8169055478309, -523.0479365515923, -525.2808338260361, -533.4630219328083,\
                                  -540.3585627829973, -521.3067663175245, -509.42569032109384, -511.40285471674554,\
                                  -511.1400904483909, -508.7950760887162, -511.4211280594908

        # plot isotope shift for each run with errorbar
        plt.errorbar(range(len(ni56_isoShift_yData)), ni56_isoShift_yData, yerr=ni56_isoShift_yData_d, label='preferred')
        # plot weighted average as red line
        plt.plot([-1, len(ni56_isoShift_yData)], [w_avg_56isoshift[0], w_avg_56isoshift[0]], 'r')
        # plot error of weighted average as red shaded box around that line
        plt.fill([-1, len(ni56_isoShift_yData), len(ni56_isoShift_yData), -1],
                 [w_avg_56isoshift[0]-w_avg_56isoshift[1], w_avg_56isoshift[0]-w_avg_56isoshift[1],
                  w_avg_56isoshift[0]+w_avg_56isoshift[1], w_avg_56isoshift[0]+w_avg_56isoshift[1]], 'r', alpha=0.2)
        # plot systematic error as lighter red shaded box around that line
        plt.fill([-1, len(ni56_isoShift_yData), len(ni56_isoShift_yData), -1],
                 [w_avg_56isoshift[0] - isoShift56_d_syst, w_avg_56isoshift[0] - isoShift56_d_syst,
                  w_avg_56isoshift[0] + isoShift56_d_syst, w_avg_56isoshift[0] + isoShift56_d_syst], 'r', alpha=0.1)
        #plt.plot(range(len(ni56_isoShift_yData)), ni56_isoShift_yData, '-o', label='preferred')
        #plt.plot(range(len(ni56_isoShift_yData)), ni56_isoShift_alt_yData, 'r-o', label='alternative')
        plt.xticks(range(len(ni56_isoShift_yData)), ni56_point_runNos, rotation=-30)
        plt.axis([-0.5, len(ni56_isoShift_yData)-0.5,
                  min(ni56_isoShift_yData)-max(ni56_isoShift_yData_d),
                  max(ni56_isoShift_yData)+max(ni56_isoShift_yData_d)])
        plt.title('Isotope Shift Ni 56-58 for all runs')
        plt.xlabel('Run Number')
        plt.ylabel('Isotope Shift  [MHz]')
        plt.legend(loc='lower right')
        plt.show()

        return ni56_point_runNos, ni56_center, ni56_center_d, ni56_isoShift_yData, ni56_isoShift_yData_d, w_avg_56isoshift, isoShift56_d_syst

    def plot_56_results(self, results_text):
        fig, ax = plt.subplots()
        # Plot Isotope shift 56-58 for all scalers to compare.
        lensc0 = len(self.results_per_scaler['scaler_0']['runNumbers_56'])
        plt.errorbar(range(lensc0), self.results_per_scaler['scaler_0']['isoShift_56-58'], c='b',
                     yerr=self.results_per_scaler['scaler_0']['isoShift_56-58_d'],
                     label='scaler 0')
        lensc1 = len(self.results_per_scaler['scaler_1']['runNumbers_56'])
        plt.errorbar(range(lensc1), self.results_per_scaler['scaler_1']['isoShift_56-58'], c='g',
                     yerr=self.results_per_scaler['scaler_1']['isoShift_56-58_d'],
                     label='scaler 1')
        lensc2 = len(self.results_per_scaler['scaler_2']['runNumbers_56'])
        plt.errorbar(range(lensc2), self.results_per_scaler['scaler_2']['isoShift_56-58'], c='r',
                     yerr=self.results_per_scaler['scaler_2']['isoShift_56-58_d'],
                     label='scaler 2')
        # Plot weighted average and errorband for all scalers
        avg_is0 = self.results_per_scaler['scaler_0']['isoShift_56-58_avg']
        avg_is0_d = self.results_per_scaler['scaler_0']['isoShift_56-58_avg_d']
        avg_is0_dsys = self.results_per_scaler['scaler_0']['isoShift_56-58_systErr']
        plt.plot([-1, lensc0], [avg_is0, avg_is0], c='b')
        plt.fill([-1, lensc0, lensc0, -1],  # statistical error
                 [avg_is0 - avg_is0_d, avg_is0 - avg_is0_d, avg_is0 + avg_is0_d, avg_is0 + avg_is0_d], 'b', alpha=0.2)
        plt.fill([-1, lensc0, lensc0, -1],  # systematic error
                 [avg_is0 - avg_is0_dsys, avg_is0 - avg_is0_dsys, avg_is0 + avg_is0_dsys, avg_is0 + avg_is0_dsys], 'b',
                 alpha=0.05)
        avg_is1 = self.results_per_scaler['scaler_1']['isoShift_56-58_avg']
        avg_is1_d = self.results_per_scaler['scaler_1']['isoShift_56-58_avg_d']
        avg_is1_dsys = self.results_per_scaler['scaler_1']['isoShift_56-58_systErr']
        plt.plot([-1, lensc1], [avg_is1, avg_is1], c='g')
        plt.fill([-1, lensc1, lensc1, -1],  # statistical error
                 [avg_is1 - avg_is1_d, avg_is1 - avg_is1_d, avg_is1 + avg_is1_d, avg_is1 + avg_is1_d], 'g', alpha=0.2)
        plt.fill([-1, lensc1, lensc1, -1],  # systematic error
                 [avg_is1 - avg_is1_dsys, avg_is1 - avg_is1_dsys, avg_is1 + avg_is1_dsys, avg_is1 + avg_is1_dsys], 'g',
                 alpha=0.05)
        avg_is2 = self.results_per_scaler['scaler_2']['isoShift_56-58_avg']
        avg_is2_d = self.results_per_scaler['scaler_2']['isoShift_56-58_avg_d']
        avg_is2_dsys = self.results_per_scaler['scaler_2']['isoShift_56-58_systErr']
        plt.plot([-1, lensc2], [avg_is2, avg_is2], c='r')
        plt.fill([-1, lensc2, lensc2, -1],
                 [avg_is2 - avg_is2_d, avg_is2 - avg_is2_d, avg_is2 + avg_is2_d, avg_is2 + avg_is2_d], 'r', alpha=0.2)
        plt.fill([-1, lensc2, lensc2, -1],  # systematic error
                 [avg_is2 - avg_is2_dsys, avg_is2 - avg_is2_dsys, avg_is2 + avg_is2_dsys, avg_is2 + avg_is2_dsys], 'r',
                 alpha=0.05)
        # plt.plot(range(len(ni56_isoShift_yData)), ni56_isoShift_yData, '-o', label='preferred')
        # plt.plot(range(len(ni56_isoShift_yData)), ni56_isoShift_alt_yData, 'r-o', label='alternative')
        plt.xticks(range(lensc0), self.results_per_scaler['scaler_0']['runNumbers_56'], rotation=-30)
        plt.axis([-0.5, lensc0 - 0.5, -580, -470])
        plt.title('Isotope Shift Ni 56-58 for all runs')
        plt.xlabel('Run Number')
        plt.ylabel('Isotope Shift  [MHz]')
        plt.legend(loc='lower right')
        props = dict(boxstyle='round', facecolor='white', alpha=1)
        ax.text(0.05, 0.95, results_text, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        plt.show()

    ''' Nickel 55 related functions: '''
    def create_stacked_files(self):
        restriction = [6315, 6502]
        # stack nickel 58 runs to new file Sum58_9999.xml. Only use calibration runs
        ni58_files, ni58_filenos = self.pick_files_from_db_by_type_and_num('%58Ni_cal%', selecttuple=[0, 6502])
        self.stack_runs('58Ni', ni58_files, binsize=1)
        # stack nickel 60 runs to new file Sum60_9999.xml. Only use calibration runs
        ni60_files, ni60_filenos = self.pick_files_from_db_by_type_and_num('%60Ni_cal%', selecttuple=[0, 6502])
        self.stack_runs('60Ni', ni60_files, binsize=1)
        # stack nickel 56 runs to new file Sum56_9999.xml
        ni56_files, ni56_filenos = self.pick_files_from_db_by_type_and_num('%56Ni%', selecttuple=[0, 6502])
        self.stack_runs('56Ni', ni56_files, binsize=1)
        # select and stack nickel 55 runs to new file Sum55_9999.xml
        ni55_files, ni55_filenos = self.pick_files_from_db_by_type_and_num('%55Ni%', selecttuple=restriction)
        self.stack_runs('55Ni', ni55_files, binsize=3)

    def stack_runs(self, isotope, files, binsize):
        ##############
        # stack runs #
        ##############
        # sum all the isotope runs
        self.time_proj_res_per_scaler = self.stack_time_projections(files)
        self.addfiles(isotope, files, binsize=binsize)

    def stack_time_projections(self, filelist):
        sumcts_sc = np.zeros(1024)  # array for one scaler
        sumcts = np.array([sumcts_sc.copy(), sumcts_sc.copy(), sumcts_sc.copy()])
        timebins = np.arange(1024)
        for files in filelist:
            filepath = os.path.join(self.datafolder, files)
            # load the spec data from file
            spec = XMLImporter(path=filepath)
            for sc_no in range(3):
                # sum time projections for each scaler
                sumcts[sc_no] += spec.t_proj[0][sc_no]
        logging.info('------- time projection fit results: --------')
        timeproj_res = {'scaler_0':{}, 'scaler_1':{}, 'scaler_2':{}}
        for sc_no in range(3):
            # fit time-projection
            ampl, sigma, center, offset = self.fit_time_projections(sumcts[sc_no], timebins)
            plt.plot(timebins, sumcts[sc_no], '.', label='scaler_{}'.format(sc_no))
            plt.plot(timebins, self.fitfunc(timebins, ampl, sigma, center, offset), '-')
            logging.info('Scaler_{}: amplitude: {}, sigma: {}, center: {}, offset:{}'
                         .format(sc_no, ampl, sigma, center, offset))
            timeproj_res['scaler_{}'.format(sc_no)] = {'sigma': sigma, 'center': center}
        plt.legend(loc='right')
        plt.show()
        return timeproj_res

    def fit_time_projections(self, cts_axis, time_axis):
        x = time_axis
        y = cts_axis
        start_pars = np.array([10*max(cts_axis), 10, np.where(cts_axis == max(cts_axis))[0], min(cts_axis)])
        print(start_pars)
        ampl, sigma, center, offset = curve_fit(self.fitfunc, x, y, start_pars)[0]
        print(ampl, sigma, center, offset)

        return ampl, sigma, center, offset

    def fitfunc(self, t, a, s, t0, o):
        """
        t: time
        t0: mid-tof
        a: cts_max
        s: sigma
        o: offset
        """
        # Gauss function
        return o + a * 1/(s*np.sqrt(2*np.pi))*np.exp(-1/2*np.power((t-t0)/s, 2))

    def addfiles_old(self, iso, filelist, scalers, binsize):
        startvoltneg = 360  # negative starting volts (don't use the -)
        scanrange = 420  # volts scanning up from startvolt
        sumcts = np.zeros(scanrange // binsize)  # should contain all the 55 scans so roughly -350 to +100
        sumabs = sumcts.copy()  # array for the absolute counts per bin. Same dimension as counts of course
        bgcounter = np.zeros(scanrange // binsize)  # array to keep track of the backgrounds
        sumvolts = np.arange(scanrange // binsize) - startvoltneg / binsize
        for files in filelist:
            filepath = os.path.join(self.datafolder, files)
            filenumber = re.split('[_.]', files)[-2]
            # get gates from stacked time projection
            sc0_res = self.time_proj_res_per_scaler['scaler_0']
            sc1_res = self.time_proj_res_per_scaler['scaler_1']
            sc2_res = self.time_proj_res_per_scaler['scaler_2']
            sig_mult = 2  # gate width in multiple sigma
            spec = XMLImporter(path=filepath,
                               softw_gates=[[-350, 0, sc0_res['center']/100 - sc0_res['sigma'] / 100 * sig_mult,
                                             sc0_res['center']/100 + sc0_res['sigma'] / 100 * sig_mult],
                                            [-350, 0, sc1_res['center'] / 100 - sc1_res['sigma'] / 100 * sig_mult,
                                             sc1_res['center'] / 100 + sc1_res['sigma'] / 100 * sig_mult],
                                            [-350, 0, sc2_res['center'] / 100 - sc2_res['sigma'] / 100 * sig_mult,
                                             sc2_res['center'] / 100 + sc2_res['sigma'] / 100 * sig_mult]])
            offst = 100  # background offset in timebins
            background = XMLImporter(path=filepath,  # sample spec of the same width, clearly separated from the timepeaks
                                     softw_gates=[[-350, 0, (sc0_res['center']-offst) / 100 - sc0_res['sigma'] / 100 * sig_mult,
                                                   (sc0_res['center']-offst) / 100 + sc0_res['sigma'] / 100 * sig_mult],
                                                  [-350, 0, (sc1_res['center']-offst) / 100 - sc1_res['sigma'] / 100 * sig_mult,
                                                   (sc1_res['center']-offst) / 100 + sc1_res['sigma'] / 100 * sig_mult],
                                                  [-350, 0, (sc2_res['center']-offst) / 100 - sc2_res['sigma'] / 100 * sig_mult,
                                                   (sc2_res['center']-offst) / 100 + sc2_res['sigma'] / 100 * sig_mult]])
            stepsize = spec.stepSize[0]
            if stepsize > 1.1*binsize:
                logging.warning('Stepsize of file {} larger than specified binsize ({}>{})!'
                                .format(files, stepsize, binsize))
            nOfSteps = spec.getNrSteps(0)
            nOfScans = spec.nrScans[0]
            nOfBunches = spec.nrBunches[0]
            voltage_x = spec.x[0]
            cts_dict = {}
            for pmts in scalers:
                cts_dict[str(pmts)] = {'cts':{}, 'bg':{}}
                cts_dict[str(pmts)]['cts'] = spec.cts[0][pmts]
                cts_dict[str(pmts)]['bg'] = background.cts[0][pmts]
            scaler_sum_cts = []
            bg_sum_cts = []
            for scnumstr, sc_cts in cts_dict.items():
                if len(scaler_sum_cts):
                    scaler_sum_cts += sc_cts['cts']
                    bg_sum_cts += sc_cts['bg']
                else:
                    scaler_sum_cts = sc_cts['cts']
                    bg_sum_cts = sc_cts['bg']
            bg_sum_totalcts = sum(bg_sum_cts)
            if bg_sum_totalcts == 0: bg_sum_totalcts = 1

            for datapoint_ind in range(len(voltage_x)):
                voltind = int(voltage_x[datapoint_ind] + startvoltneg) // binsize
                if 0 <= voltind < len(sumabs):
                    sumabs[voltind] += scaler_sum_cts[datapoint_ind]  # no normalization here
                    bgcounter[voltind] += bg_sum_totalcts / nOfSteps
            plt.plot(voltage_x, scaler_sum_cts, drawstyle='steps', label=filenumber)
        plt.show()
        sumerr = np.sqrt(sumabs)
        # prepare sumcts for transfer to xml file
        zero_ind = np.where(
            bgcounter == 0)  # find zero-values. Attention! These should only be at start and end, not middle
        bgcounter = np.delete(bgcounter, zero_ind)
        sumabs = np.delete(sumabs, zero_ind)
        sumerr = np.delete(sumerr, zero_ind)
        sumvolts = np.delete(sumvolts, zero_ind)

        plt.errorbar(sumvolts * binsize, sumabs, yerr=sumerr, fmt='.')
        plt.show()
        plt.errorbar(sumvolts * binsize, sumabs / bgcounter, yerr=sumerr / bgcounter, fmt='.')
        plt.show()

        # prepare sumcts for transfer to xml file
        scale_factor = 10000
        sumcts = np.array([sumabs / bgcounter * scale_factor]).astype(int)
        sumerr = np.array([sumerr / bgcounter * scale_factor]).astype(int)

        self.make_sumXML_file(iso, 1, sumvolts[0]*binsize, binsize, len(sumcts[0]), sumcts, sumerr)

    def addfiles(self, iso, filelist, binsize):
        startvoltneg = 360  # negative starting volts (don't use the -)
        scanrange = 420  # volts scanning up from startvolt
        sumcts = np.zeros(scanrange // binsize)  # should contain all the 55 scans so roughly -350 to +100
        sumabs = [sumcts.copy(), sumcts.copy(), sumcts.copy()]  # array for the absolute counts per bin. Same dimension as counts of course
        bgcounter = [sumcts.copy(), sumcts.copy(), sumcts.copy()]  # array to keep track of the backgrounds
        sumvolts = np.arange(scanrange // binsize) - startvoltneg / binsize
        for files in filelist:
            filepath = os.path.join(self.datafolder, files)
            filenumber = re.split('[_.]', files)[-2]
            # get gates from stacked time projection
            sc0_res = self.time_proj_res_per_scaler['scaler_0']
            sc1_res = self.time_proj_res_per_scaler['scaler_1']
            sc2_res = self.time_proj_res_per_scaler['scaler_2']
            sig_mult = 2  # gate width in multiple sigma
            spec = XMLImporter(path=filepath,
                               softw_gates=[[-350, 0, sc0_res['center']/100 - sc0_res['sigma'] / 100 * sig_mult,
                                             sc0_res['center']/100 + sc0_res['sigma'] / 100 * sig_mult],
                                            [-350, 0, sc1_res['center'] / 100 - sc1_res['sigma'] / 100 * sig_mult,
                                             sc1_res['center'] / 100 + sc1_res['sigma'] / 100 * sig_mult],
                                            [-350, 0, sc2_res['center'] / 100 - sc2_res['sigma'] / 100 * sig_mult,
                                             sc2_res['center'] / 100 + sc2_res['sigma'] / 100 * sig_mult]])
            offst = 100  # background offset in timebins
            background = XMLImporter(path=filepath,  # sample spec of the same width, clearly separated from the timepeaks
                                     softw_gates=[[-350, 0, (sc0_res['center']-offst) / 100 - sc0_res['sigma'] / 100 * sig_mult,
                                                   (sc0_res['center']-offst) / 100 + sc0_res['sigma'] / 100 * sig_mult],
                                                  [-350, 0, (sc1_res['center']-offst) / 100 - sc1_res['sigma'] / 100 * sig_mult,
                                                   (sc1_res['center']-offst) / 100 + sc1_res['sigma'] / 100 * sig_mult],
                                                  [-350, 0, (sc2_res['center']-offst) / 100 - sc2_res['sigma'] / 100 * sig_mult,
                                                   (sc2_res['center']-offst) / 100 + sc2_res['sigma'] / 100 * sig_mult]])
            stepsize = spec.stepSize[0]
            if stepsize > 1.1*binsize:
                logging.warning('Stepsize of file {} larger than specified binsize ({}>{})!'
                                .format(files, stepsize, binsize))
            nOfSteps = spec.getNrSteps(0)
            nOfScans = spec.nrScans[0]
            nOfBunches = spec.nrBunches[0]
            voltage_x = spec.x[0]
            cts_dict = {}
            for pmts in range(3):
                cts_dict[str(pmts)] = {'cts':{}, 'bg':{}}
                cts_dict[str(pmts)]['cts'] = spec.cts[0][pmts]
                cts_dict[str(pmts)]['bg'] = background.cts[0][pmts]
            scaler_sum_cts = [[], [], []]  # one list for each scaler
            bg_sum_cts = [[], [], []]  # one list for each scaler
            for scnumstr, sc_cts in cts_dict.items():
                if len(scaler_sum_cts[int(scnumstr)]):
                    scaler_sum_cts[int(scnumstr)] += sc_cts['cts']
                    bg_sum_cts[int(scnumstr)] += sc_cts['bg']
                else:
                    scaler_sum_cts[int(scnumstr)] = sc_cts['cts']
                    bg_sum_cts[int(scnumstr)] = sc_cts['bg']
            bg_sum_totalcts = [sum(bg_sum_cts[0]), sum(bg_sum_cts[1]), sum(bg_sum_cts[2])]
            for scaler in bg_sum_totalcts:
                if scaler == 0: scaler = 1

            for scaler in range(3):
                for datapoint_ind in range(len(voltage_x)):
                    voltind = int(voltage_x[datapoint_ind] + startvoltneg) // binsize
                    if 0 <= voltind < len(sumabs[scaler]):
                        sumabs[scaler][voltind] += scaler_sum_cts[scaler][datapoint_ind]  # no normalization here
                        bgcounter[scaler][voltind] += bg_sum_totalcts[scaler] / nOfSteps
                plt.plot(voltage_x, scaler_sum_cts[scaler], drawstyle='steps', label=filenumber)
        plt.show()
        sumerr = [np.sqrt(sumabs[0]), np.sqrt(sumabs[1]), np.sqrt(sumabs[2])]
        # prepare sumcts for transfer to xml file
        zero_ind = np.where(
            bgcounter[0] == 0)  # find zero-values. Attention! These should only be at start and end, not middle
        for scaler in range(3):
            bgcounter[scaler] = np.delete(bgcounter[scaler], zero_ind)
            sumabs[scaler] = np.delete(sumabs[scaler], zero_ind)
            sumerr[scaler] = np.delete(sumerr[scaler], zero_ind)
        sumvolts = np.delete(sumvolts, zero_ind)

        plt.errorbar(sumvolts * binsize, sumabs[0], yerr=sumerr[0], fmt='.')
        plt.show()
        plt.errorbar(sumvolts * binsize, sumabs[0] / bgcounter[0], yerr=sumerr[0] / bgcounter[0], fmt='.', label='scaler_0')
        plt.errorbar(sumvolts * binsize, sumabs[1] / bgcounter[1], yerr=sumerr[1] / bgcounter[1], fmt='.', label='scaler_1')
        plt.errorbar(sumvolts * binsize, sumabs[2] / bgcounter[2], yerr=sumerr[2] / bgcounter[2], fmt='.', label='scaler_2')
        plt.show()

        # prepare sumcts for transfer to xml file
        scale_factor = 3000  # This might need to be adjusted when changing the number of scalers involved.
        for scaler in range(3):
            sumabs[scaler] = (sumabs[scaler] / bgcounter[scaler] * scale_factor).astype(int)
            sumerr[scaler] = (sumerr[scaler] / bgcounter[scaler] * scale_factor).astype(int)

        self.make_sumXML_file(iso, sumvolts[0]*binsize, binsize, len(sumabs[0]), np.array(sumabs), np.array(sumerr))

    def make_sumXML_file(self, isotope, startVolt, stepSizeVolt, nOfSteps, cts_list, err_list=None):
        ####################################
        # Prepare dicts for writing to XML #
        ####################################
        file_creation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        header_dict = {'type': 'cs',
                       'isotope': isotope,
                       'isotopeStartTime': file_creation_time,
                       'accVolt': 29850,
                       'laserFreq': Physics.wavenumber(self.laser_freqs[isotope.split('_')[0]])/2,
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
        xml_name = 'Sum'+isotope.split('_')[0]+'_9999' + '.xml'
        xml_filepath = os.path.join(self.datafolder, xml_name)
        self.writeXMLfromDict(xml_dict, xml_filepath, 'BecolaData')
        self.ni55analysis_combined_files.append(xml_name)
        # add file to database
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''INSERT OR IGNORE INTO Files (file, filePath, date, type) VALUES (?, ?, ?, ?)''',
                    (xml_name, os.path.relpath(xml_filepath, self.workdir), file_creation_time, isotope+'_sum'))
        con.commit()
        cur.execute(
            '''UPDATE Files SET offset = ?, accVolt = ?,  laserFreq = ?, laserFreq_d = ?, colDirTrue = ?, 
            voltDivRatio = ?, lineMult = ?, lineOffset = ?, errDateInS = ? WHERE file = ? ''',
            ('[0]', 29850, self.laser_freqs[isotope.split('_')[0]], 0, True, str({'accVolt': 1.0, 'offset': 1.0}), 1, 0, 1,
             xml_name))
        con.commit()
        # create new isotope
        cur.execute('''SELECT * FROM Isotopes WHERE iso = ? ''', (isotope,))  # get original isotope to copy from
        mother_isopars = cur.fetchall()
        isopars_lst = list(mother_isopars[0])  # change into list to replace some values
        isopars_lst[0] = isotope+'_sum'
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
        king.calcChargeRadii(isotopes=isotopes, run=reference_run, plot_evens_seperate=False)

        # king.kingFit(alpha=378, findBestAlpha=True, run=run, find_slope_with_statistical_error=True)
        king.kingFit(alpha=361, findBestAlpha=True, run=reference_run)
        radii_alpha = king.calcChargeRadii(isotopes=isotopes, run=reference_run, plot_evens_seperate=False)
        print('radii with alpha', radii_alpha)
        # king.calcChargeRadii(isotopes=isotopes, run=run)


if __name__ == '__main__':

    analysis = NiAnalysis()
    analysis.separate_runs_analysis()
    analysis.stacked_runs_analysis()