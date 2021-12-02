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
import TildaTools as TiTs
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
        self.run_name = 'CEC_AsymVoigt_MSU'

        # create excel workbook to save some results
        path_results = 'ownCloud\\User\\Felix\\Measurements\\Nickel_online_Becola\\Analysis\\Results'
        excel_path = os.path.join(path_results, 'analysis_prefinal.xlsx')
        self.excelpath = os.path.join(user_home_folder, excel_path)
        self.excel = ExcelWriter(self.excelpath)
        self.excel.active_sheet = self.excel.wb.copy_worksheet(self.excel.wb['Template'])
        self.excel.active_sheet.title = self.run_name

        # Select runs; Format: ['run58', 'run60', 'run56']
        # to use a different lineshape you must create a new run under runs and a new linevar under lines and link the two.
        self.runs = ['CEC_AsymVoigt_MSU', 'CEC_AsymVoigt_MSU', 'CEC_AsymVoigt_MSU', 'CEC_AsymVoigt_MSU']
        self.excel.active_sheet['B1'] = str(self.runs)

        """ 
        ### Uncertainties ###
        All uncertainties that we can quantify and might want to respect
        """
        self.wavemeter_wsu30_mhz_d = 3  # Kristians wavemeter paper
        self.matsuada_volts_d = 0.05  # Rebinning and graphical Analysis TODO: get a solid estimate for this value


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
        self.nuclear_spin_and_moments = {
            '55Ni': (55, -7/2, -0.98, 0.03, 0, 0),
            '57Ni': (57, -3/2, -0.7975, 0.0014, 0, 0),
            '61Ni': (61, -3/2, -0.75002, 0.00004, 0.162, 0.015)
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
        self.restframe_trans_freq = 850343800
        self.excel.active_sheet['B3'] = self.restframe_trans_freq

        ''' literature value IS 60-58'''
        # Reference: ??
        # isotope shift of Nickel-60 with respect to Nickel-58 (=fNi60-fNi58)
        # Collaps 2017: 508.2(4)[76] MHz (Simon's PHD thesis)
        # Collaps 2016: 510.6(6)[95]MHz (Simon's PHD thesis)
        # Collaps combined: 509.4(2)[42] (28.01.20, Version a)
        # Steudel 1980: 0.01694(9) cm-1 corresponds to 507.8(27) MHz
        self.literature_IS60vs58 = 509.4
        self.literature_IS60vs58_d_stat = 0.2
        self.literature_IS60vs58_d_syst = 4.2
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
            '61Ni': (4.9005, np.sqrt(0.0010 ** 2 + 0.0017 ** 2)),
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

        # acceleration set voltage (Buncher potential), negative
        self.accVolt_set = 29883  # omit voltage sign, assumed to be negative

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
        self.isotope_colors = {58: 'black', 60: 'blue', 56: 'green', 55: 'purple'}

        # define calibration tuples:
        # do voltage calibration with these calibration pairs.
        self.calib_tuples = [(6191, 6192), (6207, 6208), (6224, 6225), (6232, 6233), (6242, 6243),
                             (6253, 6254), (6258, 6259), (6269, 6270), (6284, 6285), (6294, 6295), (6301, 6302),
                             (6310, 6311), (6323, 6324), (6340, 6342), (6362, 6363), (6395, 6396),
                             (6418, 6419), (6467, 6466), (6501, 6502)]
        self.use_handassigned = True  # use hand-assigned calibrations? If false will interpolate on time axis
        self.excel.active_sheet['K3'] = self.use_handassigned
        # current scaler variable:
        self.scalers = '[0,1,2]'

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
        scaler56_sum_syst = []

        for scalers in range(3):
            logging.info('\n'
                         '## Analysis started for scaler number {}'
                         .format(scalers))
            # write scaler to db
            self.update_scalers_in_db(scalers)

            for iso in ['58Ni', '60Ni', '56Ni']:
                # Do a first set of fits for all 58 & 60 runs without any calibration applied.
                filelist, runNo, center_freqs, center_freqs_d, center_freqs_d_syst, start_times = \
                    self.chooseAndFitRuns(iso+'%', reset=(self.accVolt_set, iso))
                zero_list = [0]*len(filelist)
                isodict = {iso:
                               {'file_names': filelist,
                                'file_numbers': runNo,
                                'file_times': start_times,
                                'scaler_{}'.format(scalers):
                                            self.make_results_dict_scaler(center_freqs, center_freqs_d, center_freqs_d_syst,
                                                                          zero_list, zero_list, zero_list,
                                                                          0, 0, 0),
                                'color': self.isotope_colors[int(iso[:2])]
                                }
                           }
                TiTs.merge_extend_dicts(self.results, isodict, overwrite=True, force_overwrite=True)
            # plot results of first fit
            self.plot_parameter_for_isos_and_scaler(['58Ni', '60Ni', '56Ni'], [scalers], 'center_fits', offset=[0, -450, 450])
            self.all_centerFreq_to_scanVolt('58Ni', [scalers])
            self.all_centerFreq_to_scanVolt('60Ni', [scalers])
            self.all_centerFreq_to_scanVolt('56Ni', [scalers])
            self.plot_parameter_for_isos_and_scaler(['58Ni', '60Ni', '56Ni'], [scalers], 'center_scanvolt')

            # self.calibrateVoltage(self.calib_tuples)
            # now tuples contain (58ref, 60ref, isoshift, isoshift_d, isoshift_d_syst, calVolt, calVoltStatErr, calVoltSystErr)

            # attach the Ni56 runs to some calibration point(s) and adjust voltage plus create new isotope with adjusted center
            # hand-assigned calibration runs; tuples of (56file, (58reference, 60reference))
            files56_withReference_tuples_handassigned = [(6202, (6191, 6192)), (6203, (6191, 6192)),
                                                         (6204, (6191, 6192)), (6211, (6224, 6225)),
                                                         (6213, (6224, 6225)),
                                                         (6214, (6224, 6225)), (6238, (6242, 6243)),
                                                         (6239, (6242, 6243)),
                                                         (6240, (6242, 6243)), (6251, (6253, 6254)),
                                                         (6252, (6253, 6254))]
            files56_withReference_tuples_handassigned = [(6202, (6191, 6192)), (6203, (6191, 6192)),
                                                         (6204, (6191, 6192)), (6211, (6207, 6208)),
                                                         (6213, (6207, 6208)),
                                                         (6214, (6207, 6208)), (6238, (6242, 6243)),
                                                         (6239, (6242, 6243)),
                                                         (6240, (6242, 6243)), (6251, (6253, 6254)),
                                                         (6252, (6253, 6254))]

            # calculate isotope shift
            if self.use_handassigned:
                self.extract_iso_shift_handassigned('56Ni', scalers, files56_withReference_tuples_handassigned, calibrated=False)
                self.plot_parameter_for_isos_and_scaler(['56Ni'], [scalers], 'shifts_iso-58')
                self.calibrateVoltage(self.calib_tuples)
                self.assign_calibration_voltage_handassigned('56Ni', '58Ni', scalers, files56_withReference_tuples_handassigned)
            else:
                self.extract_iso_shift_interp('56Ni', scalers, calibrated=False)
                self.plot_parameter_for_isos_and_scaler(['56Ni'], [scalers], 'shifts_iso-58')
                self.calibrateVoltage(self.calib_tuples)
                self.assign_calibration_voltages_interpolation('56Ni', '58Ni_ref', scalers)

            for iso in ['58Ni_cal', '60Ni_cal', '56Ni_cal']:
                # Do a second set of fits for all 56, 58 & 60 runs with calibration applied.
                # TODO: I should start fixing parameters like asymmetry for all runs within one calibration point (at least)
                filelist, runNo, center_freqs, center_freqs_d, center_freqs_d_syst, start_times = \
                    self.chooseAndFitRuns(iso+'%')
                zero_list = [0]*len(filelist)
                isodict = {iso:
                               {'file_names': filelist,
                                'file_numbers': runNo,
                                'file_times': start_times,
                                'scaler_{}'.format(scalers):
                                            self.make_results_dict_scaler(center_freqs, center_freqs_d, center_freqs_d_syst,
                                                                          zero_list, zero_list, zero_list,
                                                                          0, 0, 0),
                                'color': self.isotope_colors[int(iso[:2])]
                                }
                           }
                TiTs.merge_extend_dicts(self.results, isodict, overwrite=True, force_overwrite=True)
                if self.scalers == '[0]':
                    col = 18
                elif self.scalers == '[1]':
                    col = 25
                elif self.scalers == '[2]':
                    col = 32
                else:
                    col = 39
                if '58' in iso:
                    self.excel.last_row = 9  # return to start
                    for indx in range(len(filelist)):
                        self.excel.active_sheet.cell(row=self.excel.last_row, column=col+0, value=center_freqs[indx])
                        self.excel.active_sheet.cell(row=self.excel.last_row, column=col+1, value=center_freqs_d[indx])
                        self.excel.last_row += 1
                elif '60' in iso:
                    self.excel.last_row = 9  # return to start
                    for indx in range(len(filelist)):
                        self.excel.active_sheet.cell(row=self.excel.last_row, column=col+2+0, value=center_freqs[indx])
                        self.excel.active_sheet.cell(row=self.excel.last_row, column=col+2+1, value=center_freqs_d[indx])
                        self.excel.last_row += 1
                elif '56' in iso:
                    self.excel.last_row = 37  # return to start
                    for indx in range(len(filelist)):
                        self.excel.active_sheet.cell(row=self.excel.last_row, column=col + 0, value=center_freqs[indx])
                        self.excel.active_sheet.cell(row=self.excel.last_row, column=col + 1, value=center_freqs_d[indx])
                        self.excel.last_row += 1

            # plot results of first fit
            self.plot_parameter_for_isos_and_scaler(['58Ni_cal', '60Ni_cal', '56Ni_cal'], [scalers], 'center_fits', offset=[0, -450, 450])

            # calculate isotope shift
            if self.use_handassigned:
                self.extract_iso_shift_handassigned('56Ni_cal', scalers, files56_withReference_tuples_handassigned)
            else:
                self.extract_iso_shift_interp('56Ni_cal', scalers)

            # plot isotope shift
            self.plot_parameter_for_isos_and_scaler(['56Ni_cal'], [scalers], 'shifts_iso-58', offset=[0, 0])

            # add scaler result to total result
            sc_str = 'scaler_{}'.format(scalers)
            weight = 1/self.results['56Ni_cal'][sc_str]['avg_shift_iso-58']['d_stat'] ** 2
            # systematic uncertainties should not be reduced through weighting
            #weight_syst = 1 / self.results['56Ni_cal'][sc_str]['avg_shift_iso-58']['d_syst'] ** 2
            weight_syst = self.results['56Ni_cal'][sc_str]['avg_shift_iso-58']['d_syst']
            scaler56_weighted_sum += self.results['56Ni_cal'][sc_str]['avg_shift_iso-58']['val'] * weight
            scaler56_sum_of_weights += weight
            scaler56_sum_syst.append(weight_syst)

        # combine all 3 scalers to final result:
        self.ni56_isoShift_final = scaler56_weighted_sum/scaler56_sum_of_weights
        self.ni56_isoShift_final_d = np.sqrt(1/scaler56_sum_of_weights)
        self.ni56_isoShift_final_d_syst = sum(scaler56_sum_syst)/len(scaler56_sum_syst)
        ni56res_message = 'Isotope shift 56-58 combined: {0:.2f}({1:.0f})[{2:.0f}]MHz'\
            .format(self.ni56_isoShift_final, 100*self.ni56_isoShift_final_d, 100*self.ni56_isoShift_final_d_syst)
        print(ni56res_message)
        # write final result to database
        self.write_shift_to_combined_db('56Ni', self.runs[0],
                                        (self.ni56_isoShift_final, self.ni56_isoShift_final_d, self.ni56_isoShift_final_d_syst),
                                        'BECOLA 2018; 3 scalers separate; calibrated; ni56 runs: {}'
                                        .format(self.results['56Ni_cal']['file_numbers']))
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
        # switch to voigt profiles here...
        # self.runs = ['CEC_AsymVoigt', 'CEC_AsymVoigt', 'CEC_AsymVoigt', 'CEC_AsymVoigt']

        # combine runs to new 3-scaler files.
        self.ni55analysis_combined_files = []
        self.create_stacked_files()

        # update database to use all three scalers for analysis
        self.update_scalers_in_db('[0,1,2]')  # scalers to be used for combined analysis

        # do a batchfit of the newly created files
        for iso in ['58Ni_sum', '60Ni_sum', '56Ni_sum', '55Ni_sum']:
            for scalers in ['[0]','[1]','[2]','[0,1,2]']:
                self.update_scalers_in_db(scalers)
                scaler_name = 'scaler_{}'.format(scalers[1:-1].replace(',',''))
                # Do a first set of fits for all 58 & 60 runs without any calibration applied.
                filelist, runNo, center_freqs, center_freqs_d, center_freqs_d_syst, start_times = \
                    self.chooseAndFitRuns(iso + '%')
                if '55' in iso:
                    # extract additional hfs fit parameters
                    al = self.param_from_fitres_db(filelist[0], iso, self.runs[3], 'Al')
                    au = self.param_from_fitres_db(filelist[0], iso, self.runs[3], 'Au')
                    bl = self.param_from_fitres_db(filelist[0], iso, self.runs[3], 'Bl')
                    bu = self.param_from_fitres_db(filelist[0], iso, self.runs[3], 'Bu')
                    hfs_dict = {'Al': al, 'Au': au, 'Bl': bl, 'Bu': bu}
                else:
                    hfs_dict = None
                zero_list = [0] * len(filelist)
                isodict = {iso:
                               {'file_names': filelist,
                                'file_numbers': runNo,
                                'file_times': start_times,
                                scaler_name:
                                    self.make_results_dict_scaler(center_freqs, center_freqs_d, center_freqs_d_syst,
                                                                  zero_list, zero_list, zero_list,
                                                                  0, 0, 0, hfs_pars=hfs_dict),
                                'color': self.isotope_colors[int(iso[:2])]
                                }
                           }
                TiTs.merge_extend_dicts(self.results, isodict, overwrite=True, force_overwrite=True)
        # plot results of first fit
        # TODO: Add possibility to plot data plus fit (like in pollifit)
        self.plot_parameter_for_isos_and_scaler(['58Ni_sum', '60Ni_sum', '56Ni_sum', '55Ni_sum'], ['scaler_012'], 'center_fits')
        self.plot_parameter_for_isos_and_scaler(['58Ni_sum', '60Ni_sum', '56Ni_sum', '55Ni_sum'], ['scaler_0'], 'center_fits')

        # do voltage calibrations for the sumfiles
        for scalers in ['[0]', '[1]', '[2]', '[0,1,2]']:
            self.update_scalers_in_db(scalers)
            scaler_name = 'scaler_{}'.format(scalers[1:-1].replace(',', ''))
            self.calibrateVoltage_sumfiles(scaler_name)


        # batchfit all files again with new voltages
        for iso in ['58Ni_sum_cal', '60Ni_sum_cal', '56Ni_sum_cal', '55Ni_sum_cal']:
            for scalers in ['[0]', '[1]', '[2]', '[0,1,2]']:
                self.update_scalers_in_db(scalers)
                scaler_name = 'scaler_{}'.format(scalers[1:-1].replace(',', ''))
                # get calibration voltage
                volt = self.results[iso[:-4]][scaler_name]['acc_volts']['vals'][0]
                volt_d = self.results[iso[:-4]][scaler_name]['acc_volts']['d_stat'][0]
                volt_d_syst = self.results[iso[:-4]][scaler_name]['acc_volts']['d_syst'][0]
                # create/replace new isotopes in database
                self.create_new_isotope_in_db(iso[:-4], iso, volt)
                # Do a first set of fits for all 58 & 60 runs without any calibration applied.
                filelist, runNo, center_freqs, center_freqs_d, center_freqs_d_syst, start_times = \
                    self.chooseAndFitRuns(iso + '%', reset=(volt, iso))
                if '55' in iso:
                    # extract additional hfs fit parameters
                    al = self.param_from_fitres_db(filelist[0], iso, self.runs[3], 'Al')
                    au = self.param_from_fitres_db(filelist[0], iso, self.runs[3], 'Au')
                    bl = self.param_from_fitres_db(filelist[0], iso, self.runs[3], 'Bl')
                    bu = self.param_from_fitres_db(filelist[0], iso, self.runs[3], 'Bu')
                    hfs_dict = {'Al': al, 'Au': au, 'Bl': bl, 'Bu': bu}
                else:
                    hfs_dict = None
                zero_list = [0] * len(filelist)
                isodict = {iso:
                               {'file_names': filelist,
                                'file_numbers': runNo,
                                'file_times': start_times,
                                scaler_name:
                                    self.make_results_dict_scaler(center_freqs, center_freqs_d, center_freqs_d_syst,
                                                                  zero_list, zero_list, zero_list,
                                                                  0, 0, 0, hfs_pars=hfs_dict),
                                'color': self.isotope_colors[int(iso[:2])]
                                }
                           }
                isodict[iso][scaler_name]['acc_volts'] = {'vals': [volt],
                                                          'd_stat': [volt_d],
                                                          'd_syst': [volt_d_syst]
                                                          }
                TiTs.merge_extend_dicts(self.results, isodict, overwrite=True, force_overwrite=True)

        for scalers in ['[0]', '[1]', '[2]', '[0,1,2]']:
            self.update_scalers_in_db(scalers)
            scaler_name = 'scaler_{}'.format(scalers[1:-1].replace(',', ''))
            # extract isotope shift
            center58 = self.results['58Ni_sum_cal'][scaler_name]['center_fits']['vals']
            center58_d = self.results['58Ni_sum_cal'][scaler_name]['center_fits']['d_stat']
            center58_d_syst = self.results['58Ni_sum_cal'][scaler_name]['center_fits']['d_syst']
            center60 = self.results['60Ni_sum_cal'][scaler_name]['center_fits']['vals']
            center60_d = self.results['60Ni_sum_cal'][scaler_name]['center_fits']['d_stat']
            center60_d_syst = self.results['60Ni_sum_cal'][scaler_name]['center_fits']['d_syst']
            center56 = self.results['56Ni_sum_cal'][scaler_name]['center_fits']['vals']
            center56_d = self.results['56Ni_sum_cal'][scaler_name]['center_fits']['d_stat']
            center56_d_syst = self.results['56Ni_sum_cal'][scaler_name]['center_fits']['d_syst']
            center55 = self.results['55Ni_sum_cal'][scaler_name]['center_fits']['vals']
            center55_d = self.results['55Ni_sum_cal'][scaler_name]['center_fits']['d_stat']
            center55_d_syst = self.results['55Ni_sum_cal'][scaler_name]['center_fits']['d_syst']

            # get isotope shifts
            isoShift60 = center60[0] - center58[0]  # only one value in each of these lists
            isoShift60_d = np.sqrt(center60_d[0] ** 2 + center58_d[0] ** 2)
            isoShift60_d_syst = np.sqrt(center60_d_syst[0] ** 2 + center58_d_syst[0] ** 2)
            isoShift56 = center56[0] - center58[0]  # only one value in each of these lists
            isoShift56_d = np.sqrt(center56_d[0] ** 2 + center58_d[0] ** 2)
            isoShift56_d_syst = np.sqrt(center56_d_syst[0] ** 2 + center58_d_syst[0] ** 2)
            isoShift55 = center55[0] - center58[0]  # only one value in each of these lists
            isoShift55_d = np.sqrt(center55_d[0] ** 2 + center58_d[0] ** 2)
            isoShift55_d_syst = np.sqrt(center55_d_syst[0] ** 2 + center58_d_syst[0] ** 2)

            # TODO: add systematic uncertainty from voltage calibration
            volt = self.results['58Ni_sum_cal'][scaler_name]['acc_volts']['vals'][0]
            volt_d = self.results['58Ni_sum_cal'][scaler_name]['acc_volts']['d_stat'][0]
            volt_d_syst = self.results['58Ni_sum_cal'][scaler_name]['acc_volts']['d_syst'][0]
            diff_Doppler_58 = Physics.diffDoppler(self.restframe_trans_freq, volt, 60)
            diff_Doppler_58 = Physics.diffDoppler(self.restframe_trans_freq, volt, 58)
            diff_Doppler_56 = Physics.diffDoppler(self.restframe_trans_freq, volt, 56)
            diff_Doppler_55 = Physics.diffDoppler(self.restframe_trans_freq, volt, 55)
            delta_diff_doppler_56 = diff_Doppler_56 - diff_Doppler_58
            delta_diff_doppler_55 = diff_Doppler_55 - diff_Doppler_58

            isoShift56_d = np.sqrt((delta_diff_doppler_56 * volt_d) ** 2 + isoShift56_d ** 2)
            isoShift56_d_syst = np.sqrt((delta_diff_doppler_56 * volt_d_syst) ** 2 + isoShift56_d_syst ** 2)
            isoShift55_d = np.sqrt((delta_diff_doppler_55 * volt_d) ** 2 + isoShift55_d ** 2)
            isoShift55_d_syst = np.sqrt((delta_diff_doppler_55 * volt_d_syst) ** 2 + isoShift55_d_syst ** 2)

            # write to self.results
            self.write_into_self_results_scaler('60Ni_sum_cal', scaler_name, 'shifts_iso-58',
                                                {'vals': [isoShift60], 'd_stat': [isoShift60_d], 'd_syst': [isoShift60_d_syst]})
            self.write_into_self_results_scaler('56Ni_sum_cal', scaler_name, 'shifts_iso-58',
                                                {'vals': [isoShift56], 'd_stat': [isoShift56_d], 'd_syst': [isoShift56_d_syst]})
            self.write_into_self_results_scaler('55Ni_sum_cal', scaler_name, 'shifts_iso-58',
                                                {'vals': [isoShift55], 'd_stat': [isoShift55_d], 'd_syst': [isoShift55_d_syst]})

        # write final result to database
        self.write_shift_to_combined_db('56Ni_sum_cal', self.runs[0],
                                        (isoShift56, isoShift56_d, isoShift56_d_syst),
                                        'BECOLA 2018; 3 scalers combined; calibrated; ni56 runs summed to file: {}'
                                        .format(self.ni55analysis_combined_files[2]))
        self.write_shift_to_combined_db('55Ni_sum_cal', self.runs[0],
                                        (isoShift55, isoShift55_d, isoShift55_d_syst),
                                        'BECOLA 2018; 3 scalers combined; calibrated; ni56 runs summed to file: {}'
                                        .format(self.ni55analysis_combined_files[3]))

        self.plot_sum_results()

        # Extract A and B factors and do stuff:
        self.ni55_A_B_analysis('55Ni_sum_cal')

        ####################
        # Do the king plot #
        ####################
        # TODO: Probably need to automate reference run here? But need to adjust value in Collaps results as well...
        # TODO: Also: Do I want 56/58Ni or 56/58Ni_sum_cal
        refrun = self.runs[0]
        self.perform_king_fit_analysis(self.delta_lit_radii_58,
                                       isotopes=['55Ni_sum_cal', '56Ni_sum_cal', '56Ni', '58Ni', '59Ni', '60Ni', '61Ni',
                                                 '62Ni', '64Ni'],
                                       reference_run=refrun)

        # ###################
        # # export results  #
        # ###################
        # to_file_dict = {}
        # for keys, vals in self.results.items():
        #     # xml cannot take numbers as first letter of key
        #     to_file_dict['i' + keys] = vals
        # # add analysis parameters
        # to_file_dict['parameters'] = {'line_profiles': self.runs,
        #                               'calib_assignment': 'handassigned' if self.use_handassigned else 'interpolated'}
        # dateandtime = datetime.now()
        # filename = 'results\\' + self.run_name + '_' + dateandtime.strftime("%Y-%m-%d_%H-%M") + '.xml'
        # self.writeXMLfromDict(to_file_dict, os.path.join(self.workdir, filename), 'BECOLA_Analysis')

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

    def write_offset_to_db_lines(self, line, offset):
        """
        Since the offset can differ significantly between the isotopes, especially with the stacked runs,
        we might want to adjust that line-parameter before fitting!
        :param line: lineVar parameter to edit
        :param offset: new value for the offset parameter
        :return:
        """
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''SELECT * FROM Lines WHERE lineVar = ? ''', (line,))  # get original line to copy from
        copy_line = cur.fetchall()
        copy_shape = copy_line[0][6]
        shape_dict = ast.literal_eval(copy_shape)
        shape_dict['offset'] = offset
        copy_line_list = list(copy_line[0])
        copy_line_list[6] = str(shape_dict)
        line_new = tuple(copy_line_list)
        cur.execute('''INSERT OR REPLACE INTO Lines VALUES (?,?,?,?,?,?,?,?,?)''', line_new)
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

    ''' preparation and calibration '''

    def make_results_dict_scaler(self,
                                 centers, centers_d_stat, center_d_syst,
                                 shifts, shifts_d_stat, shifts_d_syst,
                                 avg_shift, avg_shift_d_stat, avg_shift_d_syst, hfs_pars=None):
        ret_dict = {'center_fits':
                        {'vals': centers,
                         'd_stat': centers_d_stat,
                         'd_syst': center_d_syst
                         },
                    'shifts_iso-58':
                        {'vals': shifts,
                         'd_stat': shifts_d_stat,
                         'd_syst': shifts_d_syst
                         },
                    'avg_shift_iso-58':
                        {'val': avg_shift,
                         'd_stat': avg_shift_d_stat,
                         'd_syst': avg_shift_d_syst
                         }
                    }
        if hfs_pars is not None:
            ret_dict['hfs_pars'] = hfs_pars
        return ret_dict

    def export_results(self):
        ###################
        # export results  #
        ###################
        to_file_dict = {}
        for keys, vals in self.results.items():
            # xml cannot take numbers as first letter of key
            to_file_dict['i' + keys] = vals
        # add analysis parameters
        to_file_dict['parameters'] = {'line_profiles': self.runs,
                                      'calib_assignment': 'handassigned' if self.use_handassigned else 'interpolated'}
        dateandtime = datetime.now()
        filename = 'results\\' + self.run_name + '_' + dateandtime.strftime("%Y-%m-%d_%H-%M") + '.xml'
        self.writeXMLfromDict(to_file_dict, os.path.join(self.workdir, filename), 'BECOLA_Analysis')

    def write_into_self_results_scaler(self, isotope, scaler, key, dict_insert):
        isotope_dict = self.results[isotope][scaler]
        def loop_dict(target_dict, k, insert):
            for key, vals in target_dict.items():
                if key == k:
                    target_dict[key] = insert
                else:
                    if type(vals) is dict:
                        loop_dict(vals, k, insert)
        loop_dict(isotope_dict, key, dict_insert)
        # write to results dict
        TiTs.merge_extend_dicts(self.results, {isotope: {scaler: isotope_dict}}, overwrite=True, force_overwrite=True)

    def chooseAndFitRuns(self, db_like, reset=None):
        '''

        :param db_like: str: example '58Ni%'
        :param reset: str: if a string is given, all files type will be reset to this string
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
        filearray = np.array(filelist)  # needed for batch fitting

        if reset:
            # Reset all calibration information so that pre-calib information can be extracted.
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            # Set type in files back to bare isotopes (56Ni, 58Ni, 60Ni)
            # Set accVolt in files back to nominal 29850
            cur.execute('''UPDATE Files SET accVolt = ?, type = ? WHERE type LIKE ? ''', (reset[0], reset[1], db_like))
            con.commit()
            con.close()

        # see what run to use
        # Adapt also offset parameter in lines database based on isotope!
        bg_dict = {'single': {'56': 700, '58': 2500, '60': 8000},
                   'sum': {'55': 12000, '56': 6000, '58': 45000, '60': 105000}}
        run = 0
        line_offset = 0
        if 'sum' in db_like:
            if '58' in db_like:
                run = 0
                line_offset = bg_dict['sum']['58']
            elif '60' in db_like:
                run = 1
                line_offset = bg_dict['sum']['60']
            elif '56' in db_like:
                run = 2
                line_offset = bg_dict['sum']['56']
            elif '55' in db_like:
                run = 2
                line_offset = bg_dict['sum']['55']
        else:
            if '58' in db_like:
                run = 0
                line_offset = bg_dict['single']['58']
            elif '60' in db_like:
                run = 1
                line_offset = bg_dict['single']['60']
            elif '56' in db_like:
                run = 2
                line_offset = bg_dict['single']['56']
        self.write_offset_to_db_lines(self.runs[0], line_offset)
        # do the batchfit
        BatchFit.batchFit(filearray, self.db, self.runs[run], x_as_voltage=True, softw_gates_trs=None, save_file_as='.png')
        # get fitresults (center) vs run for 58
        all_rundate = []
        all_center_MHz = []
        all_center_MHz_d = []
        all_center_MHz_d_syst = []
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
                '''SELECT pars FROM FitRes WHERE file = ? AND iso = ? AND run = ?''', (files, file_type, self.runs[run]))
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

            # systematic uncertainties:
            # wavemeter has 3MHz uncertainty. This goes directly into the center fit parameter
            # matsuada voltage read value oscillates a little over time. This is an estimate
            diffDoppler = Physics.diffDoppler(self.restframe_trans_freq, file_volt, int(db_like[:2]))
            d_syst = np.sqrt(self.wavemeter_wsu30_mhz_d ** 2 + (self.matsuada_volts_d * diffDoppler) ** 2)
            all_center_MHz_d_syst.append(d_syst)

        return filelist, runNos, all_center_MHz, all_center_MHz_d, all_center_MHz_d_syst, all_rundate

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
            '''SELECT mass FROM Isotopes WHERE iso = ? ''', (isostring, ))
        db_isopars = cur.fetchall()
        # Query laser frequency for isotope
        cur.execute(
            '''SELECT laserFreq FROM Files WHERE type = ? ''', (isostring, ))
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

    def all_centerFreq_to_scanVolt(self, isostring, scaler_list):
        """
        Converts all center frequencies of an isotope into scanVoltage values.
        :param isostring: Isotope as used in results dict
        :return:
        """
        for num in range(len(scaler_list)):
            scaler = scaler_list[num]
            if type(scaler) is int:
                scaler = 'scaler_{}'.format(scaler)
            center_freq = self.results[isostring][scaler]['center_fits']['vals']
            center_freq_d = self.results[isostring][scaler]['center_fits']['d_stat']
            center_freq_dsyst = self.results[isostring][scaler]['center_fits']['d_syst']

            center_volt, center_volt_d, center_volt_dsyst = [], [], []
            for nu, d, dsys in zip(center_freq, center_freq_d, center_freq_dsyst):
                v, v_d, v_ds = self.centerFreq_to_absVoltage(isostring, nu, d, dsys)
                center_volt.append(v-self.accVolt_set)
                center_volt_d.append(v_d)
                center_volt_dsyst.append(v_ds)

            voltdict = {'vals': center_volt,
                        'd_stat': center_volt_d,
                        'd_syst': center_volt_dsyst}
            self.results[isostring][scaler]['center_scanvolt'] = voltdict

    def plot_parameter_for_isos_and_scaler(self, isotopes, scaler_list, parameter, offset=None, overlay=None):
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
        x_type = 'file_times'  # alternative: 'file_numbers'
        for num in range(len(scaler_list)):
            scaler = scaler_list[num]
            if type(scaler) is int:
                scaler = 'scaler_{}'.format(scaler)
            for i in range(len(isotopes)):
                iso = isotopes[i]
                x_ax = self.results[iso][x_type]
                centers = self.results[iso][scaler][parameter]['vals']
                centers_d_stat = self.results[iso][scaler][parameter]['d_stat']
                centers_d_syst = self.results[iso][scaler][parameter]['d_syst']
                col = self.results[iso]['color']
                off = 0
                if offset is True:
                    avg_shift = self.results[iso][scaler]['avg_shift_iso-58']
                    off = round(avg_shift, -1)
                elif type(offset) is list:
                    # offset might be given manually per isotope
                    off = offset[i]
                # plot center frequencies in MHz:
                if off != 0:
                    plt_label = '{} (offset: {}MHz)'.format(iso, off)
                else:
                    plt_label = '{}'.format(iso)
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
                    plt.plot([x_ax[0], x_ax[-1]], [avg_shift, avg_shift], 'r', label='{} avg: {:.2f}MHz'.format(iso, avg_shift))
                    # plot error of weighted average as red shaded box around that line
                    plt.fill([x_ax[0], x_ax[-1], x_ax[-1], x_ax[0]],
                             [avg_shift - avg_shift_d, avg_shift - avg_shift_d,
                              avg_shift + avg_shift_d, avg_shift + avg_shift_d], 'r',
                             alpha=0.2)
                    # plot systematic error as lighter red shaded box around that line
                    plt.fill([x_ax[0], x_ax[-1], x_ax[-1], x_ax[0]],
                             [avg_shift - avg_shift_d_syst, avg_shift - avg_shift_d_syst,
                              avg_shift + avg_shift_d_syst, avg_shift + avg_shift_d_syst], 'r',
                             alpha=0.1)
        if overlay:
            plt.plot(x_ax, overlay, color='red')
        plt.title('{} in MHz for isotopes: {}'.format(parameter, isotopes))
        plt.ylabel('{} [MHz]'.format(parameter))
        plt.legend(loc='best')
        if x_type == 'file_times':
            plt.xlabel('date')
            days_fmt = mpdate.DateFormatter('%d.%B')
            ax.xaxis.set_major_formatter(days_fmt)
        else:
            plt.xlabel('run numbers')
        plt.xticks(rotation=45)
        plt.show()

    def calibrateVoltage(self, calibration_tuples):
        '''

        :param calibration_tuples:
        :return: calib_tuples_with_isoshift_and_calibrationvoltage:
                contains a tuple for each calibration point with entries: (58ref, 60ref, isoshift, isoshift_d, calVolt, calVoltStatErr, calVoltSystErr)
        '''
        #######################
        # Calibration process #
        #######################

        # Calibration sets of 58/60Ni
        calib_tuples = calibration_tuples
        calib_tuples_with_isoshift = []

        current_scaler_name = 'scaler_{}'.format(self.scalers[1])

        # copy the results dicts and remove all non-calibration values
        isos = ['58Ni', '60Ni']
        dicts = [{}, {}]
        for j in range(2):
            new_results_dict = self.results[isos[j]]
            indexlist = [new_results_dict['file_numbers'].index(i[j]) for i in calib_tuples]
            for keys, vals in new_results_dict.items():
                if type(vals) is list:
                    new_results_dict[keys] = [vals[i] for i in indexlist]
            for keys, vals in new_results_dict[current_scaler_name].items():
                for keys2, vals2 in vals.items():
                    if type(vals2) is list:
                        vals[keys2] = [vals2[i] for i in indexlist]
            dicts[j] = new_results_dict
        new58_dict = dicts[0]
        new60_dict = dicts[1]

        # Calculate Isotope shift for all calibration tuples and add to list.
        # also write calibration tuple data to excel
        self.excel.last_row = 9  # return to start
        for tuples in calib_tuples:
            # Get 58Nickel center fit parameter in MHz
            run58 = tuples[0]
            indx_58 = new58_dict['file_numbers'].index(run58)
            run58file = 'BECOLA_'+str(run58)+'.xml'
            center58 = self.param_from_fitres_db(run58file, '58Ni', self.runs[0], 'center')
            center58_d_syst = new58_dict[current_scaler_name]['center_fits']['d_syst'][indx_58]

            # Get 60Nickel center fit parameter in MHz
            run60 = tuples[1]
            indx_60 = new60_dict['file_numbers'].index(run60)
            run60file = 'BECOLA_' + str(run60) + '.xml'
            center60 = self.param_from_fitres_db(run60file, '60Ni', self.runs[0], 'center')
            center60_d_syst = new60_dict[current_scaler_name]['center_fits']['d_syst'][indx_60]

            # Calculate isotope shift of 60Ni with respect to 58Ni for this calibration point
            isoShift = center60[0]-center58[0]
            isoShift_d = np.sqrt(center60[1]**2+center58[1]**2)  # statistical uncertainty from fits
            isoShift_d_syst = np.sqrt(center58_d_syst ** 2 + center60_d_syst **2)
            tuple_with_isoshift = tuples + (isoShift, isoShift_d, isoShift_d_syst)
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

        # write isotope shifts for nickel 60 to dict
        ni60_shifts = [i[2] for i in calib_tuples_with_isoshift]
        ni60_shifts_d = [i[3] for i in calib_tuples_with_isoshift]
        ni60_shifts_d_syst = [i[4] for i in calib_tuples_with_isoshift]
        new60_dict[current_scaler_name]['shifts_iso-58']['vals'] = ni60_shifts
        new60_dict[current_scaler_name]['shifts_iso-58']['d_stat'] = ni60_shifts_d
        new60_dict[current_scaler_name]['shifts_iso-58']['d_syst'] = ni60_shifts_d_syst
        # TODO: this is no weighted avg. Maybe remove...
        new60_dict[current_scaler_name]['avg_shift_iso-58']['val'] = sum(ni60_shifts)/len(ni60_shifts)
        # write the new dicts to self.results
        TiTs.merge_extend_dicts(self.results, {'58Ni_ref': new58_dict}, overwrite=True, force_overwrite=True)
        TiTs.merge_extend_dicts(self.results, {'60Ni_ref': new60_dict}, overwrite=True, force_overwrite=True)

        # plot isotope shift for all calibration points (can be removed later on):
        self.plot_parameter_for_isos_and_scaler(['60Ni'], [current_scaler_name], 'shifts_iso-58')

        # Calculate resonance DAC Voltage from the 'center' positions
        calib_tuples_with_isoshift_and_calibrationvoltage = []
        average_calib_voltage = []
        # Do the voltage calibration for each tuple.
        self.excel.last_row = 9  # return to start
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
        self.ni56_average_calib_voltage = sum(average_calib_voltage[:7])/len(calib_tuples[:7])  # TODO: lets see how this does to the 56 runs


        acc_volt_dict = {'acc_volts': {'vals': [i[5] for i in calib_tuples_with_isoshift_and_calibrationvoltage],
                                      'd_stat': [i[6] for i in calib_tuples_with_isoshift_and_calibrationvoltage],
                                      'd_syst': [i[7] for i in calib_tuples_with_isoshift_and_calibrationvoltage]
                                      }
                         }
        calibrations_dict = {'58Ni_ref': {current_scaler_name: acc_volt_dict},
                             '60Ni_ref': {current_scaler_name: acc_volt_dict}
                             }
        # write to results dict
        TiTs.merge_extend_dicts(self.results, calibrations_dict, overwrite=True, force_overwrite=True)

        # plot calibration voltages:
        overlay = [29850]*len(calib_tuples)
        self.plot_parameter_for_isos_and_scaler(['58Ni_ref'], [current_scaler_name], 'acc_volts', overlay=overlay)


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

    def assign_calibration_voltages_interpolation(self, isotope, ref_isotope, scaler):
        if type(scaler) is int:
            scaler = 'scaler_{}'.format(scaler)
        # get reference run times and voltages from database
        ref_dates = self.results[ref_isotope]['file_times']
        ref_volts = self.results[ref_isotope][scaler]['acc_volts']['vals']
        ref_volts_d = self.results[ref_isotope][scaler]['acc_volts']['d_stat']
        ref_volts_d_syst = self.results[ref_isotope][scaler]['acc_volts']['d_syst']

        # get run times for isotope to be calibrated from database
        iso_dates = self.results[isotope]['file_times']

        # make floats (seconds relative to reference-time) out of the dates
        zero_time = ref_dates[0]
        ref_dates = list((t - zero_time).total_seconds() for t in ref_dates)
        iso_dates = list((t - zero_time).total_seconds() for t in iso_dates)

        # use np.interp to assign calibration voltages to each Nickel 55 run.
        interpolation = np.interp(iso_dates, ref_dates, ref_volts)
        interpolation_d = np.interp(iso_dates, ref_dates, ref_volts_d)
        interpolation_d_syst = np.interp(iso_dates, ref_dates, ref_volts_d_syst)

        # write calibration voltages back into database
        # TODO: how to assign errors to the interpolation?
        isotope_cal = '{}_cal'.format(isotope)
        voltdict = {isotope_cal: {scaler: {'acc_volts': {'vals': interpolation.tolist(),
                                                         'd_stat': interpolation_d.tolist(),
                                                         'd_syst': interpolation_d_syst.tolist()
                                                         }
                                           }
                                  }
                    }
        TiTs.merge_extend_dicts(self.results, voltdict)

        # make a quick plot of references and calibrated voltages
        plt.plot(ref_dates, ref_volts, 'o')
        plt.plot(iso_dates, interpolation, 'x')
        plt.show()

        # Write calibrations to XML database
        print('Updating self.db with new voltages now...')
        iso_names = self.results[isotope]['file_names']
        iso_numbers = self.results[isotope]['file_numbers']
        for file in iso_names:
            ind = iso_names.index(file)
            new_voltage = interpolation[ind]
            fileno = iso_numbers[ind]

            # Update 'Files' in self.db
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            cur.execute('''UPDATE Files SET accVolt = ?, type = ? WHERE file = ? ''',
                        (new_voltage, '{}_{}'.format(isotope_cal, fileno), file))
            con.commit()
            con.close()

            # Create new isotopes in self.db
            self.create_new_isotope_in_db(isotope, '{}_{}'.format(isotope_cal, fileno), new_voltage)
        print('...self.db update completed!')

    def assign_calibration_voltage_handassigned(self, isotope, ref_isotope, scaler, assigned):
        '''
        :return:
        '''
        if type(scaler) is int:
            scaler = 'scaler_{}'.format(scaler)
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

    ''' Nickel 56 related functions: '''

    def extract_iso_shift_interp(self, isotope, scaler, calibrated=True):
        if type(scaler) is int:
            scaler = 'scaler_{}'.format(scaler)
        if calibrated:
            iso58 = '58Ni_cal'
        else:
            iso58 = '58Ni'

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
            iso_volts = np.full((len(iso_numbers)), 29850)
            iso_volts_d = np.full((len(iso_numbers)), 0)
            iso_volts_d_syst = np.full((len(iso_numbers)), 0)

        # Get info for 58 Nickel:
        ni58_dates = self.results[iso58]['file_times']
        ni58_numbers = self.results[iso58]['file_numbers']
        ni58_center = self.results[iso58][scaler]['center_fits']['vals']
        ni58_center_d = self.results[iso58][scaler]['center_fits']['d_stat']
        ni58_center_d_syst = self.results[iso58][scaler]['center_fits']['d_syst']

        # TODO: isotope shift is not trivial, since the combination of 2 runs is not clear anymore.
        # I'll do an interpolation here as well...
        # make floats (seconds relative to reference-time) out of the dates
        zero_time = ni58_dates[0]
        ni58_dates = list((t - zero_time).total_seconds() for t in ni58_dates)
        iso_dates = list((t - zero_time).total_seconds() for t in iso_dates)
        # use np.interp to get a matching Nickel 58 center frequency for each Nickel 56 run.
        ni58_center_interp = np.interp(iso_dates, ni58_dates, ni58_center)
        ni58_center_d_interp = np.interp(iso_dates, ni58_dates, ni58_center_d)
        ni58_center_d_syst_interp = np.interp(iso_dates, ni58_dates, ni58_center_d_syst)


        diff_Doppler_58 = Physics.diffDoppler(self.restframe_trans_freq, np.array(iso_volts), 58)
        diff_Doppler_56 = Physics.diffDoppler(self.restframe_trans_freq, np.array(iso_volts), int(isotope[:2]))
        delta_diff_doppler = diff_Doppler_56 - diff_Doppler_58

        # calculate isotope shifts now:
        iso_shifts = np.array(iso_center) - np.array(ni58_center_interp)
        iso_shifts_d = np.sqrt(np.array(iso_center_d) ** 2 + np.array(ni58_center_d_interp) ** 2)
        iso_shifts_d = np.sqrt(iso_shifts_d**2 + (delta_diff_doppler * iso_volts_d)**2)
        iso_shifts_d_syst = np.sqrt(np.array(iso_center_d_syst) ** 2 + np.array(ni58_center_d_syst_interp) ** 2)
        # use differential doppler shift with cal voltage uncertainty to get MHz uncertainty:
        # TODO: do I need the systematic uncertainties of center fit positions here as well?
        iso_shifts_d_syst = np.sqrt((delta_diff_doppler * iso_volts_d_syst) ** 2 + iso_shifts_d_syst ** 2)

        # calculate an average value using weighted avg
        weights = 1/iso_shifts_d**2
        iso_shift_avg = np.sum(weights*iso_shifts)/np.sum(weights)
        iso_shift_avg_d = np.sqrt(1/np.sum(weights))
        # TODO: how to assign systematic uncertainties here? I think I should not weight them...
        iso_shift_avg_d_syst = sum(iso_shifts_d_syst)/len(iso_shifts_d_syst)

        # write isoshift to results dict
        shift_dict = {isotope: {scaler: {'shifts_iso-58': {'vals': iso_shifts.tolist(),
                                                           'd_stat': iso_shifts_d.tolist(),
                                                           'd_syst': iso_shifts_d_syst.tolist()},
                                         'avg_shift_iso-58': {'val': iso_shift_avg,
                                                              'd_stat': iso_shift_avg_d,
                                                              'd_syst': iso_shift_avg_d_syst}
                                         }}}
        TiTs.merge_extend_dicts(self.results, shift_dict, overwrite=True, force_overwrite=True)
        # TODO: Would it make sense to calculate back to the center DAC here, and go manually to frequency again in order to include uncertainties like voltage?

    def extract_iso_shift_handassigned(self, isotope, scaler, assigned, calibrated=True):
        if type(scaler) is int:
            scaler = 'scaler_{}'.format(scaler)
        if calibrated:
            iso58 = '58Ni_cal'
        else:
            iso58 = '58Ni'

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
            iso_volts = np.full((len(iso_numbers)), 29850)
            iso_volts_d = np.full((len(iso_numbers)), 0)
            iso_volts_d_syst = np.full((len(iso_numbers)), 0)

        # Get info for 58 Nickel:
        ni58_dates = self.results[iso58]['file_times']
        ni58_numbers = self.results[iso58]['file_numbers']
        ni58_center = self.results[iso58][scaler]['center_fits']['vals']
        ni58_center_d = self.results[iso58][scaler]['center_fits']['d_stat']
        ni58_center_d_syst = self.results[iso58][scaler]['center_fits']['d_syst']

        # create new lists that match iso_center lists in length and contain the corresponding 58 centers
        ni58_center_assigned = []
        ni58_center_d_assigned = []
        ni58_center_d_syst_assigned = []

        for tuples in assigned:
            iso_run_no = tuples[0]
            ref_run_no = tuples[1][0]

            iso_indx = iso_numbers.index(iso_run_no)
            ref_indx = ni58_numbers.index(ref_run_no)

            ni58_center_assigned.append(ni58_center[ref_indx])
            ni58_center_d_assigned.append(ni58_center_d[ref_indx])
            ni58_center_d_syst_assigned.append(ni58_center_d_syst[ref_indx])

        diff_Doppler_58 = Physics.diffDoppler(self.restframe_trans_freq, np.array(iso_volts), 58)
        diff_Doppler_56 = Physics.diffDoppler(self.restframe_trans_freq, np.array(iso_volts), int(isotope[:2]))
        delta_diff_doppler = diff_Doppler_56 - diff_Doppler_58

        # calculate isotope shifts now:
        iso_shifts = np.array(iso_center) - np.array(ni58_center_assigned)
        iso_shifts_d = np.sqrt(np.array(iso_center_d) ** 2 + np.array(ni58_center_d_assigned) ** 2)
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
        shift_dict = {isotope: {scaler: {'shifts_iso-58': {'vals': iso_shifts.tolist(),
                                                           'd_stat': iso_shifts_d.tolist(),
                                                           'd_syst': iso_shifts_d_syst.tolist()},
                                         'avg_shift_iso-58': {'val': iso_shift_avg,
                                                              'd_stat': iso_shift_avg_d,
                                                              'd_syst': iso_shift_avg_d_syst}
                                         }}}
        TiTs.merge_extend_dicts(self.results, shift_dict, overwrite=True, force_overwrite=True)

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
        lensc0 = len(self.results['56Ni_cal']['file_numbers'])
        plt.errorbar(range(lensc0), self.results['56Ni_cal']['scaler_0']['shifts_iso-58']['vals'], c='b',
                     yerr=self.results['56Ni_cal']['scaler_0']['shifts_iso-58']['d_stat'],
                     label='scaler 0')
        lensc1 = len(self.results['56Ni_cal']['file_numbers'])
        plt.errorbar(range(lensc1), self.results['56Ni_cal']['scaler_1']['shifts_iso-58']['vals'], c='g',
                     yerr=self.results['56Ni_cal']['scaler_1']['shifts_iso-58']['d_stat'],
                     label='scaler 1')
        lensc2 = len(self.results['56Ni_cal']['file_numbers'])
        plt.errorbar(range(lensc2), self.results['56Ni_cal']['scaler_2']['shifts_iso-58']['vals'], c='r',
                     yerr=self.results['56Ni_cal']['scaler_2']['shifts_iso-58']['d_stat'],
                     label='scaler 2')
        # Plot weighted average and errorband for all scalers
        avg_is = self.results['56Ni_cal']['scaler_0']['avg_shift_iso-58']['val']
        avg_is_d = self.results['56Ni_cal']['scaler_0']['avg_shift_iso-58']['d_stat']
        avg_is_d_sys = self.results['56Ni_cal']['scaler_0']['avg_shift_iso-58']['d_syst']
        plt.plot([-1, lensc0], [avg_is, avg_is], c='b')
        plt.fill([-1, lensc0, lensc0, -1],  # statistical error
                 [avg_is - avg_is_d, avg_is - avg_is_d, avg_is + avg_is_d, avg_is + avg_is_d], 'b', alpha=0.2)
        plt.fill([-1, lensc0, lensc0, -1],  # systematic error
                 [avg_is - avg_is_d_sys, avg_is - avg_is_d_sys, avg_is + avg_is_d_sys, avg_is + avg_is_d_sys], 'b',
                 alpha=0.05)
        avg_is1 = self.results['56Ni_cal']['scaler_1']['avg_shift_iso-58']['val']
        avg_is1_d = self.results['56Ni_cal']['scaler_1']['avg_shift_iso-58']['d_stat']
        avg_is1_dsys = self.results['56Ni_cal']['scaler_1']['avg_shift_iso-58']['d_syst']
        plt.plot([-1, lensc1], [avg_is1, avg_is1], c='g')
        plt.fill([-1, lensc1, lensc1, -1],  # statistical error
                 [avg_is1 - avg_is1_d, avg_is1 - avg_is1_d, avg_is1 + avg_is1_d, avg_is1 + avg_is1_d], 'g', alpha=0.2)
        plt.fill([-1, lensc1, lensc1, -1],  # systematic error
                 [avg_is1 - avg_is1_dsys, avg_is1 - avg_is1_dsys, avg_is1 + avg_is1_dsys, avg_is1 + avg_is1_dsys], 'g',
                 alpha=0.05)
        avg_is2 = self.results['56Ni_cal']['scaler_2']['avg_shift_iso-58']['val']
        avg_is2_d = self.results['56Ni_cal']['scaler_2']['avg_shift_iso-58']['d_stat']
        avg_is2_dsys = self.results['56Ni_cal']['scaler_2']['avg_shift_iso-58']['d_syst']
        plt.plot([-1, lensc2], [avg_is2, avg_is2], c='r')
        plt.fill([-1, lensc2, lensc2, -1],
                 [avg_is2 - avg_is2_d, avg_is2 - avg_is2_d, avg_is2 + avg_is2_d, avg_is2 + avg_is2_d], 'r', alpha=0.2)
        plt.fill([-1, lensc2, lensc2, -1],  # systematic error
                 [avg_is2 - avg_is2_dsys, avg_is2 - avg_is2_dsys, avg_is2 + avg_is2_dsys, avg_is2 + avg_is2_dsys], 'r',
                 alpha=0.05)
        # plt.plot(range(len(ni56_isoShift_yData)), ni56_isoShift_yData, '-o', label='preferred')
        # plt.plot(range(len(ni56_isoShift_yData)), ni56_isoShift_alt_yData, 'r-o', label='alternative')
        plt.xticks(range(lensc0), self.results['56Ni_cal']['file_numbers'], rotation=-30)
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
        self.stack_runs('58Ni', ni58_files, (-46, 14), binsize=1)
        # stack nickel 60 runs to new file Sum60_9999.xml. Only use calibration runs
        ni60_files, ni60_filenos = self.pick_files_from_db_by_type_and_num('%60Ni_cal%', selecttuple=[0, 6502])
        self.stack_runs('60Ni', ni60_files, (-46, 14), binsize=1)
        # stack nickel 56 runs to new file Sum56_9999.xml
        ni56_files, ni56_filenos = self.pick_files_from_db_by_type_and_num('%56Ni%', selecttuple=[0, 6502])
        self.stack_runs('56Ni', ni56_files, (-36, 14), binsize=1)
        # select and stack nickel 55 runs to new file Sum55_9999.xml
        ni55_files, ni55_filenos = self.pick_files_from_db_by_type_and_num('%55Ni%', selecttuple=restriction)
        self.stack_runs('55Ni', ni55_files, (-270, -30), binsize=3)

    def stack_runs(self, isotope, files, volttuple, binsize):
        ##############
        # stack runs #
        ##############
        # sum all the isotope runs
        self.time_proj_res_per_scaler = self.stack_time_projections(files)
        self.addfiles(isotope, files, volttuple, binsize)

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
        fitfunction for time projection
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

    def addfiles(self, iso, filelist, voltrange, binsize):
        startvoltneg = -voltrange[0]  # negative starting volts (don't use the -)
        scanrange = voltrange[1]-voltrange[0]  # volts scanning up from startvolt
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
            offst = 100  # background offset in timebins TODO: Maybe use multiple of sigma here?
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
                # plt.plot(voltage_x, scaler_sum_cts[scaler], drawstyle='steps', label=filenumber)
        # plt.show()
        sumerr = [np.sqrt(sumabs[0]), np.sqrt(sumabs[1]), np.sqrt(sumabs[2])]
        # prepare sumcts for transfer to xml file
        zero_ind = np.where(
            bgcounter[0] == 0)  # find zero-values. Attention! These should only be at start and end, not middle
        for scaler in range(3):
            bgcounter[scaler] = np.delete(bgcounter[scaler], zero_ind)
            sumabs[scaler] = np.delete(sumabs[scaler], zero_ind)
            sumerr[scaler] = np.delete(sumerr[scaler], zero_ind)
        sumvolts = np.delete(sumvolts, zero_ind)

        # plt.errorbar(sumvolts * binsize, sumabs[0], yerr=sumerr[0], fmt='.')
        # plt.show()
        plt.errorbar(sumvolts * binsize, sumabs[0] / bgcounter[0], yerr=sumerr[0] / bgcounter[0], fmt='.', label='scaler_0')
        plt.errorbar(sumvolts * binsize, sumabs[1] / bgcounter[1], yerr=sumerr[1] / bgcounter[1], fmt='.', label='scaler_1')
        plt.errorbar(sumvolts * binsize, sumabs[2] / bgcounter[2], yerr=sumerr[2] / bgcounter[2], fmt='.', label='scaler_2')
        plt.show()

        # prepare sumcts for transfer to xml file
        total_scale_factor = 0  # This might need to be adjusted when changing the number of scalers involved.
        # replaced scale factor by bgcounter.mean()
        for scaler in range(3):
            scale_factor = bgcounter[scaler].mean()
            sumabs[scaler] = (sumabs[scaler] / bgcounter[scaler] * scale_factor).astype(int)
            sumerr[scaler] = (sumerr[scaler] / bgcounter[scaler] * scale_factor).astype(int)
            total_scale_factor += (sumabs[scaler].max() - sumabs[scaler].min())  # we also need to scale the intensity of the isotope

        self.make_sumXML_file(iso, sumvolts[0]*binsize, binsize, len(sumabs[0]), np.array(sumabs), np.array(sumerr), scale_factor=total_scale_factor)

    def make_sumXML_file(self, isotope, startVolt, stepSizeVolt, nOfSteps, cts_list, err_list=None, scale_factor=1):
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
        if isopars_lst[3] != 0:
            # spin different from zero, several sidepeaks, adjust scaling!
            scale_factor = scale_factor/10
        isopars_lst[11] = int(scale_factor)  # change intensity scaling
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

    def calibrateVoltage_sumfiles(self, scaler):
        # extract isotope shift
        center58 = self.results['58Ni_sum'][scaler]['center_fits']['vals']
        center58_d = self.results['58Ni_sum'][scaler]['center_fits']['d_stat']
        center58_d_syst = self.results['58Ni_sum'][scaler]['center_fits']['d_syst']
        center60 = self.results['60Ni_sum'][scaler]['center_fits']['vals']
        center60_d = self.results['60Ni_sum'][scaler]['center_fits']['d_stat']
        center60_d_syst = self.results['60Ni_sum'][scaler]['center_fits']['d_syst']

        isoShift60 = center60[0] - center58[0]  # only one value in each of these lists
        isoShift60_d = np.sqrt(center60_d[0] ** 2 + center58_d[0] ** 2)
        isoShift60_d_syst = np.sqrt(center60_d_syst[0] ** 2 + center58_d_syst[0] ** 2)

        # new calibration:
        diff_Doppler_58 = Physics.diffDoppler(self.restframe_trans_freq, 29850, 58)
        diff_Doppler_60 = Physics.diffDoppler(self.restframe_trans_freq, 29850, 60)
        delta_diff_doppler = diff_Doppler_60 - diff_Doppler_58
        # calculate calibration Voltage:
        calibrated_voltage = (isoShift60 - self.literature_IS60vs58) / delta_diff_doppler + 29850
        calibrated_voltage_d = np.sqrt((isoShift60_d / delta_diff_doppler) ** 2 +
                                       (self.literature_IS60vs58_d_stat / delta_diff_doppler) ** 2)
        calibrated_voltage_d_syst = np.sqrt((isoShift60_d_syst / delta_diff_doppler) ** 2 +
                                            (self.literature_IS60vs58_d_syst / delta_diff_doppler) ** 2)

        # write voltages in self.results
        for iso in ['58Ni_sum', '60Ni_sum', '56Ni_sum', '55Ni_sum']:
            isodict = {iso: {scaler: {'acc_volts': {'vals': [calibrated_voltage],
                                                          'd_stat': [calibrated_voltage_d],
                                                          'd_syst': [calibrated_voltage_d_syst]
                                                          }
                                            }
                             }
                       }
            TiTs.merge_extend_dicts(self.results, isodict, overwrite=True, force_overwrite=True)

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
        plt.show()

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


if __name__ == '__main__':

    analysis = NiAnalysis()
    analysis.centerFreq_to_absVoltage('60Ni', 0,1,2)
    analysis.separate_runs_analysis()
    analysis.stacked_runs_analysis()
    analysis.export_results()