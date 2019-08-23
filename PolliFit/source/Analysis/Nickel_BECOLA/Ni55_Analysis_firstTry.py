"""
Created on 2019-08-21

@author: fsommer

Module Description:  Analysis of the 55-Nickel Data from BECOLA taken on 13.04.-23.04.2018
Not a stand-alone Analysis but requires the Analysis of 56 to be done before, because
it relies on the database entries for calibration voltages etc.
"""

import ast
import os
import sqlite3
from datetime import datetime
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mpdate
from scipy.optimize import curve_fit

import BatchFit
import Physics
import Tools

from Analysis.Nickel_BECOLA.ExcelWrite import ExcelWriter

from Measurement.XMLImporter import XMLImporter

class NiAnalysis():
    def __init__(self):

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

        # Pick isotopes and group
        self.isotopes = ['%sNi' % i for i in range(55, 60)]
        self.isotopes.remove('57Ni')
        self.isotopes.remove('59Ni')
        '''isotope groups'''
        self.odd_isotopes = [iso for iso in self.isotopes if int(iso[:2]) % 2]
        self.even_isotopes = [iso for iso in self.isotopes if int(iso[:2]) % 2 == 0]
        self.stables = ['58Ni', '60Ni', '61Ni', '62Ni', '64Ni']

        # Name this analysis run
        self.run_name = 'Voigt'

        # create excel workbook to save some results
        excel_path = 'ownCloud\\User\\Felix\\Measurements\\Nickel_online_Becola\\Analysis\\Results\\analysis_55_1st.xlsx'
        self.excelpath = os.path.join(user_home_folder, excel_path)
        self.excel = ExcelWriter(self.excelpath)
        self.excel.active_sheet = self.excel.wb.copy_worksheet(self.excel.wb['Template'])
        self.excel.active_sheet.title = self.run_name

        # Select runs; Format: ['run58', 'run60', 'run55']
        # to use a different lineshape you must create a new run under runs and a new linevar under lines and link the two.
        self.runs = ['CEC_Voigt_58', 'CEC_Voigt_60', 'CEC_Voigt_55']
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

        # safe run settings to workbook
        self.excel.wb.save(self.excelpath)

        #######################
        # Pick Nickel 55 runs #
        #######################
        restriction = [6315, 6500]  # optional: range of files to be included
        self.ni55_files, self.ni55_filenos = self.pickfilesfromdb('%55Ni%', selecttuple=restriction)
        self.ni55_datevoltfile = []  # list of tuples with (date, voltage, file)
        for files in self.ni55_files:
            ref_date, ref_volt = self.get_date_and_voltage(files)
            ref_datetime = datetime.strptime(ref_date, '%Y-%m-%d %H:%M:%S')
            self.ni55_datevoltfile.append((ref_datetime, ref_volt, files))
        self.ni55_datevoltfile.sort(key=lambda x: x[0])

        ######################
        # Reference Voltages #
        ######################
        # get 58 Nickel reference runs from database
        self.ni58ref_files, self.ni58ref_filenos = self.pickfilesfromdb('%58Ni_cal%')
        self.ni58ref_points = []  # list of tuples with (date, voltage, file)
        # extract timestamp and calibrated voltage for each reference point
        for files in self.ni58ref_files:
            ref_date, ref_volt = self.get_date_and_voltage(files)
            ref_datetime = datetime.strptime(ref_date, '%Y-%m-%d %H:%M:%S')
            self.ni58ref_points.append((ref_datetime, ref_volt, files))
        self.ni58ref_points.sort(key=lambda x: x[0])
        # use the timestamp of the first 58 run as a time-reference
        self.reference_time = self.ni58ref_points[0][0]
        # make floats (seconds relative to reference-time) out of the dates
        self.ni58ref_points = list((x[0]-self.reference_time, x[1], x[2]) for x in self.ni58ref_points)
        self.ni55_datevoltfile = list((x[0]-self.reference_time, x[1], x[2]) for x in self.ni55_datevoltfile)
        # split list of tuples into time and volt list
        ref_times = list(pt[0].total_seconds() for pt in self.ni58ref_points)
        ref_volts = list(pt[1] for pt in self.ni58ref_points)
        ni55_times = list(pt[0].total_seconds() for pt in self.ni55_datevoltfile)
        # use np.interp to assign calibration voltages to each Nickel 55 run.
        ni55_interpolation = np.interp(ni55_times, ref_times, ref_volts)
        self.ni55_datevoltfile = list((x[0], ni55_interpolation[self.ni55_datevoltfile.index(x)], x[2])
                                      for x in self.ni55_datevoltfile)
        # write calibration voltages back into database
        for ni55runs in self.ni55_datevoltfile:
            self.write_voltage_to_file(ni55runs)
        # make a quick plot of references and calibrated voltages
        plt.plot(ref_times, ref_volts, 'o')
        plt.plot(ni55_times, ni55_interpolation, '-x')
        plt.show()
        # sum all the Nickel 55 runs taking into account the calibration voltage.
        self.addfiles(self.ni55_files)



    def pickfilesfromdb(self, type, selecttuple=None):
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

    def get_date_and_voltage(self, filename):
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute(
            '''SELECT date, accVolt FROM Files WHERE file = ? ''', (filename,))
        fetch = cur.fetchall()
        con.close()
        file_date = fetch[0][0]
        file_volt = fetch[0][1]
        return file_date, file_volt

    def write_voltage_to_file(self, datevoltfile_tuple):
        cal_Voltage = datevoltfile_tuple[1]
        filename = datevoltfile_tuple[2]
        # Update 'Files' in self.db
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''UPDATE Files SET accVolt = ?, type = ? WHERE file = ? ''', (cal_Voltage,
                                                                                   '55Ni_cal',
                                                                                   filename))
        con.commit()
        con.close()

    def addfiles(self, filelist):
        binsize = 3
        norm = True  # normalize? True or False
        startvoltneg = 350  # negative starting volts (don't use the -)
        scanrange = 450  # volts scanning up from startvolt
        sumcts = np.zeros(scanrange//binsize)  # should contain all the 55 scans so roughly -350 to +100
        addcounter = np.zeros(scanrange//binsize)  # array to keep track of how often data was added to a bin
        sumvolts = np.arange(scanrange//binsize)-startvoltneg/binsize
        for files in filelist:
            filepath = os.path.join(self.datafolder, files)
            spec = XMLImporter(path=filepath,
                               softw_gates=[[-350, 0, 5.2, 5.4],
                                            [-350, 0, 5.394, 5.594],
                                            [-350, 0, 5.465, 5.665]])
            stepsize = spec.stepSize[0]
            nOfSteps = spec.getNrSteps(0)
            scaler0_cts = spec.cts[0][0]
            scaler1_cts = spec.cts[0][1]
            scaler2_cts = spec.cts[0][2]
            scaler_sum_cts = scaler0_cts+scaler1_cts#+scaler2_cts
            voltage_x = spec.x[0]
            calibration_voltage = self.get_date_and_voltage(files)[1]
            #voltage_x -= (calibration_voltage - 29850)
            if norm:
                nOfScans = spec.nrScans[0]
                nOfBunches = spec.nrBunches[0]
                scaler0_totalcts = sum(scaler0_cts)
                scaler1_totalcts = sum(scaler1_cts)
                scaler2_totalcts = sum(scaler2_cts)
                scaler_sum_totalcts = scaler0_totalcts+scaler1_totalcts#+scaler2_totalcts
                if scaler_sum_totalcts == 0: scaler_sum_totalcts=1
            else:
                nOfScans = 1
                scaler0_totalcts = 1
                scaler1_totalcts = 1
                scaler_sum_totalcts = 1
            scaler0_timeproj = spec.t_proj[0][0]
            for datapoint_ind in range(len(voltage_x)):
                voltind = int(voltage_x[datapoint_ind] + startvoltneg)//binsize
                if 0 < voltind < len(sumcts):
                    #sumcts[voltind] += scaler0_cts[datapoint_ind]/nOfScans
                    sumcts[voltind] += scaler_sum_cts[datapoint_ind]/(nOfScans * nOfBunches)
                    addcounter[voltind] += 1
            plt.plot(voltage_x, scaler0_cts, drawstyle='steps', label=files)
            #plt.title(filename)
            #plt.show()
            #plt.title(filename)
            #plt.plot(np.arange(len(scaler0_timeproj)), scaler0_timeproj, drawstyle='steps')
            #plt.show()
        plt.show()
        addcounter = np.where(addcounter == 0, 1, addcounter)  # remove all zeros from counter for division
        sumcts = sumcts/addcounter
        plt.plot(sumvolts*binsize, sumcts, drawstyle='steps')
        plt.show()

if __name__ == '__main__':

    analysis = NiAnalysis()