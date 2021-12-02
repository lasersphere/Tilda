"""
Created on 2019-7-26

@author: fsommer

Module Description:  import a couple of .xml files and show them together.
This should help to compare runs, find resonances and get an overview of the data.
"""
import os
import re
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from Measurement.XMLImporter import XMLImporter

from lxml import etree as ET
from XmlOperations import xmlWriteDict


class RunStacker():
    def __init__(self):

        # Set working directory and database
        ''' working directory: '''
        # get user folder to access ownCloud
        user_home_folder = os.path.expanduser("~")
        # self.workdir = 'C:\\DEVEL\\Analysis\\Ni_Analysis\\XML_Data' # old working directory
        ownCould_path = 'ownCloud\\User\\Felix\\Measurements\\Nickel_online_Becola\\Analysis\\XML_Data'
        #ownCould_path = 'ownCloud\\User\\Felix\\Measurements\\Pd_offline_PIG_MSU\\Analysis\\xml'
        self.workdir = os.path.join(user_home_folder, ownCould_path)
        ''' data folder '''
        self.datafolder = os.path.join(self.workdir, 'SumsRebinned')
        ''' database '''
        self.db = os.path.join(self.workdir, 'Ni_Becola.sqlite')

        # # TODO: also drag&drop would be nice...
        # files = ['Sum55Ni_9999.xml', 'Sum56Ni_9999.xml', 'Sum58Ni_9999.xml', 'Sum60Ni_9999.xml']  # List of file paths to analyze
        # self.files = []
        # for file in files:
        #     file = os.path.join(self.datafolder, file)
        #     self.files.append(file)

        self.pickfilesfromdb('%56Ni%')  #'%55Ni%', bounds=(6315, 6502)

        self.loadfiles()

    def loadfiles(self):
        binsize = 3
        norm = True  # normalize? True or False
        startvoltneg = 360  # negative starting volts (don't use the -)
        scanrange = 420  # volts scanning up from startvolt
        sumcts = np.zeros(scanrange//binsize)  # should contain all the 55 scans so roughly -350 to +100
        sumabs = sumcts.copy()  # array for the absolute counts per bin. Same dimension as counts of course
        bgcounter = np.zeros(scanrange // binsize)  # array to keep track of the backgrounds
        sumvolts = np.arange(scanrange//binsize)-startvoltneg/binsize
        for files in self.files:
            filenumber = re.split('[_.]', files)[-2]
            filename = 'BECOLA_'+str(filenumber)+'.xml'
            gw = 0.03  # gate width
            spec = XMLImporter(path=files,
                               softw_gates=[[-350, 0, 5.24-gw, 5.24+gw],
                                            [-350, 0, 5.42-gw, 5.42+gw],  # first scaler shifted by 1.9 compared to sc0
                                            [-350, 0, 5.50-gw, 5.50+gw]])  # second scaler shifted by 2.7 compared to sc0
            background = XMLImporter(path=files,  # sample spec of the same width, clearly separated from the timepeaks
                               softw_gates=[[-350, 0, 4.50-gw, 4.50+gw],
                                            [-350, 0, 4.69-gw, 4.69+gw],  # first scaler shifted by 1.9 compared to sc0
                                            [-350, 0, 4.77-gw, 4.77+gw]])  # second scaler shifted by 2.7 compared to sc0
            stepsize = spec.stepSize[0]
            nOfSteps = spec.getNrSteps(0)
            nOfScans = spec.nrScans[0]
            nOfBunches = spec.nrBunches[0]
            voltage_x = spec.x[0]
            scaler0_cts = spec.cts[0][0]
            scaler1_cts = spec.cts[0][1]
            scaler2_cts = spec.cts[0][2]
            bg0_cts = background.cts[0][0]
            bg1_cts = background.cts[0][1]
            bg2_cts = background.cts[0][2]
            scaler_sum_cts = scaler0_cts+scaler1_cts#+scaler2_cts
            bg_sum_cts = bg0_cts+bg1_cts#+bg2_cts
            if norm:
                scaler0_totalcts = sum(scaler0_cts)
                scaler1_totalcts = sum(scaler1_cts)
                scaler2_totalcts = sum(scaler2_cts)
                scaler_sum_totalcts = sum(scaler_sum_cts)
                bg_sum_totalcts = sum(bg_sum_cts)
                if scaler_sum_totalcts == 0: scaler_sum_totalcts = 1
                if bg_sum_totalcts ==0: bg_sum_totalcts = 1
            else:
                nOfScans = 1
                scaler0_totalcts = 1
                scaler1_totalcts = 1
                scaler_sum_totalcts = 1
                bg_sum_totalcts = 1
            #scaler0_timeproj = spec.t_proj[0][0]
            for datapoint_ind in range(len(voltage_x)):
                voltind = int(voltage_x[datapoint_ind] + startvoltneg)//binsize
                if 0 <= voltind < len(sumabs):
                    sumabs[voltind] += scaler_sum_cts[datapoint_ind]  # no normalization here
                    bgcounter[voltind] += bg_sum_totalcts / nOfSteps
            plt.plot(voltage_x, scaler_sum_cts, drawstyle='steps', label=filenumber)
        plt.show()
        sumerr = np.sqrt(sumabs)
        # prepare sumcts for transfer to xml file
        zero_ind = np.where(bgcounter == 0)  # find zero-values. Attention! These should only be at start and end, not middle
        bgcounter = np.delete(bgcounter, zero_ind)
        sumabs = np.delete(sumabs, zero_ind)
        sumerr = np.delete(sumerr, zero_ind)
        sumvolts = np.delete(sumvolts, zero_ind)

        plt.errorbar(sumvolts * binsize, sumabs, yerr=sumerr, fmt='.')
        plt.show()
        plt.errorbar(sumvolts * binsize, sumabs / bgcounter, yerr=sumerr / bgcounter, fmt='.')
        plt.show()

        # prepare sumcts for transfer to xml file
        sumcts = np.array([sumabs/bgcounter*1000]).astype(int)

        self.make_sumXML_file(1, -startvoltneg, binsize, len(sumcts[0]), sumcts)

    def pickfilesfromdb(self, type, bounds=None):
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute(
            '''SELECT file FROM Files WHERE type LIKE ? ''', (type,))
        files = cur.fetchall()
        con.close()
        # convert into np array
        filelist = [f[0] for f in files]
        self.files = []
        for file in filelist:
            fileno = int(re.split('[_.]', file)[1])
            file = os.path.join(self.datafolder, file)
            if bounds:
                if bounds[0] <= fileno <= bounds[1]:
                    self.files.append(file)

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

    def make_sumXML_file(self, nrScalers, startVolt, stepSizeVolt, nOfSteps, cts_list):
        ####################################
        # Prepare dicts for writing to XML #
        ####################################
        header_dict = {'type': 'cs',
                       'isotope': '55Ni',
                       'isotopeStartTime': '2018-04-13 12:53:06',
                       'accVolt': 29850,
                       'laserFreq': 14197.56675,
                       'nOfTracks': 1,
                       'version': 99.0}

        track_dict_header = {'trigger': {},  # Need a trigger dict!
                             'activePmtList': list(range(nrScalers)),  # Must be in form [0,1,2]
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
                             'workingTime': ['2018-04-13 12:53:06', '2018-04-13 12:53:06'],
                             'waitAfterReset1us': 0,  # looks like I need those for the importer
                             'waitForKepco1us': 0  # looks like I need this too
                             }
        track_dict_data = {
            'scalerArray_explanation': 'continously acquired data. List of Lists, each list represents the counts of '
                                       'one scaler as listed in activePmtList.Dimensions are: (len(activePmtList), '
                                       'nOfSteps), datatype: np.int32',
            'scalerArray': cts_list}


if __name__ == '__main__':

    analysis = RunStacker()