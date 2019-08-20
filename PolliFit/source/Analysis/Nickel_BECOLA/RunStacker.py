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

        # TODO: there should definitely be an option to pick files from database (say based on isotope)
        # TODO: also drag&drop would be nice...
        files = ['BECOLA_6742.xml', 'BECOLA_6744.xml', 'BECOLA_6751.xml', 'BECOLA_6753.xml']  # List of file paths to analyze
        self.files = []
        for file in files:
            file = os.path.join(self.datafolder, file)
            self.files.append(file)

        self.pickfilesfromdb('%55Ni%')

        self.loadfiles()

    def loadfiles(self):
        binsize = 3
        norm = True  # normalize? True or False
        startvoltneg = 250 # negative starting volts (don't use the -)
        scanrange = 250  # volts scanning up from startvolt
        sumcts = np.zeros(scanrange//binsize)  # should contain all the 55 scans so roughly -350 to +100
        addcounter = np.zeros(scanrange//binsize)  # array to keep track of how often data was added to a bin
        sumvolts = np.arange(scanrange//binsize)-startvoltneg/binsize
        for files in self.files:
            filename = re.split('[_.]', files)[-2]
            spec = XMLImporter(path=files)
            if norm:
                nOfScans = spec.nrScans[0]
            else:
                nOfScans = 1
            scaler0_cts = spec.cts[0][0]
            voltage_x = spec.x[0]
            scaler0_timeproj = spec.t_proj[0][0]
            for datapoint_ind in range(len(voltage_x)):
                voltind = int(voltage_x[datapoint_ind] + startvoltneg)//binsize
                if 0 < voltind < len(sumcts):
                    sumcts[voltind] += scaler0_cts[datapoint_ind]/nOfScans
                    addcounter[voltind] += 1
            plt.plot(voltage_x, scaler0_cts, drawstyle='steps')
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

    def pickfilesfromdb(self, type):
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
            if 6468 <= fileno:
                self.files.append(file)




if __name__ == '__main__':

    analysis = RunStacker()