"""

Created on '07.08.2015'

@author:'simkaufm'

"""

import csv
import sqlite3
from datetime import datetime
import os
import ast

import numpy as np

from Measurement.SpecData import SpecData
# iports from Tilda which will cause problems:
from Service.FileFormat.XmlOperations import xmlGetDataFromTrack
import Service.FolderAndFileHandling as tildaFileHandl
import Service.Formating as tildaForm
import Service.Scan.ScanDictionaryOperations as SdOp

class XMLImporter(SpecData):
    '''
    This object reads a file with tab separated values into the ScanData structure

     The first column of the file is interpreted as scanning voltage, all following as scalers
    The header has 10 lines
    '''

    def __init__(self, path):
        '''Read the file'''

        print("XMLImporter is reading file", path)
        super(XMLImporter, self).__init__()

        self.file = path

        scandict, lxmlEtree = tildaFileHandl.scanDictionaryFromXmlFile(self.file)
        self.nrTracks = scandict['isotopeData']['nOfTracks']

        self.laserFreq = scandict['isotopeData']['laserFreq']
        self.date = scandict['isotopeData']['isotopeStartTime']
        self.type = scandict['isotopeData']['type']
        self.isotope = scandict['isotopeData']['isotope']

        self.accVolt = []
        self.offset = []
        self.nrScalers = []
        self.x = []
        self.cts = []
        self.err = []
        self.stepSize = []
        self.col = []
        self.dwell = []

        for tr in SdOp.get_track_names(scandict):
            track_dict = scandict[tr]
            nOfactTrack = int(tr[5:])
            nOfsteps = track_dict['nOfSteps']
            nOfScalers = len(track_dict['activePmtList'])
            dacStart18Bit = track_dict['dacStartRegister18Bit']
            dacStepSize18Bit = track_dict['dacStepSize18Bit']
            dacStop18Bit = dacStart18Bit + (dacStepSize18Bit * nOfsteps)
            xAxis = np.arange(dacStart18Bit, dacStop18Bit, dacStepSize18Bit)
            ctsstr = xmlGetDataFromTrack(lxmlEtree, nOfactTrack, 'scalerArray')
            cts = tildaForm.numpy_array_from_string(ctsstr, (nOfScalers, nOfsteps))
            self.accVolt.append(track_dict['postAccOffsetVolt'])
            self.offset.append(track_dict['postAccOffsetVolt'])
            self.nrScalers.append(nOfScalers)
            self.x.append(xAxis)
            self.cts.append(cts)
            self.err.append(np.sqrt(cts))
            self.stepSize.append(dacStepSize18Bit)
            self.col.append(track_dict['colDirTrue'])
            self.dwell.append(track_dict['dwellTime10ns'])

    def preProc(self, db):
        print('XMLImporter is using db: ', db)
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute('''SELECT accVolt, laserFreq, line, type, voltDivRatio,
          lineMult, lineOffset FROM Files WHERE file = ?''', (os.path.split(self.file)[1],))
        data = cur.fetchall()
        print(data)
        if len(data) == 1:
            (self.accVolt, self.laserFreq, self.line, self.type, self.voltDivRatio,
             self.lineMult, self.lineOffset) = data[0]
            self.col = bool(self.col)
        else:
            raise Exception('XMLImporter: No DB-entry found!')
        #
        # for i in range(len(self.x[0])):
        #     scanvolt = self.lineMult * (self.x[0][i]) + self.lineOffset + self.offset
        #     self.x[0][i] = self.accVolt - scanvolt

        con.close()


    def export(self, db):
        con = sqlite3.connect(db)
        with con:
            con.execute('''UPDATE Files SET date = ? WHERE file = ?''', (self.date, self.file))
        con.close()

    def evalErr(self, cts, f):
        cts = cts.reshape(-1)
        for i, v in enumerate(cts):
            cts[i] = f(v)
