"""

Created on '07.08.2015'

@author:'simkaufm'

"""

import csv
import sqlite3
from datetime import datetime
import os
import ast
import Physics

import numpy as np

import TildaTools
from Measurement.SpecData import SpecData


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

        self.file = os.path.basename(path)

        scandict, lxmlEtree = TildaTools.scan_dict_from_xml_file(path)
        self.nrTracks = scandict['isotopeData']['nOfTracks']

        self.laserFreq = Physics.freqFromWavenumber(2 * scandict['isotopeData']['laserFreq'])
        self.date = scandict['isotopeData']['isotopeStartTime']
        self.type = scandict['isotopeData']['isotope']

        self.offset = 0  # should also be a list for mutliple tracks
        self.nrScalers = []
        self.x = []
        self.cts = []
        self.err = []
        self.stepSize = []
        self.col = False  # should also be a list for multiple tracks
        self.dwell = []

        for tr in TildaTools.get_track_names(scandict):
            track_dict = scandict[tr]
            nOfactTrack = int(tr[5:])
            nOfsteps = track_dict['nOfSteps']
            nOfScalers = len(track_dict['activePmtList'])
            dacStart18Bit = track_dict['dacStartRegister18Bit']
            dacStepSize18Bit = track_dict['dacStepSize18Bit']
            dacStop18Bit = dacStart18Bit + (dacStepSize18Bit * nOfsteps)
            xAxis = np.arange(dacStart18Bit, dacStop18Bit, dacStepSize18Bit, dtype=np.float)
            ctsstr = TildaTools.xml_get_data_from_track(lxmlEtree, nOfactTrack, 'scalerArray')
            cts = TildaTools.numpy_array_from_string(ctsstr, (nOfScalers, nOfsteps))
            self.offset = track_dict['postAccOffsetVolt']
            self.nrScalers.append(nOfScalers)
            self.x.append(xAxis)
            self.cts.append(cts)
            self.err.append(np.sqrt(cts))
            self.stepSize.append(dacStepSize18Bit)
            self.col = track_dict['colDirTrue']
            self.dwell.append(track_dict['dwellTime10ns'])

    def preProc(self, db):
        print('XMLImporter is using db: ', db)
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute('''SELECT type, line, offset, accVolt, laserFreq,
                        colDirTrue, voltDivRatio, lineMult, lineOffset
                        FROM Files WHERE file = ?''', (self.file,))
        data = cur.fetchall()
        if len(data) == 1:
            (self.type, self.line, self.offset, self.accVolt, self.laserFreq,
             self.col, self.voltDivRatio, self.lineMult, self.lineOffset) = data[0]
            self.col = bool(self.col)
        else:
            raise Exception('XMLImporter: No DB-entry found!')

        for j in range(len(self.x)):
            for i in range(len(self.x[j])):
                scanvolt = self.lineMult * float(self.x[j][i]) + self.lineOffset + self.offset
                print(scanvolt)
                self.x[j][i] = float(float(self.accVolt) - scanvolt)
        con.close()

    def export(self, db):
        try:
            con = sqlite3.connect(db)
            with con:
                con.execute('''UPDATE Files SET date = ?, type = ?, offset = ?,
                                laserFreq = ?, colDirTrue = ?
                                 WHERE file = ?''',
                            (self.date, self.type, self.offset,
                             self.laserFreq, self.col,
                             self.file))
            con.close()
        except Exception as e:
            print(e)

    def evalErr(self, cts, f):
        cts = cts.reshape(-1)
        for i, v in enumerate(cts):
            cts[i] = f(v)
