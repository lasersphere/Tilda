'''
Created on 08.08.2016

@author: tratajzcyk
'''

import ast
import csv
import os
import sqlite3
from datetime import datetime

import numpy as np

from Measurement.SpecData import SpecData


class ALIVEImporter(SpecData):
    '''
    This object reads a file with tab separated values into the ScanData structure
    
     The first column of the file is interpreted as scanning voltage, all following as scalers
    The header has 10 lines
    '''

    def __init__(self, path):
        '''Read the file'''
        
        print("ALIVEImporter is reading file", path)
        super(ALIVEImporter, self).__init__()

        
        self.file = os.path.basename(path)

        l = self._dimension(path)
        self.nrScalers = 1
        self.nrTracks = 1
        
        with open(path) as f:
            fmt = '%Y-%m-%d\t%H-%M-%S'
            date = self.file.split('_')[:2]
            date = str(date[0]) + '\t' + str(date[1])
            print(date)
            self.date = datetime.strptime(date, fmt)
            f.readline()
            self.informations = f.readline()
            startVolt = float(self.informations[13:self.informations.find(';', 13)])
            ind = self.informations.find('EndVoltage')
            stopVolt = float(self.informations[ind + 10:self.informations.find(';', ind + 10)])
            ind = self.informations.find('voltageStep')
            self.stepSize = float(self.informations[ind + 11:self.informations.find(';', ind + 11)])
            ind = self.informations.find('dwellTime')
            self.dwell = float(self.informations[ind + 9:self.informations.find(';', ind + 9)])*10**-6
            self.nrLoops = float(self.informations[self.informations.find('numScans', ind) + 8:self.informations.find('\n', ind + 8)])
            self.columnnames = f.readline().split(';')
            read = csv.reader(f, delimiter = ';')
            l = np.floor((stopVolt-startVolt)/self.stepSize)+1
            self.x = [np.zeros(l)]
            self.cts = [np.zeros((self.nrScalers, l))]
            self.err = [np.zeros((self.nrScalers, l))]

            for i, row in enumerate(read):
                self.x[0][i] = float(row[0])
                for j, counts in enumerate(row[1:]):
                    self.cts[0][j][i] = float(counts)
                    self.err[0][j][i] = max(np.sqrt(float(counts)), 1)

    def preProc(self, db):
        print('ALIVEImporter is using db', db)
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute('''SELECT accVolt, laserFreq, colDirTrue, line, type, voltDivRatio, lineMult, lineOffset, offset
                        FROM Files WHERE file = ?''', (self.file,))
        data = cur.fetchall()
        if len(data) == 1:
            (self.accVolt, self.laserFreq, self.col, self.line, self.type, self.voltDivRatio, self.lineMult,
                    self.lineOffset, self.offset) = data[0]
        else:
            raise Exception('ALIVEImporter: No DB-entry found!')
        self.col = bool(self.col)
        try:
            self.voltDivRatio = ast.literal_eval(self.voltDivRatio)
        except Exception as e:
            print('error while converting the voltdivratio: %s' % e)
            self.voltDivRatio = {'accVolt': 1, 'offset': 1}
            print('using now: %s' % self.voltDivRatio)
        for trackindex, tracks in enumerate(self.x):
            for xindex, x in enumerate(tracks):
                scanvolt = (self.lineMult * x + self.lineOffset + self.offset) * self.voltDivRatio['offset']
                self.x[trackindex][xindex]= self.accVolt*self.voltDivRatio['accVolt'] - scanvolt
        print(self.x)
        print(self.cts)
    
    def export(self, db):
        con = sqlite3.connect(db)
        with con:
            con.execute('''UPDATE Files SET date = ? WHERE file = ?''', (self.date, self.file))     
        con.close()
    
    
    def _dimension(self, path):
        '''returns the nr of lines and columns of the file'''
        with open(path) as f:
            for i in range(0,4):
                f.readline()
            lines = int(self.getFloat(f))
            for i in range(0,4):
                f.readline()
            cols = len(f.readline().split(';'))
        return (lines, cols)


    def getFloat(self, f):
        return float(f.readline().split(';')[1])

if __name__=='__main__':
    test_data = 'Data/2016-07-22_09-15-46_003.dat'
    project = 'test/Project'
    db = 'AliveDB.sqlite'
    project_path = os.path.normpath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir, project))
    test_data_path = os.path.normpath(os.path.join(project_path, test_data))
    meas = ALIVEImporter(test_data_path)
    db = os.path.join(project_path, db)
    print(db, os.path.isfile(db))
    meas.preProc(db)
    import MPLPlotter as MplPl
    plt = MplPl.plot(meas.getArithSpec([0], -1))
    MplPl.show(True)