'''
Created on 08.08.2016

@author: tratajzcyk
'''

import csv, ast
import sqlite3
from datetime import datetime
import os

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
            date = path.split('_')
            date = str(date[0])[-10:] + '\t' + str(date[1])
            print(date)
            self.date = datetime.strptime(date, fmt )
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
            # print('x:', self.x)
            # print('cts:', self.cts)
            # print('err:', self.err)
 
 
    def preProc(self, db):
        print('MCPimporter is using db', db)
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute('''SELECT accVolt, laserFreq, colDirTrue, line, type, voltDivRatio, lineMult, lineOffset, offset
                                        FROM Files WHERE file = ?''', (self.file,))
        data = cur.fetchall()
        if len(data) == 1:
            (self.accVolt, self.laserFreq, self.col, self.line, self.type, self.voltDivRatio, self.lineMult,
                    self.lineOffset, self.offset) = data[0]
        else:
            raise Exception('MCPImporter: No DB-entry found!')
        self.col = bool(self.col)
        self.voltDivRatio = ast.literal_eval(self.voltDivRatio)
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

