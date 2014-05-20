'''
Created on 30.04.2014

@author: hammen
'''

import csv
import sqlite3
from datetime import datetime

import numpy as np

from Measurement.SpecData import SpecData
import Physics

class TLDImporter(SpecData):
    '''
    This object reads a file with tab separated values into the ScanData structure
    
     The first column of the file is interpreted as scanning voltage, all following as scalers
    The header has 10 lines
    '''

    def __init__(self, path, db):
        '''Read the file'''
        
        print("TLDImporter is reading file", path)
        super(TLDImporter, self).__init__()
        con = sqlite3.connect(db)
        cur = con.cursor()
        
        self.path = path
        cur.execute('''SELECT accVolt, laserFreq, colDirTrue, line, type, voltDivRatio, lineMult, lineOffset, offset WHERE path = ?''', (path,))
        data = cur.fetchall()
        if len(data) == 1:
            (self.accVolt, self.laserFreq, self.colDirTrue, self.line, self.type, self.voltDivRatio, self.lineMult, self.lineOffset, self.offset) = data[0]
        else:
            raise Exception('TLDImporter: No DB-entry found!')

        l = self.dimension(path)
        self.nrScalers = l[1] - 1
        self.nrTracks = 1
        
        self.x = [np.zeros(l[0])] * self.nrTracks
        self.cts = [np.zeros((self.nrScalers, l[0]))] * self.nrTracks
        self.err = [np.zeros((self.nrScalers, l[0]))] * self.nrTracks
        
        with open(path) as f:
            self.date = datetime.strptime(f.readline(), '%d.%m.%Y\t%H:%M')
            self.offset = self.offset*self.voltDivRatio/1000
            f.readline()
            self.stepSize = self.getFloat(f)
            f.readline()
            self.nrLoops =  self.getFloat(f)
            self.dwell = self.getFloat(f)
            f.readline()
            f.readline()
            self.columnnames = f.readline().split('\t')
            read = csv.reader(f, delimiter = '\t')
            for i, row in enumerate(read):
                scanvolt = self.lineMult * (float(row[0].replace(',', '.'))/50) + self.lineOffset + self.offset
                self.x[0][i] = self.accVolt - scanvolt
                for j, counts in enumerate(row[1:]):
                    self.cts[0][j][i] = float(counts.replace(',', '.'))
                    self.err[0][j][i] = max(np.sqrt(float(counts.replace(',', '.'))), 1)
 
        
    def dimension(self, path):
        '''returns the nr of lines and columns of the file'''
        with open(path) as f:
            for i in range(0,4):
                f.readline()
            lines = int(self.getFloat(f))
            for i in range(0,4):
                f.readline()
            cols = len(f.readline().split('\t'))        
        return (lines, cols)
    
    def getFloat(self, f):
        return float(f.readline().split('\t')[1])
    