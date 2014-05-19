'''
Created on 30.04.2014

@author: hammen
'''

import csv
import Physics

import numpy as np

from Measurement.SpecData import SpecData
import Experiment as Exp

class TLDImporter(SpecData):
    '''
    This object reads a file with tab separated values into the ScanData structure
    
     The first column of the file is interpreted as scanning voltage, all following as scalers
    The header has 10 lines
    '''

    def __init__(self, path):
        '''Read the file'''
        
        print("TLDImporter is reading file", path)
        super(TLDImporter, self).__init__()
        
        self.path = path
        self.col = Exp.dirColTrue(self.time) 
        self.laserFreq = Exp.getLaserFreq()
        self.type = 'Ca'

        l = self.dimension(path)
        self.nrScalers = l[1] - 1
        self.nrTracks = 1
        
        self.x = np.zeros((self.nrTracks, l[0]))
        self.cts = np.zeros((self.nrScalers, self.nrTracks, l[0]))
        self.err = np.zeros((self.nrScalers, self.nrTracks, l[0]))
        
        with open(path) as f:
            [self.date, self.time] = f.readline().split('\t')
            self.offset = self.getFloat(f)
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
                scanvolt = Exp.lineToScan(float(row[0].replace(',', '.'))/50) + self.offset
                self.x[0][i] = Exp.getAccVolt() - scanvolt
                for j, counts in enumerate(row[1:]):
                    self.cts[j][0][i] = float(counts.replace(',', '.'))
                    self.err[j][0][i] = max(np.sqrt(float(counts.replace(',', '.'))), 1)
 
        
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
    