'''
Created on 30.04.2014

@author: hammen
'''

import csv

import numpy as np

from Measurement.SpecData import SpecData
import Experiment as Exp

class KepcoImporterTLD(SpecData):
    '''
    This object reads a file with tab separated values into the KepcoData structure
    
    In the first row of the file is the Heinzinger-Offset written.
    The first column of the file is interpreted as DAQ-voltage, the second as scanning voltage
    '''

    def __init__(self, path):
        '''Read the file'''
        
        print("KepcoImporterTLD is reading file", path)
        super(KepcoImporterTLD, self).__init__()
        
        self.path = path
        self.laserFreq = Exp.getLaserFreq(self.time)
        self.col = Exp.dirColTrue(self.time) 
        self.type = 'Kepco'
         
        l = self.dimension(path)
        self.nrScalers = l[1] - 1
        self.nrTracks = 1
        
        self.x = np.zeros((self.nrTracks, l[0]))
        self.cts = np.zeros((self.nrScalers, self.nrTracks, l[0]))
        self.err = np.zeros((self.nrScalers, self.nrTracks, l[0]))
        
        with open(path) as f:
            self.offset = float(f.readline().split('\t')[1])
            read = csv.reader(f, delimiter = '\t')
            for i, row in enumerate(read):
                self.x[0][i] = float(row[0])
                for j, scanVolt in enumerate(row[1:]):
                    self.cts[j][0][i] = (float(scanVolt) - self.offset)*Exp.getVoltDivRatio()
                    self.err[j][0][i] = 10**-4
 
        
    def dimension(self, path):
        '''returns the nr of lines and columns of the file'''
        lines = 0
        with open(path) as f:
            cols = len(f.readline().split('\t'))
            for line in f:
                lines += 1        
        return (lines, cols)
    