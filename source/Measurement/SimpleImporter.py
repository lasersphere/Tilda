'''
Created on 30.04.2014

@author: hammen
'''

import csv

import numpy as np

from Measurement.SpecData import SpecData
import Experiment as Exp

class SimpleImporter(SpecData):
    '''
    This object reads a file with tab separated values into the SpecData structure
    
    The first column of the file is interpreted as scanning voltage, all following as scalers
    '''

    def __init__(self, path):
        '''Read the file'''
        
        print("SimpleImporter is reading file", path)
        super(SimpleImporter, self).__init__()
        
        self.path = path
        
        l = self.dimension(path)
        self.nrScalers = l[1] - 1
        self.nrTracks = 1
        
        self.x = np.zeros((self.nrTracks, l[0]))
        self.cts = np.zeros((self.nrScalers, self.nrTracks, l[0]))
        self.err = np.zeros((self.nrScalers, self.nrTracks, l[0]))
        
        with open(path) as f:
            read = csv.reader(f, delimiter = '\t')
            for i, row in enumerate(read):
                self.x[0][i] = float(row[0])
                for j, counts in enumerate(row[1:]):
                    self.cts[j][0][i] = float(counts)
                    self.err[j][0][i] = max(np.sqrt(float(counts)), 1)
                    
                
        self.accVolt = Exp.getAccVolt(self.time)
        self.laser = Exp.getLaserFreq(self.time)
        self.col = Exp.dirColTrue(self.time) 
            
            
    def dimension(self, path):
        '''returns the nr of lines and columns of the file'''
        lines = 1
        with open(path) as f:
            cols = len(f.readline().split('\t'))
            for line in f:
                lines += 1
                
        return (lines, cols)