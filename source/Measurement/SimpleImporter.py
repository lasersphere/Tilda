'''
Created on 30.04.2014

@author: hammen
'''

import csv

import numpy as np

from Measurement.SpecData import SpecData

class SimpleImporter(SpecData):
    '''
    This object reads a file with tab separated values into the SpecData structure
    
    The first column of the file is interpreted as scanning voltage, all following as scalers
    '''

    def __init__(self, file):
        '''Read the file'''
        
        print("Reading file", file)
        super(SimpleImporter, self).__init__()
        
        self.file = file
        
        l = self.dimension(file)
        self.nrScalers = l[1] - 1
        self.nrTracks = 1
        
        self.x = np.zeros(l[0])
        self.cts = np.zeros((self.nrScalers, self.nrTracks, l[0]))
        self.err = np.zeros((self.nrScalers, self.nrTracks, l[0]))
        
        with open(file) as f:
            read = csv.reader(f, delimiter = '\t')
            for i, row in enumerate(read):
                self.x[i] = float(row[0])
                for j, counts in enumerate(row[1:]):
                    self.cts[j][0][i] = float(counts)
                    self.err[j][0][i] = max(np.sqrt(float(counts)), 1)
            
            
    def dimension(self, file):
        '''returns the nr of lines and columns of the file'''
        lines = 1
        with open(file) as f:
            cols = len(f.readline().split('\t'))
            for line in f:
                lines += 1
                
        return (lines, cols)