'''
Created on 30.04.2014

@author: hammen
'''

import csv
import sqlite3
import os

import numpy as np

from Tilda.PolliFit.Measurement.SpecData import SpecData

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
        self.type = 'Kepco'
        
        self.file = os.path.basename(path)
         
        l = self.dimension(path)
        self.nrScalers = l[1] - 1
        self.nrTracks = 1
        
        self.x = [np.zeros(l[0])]
        self.cts = [np.zeros((self.nrScalers, l[0]))]
        self.err = [np.zeros((self.nrScalers, l[0]))]
        
        with open(path) as f:
            for i in range(9):
                f.readline()
            read = csv.reader(f, delimiter = '\t')
            for i, row in enumerate(read):
                self.x[0][i] = float(row[0])
                for j, scanVolt in enumerate(row[1:]):
                    self.cts[0][j][i] = float(scanVolt)
                    self.err[0][j][i] = self.cts[0][j][i] * 10**-4


    def pre_process(self, db):
        print('Kepco importer is using db', db)
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute('''SELECT voltDivRatio, offset FROM Files WHERE file = ?''', (self.file,))
        data = cur.fetchall()
        if len(data) == 1:
            (self.voltDivRatio, self.offset) = data[0]
        else:
            raise Exception('KepcoImporterTLD: No DB-entry found!')
                
        for i in range(len(self.cts[0])):
            for j in range(len(self.cts[0][0])):
                self.cts[0][i][j] = (self.cts[0][i][j] - self.offset) * self.voltDivRatio
                self.err[0][i][j] = self.cts[0][i][j] * 10**-4
        
        con.close() 
        
        
    def dimension(self, path):
        '''returns the nr of lines and columns of the file'''
        
        lines = 1
        with open(path) as f:
            for i in range(9):
                f.readline()
            cols = len(f.readline().split('\t'))
            for line in f:
                lines += 1        
        return (lines, cols)
    
    def export(self, db):
        return
        con = sqlite3.connect(db)
        con.execute('''UPDATE Files SET date = ? WHERE filePath = ?''', (self.date, self.file))        
        con.close()
    