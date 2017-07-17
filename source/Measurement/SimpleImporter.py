'''
Created on 30.04.2014

@author: hammen
'''

import csv, os, sqlite3

import numpy as np

from Measurement.SpecData import SpecData


class SimpleImporter(SpecData):
    '''
    This object reads a file with tab separated values into the SpecData structure
    
    The first column of the file is interpreted as scanning voltage, all following as scalers
    '''

    def __init__(self, path):
        '''Read the file'''
        
        print("SimpleImporter is reading file", path)
        super(SimpleImporter, self).__init__()

        self.file = os.path.basename(path)
        self.path = path


        l = self.dimension(path)
        self.nrScalers = l[1] - 1
        self.nrTracks = 1
        
        self.x = [np.zeros(l[0])]
        self.cts = [np.zeros((self.nrScalers, l[0]))]
        self.err = [np.zeros((self.nrScalers, l[0]))]
        
        with open(path) as f:
            read = csv.reader(f, delimiter = '\t')
            for i, row in enumerate(read):
                self.x[0][i] = float(row[0])
                for j, counts in enumerate(row[1:]):
                    self.cts[0][j][i] = float(counts)
                    #self.err[0][j][i] = max(np.sqrt(float(counts)), 1)
                    self.err[0][j][i] = 0.01
    
    def preProc(self, db):
        print('SimpleImporter is using db', db)
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute('''SELECT accVolt, laserFreq, colDirTrue, line, type, voltDivRatio, lineMult, lineOffset, offset
                                        FROM Files WHERE file = ?''', (self.file,))
        data = cur.fetchall()
        if len(data) == 1:
            (self.accVolt, self.laserFreq, self.col, self.line, self.type, self.voltDivRatio, self.lineMult,
                    self.lineOffset, self.offset) = data[0]
        else:
            raise Exception('SimpleImporter: No DB-entry found!')

    def dimension(self, path):
        '''returns the nr of lines and columns of the file'''
        lines = 1
        with open(path) as f:
            cols = len(f.readline().split('\t'))
            for line in f:
                lines += 1
                
        return (lines, cols)

    def export(self, db):
        con = sqlite3.connect(db)
        with con:
            con.execute('''UPDATE Files SET date = ?, type = ?, offset = ?, accVolt = ?, colDirTrue = ?, voltDivRatio = ?,
                            lineMult = ?, lineOffset = ?  WHERE file = ?''', (self.date, self.type, self.offset,
                            self.accVolt, self.col, self.voltDivRatio, self.lineMult, self.lineOffset, self.file))
        con.commit()
        con.close()