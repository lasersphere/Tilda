'''
Created on 20.01.2017

@author: schmid
'''

import csv
import ast
import os
import sqlite3
from datetime import datetime


import numpy as np

import TildaTools

from Measurement.SpecData import SpecData


class BeaImporter(SpecData):
    '''
    This object reads a file with tab separated values into the SpecData structure
    
    The first column of the file is interpreted as scanning voltage, all following as scalers
    '''

    def __init__(self, path, accVolt, laserFreq, colDirTrue):
        '''Read the file'''
        
        print("BeaImporter is reading file", path)
        super(BeaImporter, self).__init__()

        self.file = os.path.basename(path)

        self.path = path 
        self.accVolt = accVolt
        self.laserFreq = laserFreq
        self.colDirTrue = colDirTrue

        self.voltDivRatio = {'accVolt': 1, 'offset': 1}
        fmt = '%Y-%m-%d\t%H-%M-%S'
        date = self.file.split('_')[:2]
        date = str(date[0]) + '\t' + str(date[1])
        print(date)
        self.date = datetime.strptime(date, fmt)
        self.offset = 0
        self.lineMult = 1
        self.lineOffset = 1
        self.voltDivRatio = '''{'accVolt': 1, 'offset': 1}'''
        self.iso = '43_Ca'

        l = self.dimension(path)
        self.nrScalers = l[1] - 1
        self.nrTracks = 1

        self.x = [np.zeros(l[0])]
        self.cts = [np.zeros((self.nrScalers, l[0]))]
        self.err = [np.zeros((self.nrScalers, l[0]))]
        print("BeaImporter is opening file", path)
        with open(path) as f:

            read = csv.reader(f, delimiter = '\t')

            for i, row in enumerate(read):
                self.x[0][i] =float(row[0])

                for j, counts in enumerate(row[1:]):
                    self.cts[0][j][i] = float(counts)

                    self.err[0][j][i] = max(np.sqrt(float(counts)), 1)

    
    def preProc(self, db):
        print('BeaImporter is using db', db)
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute('''SELECT accVolt, laserFreq, colDirTrue, line, type, voltDivRatio, lineMult, lineOffset, offset
                        FROM Files WHERE file = ?''', (self.file,))
        data = cur.fetchall()
        if len(data) == 1:
            (self.accVolt, self.laserFreq, self.col, self.line, self.type, self.voltDivRatio, self.lineMult,
                    self.lineOffset, self.offset) = data[0]
        else:
            raise Exception('BeaImporter: No DB-entry found!')
        self.col = bool(self.col)
        try:
            self.voltDivRatio = ast.literal_eval(self.voltDivRatio)
        except Exception as e:
            print('error while converting the voltdivratio: %s' % e)
            self.voltDivRatio = {'accVolt': 1, 'offset': 1}
            print('using now: %s' % self.voltDivRatio)
        for trackindex, tracks in enumerate(self.x):
            self.x[trackindex] = TildaTools.line_to_total_volt(self.x[trackindex], self.lineMult, self.lineOffset,
                                                               self.offset, self.accVolt, self.voltDivRatio)

      
    def dimension(self, path):
        '''returns the nr of lines and columns of the file'''
        lines = 1
        with open(path) as f:
            cols = len(f.readline().split('\t'))
            for line in f:
                lines += 1
                
        return (lines, cols)

    def export(self, db):
        print('export')
        con = sqlite3.connect(db)
        print('connect')
        print(self.date, self.laserFreq, self.iso, self.accVolt, self.offset, self.lineMult, self.lineOffset,
            self.voltDivRatio, self.file)
        with con:
            con.execute(
                '''UPDATE Files SET date = ?, laserFreq=?, type=?, accVolt=?, offset=?, lineMult=?, lineOffset=?, voltDivRatio=? WHERE file = ?''',
                (
                self.date, self.laserFreq, self.iso, self.accVolt, self.offset, self.lineMult, self.lineOffset,
                self.voltDivRatio, self.file))

        con.close()