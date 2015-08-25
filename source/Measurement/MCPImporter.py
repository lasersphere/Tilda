'''
Created on 24.08.2015

@author: chgorges
'''

import csv
import sqlite3
from datetime import datetime
import os

import numpy as np

from Measurement.SpecData import SpecData

class MCPImporter(SpecData):
    '''
    This object reads a file with tab separated values into the ScanData structure
    '''

    def __init__(self, path):
        '''Read the file'''
        
        print("MCPImporter is reading file", path)
        super(MCPImporter, self).__init__()

        self.file = os.path.basename(path)

        self.nrScalers = 0
        self.nrTracks = 0
        self.nrSteps = 0
        self.offset = 0
        self.lineMult = 50
        self.lineOffset = 0
        self.voltDivRatio = 1000
        self.accVolt = 30000

        self.x = [[]]
        self.cts = []
        self.err = []
        
        with open(path) as f:
            self.mcpVersion = f.readline()
            f.readline() #strange number
            fmt = '%d.%m.%Y\t%H:%M:%S'
            date = f.readline().split(' ')
            month = '.00.'
            if date[1] == 'Jan':
                    month = '.01.'
            elif date[1] == 'Feb':
                    month = '.02.'
            elif date[1] == 'Mar':
                    month = '.03.'
            elif date[1] == 'Apr':
                    month = '.04.'
            elif date[1] == 'May':
                    month = '.05.'
            elif date[1] == 'Jun':
                    month = '.06.'
            elif date[1] == 'Jul':
                    month = '.07.'
            elif date[1] == 'Aug':
                    month = '.08.'
            elif date[1] == 'Sep':
                    month = '.09.'
            elif date[1] == 'Oct':
                    month = '.10.'
            elif date[1] == 'Nov':
                    month = '.11.'
            elif date[1] == 'Dec':
                    month = '.12.'
            self.date = datetime.strptime(str(str(date[2] + month + str(date[4])[:-2] + '\t' + str(date[3]))), fmt)
            f.readline() # "???"
            f.readline()
            self.nrSteps = int(f.readline().split(',')[3])
            f.readline()
            f.readline()
            f.readline()
            limits = str(f.readline()).split(',')
            limits.pop(0)
            line = f.readline()
            while line != ',["SiclReaderObj"]\n' and line != ',["TriggerObj"]\n':
                line = f.readline()
            offsets = str(f.readline()).split(',')
            offsets[4] = str(offsets[4])[1:]
            offsets[-1] = str(offsets[-1])[:-2]
            for i in range(4, len(offsets)):
                self.offset = self.offset + float(offsets[i])
            self.offset = self.offset/len(offsets)

            self.nrLoops = len(offsets)
            f.readline()
            line = str(f.readline())
            while line != ',["PM_SpectrumObj"]\n':
                line = str(f.readline())
            self.counting(f)
            while line == ',["PM_SpectrumObj"]\n':
                f.readline()
                self.counting(f)
                line = str(f.readline())
            for i in range(0, self.nrSteps):
                self.x[0].append(float(limits[0]) + i * (float(limits[1]) - float(limits[0])) / self.nrSteps)

    def preProc(self, db):
        print('MCPimporter is using db', db)
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute('''SELECT accVolt, laserFreq, colDirTrue, line, type, voltDivRatio, lineMult, lineOffset, offset FROM Files WHERE file = ?''', (self.file,))
        data = cur.fetchall()
        if len(data) == 1:
            (self.accVolt, self.laserFreq, self.colDirTrue, self.line, self.type, self.voltDivRatio, self.lineMult, self.lineOffset, self.offset) = data[0]
        else:
            raise Exception('MCPImporter: No DB-entry found!')


        for i in range(len(self.x[0])):
            print(self.lineMult)
            scanvolt = self.lineMult * (self.x[0][i]) + self.lineOffset + self.offset
            self.x[0][i] = self.accVolt - scanvolt
        con.close()
 
    
    def export(self, db):
        con = sqlite3.connect(db)
        with con:
            con.execute('''UPDATE Files SET date = ?, offset = ?, accVolt = ?, voltDivRatio = ?, lineMult = ?, lineOffset = ?  WHERE file = ?''', (self.date, self.offset, self.accVolt, self.voltDivRatio, self.lineMult, self.lineOffset, self.file))
        con.close()
    
    def counting(self, f):
        scalerNo = f.readline().split(',')[1]
        self.nrScalers +=1
        ctscopy = []
        cts = []
        err = []
        noEnd = True
        while noEnd:
            line = f.readline()
            ctscopy.extend(line.split(','))
            if str(line)[-2] == '>':
                noEnd = False
        ctscopy[0] = ctscopy[0][1:]
        for i in range(0, len(ctscopy)):
            if ctscopy[i] != '\n' and ctscopy[i] != '>>\n' and ctscopy[i] != ' ' and ctscopy[i] != '':
                if str(ctscopy[i])[-1] == '\n':
                    if str(ctscopy[i])[-2] == '>':
                        cts.append(float(str(ctscopy[i])[:-3]))
                    else:
                        cts.append(float(str(ctscopy[i])[:-1]))
                else:
                    cts.append(float(ctscopy[i]))
                err.append(np.sqrt(float(cts[-1])))
        self.cts.append(cts)
        self.err.append(err)
        return scalerNo