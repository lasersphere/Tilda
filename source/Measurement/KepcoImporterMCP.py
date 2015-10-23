'''
Created on 30.04.2014

@author: hammen
'''

import csv
from datetime import datetime
import sqlite3
import os

import numpy as np

from Measurement.SpecData import SpecData

class KepcoImporterMCP(SpecData):
    '''
    This object reads a MCP-file into the KepcoData structure
    '''

    def __init__(self, path):
        '''Read the file'''
        
        print("KepcoImporterMCP is reading file", path)
        super(KepcoImporterMCP, self).__init__()
        
        self.path = path
        self.type = 'Kepco'
        
        self.file = os.path.basename(path)
        self.nrScalers = 2
        self.nrTracks = 1
        self.nrScalers = 0
        self.offset = 0

        self.xTemp = [[]]
        self.ctsTemp = []
        self.errTemp = []

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
            while line != ',["SiclStepObj"]\n':
                line = f.readline()
            self.counting(f)
            line = f.readline()
            while line != ',["KepcoEichungVoltageObj"]\n':
                line = f.readline()
            self.counting(f)
            self.nrSteps = len(self.ctsTemp[0])
            self.x = [np.zeros(self.nrSteps)]
            for i in range(0, self.nrSteps):
                self.x[0][i] = float(float(limits[0]) + i * (float(limits[1]) - float(limits[0])) / self.nrSteps)
            k = 0
            while k < self.nrScalers:
                for i, j in enumerate(self.ctsTemp[k].copy()):
                    self.ctsTemp[k][i] = float(j)
                    self.errTemp[k][i] = float(j)/10**4
                k +=1
            self.offset = self.ctsTemp[0][int(round(self.nrSteps/2, 0))]
            self.cts = [np.array(self.ctsTemp)]
            self.err = [np.array(self.errTemp)]

    def preProc(self, db):
        print('Kepco importer is using db', db)
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute('''SELECT voltDivRatio, offset FROM Files WHERE file = ?''', (self.file,))
        data = cur.fetchall()
        if len(data) == 1:
            (self.voltDivRatio, self.offset) = data[0]
        else:
            raise Exception('KepcoImporterMCP: No DB-entry found!')
                
        for i in range(len(self.cts[0])):
            for j in range(len(self.cts[0][0])):
                self.cts[0][i][j] = (self.cts[0][i][j] - self.offset) * self.voltDivRatio
                self.err[0][i][j] = self.cts[0][i][j] * 10**-4
        con.close()
    
    def export(self, db):
        con = sqlite3.connect(db)
        with con:
            con.execute('''UPDATE Files SET date = ?, offset = ?, type = ? WHERE file = ?''', (self.date, self.offset, self.type, self.file))
        con.close()

    def counting(self, f):
        firstLine = str(f.readline()).split(',')
        scalerNo = self.nrScalers
        self.nrScalers +=1
        ctscopy = []
        cts = []
        err = []
        i = 1
        while firstLine[i][0] != '<':
            i += 1
        cts.extend([float(firstLine[i][1:]),float(firstLine[i+1]),float(firstLine[i+2]),float(firstLine[i+3]),float(firstLine[i+4])])
        err.extend([float(firstLine[i][1:])/10**4,float(firstLine[i+1])/10**4,float(firstLine[i+2])/10**4,float(firstLine[i+3])/10**4,float(firstLine[i+4])/10**4])
        noEnd = True
        while noEnd:
            line = f.readline()
            ctscopy.extend(line.split(','))
            if str(line)[-2] == '>':
                noEnd = False
        for i in range(0, len(ctscopy)):
            if ctscopy[i] != '\n' and ctscopy[i] != '>\n' and ctscopy[i] != ' ':
                if str(ctscopy[i])[-1] == '\n':
                    if str(ctscopy[i])[-2] == '>':
                        pass
                    else:
                        cts.append(float(str(ctscopy[i])[:-1]))
                        err.append(((float(cts[-1])*10**-4)))
                else:
                    cts.append(float(ctscopy[i]))
                    err.append(((float(cts[-1])*10**-4)))
        self.ctsTemp.append(cts)
        self.errTemp.append(err)
        return scalerNo