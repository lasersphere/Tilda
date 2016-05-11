'''
Created on 02.05.2015

@author: skaufmann
'''

import csv, ast
import sqlite3
from datetime import datetime
import os
import re

import numpy as np

from Measurement.SpecData import SpecData

class MCPHrsImporter(SpecData):
    '''
    This object reads a file with tab separated values into the ScanData structure
    '''

    def __init__(self, path):
        '''Read the file
        '''
        
        print("MCPImporter is reading file", path)
        super(MCPHrsImporter, self).__init__()

        self.file = os.path.basename(path)

        self.nrScalers = 0
        self.nrTracks = 0
        self.nrSteps = 0
        self.offset = 0
        self.cts = []

        with open(path) as f:
            file_as_str = str(f.read().replace('\n', '').replace('\"', ''))
            volts = self.find_data_list_in_str(file_as_str, 'SiclReaderObj')
            self.acc_volt = np.mean(volts[0][volts[1].index('lan[A-34461A-06386]:inst0')])
            prema = np.mean(self.find_data_list_in_str(file_as_str, 'PremaVoltageObj')[0])
            agilent = np.mean(volts[0][volts[1].index('lan[A-34461A-06287]:inst0')])
            d_prema_agilent = prema - agilent
            self.offset = np.mean([prema, agilent])
            scalers = self.find_data_list_in_str(file_as_str, 'PM_SpectrumObj')
            self.cts.append(scalers[0])
            self.activePmtList = scalers[1]
            self.nrOfScalers = len(scalers[1])
            self.nrLoops = len(volts[0])

            scans, self.nrScans, self.nrOfSteps = self.get_nr_of_scans(file_as_str)
            # self.x =
            raise Exception
        f.close()

    def preProc(self, db):
        print('MCPimporter is using db', db)
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute('''SELECT accVolt, laserFreq, colDirTrue, line, type, voltDivRatio, lineMult, lineOffset, offset FROM Files WHERE file = ?''', (self.file,))
        data = cur.fetchall()
        if len(data) == 1:
            (self.accVolt, self.laserFreq, self.col, self.line, self.type, self.voltDivRatio, self.lineMult, self.lineOffset, self.offset) = data[0]
            self.col = bool(self.col)
        else:
            raise Exception('MCPImporter: No DB-entry found!')


        for i in range(len(self.x[0])):
            scanvolt = self.lineMult * self.x[0][i] + self.lineOffset + self.offset * self.voltDivRatio
            self.x[0][i] = self.accVolt - scanvolt
        con.close()
        self.cts = [np.array(self.ctsTemp)]
        self.err = [np.array(self.errTemp)]
    
    def export(self, db):
        con = sqlite3.connect(db)
        with con:
            con.execute('''UPDATE Files SET date = ?, offset = ?, accVolt = ?, voltDivRatio = ?, lineMult = ?, lineOffset = ?  WHERE file = ?''', (self.date, self.offset, self.accVolt, self.voltDivRatio, self.lineMult, self.lineOffset, self.file))
        con.close()
    
    def find_data_list_in_str(self, orig_str, obj_name_str, multiple_occurence=2, data_begin_str='<', data_end_str='>>'):
        l_ind = [m.start() for m in re.finditer(obj_name_str, orig_str)]
        if multiple_occurence >= 2:
            del l_ind[::multiple_occurence]  # every object is mentioned twice in mcp file
        ret = ''
        names = []
        for ind in l_ind:
            names.append(orig_str[orig_str.find(',', ind) + 1:orig_str.find(',', orig_str.find(',', ind) + 1)])
            ret += orig_str[
                   orig_str.find(data_begin_str, ind) + len(data_begin_str):orig_str.find(data_end_str, ind)]
            ret += '\t'
        # ret = ret[:-1]
        ret = ret.split('\t')[:-1]
        ret2 = []
        for ind, vals_str in enumerate(ret):
            if '.' in vals_str:
                ret2.append(np.fromstring(ret[ind], dtype=float, sep=','))
            else:
                ret2.append(np.fromstring(ret[ind], dtype=int, sep=','))
        return ret2, names

    def get_date(self, mcp_file_as_string):
        date = mcp_file_as_string[mcp_file_as_string.find('@<') + 2:
        mcp_file_as_string.find(',', mcp_file_as_string.find('@<'))]
        date_t = datetime.strptime(date, '%a %b %d %H:%M:%S %Y')
        new_fmt = date_t.strftime('%Y-%M-%d %H:%M:%S')
        return new_fmt


    def get_nr_of_scans(self, mcp_file_as_string):
        ind = mcp_file_as_string.find('<<')  # will not work for multiple tracks like this
        ind2 = mcp_file_as_string.find('<', ind + 2)
        lis = mcp_file_as_string[ind:ind2].split(',')[1:-1]
        scans = int(lis[0])
        completed_scans = int(lis[1])
        steps = int(lis[2])
        return (scans, completed_scans, steps)