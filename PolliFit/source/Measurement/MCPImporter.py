'''
Created on 02.05.2015

@author: skaufmann, chgorges
'''

import csv, ast
import sqlite3
from datetime import datetime
import os
import re
import copy

import numpy as np

from Measurement.SpecData import SpecData

class MCPImporter(SpecData):
    '''
    This object reads a file with tab separated values into the ScanData structure
    '''

    def __init__(self, path):
        '''Read the file
        '''
        
        print("MCPImporter is reading file", path)
        super(MCPImporter, self).__init__()

        self.file = os.path.basename(path)
        self.type = ''
        self.nrScalers = []
        self.nrTracks = 0
        self.nrSteps = 0
        self.offset = 0
        self.cts = []
        self.activePMTlist = []
        self.err = np.array([[]])
        self.x = []

        with open(path) as f:
            file_as_str = str(f.read().replace('\n', '').replace('\"', ''))
            self.file_as_str = file_as_str
            if self.find_data_list_in_str(file_as_str, 'SiclStepObj')[0] != []:
                self.type = 'Kepco'
                self.nrScalers = [2]
                scans, self.nrScans, self.nrSteps, limits = self.get_scan_pars(file_as_str)
                self.date = self.get_date(file_as_str)

                for index, stepnumber in enumerate(self.nrSteps):
                    self.x.append(np.linspace(float(limits[index][0]),float(limits[index][1]),num=stepnumber))
                dataset = file_as_str.split('>>>>')[:-1]
                for track in dataset:
                    data = self.find_data_list_in_str(track,'SiclStepObj')
                    self.cts.append(data[0])
                    self.activePMTlist.append(data[1])
                    data = self.find_data_list_in_str(track,'KepcoEichungVoltageObj')
                    self.cts.append(data[0])
                    self.activePMTlist.append(data[1])
                    self.nrTracks +=1
                self.cts[0][0] = np.delete(self.cts[0][0], -1)
                if self.nrSteps[0] % 2 == 0:
                    self.offset = self.cts[0][0][self.nrSteps[0]/2]
                else:
                    self.offset = np.mean([self.cts[0][0][self.nrSteps[0]/2-1],self.cts[0][0][self.nrSteps[0]/2+1]])

                self.cts[0].append(np.array(np.zeros(self.nrSteps[0])))
                for i, ct in enumerate(self.cts[1][0]):
                    self.cts[0][1][i] = ct
                self.cts = np.delete(self.cts, 1)
                self.err = np.array(copy.deepcopy(self.cts))
                for i,ctarray in enumerate(self.cts):
                    for j, cts in enumerate(ctarray):
                        self.err[i][j] = cts*0.01/100+0.00005
            else:
                self.ele = self.file.split('_')
                self.type = self.ele[0][:-2] + '_' + self.ele[0][-2:]
                volts = self.find_data_list_in_str(file_as_str, 'SiclReaderObj')
                self.accVolt = np.mean(volts[0][volts[1].index('lan[A-34461A-06386]:inst0')])
                prema = np.mean(self.find_data_list_in_str(file_as_str, 'PremaVoltageObj')[0])
                agilent = np.mean(volts[0][volts[1].index('lan[A-34461A-06287]:inst0')])
                d_prema_agilent = prema - agilent
                self.offset = np.mean([prema, agilent])

                scans, self.nrScans, self.nrSteps, limits = self.get_scan_pars(file_as_str)

                self.nrLoops = len(volts[0])
                self.date = self.get_date(file_as_str)
                self.col = True

                for index,stepnumber in enumerate(self.nrSteps):
                    self.x.append(np.linspace(float(limits[index][0]),float(limits[index][1]),num=stepnumber))

                dataset = file_as_str.split('>>>>')[:-1]
                for track in dataset:
                    data = self.find_data_list_in_str(track,'PM_SpectrumObj')
                    self.cts.append(data[0])
                    self.activePMTlist.append(data[1])
                    self.nrTracks +=1

                pmts_flat = [item for sublist in self.activePMTlist for item in sublist]
                pmts_ok = [pmt_name for pmt_name in self.activePMTlist[0] if pmts_flat.count(pmt_name) == self.nrTracks]
                self.cts = [[pmt for pmt_ind, pmt in enumerate(self.cts[tr_ind]) if self.activePMTlist[tr_ind][pmt_ind] in pmts_ok]
                 for tr_ind, tr in enumerate(self.cts)]

                self.nrScalers = [len(pmts_ok) for i in self.cts]
                self.err = copy.deepcopy(self.cts)
                for i,ctarray in enumerate(self.cts):
                    for j, cts in enumerate(ctarray):
                        self.err[i][j] = np.sqrt(cts)
            f.close()

    def preProc(self, db):
        print('MCPimporter is using db', db)
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute('''SELECT accVolt, laserFreq, colDirTrue, line, type, voltDivRatio, lineMult, lineOffset, offset
                                        FROM Files WHERE file = ?''', (self.file,))
        data = cur.fetchall()
        if len(data) == 1:
            (self.accVolt, self.laserFreq, self.col, self.line, self.type, self.voltDivRatio, self.lineMult,
                    self.lineOffset, self.offset) = data[0]
        else:
            raise Exception('MCPImporter: No DB-entry found!')
        if self.type == 'Kepco':
            for trackindex, tracks in enumerate(self.cts):
                for ctindex, ct in enumerate(tracks):
                    for i, j in enumerate(ct):
                        self.cts[trackindex][ctindex][i] = (j - self.offset)
        else:
            self.col = bool(self.col)
            self.voltDivRatio = ast.literal_eval(self.voltDivRatio)
            for trackindex, tracks in enumerate(self.x):
                for xindex, x in enumerate(tracks):
                    scanvolt = (self.lineMult * x + self.lineOffset + self.offset) * self.voltDivRatio['offset']
                    self.x[trackindex][xindex]= self.accVolt*self.voltDivRatio['accVolt'] - scanvolt
            '''If the numbers of scans for the tracks are different, it will be normed to the minimal number of scans:'''
            self.norming()
        con.close()

    def export(self, db):
        con = sqlite3.connect(db)
        with con:
            con.execute('''UPDATE Files SET date = ?, type = ?, offset = ?, accVolt = ?, colDirTrue = ?, voltDivRatio = ?,
                            lineMult = ?, lineOffset = ?  WHERE file = ?''', (self.date, self.type, self.offset,
                            self.accVolt, self.col, self.voltDivRatio, self.lineMult, self.lineOffset, self.file))
        con.commit()
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
        new_fmt = date_t.strftime('%Y-%m-%d %H:%M:%S')
        return new_fmt


    def get_scan_pars(self, mcp_file_as_string):
        scans = []
        completed_scans = []
        steps = []
        limits = []
        ind = 0
        while ind!= -1:
            ind = mcp_file_as_string.find('<,', ind)
            ind2 = mcp_file_as_string.find('<', ind + 2)
            lis = mcp_file_as_string[ind:ind2].split(',')[1:-1]
            indlim = mcp_file_as_string.find('<LineVoltageSweepObj,', ind)
            indlim2 = mcp_file_as_string.find('>', indlim)
            if ind != -1:
                scans.append(int(lis[0]))
                completed_scans.append(int(lis[1]))
                steps.append(int(lis[2]))
                limits.append(mcp_file_as_string[indlim:indlim2].split(',')[1:3])
                ind +=2
        return (scans, completed_scans, steps, limits)

    def norming(self):
        for trackindex, track in enumerate(self.cts):
            for ctIndex, ct in enumerate(track):
                self.cts[trackindex][ctIndex] = ct*np.min(self.nrScans)/self.nrScans[trackindex]
                self.err[trackindex][ctIndex] = self.err[trackindex][ctIndex]*np.min(self.nrScans)/self.nrScans[trackindex]
