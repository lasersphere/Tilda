'''
Created on 02.05.2015

@author: skaufmann, chgorges
'''

import ast
import copy
import os
import re
import sqlite3
from datetime import datetime

import numpy as np

import TildaTools

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
        self.nrBunches = []  # list for each track an integer with the number for
        #  bunches per step for this track as it is unknown in MCP this is always 1
        self.nrTracks = 0
        self.nrSteps = 0
        self.offset = 0
        self.offset_by_dev = {}  # dict of all devs and there mean values
        self.cts = []
        self.activePMTlist = []
        self.err = np.array([[]])
        self.x = []
        self.post_acc_offset_volt_control = []  # which fluke?


        with open(path) as f:
            file_as_str = str(f.read().replace('\n', '').replace('\"', ''))
            self.file_as_str = file_as_str
            self.version = self.get_version(self.file_as_str)
            if self.find_data_list_in_str(file_as_str, 'SiclStepObj')[0] != []:
                # its a Kepco scan and must be treated different!
                self.type = 'Kepco'
                self.nrScalers = [2]
                scans, self.nrScans, self.nrSteps, limits = self.get_scan_pars(file_as_str)
                self.date = self.get_date(file_as_str)

                for index, stepnumber in enumerate(self.nrSteps):
                    self.x.append(np.linspace(float(limits[index][0]), float(limits[index][1]), num=stepnumber))
                dataset = file_as_str.split('>>>>')[:-1]
                for track in dataset:
                    fluke_num = self.find_data_list_in_str(track, 'FlukeSwitchObj', data_begin_str=',')[0][0][0]
                    self.post_acc_offset_volt_control.append(fluke_num)
                    data = self.find_data_list_in_str(track, 'SiclStepObj')
                    self.cts.append(data[0])
                    self.activePMTlist.append(data[1])
                    data = self.find_data_list_in_str(track, 'KepcoEichungVoltageObj')
                    self.cts.append(data[0])
                    self.activePMTlist.append(data[1])
                    self.nrTracks += 1
                self.cts[0][0] = np.delete(self.cts[0][0], -1)
                self.offset = 0  # must be corrected later on, when fitting.

                self.cts[0].append(np.array(np.zeros(self.nrSteps[0])))
                for i, ct in enumerate(self.cts[1][0]):
                    self.cts[0][1][i] = ct
                self.cts = np.delete(self.cts, 1)
                self.err = np.array(copy.deepcopy(self.cts))
                for i,ctarray in enumerate(self.cts):
                    for j, cts in enumerate(ctarray):
                        self.err[i][j] = cts*0.01/100+0.00005
            else:
                # normal spectroscopy measurement
                self.ele = self.file.split('_')
                self.type = self.ele[0][:-2] + '_' + self.ele[0][-2:]
                volts = self.find_data_list_in_str(file_as_str, 'SiclReaderObj')
                accVoltDevice = 'lan[A-34461A-06386]:inst0'
                offsetDevice = 'lan[A-34461A-06287]:inst0'
                if accVoltDevice in volts[1]:
                    self.accVolt = np.mean(volts[0][volts[1].index(accVoltDevice)])
                else:
                    self.accVolt = None
                prema = np.mean(self.find_data_list_in_str(file_as_str, 'PremaVoltageObj')[0])
                print(prema)
                if prema != None:
                    if offsetDevice in volts[1]:
                        agilent = np.mean(volts[0][volts[1].index(offsetDevice)])
                        self.offset = np.mean([prema, agilent])
                        self.offset_by_dev = {'prema': prema, 'agilent': agilent}
                    else:
                        self.offset = prema
                else:
                    self.offset = np.mean(volts[0][volts[1].index(offsetDevice)])
                scans, self.nrScans, self.nrSteps, limits = self.get_scan_pars(file_as_str)

                self.nrLoops = len(volts[0])
                self.date = self.get_date(file_as_str)
                self.col = True

                for index, stepnumber in enumerate(self.nrSteps):
                    self.x.append(np.linspace(float(limits[index][0]), float(limits[index][1]), num=stepnumber))

                dataset = file_as_str.split('>>>>')[:-1]
                for track in dataset:
                    fluke_num = self.find_data_list_in_str(track, 'FlukeSwitchObj', data_begin_str=',')
                    if fluke_num[0]:  # fluke switch might not be changed in every track!
                        fluke_num = fluke_num[0][0][0]
                    else:
                        print('fluke num: %s %s ' % fluke_num)
                        fluke_num = -1
                    self.post_acc_offset_volt_control.append(fluke_num)
                    data = self.find_data_list_in_str(track, 'PM_SpectrumObj')
                    self.cts.append(data[0])
                    self.activePMTlist.append(data[1])
                    self.nrTracks += 1
                    self.nrBunches += 1,

                # remove scalers, which are not used in all tracks:
                pmts_flat = [item for sublist in self.activePMTlist for item in sublist]
                pmts_ok = [pmt_name for pmt_name in self.activePMTlist[0] if pmts_flat.count(pmt_name) == self.nrTracks]
                self.cts = [
                    [pmt for pmt_ind, pmt in enumerate(self.cts[tr_ind])
                     if self.activePMTlist[tr_ind][pmt_ind] in pmts_ok]
                    for tr_ind, tr in enumerate(self.cts)]
                self.nrScalers = [len(pmts_ok) for i in self.cts]
                self.activePMTlist = [pmts_ok for i in self.cts]
                self.err = copy.deepcopy(self.cts)
                for i, ctarray in enumerate(self.cts):
                    for j, cts in enumerate(ctarray):
                        self.err[i][j] = np.sqrt(np.abs(cts))
                        self.err[i][j][self.err[i][j] == 0.0] = 1
            f.close()

    def preProc(self, db, from_input=False):
        if not from_input:
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
            #  there is no pre scan mesasurement for the offset in kepco scan
            #  so finding the value for dac volt = 0V must be done by a fit.
            self.offset = self.find_offset_for_kepco()
            for trackindex, tracks in enumerate(self.cts):
                for ctindex, ct in enumerate(tracks):
                    for i, j in enumerate(ct):
                        self.cts[trackindex][ctindex][i] = (j - self.offset[ctindex])

        else:
            self.col = bool(self.col)
            self.voltDivRatio = ast.literal_eval(self.voltDivRatio)
            for trackindex, tracks in enumerate(self.x):
                # for xindex, x in enumerate(tracks):
                #     if isinstance(self.voltDivRatio['offset'], float):  # just one number
                #         scanvolt = (self.lineMult * x + self.lineOffset + self.offset) * self.voltDivRatio['offset']
                #     else:  # offset should be a dictionary than
                #         vals = list(self.voltDivRatio['offset'].values())
                #         mean_offset_div_ratio = np.mean(vals)
                #         # treat each offset with its own divider ratio
                #         mean_offset = np.mean([val * self.offset_by_dev[key] for key, val in
                #                                self.voltDivRatio['offset'].items()])
                #         scanvolt = (self.lineMult * x + self.lineOffset) * mean_offset_div_ratio + mean_offset
                #     self.x[trackindex][xindex] = self.accVolt*self.voltDivRatio['accVolt'] - scanvolt
                self.x[trackindex] = TildaTools.line_to_total_volt(self.x[trackindex], self.lineMult, self.lineOffset,
                                                                   self.offset, self.accVolt, self.voltDivRatio,
                                                                   offset_by_dev_mean=self.offset_by_dev)
            '''If the numbers of scans for the tracks are different, it will be normed to the minimal number of scans:'''
            # print(self.x)
            # print(self.cts)
            # print(self.nrScalers)
            self.norming()
            self.x_units = self.x_units_enums.total_volts
        if not from_input:
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
            if (orig_str.find(data_end_str, ind) == -1):
            #for the last scaler, there is now '>>' at the end, so find() return is -1 and the last digit of the counts will be lost...
                ret += orig_str[orig_str.find(data_begin_str, ind) + len(data_begin_str):len(orig_str)]
            else:
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

    def get_version(self, mcp_file_as_string):
        version = mcp_file_as_string[mcp_file_as_string.find('Version') + 8:
        mcp_file_as_string.find('Version') + 18]
        return version

    def get_scan_pars(self, mcp_file_as_string):
        scans = []
        completed_scans = []
        steps = []
        limits = []
        # first track in mcp is opened by '<<'
        # following tracks are opened by '>,<,
        # those are followed by:
        # 'unknown_string, scans, completed_scans, steps, unknown_int, unknown_int, unknown_int,<
        ind = mcp_file_as_string.find('<<', 0)
        ind2 = mcp_file_as_string.find('<', ind + 2)

        while ind != -1:
            if scans:  # not in first call/track
                ind = mcp_file_as_string.find('>,<', ind)
                ind = ind if ind == -1 else ind + 3  # when no track can be found anymore yield ind = -1
                ind2 = mcp_file_as_string.find('<', ind + 2)
            lis = mcp_file_as_string[ind:ind2]
            lis = lis.split(',')[1:-1]
            indlim = mcp_file_as_string.find('<LineVoltageSweepObj,', ind)
            indlim2 = mcp_file_as_string.find('>', indlim)
            if ind != -1:
                scans.append(int(lis[0]))
                completed_scans.append(int(lis[1]))
                steps.append(int(lis[2]))
                # limits.append(mcp_file_as_string[indlim:indlim2].split(',')[1:3])
                if indlim != -1:
                    limits.append(mcp_file_as_string[indlim:indlim2].split(',')[1:3])
                else:
                    # -> no LineVoltageSweepObj included
                    # might be a release curve
                    indtrig = mcp_file_as_string.find('<TriggerObj,', ind)
                    indtrig2 = mcp_file_as_string.find('>', indtrig)
                    start_time = 10  # bascially one time bin TODO actually read from file
                    end_time = steps[-1] * start_time
                    limits.append([start_time, end_time])
                ind += 3
        return (scans, completed_scans, steps, limits)

    def norming(self):
        for trackindex, track in enumerate(self.cts):
            for ctIndex, ct in enumerate(track):
                min_nr_of_scan = max(np.min(self.nrScans), 1)  # maybe there is a track with 0 complete scans
                nr_of_scan_this_track = self.nrScans[trackindex]
                if nr_of_scan_this_track:
                    self.cts[trackindex][ctIndex] = ct * min_nr_of_scan / nr_of_scan_this_track
                    self.err[trackindex][ctIndex] = self.err[trackindex][ctIndex] * min_nr_of_scan / nr_of_scan_this_track

    def find_offset_for_kepco(self):
        """ find the offset of the measurement for each multimeter and return a list of the offsets for each dmm """
        from Spectra.Straight import Straight
        from SPFitter import SPFitter
        offsets = []
        print('preprocessing fits of MCPImporter for Kepco scan in order to find offset')
        for sc_ind, cts in enumerate(self.cts[0]):  # assume only one track
            st = ([sc_ind], 0)
            fitter = SPFitter(Straight(), self, st)
            fitter.fit()
            offsets.append(fitter.par[0])  # the b parameter is the offset
        return offsets
