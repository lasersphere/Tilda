"""

Created on '07.08.2015'

@author:'simkaufm'

"""

import csv
import sqlite3
from datetime import datetime
import os
import Service.Formating as Form
import Physics

import numpy as np

import TildaTools
from Measurement.SpecData import SpecData


class XMLImporter(SpecData):
    '''
    This object reads a file with tab separated values into the ScanData structure

     The first column of the file is interpreted as scanning voltage, all following as scalers
    The header has 10 lines
    '''

    def __init__(self, path):
        '''Read the file'''

        print("XMLImporter is reading file", path)
        super(XMLImporter, self).__init__()

        self.file = os.path.basename(path)

        scandict, lxmlEtree = TildaTools.scan_dict_from_xml_file(path)
        self.nrTracks = scandict['isotopeData']['nOfTracks']

        self.laserFreq = Physics.freqFromWavenumber(2 * scandict['isotopeData']['laserFreq'])
        self.date = scandict['isotopeData']['isotopeStartTime']
        self.type = scandict['isotopeData']['isotope']
        self.seq_type = scandict['isotopeData']['type']

        self.accVolt = scandict['isotopeData']['accVolt']

        self.offset = 0  # should also be a list for mutliple tracks
        self.nrScalers = []
        self.x = Form.create_x_axis_from_scand_dict(scandict, as_voltage=True)  # x axis, voltage
        self.cts = []  # countervalues
        self.err = []  # error to the countervalues
        self.stepSize = []
        self.col = False  # should also be a list for multiple tracks
        self.dwell = []
        if self.seq_type == 'trs':
            self.t = Form.create_time_axis_from_scan_dict(scandict)  # time axis, 10ns resolution
            self.t_proj = []
            self.time_res = []




        for tr_ind, tr_name in enumerate(TildaTools.get_track_names(scandict)):
            track_dict = scandict[tr_name]
            nOfactTrack = int(tr_name[5:])
            nOfsteps = track_dict['nOfSteps']
            nOfBins = track_dict.get('nOfBins')
            nOfScalers = len(track_dict['activePmtList'])
            dacStepSize18Bit = track_dict['dacStepSize18Bit']

            self.nrScalers.append(nOfScalers)
            self.stepSize.append(dacStepSize18Bit)
            self.col = track_dict['colDirTrue']
            self.offset = track_dict['postAccOffsetVolt']
            if track_dict.get('postAccOffsetVoltControl') == 0:
                self.offset = 0

            if self.seq_type == 'trs':
                cts_shape = (nOfScalers, nOfsteps, nOfBins)
                v_proj = TildaTools.xml_get_data_from_track(
                    lxmlEtree, nOfactTrack, 'voltage_projection', (nOfScalers, nOfsteps))
                t_proj = TildaTools.xml_get_data_from_track(
                    lxmlEtree, nOfactTrack, 'time_projection', (nOfsteps, nOfBins))
                scaler_array = TildaTools.xml_get_data_from_track(
                    lxmlEtree, nOfactTrack, 'scalerArray', cts_shape)
                self.time_res.append(scaler_array)
                if v_proj is None or t_proj is None:
                    v_proj, t_proj = Form.gate_one_track(
                        tr_ind, nOfactTrack, scandict, self.time_res, self.t, self.x, [])[0]
                self.cts.append(v_proj)
                self.err.append(np.sqrt(v_proj))
                self.t_proj.append(t_proj)
                self.time_res.append(scaler_array)
                gates = track_dict['softwGates']
                dwell = [g[3] - g[2] for g in gates]
                self.dwell.append(dwell)

            elif self.seq_type == 'cs':
                cts_shape = (nOfScalers, nOfsteps)
                scaler_array = TildaTools.xml_get_data_from_track(
                    lxmlEtree, nOfactTrack, 'scalerArray', cts_shape)
                self.cts.append(scaler_array)
                self.err.append(np.sqrt(scaler_array))
                self.dwell.append(track_dict.get('dwellTime10ns'))

    def preProc(self, db):
        print('XMLImporter is using db: ', db)
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute('''SELECT type, line, offset, accVolt, laserFreq,
                        colDirTrue, voltDivRatio, lineMult, lineOffset
                        FROM Files WHERE file = ?''', (self.file,))
        data = cur.fetchall()
        if len(data) == 1:
            (self.type, self.line, self.offset, self.accVolt, self.laserFreq,
             self.col, self.voltDivRatio, self.lineMult, self.lineOffset) = data[0]
            self.col = bool(self.col)
        else:
            raise Exception('XMLImporter: No DB-entry found!')

        for j in range(len(self.x)):
            for i in range(len(self.x[j])):
                scanvolt = self.lineMult * float(self.x[j][i]) + self.lineOffset + self.offset
                self.x[j][i] = float(float(self.accVolt) - scanvolt)
        con.close()

    def export(self, db):
        try:
            con = sqlite3.connect(db)
            with con:
                con.execute('''UPDATE Files SET date = ?, type = ?, offset = ?,
                                laserFreq = ?, colDirTrue = ?, accVolt = ?
                                 WHERE file = ?''',
                            (self.date, self.type, self.offset,
                             self.laserFreq, self.col, self.accVolt,
                             self.file))
            con.close()
        except Exception as e:
            print(e)

    def evalErr(self, cts, f):
        cts = cts.reshape(-1)
        for i, v in enumerate(cts):
            cts[i] = f(v)
