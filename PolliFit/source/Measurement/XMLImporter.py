"""

Created on '07.08.2015'

@author:'simkaufm'

"""

import sqlite3
import os
import Service.Formating as Form
import Physics

import numpy as np

import TildaTools
from Measurement.SpecData import SpecData


class XMLImporter(SpecData):
    '''
    This Module Reads the .xml files.

    .xml file structure is:
    -<TrigaLaserData>
        -<header>
            <accVolt>0</accVolt>
            <isotope>Ni</isotope>
            <isotopeStartTime>2016-04-14 23:32:10</isotopeStartTime>
            <laserFreq>0</laserFreq>
            <nOfTracks>1</nOfTracks>
            <type>tipa</type>
            <version>1.08</version>
        </header>
        -<tracks>
            -<track0>
                -<header>
                    <activePmtList>[0, 1, 2, 3]</activePmtList>
                    <colDirTrue>True</colDirTrue>
                    <dacStartRegister18Bit>0</dacStartRegister18Bit>
                    <dacStartVoltage>-9.993201</dacStartVoltage>
                    <dacStepSize18Bit>1</dacStepSize18Bit>
                    <dacStepsizeVoltage>0.001572</dacStepsizeVoltage>
                    <dacStopVoltage>-9.992896</dacStopVoltage>
                    <invertScan>False</invertScan>
                    <nOfBins>1000</nOfBins>
                    <nOfBunches>1</nOfBunches>
                    <nOfCompletedSteps>45</nOfCompletedSteps>
                    <nOfScans>None</nOfScans>
                    <nOfSteps>5</nOfSteps>
                    <softBinWidth_ns>100</softBinWidth_ns>
                    <softwGates>[[-10, 10, 0, 10000], [0, 4, 0.0, 9900.0]]</softwGates>
                    <trigger>{'trigDelay10ns': 10000, 'type': 'SingleHit'}</trigger>
                    <workingTime>None</workingTime>
                </header>
                +<data>
            </track0>
        </tracks>
        </TrigaLaserData>
    '''

    def __init__(self, path, x_as_volt=True):
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
        self.active_pmt_list = []
        if self.seq_type in ['tipa', 'tipadummy']:
            x_as_volt = False
        self.x = Form.create_x_axis_from_scand_dict(scandict, as_voltage=x_as_volt)  # x axis, voltage
        self.cts = []  # countervalues
        self.err = []  # error to the countervalues
        self.stepSize = []
        self.col = False  # should also be a list for multiple tracks
        self.dwell = []
        self.softw_gates = None
        self.track_names = TildaTools.get_track_names(scandict)

        for tr_ind, tr_name in enumerate(TildaTools.get_track_names(scandict)):
            track_dict = scandict[tr_name]
            nOfactTrack = int(tr_name[5:])
            nOfsteps = track_dict['nOfSteps']
            nOfBins = track_dict.get('nOfBins')
            nOfScalers = len(track_dict['activePmtList'])
            self.active_pmt_list.append(track_dict['activePmtList'])

            dacStepSize18Bit = track_dict['dacStepSize18Bit']

            self.nrScalers.append(nOfScalers)
            self.stepSize.append(dacStepSize18Bit)
            self.col = track_dict['colDirTrue']
            self.offset = track_dict['postAccOffsetVolt']
            if track_dict.get('postAccOffsetVoltControl') == 0:
                self.offset = 0

            if self.seq_type == 'trs' or self.seq_type == 'tipa':
                self.softBinWidth_ns = track_dict.get('softBinWidth_ns', 10)
                self.t = Form.create_time_axis_from_scan_dict(scandict)  # force 10 ns resolution
                self.t_proj = []
                self.time_res = []
                cts_shape = (nOfScalers, nOfsteps, nOfBins)
                v_proj = TildaTools.xml_get_data_from_track(
                    lxmlEtree, nOfactTrack, 'voltage_projection', (nOfScalers, nOfsteps))
                t_proj = TildaTools.xml_get_data_from_track(
                    lxmlEtree, nOfactTrack, 'time_projection', (nOfScalers, nOfBins))
                scaler_array = TildaTools.xml_get_data_from_track(
                    lxmlEtree, nOfactTrack, 'scalerArray', cts_shape)
                self.time_res.append(scaler_array)
                if v_proj is None or t_proj is None:
                    v_proj, t_proj = Form.gate_one_track(
                        tr_ind, nOfactTrack, scandict, self.time_res, self.t, self.x, [])[0]
                self.cts.append(v_proj)
                self.err.append(np.sqrt(v_proj))
                self.err[-1][self.err[-1] < 1] = 1  # remove 0's in the error
                self.t_proj.append(t_proj)
                self.softw_gates = track_dict['softwGates']
                dwell = [g[3] - g[2] for g in self.softw_gates]
                self.dwell.append(dwell)

            elif self.seq_type == 'cs':
                cts_shape = (nOfScalers, nOfsteps)
                scaler_array = TildaTools.xml_get_data_from_track(
                    lxmlEtree, nOfactTrack, 'scalerArray', cts_shape)
                self.cts.append(scaler_array)
                self.err.append(np.sqrt(scaler_array))
                self.dwell.append(track_dict.get('dwellTime10ns'))

        print('%s was successfully imported' % self.file)

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

    def get_scaler_step_and_bin_num(self, track_ind):
        """ returns a tuple: (nOfScalers, nOfSteps, nOfBins) """
        if self.seq_type in ['trs', 'trsdummy', 'tipa', 'tipadummy']:
            return self.time_res[track_ind].shape
        else:
            return self.nrScalers[track_ind], self.getNrSteps(track_ind), -1
