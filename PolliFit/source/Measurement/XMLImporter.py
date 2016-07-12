"""

Created on '07.08.2015'

@author:'simkaufm'

"""

import os
import sqlite3

import numpy as np

import Physics
import TildaTools
from Measurement.SpecData import SpecData
from Service.VoltageConversions import VoltageConversions as VCon


class XMLImporter(SpecData):
    """
    This Module Reads the .xml files or reads from a given scan_dictionary.
    """

    def __init__(self, path=None, x_as_volt=True, scan_dict=None):
        '''Read the file'''

        super(XMLImporter, self).__init__()
        if path is not None:
            print("XMLImporter is reading file", path)
            self.file = os.path.basename(path)
            scandict, lxmlEtree = TildaTools.scan_dict_from_xml_file(path)
        else:
            print("XMLImporter is reading from scan_dictionary", scan_dict)
            lxmlEtree = None
            scandict = scan_dict
            self.file = os.path.basename(scandict['pipeInternals']['activeXmlFilePath'])

        self.nrTracks = scandict['isotopeData']['nOfTracks']

        self.laserFreq = Physics.freqFromWavenumber(2 * scandict['isotopeData']['laserFreq'])
        self.date = scandict['isotopeData']['isotopeStartTime']
        self.type = scandict['isotopeData']['isotope']
        self.seq_type = scandict['isotopeData']['type']

        self.accVolt = scandict['isotopeData']['accVolt']
        self.offset = None
        dmms_dict = scandict['measureVoltPars'].get('dmms', None)
        if dmms_dict is not None:
            offset = []
            acc_volt = []
            for dmm_name, dmm_dict in dmms_dict.items():
                for key, val in dmm_dict.items():
                    if key == 'preScanRead':
                        if isinstance(val, str):
                            val = float(val)
                        if dmm_dict.get('assignment') == 'offset':
                            offset.append(val)
                        elif dmm_dict.get('assignment') == 'accVolt':
                            acc_volt.append(val)
            if np.any(offset):
                self.offset = np.mean(offset)  # will be overwritten below!
            if np.any(acc_volt):
                self.accVolt = np.mean(acc_volt)
        self.nrScalers = []
        self.active_pmt_list = []
        # if self.seq_type in ['tipa', 'tipadummy', 'kepco']:
        #     x_as_volt = False
        print('axaxis as voltage:', x_as_volt)
        self.x = TildaTools.create_x_axis_from_file_dict(scandict, as_voltage=x_as_volt)  # x axis, voltage
        self.cts = []  # countervalues, this is the voltage projection here
        self.err = []  # error to the countervalues
        self.t_proj = []  # time projection only for time resolved
        self.time_res = []  # time resolved matrices only for time resolved measurments

        self.stepSize = []
        self.col = False  # should also be a list for multiple tracks
        self.dwell = []
        self.softw_gates = []
        self.track_names = TildaTools.get_track_names(scandict)
        self.softBinWidth_ns = []


        ''' operations on each track: '''
        for tr_ind, tr_name in enumerate(TildaTools.get_track_names(scandict)):
            track_dict = scandict[tr_name]
            start = scandict[tr_name]['dacStartVoltage']
            stop = scandict[tr_name]['dacStopVoltage']
            step = scandict[tr_name]['dacStepsizeVoltage']
            start_reg = scandict[tr_name]['dacStartRegister18Bit']
            step_reg = scandict[tr_name]['dacStepSize18Bit']
            stop_reg = scandict[tr_name].get('dacStopRegister18Bit', None)


            nOfactTrack = int(tr_name[5:])
            nOfsteps = track_dict['nOfSteps']
            nOfBins = track_dict.get('nOfBins')
            nOfScalers = len(track_dict['activePmtList'])
            self.active_pmt_list.append(track_dict['activePmtList'])

            print('---------------------------')
            print('voltage debugging, ', tr_name)
            print('x_start: ', start)
            print('x_stop: ', stop)
            print('x_stepsize: ', step)
            new_step = (stop - start)/(nOfsteps - 1)
            print('x_stepsize calc (stop - start)/(nOfsteps-1): ', new_step)
            print('x_stop_calc (x_start + x_step * (nOfsteps - 1) ', start + step * (nOfsteps - 1))
            print('x_start + x_stepsize calc * (nOfsteps -1)', start + new_step * (nOfsteps - 1))
            print('---------------------------')
            print('voltage debugging, ', tr_name)
            print('x_start_reg: ', start_reg, VCon.get_voltage_from_18bit(start_reg))
            print('x_stop: ', stop_reg, VCon.get_voltage_from_18bit(stop_reg))
            print('x_stepsize: ', step_reg, VCon.get_stepsize_in_volt_from_18bit(step_reg), VCon.get_voltage_from_18bit(step_reg + int(2 ** 17)))
            print('x_stepsize calc (stop - start)/(nOfsteps-1): ', (stop_reg - start_reg)/(nOfsteps - 1))
            print('x_stop_calc (x_start + x_step * (nOfsteps - 1) ', start_reg + step_reg * (nOfsteps - 1))


            dacStepSize18Bit = track_dict['dacStepSize18Bit']

            self.nrScalers.append(nOfScalers)
            self.stepSize.append(dacStepSize18Bit)
            self.col = track_dict['colDirTrue']
            if self.offset is None:
                self.offset = track_dict['postAccOffsetVolt']
                if track_dict.get('postAccOffsetVoltControl') == 0:
                    self.offset = 0

            if self.seq_type in ['trs', 'tipa', 'trsdummy']:
                self.softBinWidth_ns.append(track_dict.get('softBinWidth_ns', 10))
                self.t = TildaTools.create_t_axis_from_file_dict(scandict)  # force 10 ns resolution
                cts_shape = (nOfScalers, nOfsteps, nOfBins)
                scaler_array = TildaTools.xml_get_data_from_track(
                    lxmlEtree, nOfactTrack, 'scalerArray', cts_shape)
                v_proj = TildaTools.xml_get_data_from_track(
                    lxmlEtree, nOfactTrack, 'voltage_projection', (nOfScalers, nOfsteps),
                    direct_parent_ele_str='projections')
                t_proj = TildaTools.xml_get_data_from_track(
                    lxmlEtree, nOfactTrack, 'time_projection', (nOfScalers, nOfBins),
                    direct_parent_ele_str='projections')
                self.time_res.append(scaler_array)
                if v_proj is None or t_proj is None:
                    print('projections not found, gating data now.')
                    v_proj, t_proj = TildaTools.gate_one_track(
                        tr_ind, nOfactTrack, scandict, self.time_res, self.t, self.x, [])[0]
                self.cts.append(v_proj)
                self.err.append(np.sqrt(v_proj))
                self.err[-1][self.err[-1] < 1] = 1  # remove 0's in the error
                self.t_proj.append(t_proj)
                self.softw_gates.append(track_dict['softwGates'])
                dwell = [g[3] - g[2] for g in track_dict['softwGates']]
                self.dwell.append(dwell)

            elif self.seq_type in ['cs', 'csdummy']:
                cts_shape = (nOfScalers, nOfsteps)
                scaler_array = TildaTools.xml_get_data_from_track(
                    lxmlEtree, nOfactTrack, 'scalerArray', cts_shape)
                self.cts.append(scaler_array)
                self.err.append(np.sqrt(scaler_array))
                self.dwell.append(track_dict.get('dwellTime10ns'))

            elif self.seq_type in ['kepco']:
                meas_volt_dict = scandict['measureVoltPars']
                dmms_dict = meas_volt_dict['dmms']
                dmm_names = list(sorted(dmms_dict.keys()))
                self.nrScalers = [len(dmm_names)]
                self.active_pmt_list = [dmm_names]
                cts_shape = (self.nrScalers[0], nOfsteps)
                dmm_volt_array = TildaTools.xml_get_data_from_track(
                    lxmlEtree, nOfactTrack, 'scalerArray', cts_shape, np.float, default_val=np.nan)
                self.cts.append(dmm_volt_array)
                err = []
                for ind, dmm_name in enumerate(dmm_names):
                    read_acc = 1
                    range_acc = 1
                    if isinstance(dmms_dict[dmm_name]['accuracy'], str):
                        read_acc, range_acc = eval(dmms_dict[dmm_name]['accuracy'])
                    elif isinstance(dmms_dict[dmm_name]['accuracy'], tuple):
                        read_acc, range_acc = dmms_dict[dmm_name]['accuracy']
                    err.append(dmm_volt_array[ind] * read_acc + range_acc)
                self.err.append(err)

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
        if self.seq_type not in ['kepco']:  # do not change the x axis for a kepco scan!
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

# import Service.Scan.draftScanParameters as dft
# import Service.Formating as Form
# test = XMLImporter(None, False, dft.draftScanDict)
# a = test.t_proj[0]
# test = Form.time_rebin_all_spec_data(test, 20)
# b = test.t_proj[0]  # this works
# print(len(b))

# from file:
# for file_num in range(169, 172):
#     test_file = 'D:\lala\sums\Test_kepco_%s.xml' % file_num
#     file_xml = XMLImporter(test_file, False)
#     print(file_num, 'offset: ', file_xml.offset, 'accVolt: ', file_xml.accVolt)
