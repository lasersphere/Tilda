"""

Created on '07.08.2015'

@author:'simkaufm'

"""

import ast
import os
import sqlite3

import numpy as np

import Physics
import TildaTools
from Measurement.SpecData import SpecData


class XMLImporter(SpecData):
    """
    This Module Reads the .xml files or reads from a given scan_dictionary.
    """

    def __init__(self, path=None, x_as_volt=True, scan_dict=None, softw_gates=None):
        '''Read the file'''

        self.file = None

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

        if 'AD5781' in self.type or 'ad5781' in self.type or 'dac_calibration' in self.type:
            print('XMLIMporter assumes this a calibration measurement of the DAC,\n'
                  ' therefore the x-axis will be set to units of DAC registers.\n'
                  'key words therefore are: AD5781, ad5781, dac_calibration\n'
                  'do not use those for the isotope name if you do not want this!\n')
            x_as_volt = False  # assume this is a gauge measurement of the DAC, so set the x axis in DAC registers

        self.accVolt = scandict['isotopeData']['accVolt']
        self.offset = None
        dmms_dict = scandict['measureVoltPars']['preScan'].get('dmms', None)
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
        self.nrScalers = []  # number of scalers for this track
        self.active_pmt_list = []  # list of scaler/pmt names for this track
        # if self.seq_type in ['tipa', 'tipadummy', 'kepco']:
        # x_as_volt = False
        print('axaxis as voltage:', x_as_volt)
        self.x = TildaTools.create_x_axis_from_file_dict(scandict, as_voltage=x_as_volt)  # x axis, voltage
        self.cts = []  # countervalues, this is the voltage projection here
        self.err = []  # error to the countervalues
        self.t_proj = []  # time projection only for time resolved
        self.time_res = []  # time resolved matrices only for time resolved measurements
        self.time_res_zf = []  # time resolved list of pmt events in form of indices, zf is for zero free,
        #  therefore in this list are only events really happened, indices might be missing.
        #  list contains numpy arrays with structure: ('sc', 'step', 'time', 'cts')
        #  indices in list correspond to track indices

        self.stepSize = []
        self.col = False  # should also be a list for multiple tracks
        self.dwell = []
        self.softw_gates = []
        self.track_names = TildaTools.get_track_names(scandict)
        self.softBinWidth_ns = []

        cts_shape = []
        ''' operations on each track: '''
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
            if self.offset is None:
                self.offset = track_dict['postAccOffsetVolt']
                if track_dict.get('postAccOffsetVoltControl') == 0:
                    self.offset = 0

            if self.seq_type in ['trs', 'tipa', 'trsdummy']:
                self.softBinWidth_ns.append(track_dict.get('softBinWidth_ns', 10))
                self.t = TildaTools.create_t_axis_from_file_dict(scandict)  # force 10 ns resolution
                cts_shape.append((nOfScalers, nOfsteps, nOfBins))
                scaler_array = TildaTools.xml_get_data_from_track(
                    lxmlEtree, nOfactTrack, 'scalerArray', cts_shape[tr_ind])
                v_proj = TildaTools.xml_get_data_from_track(
                    lxmlEtree, nOfactTrack, 'voltage_projection', (nOfScalers, nOfsteps),
                    direct_parent_ele_str='projections')
                t_proj = TildaTools.xml_get_data_from_track(
                    lxmlEtree, nOfactTrack, 'time_projection', (nOfScalers, nOfBins),
                    direct_parent_ele_str='projections')
                if isinstance(scaler_array[0], np.void):  # this is zero free data

                    # this fails somewhere for the second track
                    self.time_res_zf.append(scaler_array)
                    time_res_classical_tr = TildaTools.zero_free_to_non_zero_free(self.time_res_zf, cts_shape)[tr_ind]
                    self.time_res.append(time_res_classical_tr)
                else:  # classic full matrix array
                    self.time_res.append(scaler_array)
                print('until here ok')

                if v_proj is None or t_proj is None or softw_gates is not None:
                    print('projections not found, or software gates set by hand, gating data now.')
                    if softw_gates is not None:
                        scandict[tr_name]['softwGates'] = softw_gates[tr_ind]
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
                meas_volt_dict = scandict['measureVoltPars']['duringScan']
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
        if self.seq_type not in ['kepco']:  # do not change the x axis for a kepco scan!
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
            self.voltDivRatio = ast.literal_eval(self.voltDivRatio)
            for tr_ind, track in enumerate(self.x):
                scanvolt = (self.lineMult * self.x[tr_ind] + self.lineOffset + self.offset) * self.voltDivRatio['offset']
                self.x[tr_ind] = self.accVolt * self.voltDivRatio['accVolt'] - scanvolt
        elif self.seq_type == 'kepco':  # correct kepco scans by the measured offset before the scan.
            cur.execute('''SELECT offset FROM Files WHERE file = ?''', (self.file,))
            data = cur.fetchall()
            if len(data) == 1:
                self.offset = data[0]
            for tr_ind, cts_tr in enumerate(self.cts):
                self.cts[tr_ind] = self.cts[tr_ind] - self.offset
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
        """ returns a tuple: (nOfScalers, nOfSteps, nOfBins)
        or if track == -1 go through all tracks and append those tuples for each track """
        if track_ind == -1:
            return [self.get_scaler_step_and_bin_num(tr_ind) for tr_ind, x_tr in enumerate(self.x)]
        n_of_scalers_tr = self.nrScalers[track_ind]
        n_of_steps_tr = self.x[track_ind].size
        if self.seq_type in ['trs', 'trsdummy', 'tipa', 'tipadummy']:
            n_of_bins_tr = self.t[track_ind].size
        else:
            n_of_bins_tr = -1
        return n_of_scalers_tr, n_of_steps_tr, n_of_bins_tr

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
