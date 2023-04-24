"""

Created on '07.08.2015'

@author:'simkaufm'

"""

import ast
import logging
import os
import sqlite3
import datetime

import numpy as np

from Tilda.PolliFit import TildaTools, Physics
from Tilda.PolliFit.Measurement.SpecData import SpecData
from Tilda.Service.Scan.draftScanParameters import draft_scan_device
import Tilda.Service.VoltageConversions.VoltageConversions as VCon


METADATA_CHANNELS = dict(
    accVolt=['agilent', 'pxi', 'hv', 'voltage', 'accvolt'],
    offset=['agilent', 'pxi', 'offset'],
    frequency=['laser_freq_mult', 'frequency']
)


class XMLImporter(SpecData):
    """
    This Module Reads the .xml files or reads from a given scan_dictionary.
    """

    def __init__(self, path=None, x_as_volt=True, scan_dict=None, softw_gates=None):
        """
        Class representing an XML file.

        :param path: str, path to the xml file, set None to load from scan_dict
        :param x_as_volt: bool, True for voltage, False for DAC register
        :param scan_dict: dict, contains all scan information,
         can be used to create a xmlimporter object instead of a .xml file
        :param softw_gates:
            None: read gates from File
            tuple: (db_str, run_str) -> read software gates from db
            list: [
            [[tr0_sc0_vmin, tr0_sc0_vmax, tr0_sc0_t_min, tr0_sc0_tmax], [other sc]],
            [[tr1_sc0_vmin, tr1_sc0_vmax, tr1_sc0_t_min, tr1_sc0_tmax], [other sc]]
            ]
            also possible list: (use same gates for all tracks)
            [[tr0_sc0_vmin, tr0_sc0_vmax, tr0_sc0_t_min, tr0_sc0_tmax], [other sc]]
        """
        super(XMLImporter, self).__init__()

        self.file = None
        if path is not None:
            logging.info('XMLImporter is reading file ' + path)
            self.file = os.path.basename(path)
            scan_dict, lxml_etree = TildaTools.scan_dict_from_xml_file(path)
        else:
            logging.info('XMLImporter is reading from scan_dictionary ' + str(scan_dict))
            lxml_etree = None
            self.file = os.path.basename(scan_dict['pipeInternals']['activeXmlFilePath'])
            # path = scan_dict['pipeInternals']['activeXmlFilePath']

        self.nrTracks = scan_dict['isotopeData']['nOfTracks']
        self.date = scan_dict['isotopeData']['isotopeStartTime']  # might be overwritten below
        # by the mid-time of the iso
        self.date_d = 0.0  # uncertainty of the date in s might be overwritten
        # at the end of tracks readout should be file_length / 2
        self.type = scan_dict['isotopeData']['isotope']
        self.seq_type = scan_dict['isotopeData']['type']
        self.version = scan_dict['isotopeData']['version']
        self.version_list = [int(n) for n in self.version.split('.')]
        if len(self.version_list) < 3:
            self.version_list.append(0)
        self.dac_calibration_measurement = False

        self.trigger = []  # list of triggers for each track

        self.offset_by_dev = [{}]  # list (track_indexed) of dicts for a list of measured offset voltages
        #  key is device name value is list, which is split into pre scan and post scan values

        if 'AD5781' in self.type or 'ad5781' in self.type or 'dac_calibration' in self.type:
            logging.warning('--------------------------WARNING----------------------------------\n'
                            'XMLIMporter assumes this a calibration measurement of the DAC,\n'
                            ' therefore the x-axis will be set to units of DAC registers.\n'
                            'key words therefore are: AD5781, ad5781, dac_calibration\n'
                            'do not use those for the isotope name if you do not want this!\n'
                            '--------------------------WARNING----------------------------------\n')
            x_as_volt = False  # assume this is a gauge measurement of the DAC, so set the x-axis in DAC registers
            self.dac_calibration_measurement = True

        ''' Get voltage and frequency metadata from scan_dict or XML file '''
        self.meta_data_channels = dict(accVolt='', offset='', frequency='')
        # take the GUI value if it was not measured.
        self.accVolt, self.accVolt_d = scan_dict['isotopeData']['accVolt'], 0.
        voltages = self.get_metadata_measurement(scan_dict, 'accVolt')
        if voltages and any(d for d in voltages):
            voltages = [u for u_track in voltages for u in u_track]  # Currently there is only 1 voltage allowed.
            self.accVolt = np.mean(voltages)
            self.accVolt_d = np.std(voltages, ddof=1) if len(voltages) > 1 else 0.

        # take the GUI value if it was not measured.
        self.offset = [scan_dict[track]['postAccOffsetVolt'] for track in TildaTools.get_track_names(scan_dict)]
        # self.offset_by_dev, self.offset_by_dev_mean, self.offset = self.get_dmm_measurement(scan_dict, 'offset')
        self.offset_by_dev = [{} for _ in TildaTools.get_track_names(scan_dict)]  # TODO: Get rid of this or implement.
        self.offset_by_dev_mean = [{} for _ in TildaTools.get_track_names(scan_dict)]
        offsets = self.get_metadata_measurement(scan_dict, 'offset')
        if offsets and all(d for d in offsets):
            self.offset = [np.mean(d) for d in offsets]  # Offset voltages are required per track.

        # take the GUI value if it was not measured.
        self.laserFreq, self.laserFreq_d = Physics.freqFromWavenumber(scan_dict['isotopeData']['laserFreq']), 0.
        frequencies = self.get_metadata_measurement(scan_dict, 'frequency')
        if frequencies and any(d for d in frequencies):
            # Currently there is only 1 frequency allowed.
            frequencies = [f for f_track in frequencies for f in f_track]
            self.laserFreq = np.mean(frequencies)
            self.laserFreq_d = np.std(frequencies, ddof=1) if len(frequencies) > 1 else 0.

        self.nrScalers = []  # number of scalers for this track
        self.active_pmt_list = []  # list of scaler/pmt names for this track
        # if self.seq_type in ['tipa', 'tipadummy', 'kepco']:
        # x_as_volt = False
        logging.debug('axaxis as voltage: %s ' % x_as_volt)
        self.x = TildaTools.create_x_axis_from_file_dict(scan_dict, as_voltage=x_as_volt)  # x axis, voltage
        self.x_dac = TildaTools.create_x_axis_from_file_dict(scan_dict, as_voltage=False)  # handy for importing files
        self.cts = []  # countervalues, this is the voltage projection here
        self.err = []  # error to the countervalues
        self.t_proj = []  # time projection only for time resolved
        self.time_res = []  # time resolved matrices only for time resolved measurements
        self.time_res_zf = []  # time resolved list of pmt events in form of indices, zf is for zero free,
        #  therefore in this list are only events really happened, indices might be missing.
        #  list contains numpy arrays with structure: ('sc', 'step', 'time', 'cts')
        #  indices in list correspond to track indices

        # in some special cases (e.g. combined data) errors might need to be given externally
        self.time_res_err = []  # non-standard error time resolved matrices only for time resolved measurements
        self.time_res_zf_err = []  # time resolved list of non-standard errors in form of indices, zf is for zero free,

        self.stepSize = []
        self.col = False  # should also be a list for multiple tracks
        self.dwell = []
        self.softw_gates = []
        self.track_names = TildaTools.get_track_names(scan_dict)
        logging.debug('track_names are: %s ' % self.track_names)
        self.softBinWidth_ns = []
        self.post_acc_offset_volt_control = []  # which heinzinger / Fluke
        self.wait_for_kepco_1us = []
        self.wait_after_reset_1us = []
        self.working_time = []
        self.nrScans = []
        self.nrBunches = []  # list for each track an integer with the number fo bunches per step for this track

        cts_shape = []
        self.measureVoltPars = []
        self.tritonPars = []
        self.sqlPars = []
        self.outbitsPars = []

        ''' Operations on each track: '''
        for tr_ind, tr_name in enumerate(TildaTools.get_track_names(scan_dict)):

            track_dict = scan_dict[tr_name]
            scan_dev_dict_tr = track_dict.get('scanDevice', draft_scan_device)
            self.scan_dev_dict_tr_wise.append(scan_dev_dict_tr)
            # overwrite with last of tracks, but should be the same unit for all tracks anyhow (hopefully)
            self.x_units = self.x_units_enums[scan_dev_dict_tr['stepUnitName']]
            self.measureVoltPars.append(track_dict.get('measureVoltPars', {}))
            self.tritonPars.append(track_dict.get('triton', {}))
            self.sqlPars.append(track_dict.get('sql', {}))
            self.outbitsPars.append(track_dict.get('outbits', {}))
            self.col = track_dict['colDirTrue']

            n_act_track = int(tr_name[5:])
            n_steps = track_dict['nOfSteps']
            self.nrSteps.append(n_steps)
            n_bins = track_dict.get('nOfBins')
            n_scalers = len(track_dict['activePmtList'])
            self.nrBunches += track_dict.get('nOfBunches', 1),
            self.active_pmt_list.append(track_dict['activePmtList'])

            self.invert_scan.append(track_dict['invertScan'])
            self.post_acc_offset_volt_control.append(track_dict['postAccOffsetVoltControl'])

            if track_dict.get('waitAfterReset25nsTicks', None) is not None:
                # this was named like this before version 1.19
                wait_1us = track_dict['waitAfterReset25nsTicks'] * 25 / 1000
                self.wait_after_reset_1us.append(wait_1us)
            if track_dict.get('waitAfterReset1us', None) is not None:
                self.wait_after_reset_1us.append(track_dict['waitAfterReset1us'])

            if track_dict.get('waitForKepco25nsTicks', None) is not None:
                wait_1us = track_dict['waitForKepco25nsTicks'] * 25 / 1000
                self.wait_for_kepco_1us.append(wait_1us)
            if track_dict.get('waitForKepco1us', None) is not None:
                self.wait_for_kepco_1us.append(track_dict['waitForKepco1us'])

            self.working_time.append(track_dict['workingTime'])
            self.nrScans.append(track_dict['nOfCompletedSteps'] // n_steps)
            self.nrLoops.append(max([self.nrScans[-1], 1]))

            dac_step_size18_bit = track_dict.get('dacStepSize18Bit', None)  # leave in for backwards_comp
            if dac_step_size18_bit is None or dac_step_size18_bit == {}:  # TODO: not nice...
                # TODO: copy/paste from laptop. dacStepsizeVoltage is set correctly in file.
                #  So why load through 'start'?
                # step_size = track_dict['dacStepsizeVoltage']  # OLD. does not exist in dummy scan dicts (and other??)
                step_size = scan_dev_dict_tr.get('stepSize', 1)
            else:
                step_size = VCon.get_stepsize_in_volt_from_bits(dac_step_size18_bit)
            self.stepSize.append(step_size)

            if track_dict.get('trigger', {}).get('meas_trigger', None) is not None:
                self.trigger.append(track_dict.get('trigger', None))
            else:
                self.trigger.append({
                                    'meas_trigger': track_dict.get('trigger', None),
                                    'step_trigger': track_dict.get('step_trigger', None),
                                    'scan_trigger': track_dict.get('scan_trigger', None)
                                    })

            self.nrScalers.append(n_scalers)
            self.col = track_dict['colDirTrue']

            if self.seq_type in ['trs', 'tipa', 'trsdummy']:
                self.softBinWidth_ns.append(track_dict.get('softBinWidth_ns', 10))
                self.t = TildaTools.create_t_axis_from_file_dict(scan_dict, with_delay=True)  # force 10 ns resolution
                cts_shape.append((n_scalers, n_steps, n_bins))
                scaler_array = TildaTools.xml_get_data_from_track(
                    lxml_etree, n_act_track, 'scalerArray', cts_shape[tr_ind])
                # maybe the file has non-standard errors. Try to import them. Fail will return NONE:
                error_array = TildaTools.xml_get_data_from_track(
                    lxml_etree, n_act_track, 'errorArray', cts_shape[tr_ind], create_if_no_root_ele=False)

                v_proj = TildaTools.xml_get_data_from_track(
                    lxml_etree, n_act_track, 'voltage_projection', (n_scalers, n_steps),
                    direct_parent_ele_str='projections')

                t_proj = None
                if isinstance(scaler_array[0], np.void):  # this is zero free data
                    self.time_res_zf.append(scaler_array)
                    time_res_classical_tr = TildaTools.zero_free_to_non_zero_free(self.time_res_zf, cts_shape)[tr_ind]
                    self.time_res.append(time_res_classical_tr)
                else:  # classic full matrix array
                    self.time_res.append(scaler_array)
                    zf_data_tr = TildaTools.non_zero_free_to_zero_free([scaler_array])[0]
                    self.time_res_zf.append(zf_data_tr)
                if error_array is not None:
                    if isinstance(error_array[0], np.void):  # this is zero free data
                        self.time_res_zf_err.append(error_array)
                        time_res_err_classical_tr = TildaTools.zero_free_to_non_zero_free(self.time_res_zf_err,
                                                                                          cts_shape)[tr_ind]
                        self.time_res_err.append(time_res_err_classical_tr)
                    else:  # classic full matrix array
                        self.time_res_err.append(error_array)
                        zf_err_tr = TildaTools.non_zero_free_to_zero_free([error_array])[0]
                        self.time_res_zf_err.append(zf_err_tr)

                if softw_gates is None:
                    # TODO: Change copy/paste from laptop. Should be solved elegantly and tested.
                    # Reason for change: sometimes voltptoj from file is bad and we want a new projection.
                    softw_gates = track_dict['softwGates']
                if v_proj is None or t_proj is None or softw_gates is not None:
                    logging.info(' while importing: projections not found,'
                                 ' or software gates set by hand, gating data now.')
                    if softw_gates is not None:
                        if isinstance(softw_gates, tuple):
                            # if the software gates are given as a tuple it should consist of:
                            # (db_str, run_str)
                            new_gates = TildaTools.get_software_gates_from_db(softw_gates[0],
                                                                              self.type, softw_gates[1], track=tr_ind)
                            if new_gates is not None:
                                # when db states -> use file,
                                # software gates from file will not be overwritten
                                scan_dict[tr_name]['softwGates'] = new_gates
                        else:
                            try:  # for more than three tracks, list index can go out off range
                                if isinstance(softw_gates[tr_ind][0], list):
                                    # software gates are defined for each track individually
                                    scan_dict[tr_name]['softwGates'] = softw_gates[tr_ind]
                                else:
                                    # software gates are only defined for one track
                                    # -> one dimension less than for all tracks.
                                    # -> need to be copied for the others
                                    scan_dict[tr_name]['softwGates'] = softw_gates
                            except IndexError:
                                # index error means its not per track!
                                scan_dict[tr_name]['softwGates'] = softw_gates

                    v_proj, t_proj = TildaTools.gate_one_track(
                        tr_ind, n_act_track, scan_dict, self.time_res, self.t, self.x, [])[0]
                self.cts.append(v_proj)

                if error_array is not None:
                    # errors are explicitly given. Use those.
                    # first gate the errors
                    v_proj_err, t_proj_err = TildaTools.gate_one_track(
                        tr_ind, n_act_track, scan_dict, [np.square(tr_arr) for tr_arr in self.time_res_err],
                        self.t, self.x, [])[0]
                    # square errors first, then sum along the projection, now take the sqrt again.
                    self.err.append(np.sqrt(v_proj_err))
                else:
                    # if no errors were specified, use standard errors
                    self.err.append(np.sqrt(v_proj))
                self.err[-1][self.err[-1] < 1] = 1  # remove 0's in the error
                self.t_proj.append(t_proj)
                self.softw_gates.append(track_dict['softwGates'])
                dwell = [g[3] - g[2] for g in track_dict['softwGates']]
                self.dwell.append(dwell)

            elif self.seq_type in ['cs', 'csdummy']:
                cts_shape = (n_scalers, n_steps)
                scaler_array = TildaTools.xml_get_data_from_track(
                    lxml_etree, n_act_track, 'scalerArray', cts_shape)
                self.cts.append(scaler_array)
                # maybe the file has non-standard errors. Try to import them. Fail will return NONE:
                error_array = TildaTools.xml_get_data_from_track(
                    lxml_etree, n_act_track, 'errorArray', cts_shape, create_if_no_root_ele=False)
                if error_array is None:
                    # if no errors were specified, use standard errors
                    self.err.append(np.sqrt(np.abs(scaler_array)))
                else:
                    # errors are explicitly given. Use those.
                    self.err.append(error_array)
                self.err[-1][self.err[-1] < 1] = 1  # remove 0's in the error
                self.dwell.append(track_dict.get('dwellTime10ns'))

            elif self.seq_type in ['kepco']:
                if self.version_list[0] <= 1 and self.version_list[1] <= 12:
                    # kept this for older versions
                    meas_volt_dict = scan_dict['measureVoltPars']['duringScan']
                else:
                    meas_volt_dict = scan_dict['track0']['measureVoltPars']['duringScan']
                dmms_dict = meas_volt_dict['dmms']
                dmm_names = list(sorted(dmms_dict.keys()))
                self.nrScalers = [len(dmm_names)]
                self.active_pmt_list = [dmm_names]
                cts_shape = (self.nrScalers[0], n_steps)
                dmm_volt_array = TildaTools.xml_get_data_from_track(
                    lxml_etree, n_act_track, 'scalerArray', cts_shape, float, default_val=np.nan)
                # TODO: np.nan might be causing problems!
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

        self.convert_date_to_mid_time()
        logging.info('%s was successfully imported' % self.file)

    def pre_process(self, db):
        con = None
        try:
            logging.info('XMLImporter is using db: %s' % db)
            con = sqlite3.connect(db)
            if self.seq_type not in ['kepco']:  # do not change the x axis for a kepco scan!
                db_ret = TildaTools.select_from_db(db, 'type, line, offset, accVolt, laserFreq, colDirTrue,'
                                                       ' voltDivRatio, lineMult, lineOffset, laserFreq_d',
                                                   'Files', [['file'], [self.file]])
                if db_ret is None:
                    raise Exception('XMLImporter: No DB-entry found!')
                if len(db_ret) == 1:
                    (self.type, self.line, self.offset, self.accVolt, self.laserFreq,
                     self.col, self.voltDivRatio, self.lineMult, self.lineOffset, self.laserFreq_d) = db_ret[0]
                    self.col = bool(self.col)
                    # should be a string of a list of offset values for each track:
                    if isinstance(self.offset, float):
                        # old databases might still have just one value for the offset in the db
                        self.offset = [self.offset] * self.nrTracks
                    elif isinstance(self.offset, str):
                        self.offset = ast.literal_eval(self.offset)
                else:
                    raise Exception('XMLImporter: No DB-entry found!')
                try:
                    self.voltDivRatio = ast.literal_eval(self.voltDivRatio)
                except Exception as e:
                    logging.error('error, converting voltage divider ratio from db, error is: ' + str(e), exc_info=True)
                    logging.info('setting voltage divider ratio to 1 !')
                    self.voltDivRatio = {'offset': 1.0, 'accVolt': 1.0, 'laserFreq': 1.0}
                if 'CounterDrift' in self.scan_dev_dict_tr_wise[0]['name']:
                    for tr_ind, track in enumerate(self.x):
                        self.x[tr_ind] = track * self.lineMult + self.lineOffset
                    self.x_units = self.x_units_enums.frequency_mhz
                else:
                    for tr_ind, track in enumerate(self.x):
                        print('trind:', tr_ind)
                        print('offset:', self.offset)
                        self.x[tr_ind] = TildaTools.line_to_total_volt(
                            self.x[tr_ind], self.lineMult, self.lineOffset, self.offset[tr_ind], self.accVolt,
                            self.voltDivRatio, offset_by_dev_mean=self.offset_by_dev_mean[tr_ind])
                    self.x_units = self.x_units_enums.total_volts
                self.laserFreq *= self.voltDivRatio['laserFreq']
                self.laserFreq_d *= self.voltDivRatio['laserFreq']

            elif self.seq_type == 'kepco':  # correct kepco scans by the measured offset before the scan.
                db_ret = TildaTools.select_from_db(db, 'offset', 'Files', [['file'], [self.file]])
                # cur.execute('''SELECT offset FROM Files WHERE file = ?''', (self.file,))
                # data = cur.fetchall()
                if db_ret is None:
                    raise Exception('XMLImporter: No DB-entry found!')
                if self.dac_calibration_measurement:
                    self.offset = 0
                else:  # get the offset from the database or leave it as it is from import always prefer db
                    if len(db_ret) == 1:
                        offset_db = db_ret[0]
                        self.offset = offset_db
                for tr_ind, cts_tr in enumerate(self.cts):
                    if len(self.active_pmt_list[tr_ind]):
                        for dmm_ind, dmm_name in enumerate(self.active_pmt_list[tr_ind]):
                            offset_dmm_tr = self.offset_by_dev_mean[tr_ind][dmm_name]
                            self.cts[tr_ind][dmm_ind] = self.cts[tr_ind][dmm_ind] - offset_dmm_tr
                            logging.debug('preprocessing kepco x-axis: dmm_name: %s, track: %s, offset_dmm_tr; %s'
                                          % (dmm_name, tr_ind, offset_dmm_tr))
                    else:
                        self.cts[tr_ind] = self.cts[tr_ind] - self.offset

            con.close()
        except Exception as e:
            logging.error(
                'error while preprocessing file %s, error is: %s, check db values!' % (self.file, e), exc_info=True)
            if con is not None:
                con.close()

    def export(self, db):
        try:
            self.convert_date_to_mid_time()
            con = sqlite3.connect(db)
            col = 1 if self.col else 0
            with con:
                # print('exporting:')
                # print((self.date, self.type, str(self.offset),
                #        self.laserFreq, col, self.accVolt, self.laserFreq_d,
                #        self.file, self.date_d))
                con.execute('''UPDATE Files SET date = ?, type = ?, offset = ?,
                                laserFreq = ?, colDirTrue = ?, accVolt = ?, laserFreq_d = ?, errDateInS = ?
                                 WHERE file = ?''',
                            (self.date, self.type, str(self.offset),
                             self.laserFreq, col, self.accVolt, self.laserFreq_d, self.date_d,
                             self.file))
            con.close()
        except Exception as e:
            logging.error('error while exporting values from file %s, error is: %s' % (self.file, e), exc_info=True)

    @staticmethod
    def eval_err(cts, f):
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

    def get_data_str(self, dev):
        if dev == 'dmms':
            return 'preScanRead' if self.version_list[0] <= 1 and self.version_list[1] <= 17 else 'readings'
        return 'data'

    def find_metadata_key(self, meta_dict, mtype):
        # TODO: Let the user choose the channel.
        dev_ch = self.meta_data_channels[mtype]
        if dev_ch:
            data_key = self.get_data_str(dev_ch[:dev_ch.find('.')])
            return dev_ch, data_key

        for _ch in METADATA_CHANNELS[mtype]:
            for dev_ch, ch_dict in meta_dict.items():
                i = dev_ch.find('.')
                dev = dev_ch[:i]
                ch = dev_ch[(i + 1):]
                data_key = self.get_data_str(dev)
                if _ch in ch.lower():
                    if dev == 'dmms' and ch_dict.get('assignment', 'accVolt') != mtype:
                        continue
                    if dev != 'dmms' and mtype == 'offset' and _ch in {'agilent', 'pxi'}:
                        continue  # We do not want to interpret the HV as an offset.
                    # self.meta_data_channels[mtype] = dev_ch
                    return dev_ch, data_key
        return None, None

    def get_metadata_measurement_pre_dur_post_track(self, scan_dict, pre_dur_post, track, mtype):
        tr_dict = scan_dict[track]
        meta_dict = []
        dmm = {'dmms.{}'.format(ch): ch_dict
               for ch, ch_dict in tr_dict.get('measureVoltPars', {}).get(pre_dur_post, {}).get('dmms', {}).items()}
        meta_dict.append(dmm)
        _triton = tr_dict.get('triton', {}).get(pre_dur_post, {})
        triton = {}
        for dev, dev_dict in _triton.items():
            if 'data' in list(dev_dict.keys()):
                triton[dev] = dev_dict
            else:
                for ch, ch_dict in dev_dict.items():
                    triton['{}.{}'.format(dev, ch)] = ch_dict
        meta_dict.append(triton)
        meta_dict.append(tr_dict.get('sql', {}).get(pre_dur_post, {}))
        meta_dict = TildaTools.merge_dicts(*meta_dict)
        dev_ch, data_key = self.find_metadata_key(meta_dict, mtype)
        if dev_ch is not None:
            return meta_dict[dev_ch].get(data_key, [])
        return []

    def get_metadata_measurement_track(self, scan_dict, track, mtype):
        data = self.get_metadata_measurement_pre_dur_post_track(scan_dict, 'duringScan', track, mtype)
        if not data:
            data = self.get_metadata_measurement_pre_dur_post_track(scan_dict, 'preScan', track, mtype) \
                + self.get_metadata_measurement_pre_dur_post_track(scan_dict, 'postScan', track, mtype)
        return data

    def get_metadata_measurement(self, scan_dict, mtype):
        return [self.get_metadata_measurement_track(scan_dict, track, mtype)
                for track in TildaTools.get_track_names(scan_dict)]

    def get_dmm_measurement(self, scandict, assignment='offset'):
        """
        before version 1.18, the offset and accvolt voltages were only measured once before the scan.
        There was NO track wise measurement possible.
        All tracks were assumed to have the same offset.
        """
        offset_by_dev = []  # track wise dicts for offsets measured by devs. key is dmm_name
        offset_by_dev_mean = []  # same as offset by dev but instead of storing all values,
        #  here only a mean is stored for each dev
        offset_vals_list = []  # track wise all offset values
        offset_mean = []  # track wise mean values of offset for all devices with offset assignment
        set_value_list = []
        if self.version_list[0] <= 1 and self.version_list[1] <= 18:
            # only prescan available and only before first track
            # in order to have a value for each track, copy this existing one:
            dmms_dict_list = [
                                 (scandict.get('measureVoltPars', {}).get('preScan', {}).get('dmms', {}),
                                  {},
                                  {})] * self.nrTracks
        else:
            dmms_dict_list = []
            # consist of a tuple of dicts for each track
            # [(tr0_dmm_pre_scan_read_dict, tr0_dmm_post_scan_read_dict), ...]
            for key, track_d in sorted(scandict.items()):
                if 'track' in key:
                    dmms_dict_list.append(
                        (track_d.get('measureVoltPars', {}).get('preScan', {}).get('dmms', {}),
                         track_d.get('measureVoltPars', {}).get('duringScan', {}).get('dmms', {}),
                         track_d.get('measureVoltPars', {}).get('postScan', {}).get('dmms', {}))
                    )
                    # if no measurements were taken an empty dict is appended.
                    if assignment == 'offset':
                        set_value_list += track_d.get('postAccOffsetVolt', 0.0),
                    elif assignment == 'accVolt':
                        set_value_list += scandict.get('isotopeData', {}).get('accVolt', 0.0),

        # check if any measurement was taken at all
        measurement_taken = any([any(each[0]) or any(each[1]) or any(each[2]) for each in dmms_dict_list])
        if measurement_taken:
            measurement_taken = any([v == assignment for track_tuple in dmms_dict_list
                                     for predurpost_dict in track_tuple for dmm_dict in predurpost_dict.values()
                                     for k, v in dmm_dict.items() if k == 'assignment'])
        # now includes an assignment check.

        if measurement_taken:
            # at least in one track the offset/accvolt voltage was measured

            # for backwards compability:
            read_key = 'preScanRead' if self.version_list[0] <= 1 and self.version_list[1] <= 17 else 'readings'

            for tr_ind, each in enumerate(dmms_dict_list):
                offset_by_dev.append({})
                offset_by_dev_mean.append({})
                offset_vals_list.append([])

                if each[0] == {} and each[1] == {} and each[2] == {}:
                    # no measurement was taken for this track, copy from the track before.
                    # this will fail when voltage is not measured in track0 but e.g. track1
                    # dont do this ;)
                    offset_by_dev[tr_ind] = offset_by_dev[tr_ind - 1]
                    offset_vals_list[tr_ind] = offset_vals_list[tr_ind - 1]
                else:
                    for pre_dur_post_ind, post_pre_dict in enumerate(each):
                        # post_pre_dict will be {} if no measurement was taken in this track.
                        for dmm_name, dmm_dict in post_pre_dict.items():
                            if offset_by_dev[tr_ind].get(dmm_name, None) is None:
                                # initialise an empty list for pre, during and post scan values.
                                offset_by_dev[tr_ind][dmm_name] = [[], [], []]
                            for key, val in dmm_dict.items():
                                if key == read_key:
                                    if isinstance(val, str):
                                        val = ast.literal_eval(val)
                                    if dmm_dict.get('assignment') == assignment:
                                        if isinstance(val, list):
                                            offset_vals_list[tr_ind] += val  # append to list
                                            offset_by_dev[tr_ind][dmm_name][pre_dur_post_ind] += val
                                        else:
                                            offset_vals_list[tr_ind].append(val)
                                            offset_by_dev[tr_ind][dmm_name][pre_dur_post_ind].append(val)
                for dmm_name, offset_list_dmm in offset_by_dev[tr_ind].items():
                    # get mean value for this dmm in this track
                    offset_by_dev_flat = [item for sublist in offset_list_dmm for item in sublist]
                    if len(offset_by_dev_flat):
                        offset_by_dev_mean[tr_ind][dmm_name] = np.mean(offset_by_dev_flat)
                if len(offset_vals_list[tr_ind]):
                    # mean of all dmms for this track
                    offset_mean += np.mean(offset_vals_list[tr_ind]),
        else:
            # no measurement was taken at all -> take set values
            offset_mean = set_value_list
            offset_by_dev_mean = [{'setValue': each} for each in set_value_list]
            offset_by_dev = [{'setValue': [[each], [each]]} for each in set_value_list]
        return offset_by_dev, offset_by_dev_mean, offset_mean

    def convert_date_to_mid_time(self):
        """
        try to get the mid time of the file and overwrite self.date with this mid time
        self.date_d will be half the length of the whole file.
        If this fails, self.date will not be overwritten and self.date_d will be zero
        :return: None
        """
        work_time_flat_date_time = []
        time_format = '%Y-%m-%d %H:%M:%S'

        if self.working_time is not None:
            if None not in self.working_time:
                # TildaTools.create_scan_dict_from_spec_data initialises
                # the working_time with a list of [None] * nrTracks
                try:
                    work_time_flat_date_time = [datetime.datetime.strptime(w_time_str, time_format)
                                                for tr_work_t_list in self.working_time
                                                for w_time_str in tr_work_t_list]
                except Exception as e:
                    logging.warning('could not convert the working time time stamps to a common'
                                    ' mid time in file %s.\nThe working times are: %s'
                                    '\nerror message is: %s' % (self.file, str(self.working_time), e))
                if len(work_time_flat_date_time) > 1:
                    work_time_flat_date_time_float = [work_t_dt.timestamp() for work_t_dt in work_time_flat_date_time]
                    iso_start_t = np.min(work_time_flat_date_time_float)
                    iso_stop_t = np.max(work_time_flat_date_time_float)
                    diff = iso_stop_t - iso_start_t
                    err_date = diff / 2
                    mid_iso_t = iso_start_t + err_date
                    mid_iso_t_dt = datetime.datetime.fromtimestamp(mid_iso_t)
                    mid_iso_t_dt_str = mid_iso_t_dt.strftime(time_format)
                    self.date = mid_iso_t_dt_str
                    self.date_d = err_date  # in seconds


# import Tilda.Service.Scan.draftScanParameters as dft
# import Tilda.Service.Formatting as Form
# test = XMLImporter(None, False, dft.draftScanDict)
# a = test.t_proj[0]
# test = Form.time_rebin_all_spec_data(test, 20)
# b = test.t_proj[0]  # this works
# print(len(b))

# from file:
# for file_num in range(169, 172):
if __name__ == '__main__':
    # meas = XMLImporter('E:\\temp2\\data\\137Ba_acol_cs_run511.xml')
    meas = XMLImporter(
        'C:\\Users\\Laura Renth\\OwnCloud\\Projekte\\COLLAPS\\Nickel'
        '\\Measurement_and_Analysis_Simon\\Ni_workspace2017\\Ni_2017\\sums\\70_Ni_trs_run268_plus_run312.xml')
    print(meas.date, meas.date_d)
