"""

Created on '07.08.2015'

@author:'simkaufm'

"""

import ast
import logging
import os
import sqlite3

import numpy as np

import Physics
import Tools
import TildaTools
from Measurement.SpecData import SpecData


class XMLImporter(SpecData):
    """
    This Module Reads the .xml files or reads from a given scan_dictionary.
    """

    def __init__(self, path=None, x_as_volt=True, scan_dict=None, softw_gates=None):
        """
        read the xml file
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

        self.file = None

        super(XMLImporter, self).__init__()
        if path is not None:
            logging.info("XMLImporter is reading file " + path)
            self.file = os.path.basename(path)
            scandict, lxmlEtree = TildaTools.scan_dict_from_xml_file(path)
        else:
            logging.info("XMLImporter is reading from scan_dictionary " + str(scan_dict))
            lxmlEtree = None
            scandict = scan_dict
            self.file = os.path.basename(scandict['pipeInternals']['activeXmlFilePath'])
            path = scandict['pipeInternals']['activeXmlFilePath']  # needed for get_frequency_measurement(...)

        self.nrTracks = scandict['isotopeData']['nOfTracks']

        self.laserFreq, self.laserFreq_d = Physics.freqFromWavenumber(2 * scandict['isotopeData']['laserFreq']), 0.0
        self.date = scandict['isotopeData']['isotopeStartTime']
        self.type = scandict['isotopeData']['isotope']
        self.seq_type = scandict['isotopeData']['type']
        self.version = scandict['isotopeData']['version']
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
            x_as_volt = False  # assume this is a gauge measurement of the DAC, so set the x axis in DAC registers
            self.dac_calibration_measurement = True

        self.offset_by_dev, self.offset_by_dev_mean, self.offset = self.get_dmm_measurement(scandict, 'offset')
        self.acc_volt_by_dev, self.acc_volt_by_dev_mean, self.accVolt = self.get_dmm_measurement(scandict, 'accVolt')
        if len(self.accVolt):
            self.accVolt = np.mean(self.accVolt)  # accvolt is assumed to be constant all the time. -> just one float
        else:
            # take the value as set from the gui if dmms did not measure it.
            self.accVolt = scandict['isotopeData']['accVolt']

        self.nrScalers = []  # number of scalers for this track
        self.active_pmt_list = []  # list of scaler/pmt names for this track
        # if self.seq_type in ['tipa', 'tipadummy', 'kepco']:
        # x_as_volt = False
        logging.debug('axaxis as voltage: %s ' % x_as_volt)
        self.x = TildaTools.create_x_axis_from_file_dict(scandict, as_voltage=x_as_volt)  # x axis, voltage
        self.x_dac = TildaTools.create_x_axis_from_file_dict(scandict, as_voltage=False)  # handy for importing files
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
        logging.debug('track_names are: %s ' % self.track_names)
        self.softBinWidth_ns = []
        self.invert_scan = []
        self.post_acc_offset_volt_control = []  # which heinzinger / Fluke
        self.wait_for_kepco_1us = []
        self.wait_after_reset_1us = []
        self.working_time = []
        self.nrScans = []
        self.nrBunches = []  # list for each track an integer with the number fo bunches per step for this track

        cts_shape = []
        self.measureVoltPars = []
        self.tritonPars = []
        self.outbitsPars = []
        ''' operations on each track: '''
        for tr_ind, tr_name in enumerate(TildaTools.get_track_names(scandict)):


            track_dict = scandict[tr_name]
            self.measureVoltPars.append(track_dict.get('measureVoltPars', {}))
            self.tritonPars.append(track_dict.get('triton', {}))
            self.outbitsPars.append(track_dict.get('outbits', {}))
            self.col = track_dict['colDirTrue']

            nOfactTrack = int(tr_name[5:])
            nOfsteps = track_dict['nOfSteps']
            nOfBins = track_dict.get('nOfBins')
            nOfScalers = len(track_dict['activePmtList'])
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
            self.nrScans.append(track_dict['nOfCompletedSteps'] // nOfsteps)

            dacStepSize18Bit = track_dict['dacStepSize18Bit']

            if track_dict.get('trigger', {}).get('meas_trigger', None) is not None:
                self.trigger.append(track_dict.get('trigger', None))
            else:
                self.trigger.append({
                                    'meas_trigger': track_dict.get('trigger', None),
                                    'step_trigger': track_dict.get('step_trigger', None),
                                    'scan_trigger': track_dict.get('scan_trigger', None)
                                    })

            self.nrScalers.append(nOfScalers)
            self.stepSize.append(dacStepSize18Bit)
            self.col = track_dict['colDirTrue']
            if self.seq_type in ['trs', 'tipa', 'trsdummy']:
                self.softBinWidth_ns.append(track_dict.get('softBinWidth_ns', 10))
                self.t = TildaTools.create_t_axis_from_file_dict(scandict, with_delay=True)  # force 10 ns resolution
                cts_shape.append((nOfScalers, nOfsteps, nOfBins))
                scaler_array = TildaTools.xml_get_data_from_track(
                    lxmlEtree, nOfactTrack, 'scalerArray', cts_shape[tr_ind])
                v_proj = TildaTools.xml_get_data_from_track(
                    lxmlEtree, nOfactTrack, 'voltage_projection', (nOfScalers, nOfsteps),
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

                if v_proj is None or t_proj is None or softw_gates is not None:
                    logging.info(' while importing: projections not found,'
                                    ' or software gates set by hand, gating data now.')
                    if softw_gates is not None:
                        if isinstance(softw_gates, tuple):
                            # if the software gates are given as a tuple it should consist of:
                            # (db_str, run_str)
                            new_gates = TildaTools.get_software_gates_from_db(softw_gates[0],
                                                                              self.type, softw_gates[1])
                            if new_gates is not None:
                                # when db states -> use file,
                                # software gates from file will not be overwritten
                                scandict[tr_name]['softwGates'] = new_gates
                        else:
                            if isinstance(softw_gates[tr_ind][0], list):
                                # software gates are defined for each track individually
                                scandict[tr_name]['softwGates'] = softw_gates[tr_ind]
                            else:
                                # software gates are only defined for one track
                                # -> one dimension less than for all tracks.
                                # -> need to be copied for the others
                                scandict[tr_name]['softwGates'] = softw_gates
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
                self.err.append(np.sqrt(np.abs(scaler_array)))
                self.err[-1][self.err[-1] < 1] = 1  # remove 0's in the error
                self.dwell.append(track_dict.get('dwellTime10ns'))

            elif self.seq_type in ['kepco']:
                if float(self.version) <= 1.12:
                    # kept this for older versions
                    meas_volt_dict = scandict['measureVoltPars']['duringScan']
                else:
                    meas_volt_dict = scandict['track0']['measureVoltPars']['duringScan']
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

        self.laserFreq, self.laserFreq_d = self.get_frequency_measurement(path, self.tritonPars)
        logging.info('%s was successfully imported' % self.file)

    def preProc(self, db):
        try:
            logging.info('XMLImporter is using db: %s' % db)
            con = sqlite3.connect(db)
            cur = con.cursor()
            if self.seq_type not in ['kepco']:  # do not change the x axis for a kepco scan!
                db_ret = TildaTools.select_from_db(
                    db, 'type, line, offset, accVolt, laserFreq, colDirTrue, voltDivRatio, lineMult, lineOffset',
                    'Files', [['file'], [self.file]])
                if db_ret is None:
                    raise Exception('XMLImporter: No DB-entry found!')
                if len(db_ret) == 1:
                    (self.type, self.line, self.offset, self.accVolt, self.laserFreq,
                     self.col, self.voltDivRatio, self.lineMult, self.lineOffset) = db_ret[0]
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
                    self.voltDivRatio = {'offset': 1.0, 'accVolt': 1.0}
                for tr_ind, track in enumerate(self.x):
                    self.x[tr_ind] = Tools.line_to_total_volt(self.x[tr_ind], self.lineMult, self.lineOffset,
                                                              self.offset[tr_ind], self.accVolt, self.voltDivRatio,
                                                              offset_by_dev_mean=self.offset_by_dev_mean[tr_ind])
                self.norming()
                self.x_units = self.x_units_enums.total_volts
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
            con.close()

    def export(self, db):
        try:
            con = sqlite3.connect(db)
            col = 1 if self.col else 0
            with con:
                con.execute('''UPDATE Files SET date = ?, type = ?, offset = ?,
                                laserFreq = ?, colDirTrue = ?, accVolt = ?, laserFreq_d = ?
                                 WHERE file = ?''',
                            (self.date, self.type, str(self.offset),
                             self.laserFreq, col, self.accVolt, self.laserFreq_d,
                             self.file))
            con.close()
        except Exception as e:
            logging.error('error while exporting values from file %s, error is: %s' % (self.file, e), exc_info=True)

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

    def norming(self):
        # TODO this is copied from MCP, still the dwell is not implemented in this!
        for trackindex, track in enumerate(self.cts):
            track = track.astype(np.float32)
            self.cts[trackindex] = track
            for ctIndex, ct in enumerate(track):
                min_nr_of_scan = max(np.min(self.nrScans), 1)  # maybe there is a track with 0 complete scans
                nr_of_scan_this_track = self.nrScans[trackindex]
                if nr_of_scan_this_track:
                    # dtype int32 causes problems here!
                    self.cts[trackindex][ctIndex] = ct * min_nr_of_scan / nr_of_scan_this_track
                    self.err[trackindex][ctIndex] = self.err[trackindex][
                                                        ctIndex] * min_nr_of_scan / nr_of_scan_this_track

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
        if float(self.version) <= 1.18:
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
            # at least in one track the offset/accvolt voltage was measured

            # for backwards compability:
            read_key = 'preScanRead' if float(self.version) <= 1.17 else 'readings'

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

    def get_frequency_measurement(self, path, scan_triton_dict):
        """
        It is assumed the frequency has been measured before and/or after first measurement track since
        frequency measurements per track like [f1, f2, ..] is not (yet) supported.
        The frequency measurement is measured by Triton device FrequencyComb1 and/or FrequencyComb2. The first found
        frequency is used for the DB, every found frequency is written into the console.
        The frequency is calculated by averaging preScan and (if taken) postScan values.
        :param path: str, path of the imported file
        :param scan_triton_dict:  dict, scan parameters
        :return: laser_freq, laser_freq_d
        """
        meas_taken = False
        freq_list = []  # holds frequency track_wise
        freq_err_list = []  # holds error of frequency track_wise
        freqs_by_dev = []  # holds mean value of each comb for each track,
        # each track will be dict: {'combkey': (mean_val, mean_err), ... }
        # print(scan_triton_dict)
        for track in scan_triton_dict:
            freq_comb_data_tr = {}
            # make sure there is indeed a pre during post scan measurement and not None
            # if track is not None:
            pre_scan_dict = track.get('preScan', {})
            pre_scan_dict = {} if pre_scan_dict is None else pre_scan_dict
            during_scan_dict = track.get('duringScan', {})
            during_scan_dict = {} if during_scan_dict is None else during_scan_dict
            post_scan_dict = track.get('postScan', {})
            post_scan_dict = {} if post_scan_dict is None else post_scan_dict

            for key, val_dict in pre_scan_dict.items():
                if 'Comb' in key:
                    freq_comb_data_tr[key] = {}
                    freq_comb_data_tr[key]['aCol'] = val_dict.get('comb_freq_acol', {}).get('data', [])
                    freq_comb_data_tr[key]['col'] = val_dict.get('comb_freq_col', {}).get('data', [])

            # now check during scan and append values to existing list or create new

            for key, val_dict in during_scan_dict.items():
                if 'Comb' in key:
                    if not freq_comb_data_tr.get(key, {}):
                        # no measurement was done pre scan for this comb, create empty dict.
                        freq_comb_data_tr[key] = {}
                    for col_a_col_key in ['aCol', 'col']:
                        if not len(freq_comb_data_tr[key].get(col_a_col_key, [])):
                            # the key was not existing yet, create or overwrite empty list
                            freq_comb_data_tr[key][col_a_col_key] = []
                    freq_comb_data_tr[key]['aCol'] += val_dict.get('comb_freq_acol', {}).get('data', [])
                    freq_comb_data_tr[key]['col'] += val_dict.get('comb_freq_col', {}).get('data', [])
                    # freq_comb_data_tr[key]['col'] = [x+203 for x in freq_comb_data_tr[key]['col']]# 203 MHz just due to AOM, REMOVE AFTERWARDS; NO PUSH!!
                    # print("freq_comb_data_tr[key]['col']: ", freq_comb_data_tr[key]['col'])

            # now check post scan and append values to existing list or create new

            for key, val_dict in post_scan_dict.items():
                if 'Comb' in key:
                    if not freq_comb_data_tr.get(key, {}):
                        # no measurement was done before for this comb, create empty dict.
                        freq_comb_data_tr[key] = {}
                    for col_a_col_key in ['aCol', 'col']:
                        if not len(freq_comb_data_tr[key].get(col_a_col_key, [])):
                            # the key was not existing yet, create or overwrite empty list
                            freq_comb_data_tr[key][col_a_col_key] = []
                    freq_comb_data_tr[key]['aCol'] += val_dict.get('comb_freq_acol', {}).get('data', [])
                    freq_comb_data_tr[key]['col'] += val_dict.get('comb_freq_col', {}).get('data', [])

            # now take the mean value for this track:
            combs_freq_mean_tr = {}  # list for the mean value of all combs that have a reading for this track
            for comb_key, comb_a_col_col_dict in freq_comb_data_tr.items():
                # im interested in col/aCol frequency reading:
                col_a_col_key = 'col' if self.col else 'aCol'
                # get the mean value from one comb in this track
                comb_mean = np.mean(comb_a_col_col_dict.get(col_a_col_key, [0]))
                comb_err = np.std(comb_a_col_col_dict.get(col_a_col_key, [0]))
                combs_freq_mean_tr[comb_key] = (comb_mean, comb_err)
            freqs_by_dev.append(combs_freq_mean_tr)

            # get the mean of all comb readings for this track:
            if any(combs_freq_mean_tr):
                means = []
                err_means = []
                for comb_key, read_tupl in combs_freq_mean_tr.items():
                    means.append(read_tupl[0])
                    err_means.append(read_tupl[1])
                freq_list.append(np.mean(means))
                freq_err_list.append(np.mean(err_means))
                meas_taken = True
            else:
                freq_list.append(-1)  # append negative value to indicate there was no measurement in this track
                freq_err_list.append(-1)

        # create ONE mean value for whole file:
        if meas_taken:
            valid_freq_meas = []
            valid_freq_meas_errs = []
            for i, each in enumerate(freq_list):
                if each > 0.0:
                    valid_freq_meas.append(each)
                    valid_freq_meas_errs.append(freq_err_list[i])
            laser_freq = (np.mean(valid_freq_meas) / 1000000)  # in MHz
            laser_freq_d = np.mean(valid_freq_meas_errs) / 1000000  # in MHz
            logging.debug('getting frequency from path: %s' % path)
            (dir, file) = os.path.split(path)
            (filename, end) = os.path.splitext(file)
            logging.debug('Measured Frequencies in ' + str(file) + ' :')
            for tr in freqs_by_dev:
                for comb_key, read_tuple in sorted(tr.items()):
                    logging.debug('%s : %.3f +/- %.3f MHz' % (comb_key, read_tupl[0] / 1000000, read_tupl[1] / 1000000))
        else:
            logging.info('no comb measurement detected, using laserfreq given from GUI')
            laser_freq = self.laserFreq
            laser_freq_d = self.laserFreq_d

        return laser_freq, laser_freq_d






# import Service.Scan.draftScanParameters as dft
# import Service.Formating as Form
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
        'E:\\Workspace\\OwnCloud\\Projekte\\COLLAPS\\Nickel'
        '\\Measurement_and_Analysis_Simon\\Ni_workspace\\TiPaData\\Ni_tipa_024.xml')
    print(meas.laserFreq)
