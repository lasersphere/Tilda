"""

Created on '30.09.2015'

@author:'simkaufm'

"""

import ast
import logging
import os
import gc
from copy import deepcopy
from datetime import datetime
from datetime import timedelta

from PyQt5 import QtCore
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QApplication

import Application.Config as Cfg
import Service.DatabaseOperations.DatabaseOperations as DbOp
import Service.FileOperations.FolderAndFileHandling as FileHandl
import Service.Scan.ScanDictionaryOperations as SdOp
import Service.Scan.draftScanParameters as Dft
from Measurement.XMLImporter import XMLImporter
import TildaTools
from Application.Main.MainState import MainState
from Service.AnalysisAndDataHandling.DisplayData import DisplayData
from Service.Scan.ScanMain import ScanMain
from Service.SimpleCounter.SimpleCounter import SimpleCounterControl


class Main(QtCore.QObject):
    # this will equal the number of completed steps in the active track:
    scan_prog_call_back_sig_pipeline = QtCore.pyqtSignal(int)
    # close_spec_display = QtCore.pyqtSignal(str)

    # signal which will be emitted from the pipeline (for now only kepco) if the scan is completed.
    scan_complete_callback = QtCore.pyqtSignal(bool)

    # string which can be connected to in order to get info / warninings from main
    info_warning_string_main_signal = QtCore.pyqtSignal(str)

    # Callbacks for a live plot
    # these callbacks should be called from the pipeline:
    # for incoming new data:
    new_data_callback = QtCore.pyqtSignal(XMLImporter)
    # if a new track is started call:
    # the tuple is of form: ((tr_ind, tr_name), (pmt_ind, pmt_name))
    new_track_callback = QtCore.pyqtSignal(tuple)
    # when the pipeline wants to save, this is emitted and it send the pipeData as a dict
    save_callback = QtCore.pyqtSignal(dict)
    # for incoming new dmm or triton data
    pre_post_meas_data_dict_callback = QtCore.pyqtSignal(dict)

    # dict, fit result plot data callback
    # -> this can be emitted from a node to send a dict containing fit results:
    # 'plotData': tuple of ([x], [y]) values to plot a fit result.
    # 'result': list of result-tuples (name, pardict, fix)
    fit_results_dict_callback = QtCore.pyqtSignal(dict)

    # signal to request updated gated data from the pipeline.
    # list: software gates [[[tr0_sc0_vMin, tr0_sc0_vMax, tr0_sc0_tMin, tr0_sc0_tMax], [tr0_sc1_...
    # int: track_index to rebin -1 for all
    # list: software bin width in ns for each track
    # bool: plot bool to force a plotting even if nothing has changed.
    new_gate_or_soft_bin_width = QtCore.pyqtSignal(list, int, list, bool)

    # save request
    save_request = QtCore.pyqtSignal()

    # progress dict coming from the main
    live_plot_progress_callback = QtCore.pyqtSignal(dict)

    def __init__(self):
        super(Main, self).__init__()
        self.m_state = MainState.init
        self.database = None  # path of the sqlite3 database
        self.working_directory = None  # path of the working directory, containig the database etc.
        self.measure_voltage_pars = Dft.draftMeasureVoltPars
        # dict containing all parameters for the voltage measurement.
        # default is: draftMeasureVoltPars = {'measVoltPulseLength25ns': 400, 'measVoltTimeout10ns': 100}
        self.laserfreq = 0  # laser frequency in cm-1
        self.acc_voltage = 0  # acceleration voltage of the source in volts
        self.simple_counter_inst = None
        self.cmd_queue = None
        self.jobs_to_do_when_idle_queue = []
        self.autostart_dict = {}  # dict containing all infos from the autostart.xml file keys are: workingDir,
        # autostartDevices: {dmms: {name: address}, powersupplies: {name:address}}

        # dict of pyqtSignals(dict) for sending the status of the power supply to the correspnding gui
        self.power_sup_stat_callback_signals = {}
        # pyqtSignal for sending the status to the gui, if there is one connected:
        self.main_ui_status_call_back_signal = None
        # pyqtSignal for sending the scan progress to the gui while scanning.
        self.scan_prog_call_back_sig_gui = None
        self.scan_prog_call_back_sig_pipeline.connect(self.update_scan_progress)
        # tuple of three callbacks which are needed for the live plot gui
        #  and which are emitted from the pipeline. Therefore those must be available when initialising the pipeline.

        self.scan_main = ScanMain()
        self.iso_scan_process = None
        self.scan_pars = {}  # {iso0: scan_dict, iso1: scan_dict} -> iso is unique
        self.scan_progress = {}  # {activeIso: str, activeTrackNum: int, completedTracks: list, nOfCompletedSteps: int}
        # nOfCompletedSteps is only for the active track!
        self.scan_yields_complete = False
        self.scan_complete_callback.connect(self.set_scan_yields_complete_callback)
        self.scan_start_time = None
        self.abort_scan = False
        self.halt_scan = False
        self.sequencer_status = None
        self.fpga_status = None

        self.triton_status = None

        self.remove_active_iso_after_scan_complete = False

        self.pre_scan_measurement_start_time = datetime.now()
        self.pre_scan_measurement_timeout_s = timedelta(seconds=60)

        self.dmm_gui_callback = None
        self.last_dmm_reading_datetime = datetime.now()  # storage for the last reading time of the dmms
        self.dmm_periodic_reading_interval = timedelta(seconds=5)

        self.displayed_data = {}  # dict of all displayed files. complete filename is key.

        self.set_state(MainState.idle)
        self.autostart()

    """ cyclic function """

    def cyclic(self):
        """
        cyclic function called regularly by the QtTimer initiated in TildaStart.py
        This will control the main
        """
        st = datetime.now()
        self.get_triton_log()
        if self.m_state[0] is MainState.idle:
            self.get_fpga_and_seq_state()
            self.read_dmms(reading_interval=self.dmm_periodic_reading_interval)
            self.work_on_next_job_during_idle()
            return True

        elif self.m_state[0] is MainState.error:
            self.get_fpga_and_seq_state()
            self.read_dmms(reading_interval=self.dmm_periodic_reading_interval)

        elif self.m_state[0] is MainState.starting_simple_counter:
            self._start_simple_counter(*self.m_state[1])
        elif self.m_state[0] is MainState.simple_counter_running:
            self.read_dmms(reading_interval=self.dmm_periodic_reading_interval)
            self._read_data_simple_counter()
        elif self.m_state[0] is MainState.stop_simple_counter:
            self._stop_simple_counter()

        elif self.m_state[0] is MainState.init_power_supplies:
            self._init_power_sups(self.m_state[1])
        elif self.m_state[0] is MainState.setting_power_supply:
            self._set_power_supply_voltage(*self.m_state[1])
        elif self.m_state[0] is MainState.reading_power_supply:
            self._power_supply_status(*self.m_state[1])
        elif self.m_state[0] is MainState.set_output_power_sup:
            self._set_power_sup_outp(*self.m_state[1])

        elif self.m_state[0] is MainState.preparing_scan:
            self._start_scan(self.m_state[1])
        elif self.m_state[0] is MainState.setting_switch_box:
            self._setting_switch_box(*self.m_state[1])
        elif self.m_state[0] is MainState.measure_pre_scan:
            self._measure_pre_and_post_scan(*self.m_state[1])
        elif self.m_state[0] is MainState.load_track:
            self._load_track()
            self.get_fpga_and_seq_state()
        elif self.m_state[0] is MainState.scanning:
            self.read_dmms(feed_to_pipe=True) #TODO: Also read_triton
            self._scanning()
            self.get_fpga_and_seq_state()
        elif self.m_state[0] is MainState.saving:
            self._stop_sequencer_and_save(*self.m_state[1])

        elif self.m_state[0] is MainState.init_dmm:
            self._init_dmm(*self.m_state[1])
        elif self.m_state[0] is MainState.config_dmm:
            self._config_and_arm_dmm(*self.m_state[1])
        elif self.m_state[0] is MainState.request_dmm_config_pars:
            self._request_dmm_config_pars(*self.m_state[1])
        elif self.m_state[0] is MainState.deinit_dmm:
            self._deinit_dmm(self.m_state[1])
        elif self.m_state[0] is MainState.triton_unsubscribe:
            self._triton_unsubscribe_all()
        elapsed = datetime.now() - st
        if elapsed.microseconds > 50000:
            logging.warning('cyclic execution took longer than 50ms, it took: %.1f ms, state is: %s'
                          % (elapsed.microseconds / 1000, self.m_state[0].name))

    """ main functions """

    def close_main(self):
        """
        will deinitialize all active power supplies,
        set 0V on the DAC and turn off all fpga outputs
        will be called after completion of main() in TildaStart
        """
        logging.debug('closing main now')
        self.scan_main.close_scan_main()

    def set_state(self, req_state, val=None, only_if_idle=False, queue_if_not_idle=True):
        """
        this will set the state of the main to req_state
        :return: bool, True if success
        """
        if only_if_idle:
            if self.m_state[0] is MainState.idle:
                self.m_state = req_state, val
                self.send_state()
                logging.debug('main changed state to ' + str(self.m_state[0].name) + ' val is: ' + str(val))
                return True
            else:
                if queue_if_not_idle:
                    self.jobs_to_do_when_idle_queue.append((req_state, val))
                    logging.warning(
                        'added %s to jobs that will be done when returning to idle, current jobs are: %s'
                        % (req_state, self.jobs_to_do_when_idle_queue))
                else:
                    logging.error('main is not in idle state, could not change state to: %s,\n current state is: %s',
                                  req_state, str(self.m_state[0].name))
                return False
        else:
            self.m_state = req_state, val
            self.send_state()
            logging.debug('main changed state to ' + str(self.m_state[0].name) + ' val is: ' + str(val))
            return True

    def work_on_next_job_during_idle(self):
        """
        this will set the main state to next item in the self.jobs_to_do_when_idle_queue
         and remove this job from the list
        """
        try:
            if len(self.jobs_to_do_when_idle_queue):
                new_state = self.jobs_to_do_when_idle_queue.pop(0)
                logging.debug('working on next item in joblist: ' + str(new_state))
                self.set_state(*new_state, only_if_idle=True)
        except Exception as e:
            print('work_on_next_job_during_idle  error : ', e)

    def gui_status_subscribe(self, callback_signal_from_gui):
        """
        a gui can connect to to the stat_dict of the main via a callback_signal.
        this is stored in self.main_ui_status_call_back_signal and if it is not none,
        the status is emitted as soon as self.send_state() is called.
        """
        self.main_ui_status_call_back_signal = callback_signal_from_gui

    def gui_status_unsubscribe(self):
        """
        unsubscribes a gui by setting self.main_ui_status_call_back_signal = None
        """
        self.main_ui_status_call_back_signal = None

    def gui_live_plot_subscribe(self):
        """
        return the callbacks which are connected to the pipeline foa a gui to subscribe to.
        It makes sense to keep them alive as long as the main exists.
        """
        return (self.new_data_callback, self.new_track_callback,
                self.save_request, self.new_gate_or_soft_bin_width,
                self.fit_results_dict_callback, self.live_plot_progress_callback,
                self.pre_post_meas_data_dict_callback)

    def send_state(self):
        """
        if a gui is subscribed via a call back signal in self.main_ui_status_call_back_signal.
        This function will emit a status dictionary containing the following keys:
        status_dict keys: ['workdir', 'status', 'database', 'laserfreq', 'accvolt', 'sequencer_status', 'fpga_status']
        """
        if self.main_ui_status_call_back_signal is not None:
            stat_dict = {
                'workdir': self.working_directory,
                'status': str(self.m_state[0].name),
                'database': self.database,
                'laserfreq': self.laserfreq,
                'accvolt': self.acc_voltage,
                'sequencer_status': self.sequencer_status,
                'fpga_status': self.fpga_status,
                'dmm_status': self.get_dmm_status(),
                'triton_status': str(self.triton_status)
            }
            self.main_ui_status_call_back_signal.emit(stat_dict)

    def get_sequencer_state(self):
        sequencer_state = self.scan_main.read_sequencer_status()
        if sequencer_state != self.sequencer_status:
            self.sequencer_status = sequencer_state
            self.send_state()

    def get_fpga_state(self):
        fpga_state = self.scan_main.read_fpga_status()
        if fpga_state != self.fpga_status:
            self.fpga_status = fpga_state
            self.send_state()

    def get_fpga_and_seq_state(self):
        self.get_fpga_state()
        self.get_sequencer_state()

    def load_spectra_to_main(self, file, gui=None, loaded_spec=None):
        """
        will be used for displaying a spectra.
        Later scan parameters from file can be loaded etc. to sum up more data etc.
        """
        try:
            self.displayed_data[file] = DisplayData(file, gui=gui, x_as_volt=True, loaded_spec=loaded_spec)
        except Exception as e:
            logging.error('Exception while loading file %s, exception is: %s' % (file, str(e)), exc_info=True)

    def close_spectra_in_main(self, file):
        """
        call this to remove the corresponding file from the list of active files.
        """
        # logging.debug('removing spectra %s from view' % file)
        self.displayed_data[file].close_display_data()
        del self.displayed_data[file]
        gc.collect()  # needs to be collected otherwise memory is piled up.
        logging.info('removed spectra %s from main' % file)
        logging.debug('remaining displayed_data is: ' + str(self.displayed_data))

    def autostart(self):
        """
        this will be called during init. call this in order to load a device or so from startup.
        """
        # lala = {'autostartDevices': {'dmms': None, 'powersupplies': None}, 'workingDir': 'E:\TildaDebugging'}
        path = FileHandl.write_to_auto_start_xml_file()
        root_ele, autostart_tpl = FileHandl.load_auto_start_xml_file(path)
        self.autostart_dict = autostart_tpl[1]
        workdir = self.autostart_dict.get('workingDir', False)
        if workdir:
            if os.path.isdir(workdir):
                self.work_dir_changed(workdir)

        dmms_dict = self.autostart_dict.get('autostartDevices', {}).get('dmms', False)
        if dmms_dict:
            try:
                dmms_dict = ast.literal_eval(dmms_dict)
                for dmm_type, dmm_address in dmms_dict.items():
                    try:
                        self.init_dmm(dmm_type, dmm_address)
                    except Exception as e:
                        print('error %s in autostart() of Main.py while starting: %s on address %s' %
                              (e, dmm_type, dmm_address))
            except Exception as e:
                print('error %s in autostart() of Main.py while trying to convert the following string: %s' %
                      (e, dmms_dict))
        pre_scan_timeout = self.autostart_dict.get('preScanTimeoutS', None)
        if pre_scan_timeout is not None:
            self.pre_scan_timeout_changed(float(pre_scan_timeout))
        power_sup_dict = self.autostart_dict.get('autostartDevices', {}).get('powersupplies', False)
        if power_sup_dict:
            print('automatic start of power supplies not included yet.')
        laser_freq = self.autostart_dict.get('laserFreq', False)
        if laser_freq:
            self.laser_freq_changed(float(laser_freq))
        acc_volt = self.autostart_dict.get('accVolt', False)
        if acc_volt:
            self.acc_volt_changed(float(acc_volt))
        # self.init_dmm('Ni4071', 'PXI1Slot5')
        # self.init_dmm('dummy', 'somewhere')

    def pre_scan_timeout_changed(self, timeout_s):
        """ changes the pre scan timeout which is used to cap
         the pre scan measurement if nto enough values come in etc. """
        self.pre_scan_measurement_timeout_s = timedelta(seconds=timeout_s)
        self.autostart_dict['preScanTimeoutS'] = timeout_s
        FileHandl.write_to_auto_start_xml_file(self.autostart_dict)

    """ operations on self.scan_pars dictionary """

    def remove_track_from_scan_pars(self, iso, track):
        """
        remove a track from the given isotope dictionary.
        """
        self.scan_pars.get(iso).pop(track)
        self.scan_pars[iso]['isotopeData']['nOfTracks'] += -1
        self.scan_pars[iso]['isotopeData']['nOfTracks'] = max(0, self.scan_pars[iso]['isotopeData']['nOfTracks'])

    def add_next_track_to_iso_in_scan_pars(self, iso):
        """
        this will look for iso in self.scan_pars and add a new track with lowest possible number.
        If there is a track with this number available in the database, load from there.
        Otherwise copy from another track.
        """
        logging.debug('adding track')
        scan_d = self.scan_pars.get(iso)  # link to the isotope
        iso = scan_d.get('isotopeData').get('isotope')
        seq_type = scan_d.get('isotopeData').get('type')
        next_track_num, track_num_list = SdOp.get_available_tracknum(scan_d)
        track_name = 'track' + str(next_track_num)
        if seq_type == 'kepco':  # only one track for a kepco scan
            if next_track_num > 0:
                print('only one track allowed for kepco scan')
                return None
        scand_from_db = DbOp.extract_track_dict_from_db(self.database, iso, seq_type, next_track_num)
        if scand_from_db is not None:
            logging.debug('adding track' + str(next_track_num) + ' from database')
            logging.debug('scan dict is: ' + str(scand_from_db))
            scan_d[track_name] = scand_from_db[track_name]
        else:
            track_to_copy_from = 'track' + str(max(track_num_list))
            logging.debug('adding track' + str(next_track_num) + ' copying values from: ' + track_to_copy_from)
            scan_d[track_name] = deepcopy(scan_d[track_to_copy_from])
        tracks, track_num_list = TildaTools.get_number_of_tracks_in_scan_dict(scan_d)
        scan_d['isotopeData']['nOfTracks'] = tracks

    def laser_freq_changed(self, laser_freq):
        """
        store the laser frequency in self.laserfreq and send the new status dict to subscribed GUIs.
        :param laser_freq: dbl, in cm-1
        """
        self.laserfreq = laser_freq
        self.autostart_dict['laserFreq'] = laser_freq
        FileHandl.write_to_auto_start_xml_file(self.autostart_dict)
        self.send_state()

    def acc_volt_changed(self, acc_volt):
        """
        store the acceleration voltage in self.acc_voltage and send the new status dict to subscribed GUIs.
        :param acc_volt: dbl, in units of volt
        """
        self.acc_voltage = acc_volt
        self.autostart_dict['accVolt'] = acc_volt
        FileHandl.write_to_auto_start_xml_file(self.autostart_dict)
        self.send_state()

    """ file operations """

    def work_dir_changed(self, workdir_str):
        """
        Sets the working directory in which the main sqlite database is stored.
        """
        if workdir_str == '':  # answer of dialog when cancel is pressed
            return None
        try:
            self.working_directory = os.path.normpath(workdir_str)
            self.database = os.path.normpath(os.path.join(workdir_str, os.path.split(workdir_str)[1] + '.sqlite'))
            DbOp.createTildaDB(self.database)
            logging.debug('working directory has been set to: ' + str(workdir_str))
        except Exception as e:
            logging.error('while loading db from: ' + workdir_str + ' this happened:' + str(e))
            self.database = None
            self.working_directory = None
        finally:
            self.autostart_dict['workingDir'] = self.working_directory
            FileHandl.write_to_auto_start_xml_file(self.autostart_dict)
            self.send_state()
            return self.working_directory

    """ sequencer operations """

    def start_scan(self, iso_name):
        """
        the given isotope scan dictionary will be completed with global informations, which are valid for all isotopes,
        such as:
        workingDirectory, version, laserFreq
        then the bitfile is loaded to the fpga and the first track is started for scanning.
        the state will therefore be changed to scanning
        """
        self.set_state(MainState.preparing_scan, iso_name, only_if_idle=True)

    def _start_scan(self, iso_name):
        """
        the given isotope scan dictionary will be completed with global informations, which are valid for all isotopes,
        such as:
        workingDirectory, version, measureVoltPars, laserFreq
        then the bitfile is loaded to the fpga and the first track is started for scanning.
        the state will therefore be changed to scanning
        """
        self.abort_scan = False
        self.halt_scan = False
        self.scan_start_time = datetime.now()
        self.scan_progress['activeIso'] = iso_name
        self.scan_progress['completedTracks'] = []
        self.scan_pars[iso_name] = self.add_global_infos_to_scan_pars(iso_name)
        logging.debug('will scan: ' + iso_name + str(sorted(self.scan_pars[iso_name])))
        self.send_info('starting_scan')
        if self.scan_main.prepare_scan(self.scan_pars[iso_name]):
            # will be true if sequencer could be started.
            self.set_state(MainState.setting_switch_box, (True, None, False))
        else:
            self.set_state(MainState.idle)

    def _setting_switch_box(self, first_call=False, desired_state=None, scan_complete=False):
        """
        this will be called in 'setting_switch_box' state.
        It will exit as soon as the switchbox is set to the right value.
        The next state is 'measure_pre_scan'
        :param first_call: bool, True for first call, this will command the fpga to change the state of the setbox.
        :param desired_state: int, desired state of the switchbox 0-4, None to get from scan pars
        """
        switch_box_settle_time_s = 2.0
        if desired_state is None and not scan_complete:
            iso_name = self.scan_progress['activeIso']
            n_of_tracks, list_of_track_nums = TildaTools.get_number_of_tracks_in_scan_dict(self.scan_pars[iso_name])
            active_track_num = min(set(list_of_track_nums) - set(self.scan_progress['completedTracks']))
            # logging.debug('list of track nums is: %s, completed tracks: %s, active track num is: %s'
            #               % (list_of_track_nums, self.scan_progress['completedTracks'], active_track_num))
            scan_dict = self.scan_pars[iso_name]
        else:
            # desired state is given by function call and not the scan dict.
            # Desired state will be stored in the main state
            active_track_num = -1
            scan_dict = {}
        if first_call:
            if scan_complete:
                self.send_info('scan_complete')
                desired_state = 4  # overwrite on scan complete, always go to loading after scan complete
                if self.remove_active_iso_after_scan_complete:
                    act_iso = deepcopy(self.scan_progress['activeIso'])
                    logging.info('after scan completion iso %s will now be deleted from the scan pars.' % act_iso)
                    self.scan_progress['activeIso'] = ''
                    self.remove_iso_from_scan_pars(act_iso)
                    self.remove_active_iso_after_scan_complete = False
                self.scan_progress['activeIso'] = ''
            self.scan_main.set_post_acc_switch_box(scan_dict, active_track_num, desired_state)
            if active_track_num >= 0 and not scan_complete:
                # only update active track num when loading from track dict.
                self.scan_progress['activeTrackNum'] = active_track_num
            self.set_state(MainState.setting_switch_box, (False, desired_state, scan_complete))
            return False
        if desired_state is None:
            # must only be called after the desired_state = None has ben stored in the main_state
            desired_state = self.scan_pars[iso_name]['track' + str(active_track_num)]['postAccOffsetVoltControl']
            switch_box_settle_time_s = scan_dict.get(
                'track' + str(self.scan_progress['activeTrackNum']), {}).get('measureVoltPars', {})\
                .get('preScan', {}).get('switchBoxSettleTimeS', 5.0)
        # print('switchbox_settle_time is: %s' % switch_box_settle_time_s)
        # logging.debug('desired state of hsb is: ' + str(des_state))
        done, currentState, desired_state = self.scan_main.post_acc_switch_box_is_set(desired_state,
                                                                                      switch_box_settle_time_s)
        if self.abort_scan:
            logging.info('aborted setting the switch box, aborting scan,'
                         ' setting switchbox to loading state, return to idle')
            self.scan_main.stop_measurement(True, True)
            self.abort_scan = False
            self.set_state(MainState.setting_switch_box, (True, 4, True))
            return False
        if done:
            if scan_complete:
                # scan completed, set switch box to loading state.
                # will go to idle state afterwards.
                self.set_state(MainState.idle)
            else:  # begin with the next track
                # scan_complete False because idle otherwise (above).
                self.set_state(MainState.measure_pre_scan, (True, 'preScan', False))

    def _measure_pre_and_post_scan(self, first_call=False, pre_post_scan_str='preScan', scan_complete=False):
        """
        this function is called within the state 'measure_pre_scan'.
            on first call:
             it will set the fpga to measure offset state and fire a software trigger to the dmms
            on other calls:
                will try to get a voltage reading from the dmms
                when all dmms are read it will go to the next state
            -> dmms are triggered by software and voltmeter-complete TTL-from dmm is ignored.
        next state will be 'load_track'
        :param first_call: bool, True if this is the first call
        :param pre_post_scan_str: str, preScan or postScan corresponding if this is a pre or post scan measurement
        """
        iso_name = self.scan_progress['activeIso']
        active_track_num = self.scan_progress['activeTrackNum']
        act_track_name = 'track' + str(active_track_num)

        if first_call:
            # on first call set the fpga to measure offset state and setup the dmms and the triton listener
            # then software trigger the dmms
            self.scan_main.prepare_dmms_for_scan(
                self.scan_pars[iso_name][act_track_name]['measureVoltPars'].get(pre_post_scan_str, {}).get('dmms', {}))
            self.scan_main.prepare_triton_listener_for_scan(
                self.scan_pars[iso_name][act_track_name].get('triton', {}), pre_post_scan_str, act_track_name)
            if self.scan_main.start_pre_scan_measurement(self.scan_pars[iso_name], act_track_name, pre_post_scan_str):
                self.pre_scan_measurement_start_time = datetime.now()
                self.set_state(MainState.measure_pre_scan, (False, pre_post_scan_str, scan_complete))   # set first call to false!
            else:
                # scan main returns False -> no pre scan measurement required!
                if pre_post_scan_str == 'postScan':
                    self.set_state(MainState.setting_switch_box, (True, None, scan_complete))  # scan_complete=True
                    # after this has ben completed, it will go to idle
                else:
                    # otherwise load next track
                    self.set_state(MainState.load_track)
            # emit the scan_pars to the pre_post_live_data ui
                    self.pre_post_meas_data_dict_callback.emit(self.scan_pars[iso_name])
        else:  # this will periodically read the dmms and triton until all dmms returned a measurement
            if self.abort_scan:
                logging.info('ABORT was pressed. Aborting pre scan measurement, aborting scan,'
                             ' setting switchbox to loading state, return to idle')
                self.scan_main.stop_measurement(True, True)
                self.abort_scan = False
                self.set_state(MainState.setting_switch_box, (True, 4, True))
            else:  # read dmms & triton devices until all values are there.
                # check timeout
                time_since_start = datetime.now() - self.pre_scan_measurement_start_time
                if self.pre_scan_measurement_timeout_s < time_since_start:  # timed out
                    logging.warning('--------- WARNING ----------\n'
                                    'pre scan measurement timed out after %s s. but timeout'
                                    ' is set to %s s. Anyhow continuing with scan now.\n'
                                    '--------- WARNING ----------'
                                    % (time_since_start.seconds, self.pre_scan_measurement_timeout_s.seconds))
                    self.send_info('pre_scan_timeout')
                    if pre_post_scan_str == 'postScan':
                        # if this was a post scan measurement, scan is complete -> set switch box
                        self.set_state(MainState.setting_switch_box, (True, None, scan_complete))  # scan_complete=True
                        # after this has ben completed, it will go to idle
                    else:
                        # otherwise load next track
                        self.set_state(MainState.load_track)
                else:
                    # not timed out, check if all values are measured yet:
                    # note: if dmms / triton log complete they will be set to duringScan pars (or periodic (for dmms))
                    if self.scan_main.prescan_measurement(
                            scan_dict=self.scan_pars[iso_name], dmm_reading=self.read_dmms(False),
                            pre_during_post_scan_str=pre_post_scan_str,
                            tr_name='track' + str(self.scan_progress['activeTrackNum'])):
                        if pre_post_scan_str == 'postScan':
                            # if this was a post scan measurement, scan is complete -> set switch box
                            self.set_state(MainState.setting_switch_box, (True, None, scan_complete))  # scan_complete=True
                            # after this has ben completed, it will go to idle
                        else:
                            self.set_state(MainState.load_track)

    def add_global_infos_to_scan_pars(self, iso_name):
        """
        this will update the self.scan_pars[isoname] dict with the global informations.
        :param iso_name: str, name of the isotope
        :return: dict, completed dictionary
        """
        self.scan_pars[iso_name] = SdOp.fill_meas_complete_dest(self.scan_pars[iso_name])
        self.scan_pars[iso_name] = SdOp.add_missing_voltages(self.scan_pars[iso_name])
        self.scan_pars[iso_name]['pipeInternals']['workingDirectory'] = self.working_directory
        self.scan_pars[iso_name]['isotopeData']['version'] = Cfg.version
        self.scan_pars[iso_name]['isotopeData']['laserFreq'] = self.laserfreq
        self.scan_pars[iso_name]['isotopeData']['accVolt'] = self.acc_voltage
        for key, track_d in self.scan_pars[iso_name].items():
            #  fill empty triton dicts to each track if there is not one already
            if 'track' in key:
                if self.scan_pars[iso_name][key].get('triton', None) is None:
                    self.scan_pars[iso_name][key]['triton'] = {}
        if self.scan_pars[iso_name]['isotopeData']['type'] == 'kepco':
            track_num, list_of_tracknums = TildaTools.get_number_of_tracks_in_scan_dict(self.scan_pars[iso_name])
            if track_num > 1:
                [self.scan_pars[iso_name].pop('track%s' % track_num) for track_num in list_of_tracknums[1:]]
            self.scan_pars[iso_name]['track0']['nOfScans'] = 1  # force to one scan!
        return self.scan_pars[iso_name]

    def halt_scan_func(self, halt_bool):
        """
        this will set the halt boolean on the fpga True/False.
        This will cause the FPGA to stop after completion of the next Scan if True.
        The FPGA therefore will end up in 'error' state.
        """
        self.halt_scan = halt_bool
        self.scan_main.halt_scan(self.halt_scan)

    def update_scan_progress(self, number_of_completed_steps=None):
        """
        will be updated from the pipeline via Qt callback signal.
        number_of_completed_steps is just for the current track.
        """
        if number_of_completed_steps is not None:
            self.scan_progress['nOfCompletedSteps'] = number_of_completed_steps
        progress_dict = self.scan_main.calc_scan_progress(self.scan_progress,
                                                          self.scan_pars[self.scan_progress['activeIso']],
                                                          self.scan_start_time)
        if progress_dict is not None:
            if self.live_plot_progress_callback is not None:
                self.live_plot_progress_callback.emit(progress_dict)

    def subscribe_to_scan_prog(self):
        """
        the scanProgressUi can poll the correct pyqtsignal for getting a progress update here.
        simply connect to it.
        """
        return self.live_plot_progress_callback

    def unsubscribe_from_scan_prog(self):
        """
        sets self.scan_prog_call_back_sig_gui = None
        """
        self.scan_prog_call_back_sig_gui = None

    def _load_track(self):
        """
        called via state 'load_track'
        this will prepare the pipeline for the next track
        and then start the fpga with this track.
        will switch state to 'scanning' or 'error'
        """
        iso_name = self.scan_progress['activeIso']
        active_track_num = self.scan_progress['activeTrackNum']

        is_this_first_track = len(self.scan_progress['completedTracks']) == 0
        if is_this_first_track:  # initialise the pipeline on first track
            self.scan_main.init_analysis_thread(
                self.scan_pars[iso_name], self.scan_prog_call_back_sig_pipeline,
                live_plot_callback_tuples=(self.new_data_callback, self.new_track_callback,
                                           self.save_request, self.new_gate_or_soft_bin_width,
                                           self.pre_post_meas_data_dict_callback),
                fit_res_dict_callback=self.fit_results_dict_callback,
                scan_complete_callback=self.scan_complete_callback,
            )
        self.scan_main.prep_track_in_pipe(active_track_num, active_track_num)
        if self.scan_main.start_measurement(self.scan_pars[iso_name], active_track_num):
            self.set_state(MainState.scanning)
        else:
            logging.error('could not start scan on fpga')
            self.set_state(MainState.error)

    def _scanning(self):
        """
        will be called when in state 'scanning'
        will always feed data to the pipeline in scan_main.
        will change to 'load_track', when no data is available anymore
        AND the state is not measuring state anymore.
        """
        if self.abort_scan:  # abort the scan and return to idle state
            logging.info('\t ABORT was pressed during scan.'
                         'Now: Abort scan -> saving -> post scan measurement -> setting switchbox -> idle')
            self.scan_main.abort_scan()
            self.abort_scan = False
            complete_stop = True
            self.set_state(MainState.saving, (complete_stop, True))
            return None
        elif not self.scan_main.read_data():  # read_data() yields False if no Elements can be read from fpga
            if not self.scan_main.check_scanning():  # check if fpga is still in scanning state
                if self.scan_pars[self.scan_progress['activeIso']]['isotopeData']['type'] == 'kepco':
                    # for now this feedback of the pipeline when the scan is complete
                    # is only implemented for a kepco scan, but for the future also other pipelines might make sense.
                    if self.scan_yields_complete or self.halt_scan:  # scan done -> normal exit etc
                        pass
                    else:  # scan not done -> keep scanning
                        return None
                if self.halt_scan:  # scan was halted
                    self.halt_scan_func(False)  # set halt variable to false afterwards, also on fpga
                    complete_stop = True
                    self.set_state(MainState.saving, (complete_stop, True))
                else:  # normal exit after completion of each track
                    self.scan_progress['completedTracks'].append(self.scan_progress['activeTrackNum'])
                    tracks, tr_l = TildaTools.get_number_of_tracks_in_scan_dict(
                        self.scan_pars[self.scan_progress['activeIso']])
                    complete_stop = len(self.scan_progress['completedTracks']) == tracks  # scan complete
                    logging.debug(
                        'number of completed tracks: %d, %s, total number of tracks: %d, stopping: %s'
                        % (len(self.scan_progress['completedTracks']),
                           self.scan_progress['completedTracks'], tracks, complete_stop))
                    self.set_state(MainState.saving, (complete_stop, True))
            else:  # fpga is still scanning, so keep reading data (see above)
                return None

    def _stop_sequencer_and_save(self, complete_stop=False, first_call=True):
        """
        will be called in state 'saving'
        """
        if first_call:
            if complete_stop:
                logging.info('saving...')
            QApplication.setOverrideCursor(QCursor(QtCore.Qt.WaitCursor))  # ignore warning
            self.scan_main.stop_measurement(complete_stop=complete_stop, clear=complete_stop)  # stop pipeline and clear
            self.set_state(MainState.saving, (complete_stop, False))  # go back to saving until analysis is complete
        else:
            if self.scan_main.analysis_done_check():  # when done with analysis, leave state
                QApplication.restoreOverrideCursor()  # ignore warning
                self.set_state(MainState.measure_pre_scan, (True, 'postScan', complete_stop))

    def set_scan_yields_complete_callback(self, complete_bool):
        self.scan_yields_complete = complete_bool

    def pause_scan(self, pause_scan_bool=None):
        """
        This will pause the scan with a loop in the handshake.
        Use this, if the laser jumped or so and you want to continue on the data.
        :param pause_bool: bool, None if you want to toggle
        """
        return self.scan_main.pause_scan(pause_scan_bool)

    """ simple counter """

    def start_simple_counter(self, act_pmt_list, datapoints, callback_sig, sample_interval):
        self.set_state(MainState.starting_simple_counter,
                       (act_pmt_list, datapoints, callback_sig, sample_interval), only_if_idle=True)

    def _start_simple_counter(self, act_pmt_list, datapoints, callback_sig, sample_interval):
        if self.scan_main.sequencer is not None:
            self.scan_main.deinit_fpga()
        self.simple_counter_inst = SimpleCounterControl(act_pmt_list, datapoints, callback_sig, sample_interval)
        ret = self.simple_counter_inst.run()
        if ret:
            pass
        else:
            print('while starting the simple counter bitfile, something did not work.')
            print('don\'t worry, starting DUMMY Simple Counter now.')
            self.simple_counter_inst.run_dummy()
        self.set_state(MainState.simple_counter_running)

    def _read_data_simple_counter(self):
        self.simple_counter_inst.read_data()

    def simple_counter_post_acc(self, state_name):
        """
        sets the post acceleration control to the given state.
        beware of the switching time in seconds.
        """
        self.simple_counter_inst.set_post_acc_control(state_name)

    def get_simple_counter_post_acc(self):
        """
        :return: post_acc_state, post_acc_name
        """
        return self.simple_counter_inst.get_post_acc_control()

    def simple_counter_set_dac_volt(self, volt_dbl):
        """
        set the dac voltage
        """
        self.simple_counter_inst.set_dac_volt(volt_dbl)

    def stop_simple_counter(self):
        self.set_state(MainState.stop_simple_counter)

    def _stop_simple_counter(self):
        fpga_status = self.simple_counter_inst.stop()
        self.simple_counter_inst = None
        logging.debug('fpga status after deinit is: ' + str(fpga_status))
        self.set_state(MainState.idle)

    """ postaccleration power supply functions """

    def subscribe_to_power_sub_status(self, callback_signal, key):
        """
        can be used for a gui to subscribe to the status of the power supply
        :param callback_signal: QtCore.pyqtSignal(dict)
        :param key: str, key which will be used to subscribe and unsubscribe!
        """
        self.power_sup_stat_callback_signals[key] = callback_signal
        return key

    def un_subscribe_to_power_sub_status(self, key):
        """
        unsubscribe from status of the power supplies
        :param key: str, key which will be used to subscribe and unsubscribe!
        """
        if key in self.power_sup_stat_callback_signals.keys():
            self.power_sup_stat_callback_signals.pop(key)

    def init_power_sups(self, call_back_signal=None):
        """
        initializes all power supplies and reads the status afterwards.
        only changes state, when in idle
        """
        self.set_state(MainState.init_power_supplies, call_back_signal, only_if_idle=True)

    def _init_power_sups(self, call_back_signal=None):
        """
        initializes all power supplies and reads the status afterwards.
        """
        self.scan_main.init_post_accel_pwr_supplies()
        self.set_state(MainState.reading_power_supply, ('all', call_back_signal))

    def set_power_supply_voltage(self, power_supply, volt, call_back_signal=None):
        """
        this will request a change in the state in order to set the requested voltage.
        power_supply -> self.requested_power_supply
        volt -> self.requested_voltage
        """
        self.set_state(MainState.setting_power_supply, (power_supply, volt, call_back_signal), True,
                       queue_if_not_idle=False)

    def _set_power_supply_voltage(self, name, volt, call_back_signal=None):
        """
        this will actually call to the power supply
        will set the Output voltage of the desired power supply,
        as stated in self.requested_power_supply to the requested voltage
        """
        self.scan_main.set_post_accel_pwr_supply(name, volt)
        self.set_state(MainState.reading_power_supply, (name, call_back_signal))

    def set_power_sup_outp(self, name, outp, call_back_signal=None):
        """
        change state
        """
        self.set_state(MainState.set_output_power_sup, (name, outp, call_back_signal), True)

    def _set_power_sup_outp(self, name, outp, call_back_signal=None):
        """
        set the output
        """
        self.scan_main.set_post_accel_pwr_spply_output(name, outp)
        self.set_state(MainState.reading_power_supply, (name, call_back_signal))

    def power_supply_status(self, power_supply, call_back_sig):
        """
        returns a dict containing the status of the power supply,
        keys are: name, programmedVoltage, voltageSetTime, readBackVolt
        """
        self.set_state(MainState.reading_power_supply, (power_supply, call_back_sig), True)

    def _power_supply_status(self, name, call_back_sig=None):
        """
        connects to the requested power supply and writes the status of the given power supply into
        self.requested_power_supply_status
        """
        stat = self.scan_main.get_status_of_pwr_supply(name)
        if call_back_sig is not None and isinstance(call_back_sig, str):
            call_back_sig = self.power_sup_stat_callback_signals.get(call_back_sig, None)
            if call_back_sig is not None:
                call_back_sig.emit(stat)
        self.set_state(MainState.idle)

    """ database functions """

    def get_available_isos_from_db(self, seq_type):
        """
        connects to the database defined by self.database
        :return: list, name of all available isotopes
        """
        isos = DbOp.check_for_existing_isos(self.database, seq_type)
        return isos

    def get_isotope_settings_from_db(self, iso):
        """ go into the Isotopes table in the database and return all values for this isotope """
        iso = DbOp.get_iso_settings(self.database, iso)
        return iso

    def update_iso_in_db(self, iso, settings_list):
        """ edit or create the iso with the settings in settings_list to Isotopes """
        DbOp.update_iso_settings(self.database, iso, settings_list)

    def add_new_iso_to_db(self, iso, seq_type, exisiting_iso=None):
        """
        add a new isotope of type seq, to the database, will try to copy from existing iso
        """
        return DbOp.add_new_iso(self.database, iso, seq_type, exisiting_iso)

    def add_iso_to_scan_pars(self, iso, seq_type):
        """
        connect to the database and add all tracks with given isotope and sequencer type
        to self.scan_pars.
        :return: str, key of new isotope.
        """
        scand = DbOp.extract_all_tracks_from_db(self.database, iso, seq_type)
        key = iso + '_' + seq_type
        self.scan_pars[key] = scand
        logging.debug('scan_pars are: ' + str(self.scan_pars))
        return key

    def add_iso_to_scan_pars_no_database(self, scan_dict):
        """
        will add the scan_dict to self.scan_pars,
        WITHOUT accessing the database first.
        This is useful when loading settings from file instead of db
        :param scan_dict: dict, scan_dict, see ..\Service\Scan\draftScanParameters.py
        :return: str, name of isotope
        """
        iso = scan_dict['isotopeData']['isotope']
        seq_type = scan_dict['isotopeData']['type']
        key = iso + '_' + seq_type
        self.scan_pars[key] = scan_dict
        logging.debug('scan_pars are: ' + str(self.scan_pars))
        return key

    def remove_iso_from_scan_pars(self, iso_seqtype):
        """
        this will remove the dictionary named 'iso_seqtype' from self.scan_pars
        """
        if iso_seqtype != self.scan_progress.get('activeIso', 'None'):
            self.scan_pars.pop(iso_seqtype)
            logging.debug('removed iso %s from scan pars. scan_pars are: %s' % (iso_seqtype, str(self.scan_pars)))
        else:
            self.remove_active_iso_after_scan_complete = True
            logging.info('could not remove %s because this is the current scan parameter.'
                         ' Will remove this after scan completed.' % iso_seqtype)

    def remove_old_dmm_triton_from_scan_pars(self, iso):
        """
        overwrites all dmm and triton 'readings' and 'data' entries for a fresh scan.
        """
        is_there_something_to_remove = 0
        for keys in self.scan_pars[iso]:
            if 'track' in keys:
                for predurpos in self.scan_pars[iso][keys]['measureVoltPars']:
                    for dmm_names, dmm_dicts in \
                            self.scan_pars[iso][keys]['measureVoltPars'][predurpos].get('dmms', {}).items():
                        for entries in dmm_dicts:
                            if entries == 'readings' and len(dmm_dicts[entries]) > 0:
                                is_there_something_to_remove += 1
                                dmm_dicts[entries] = []
                            elif entries == 'aquiredPreScan':
                                dmm_dicts[entries] = 0
                for predurpos in self.scan_pars[iso][keys]['triton']:
                    for dev_names, dev_dicts in self.scan_pars[iso][keys]['triton'].get(predurpos, {}).items():
                        for channels, ch_dicts in dev_dicts.items():
                            for entries in ch_dicts:
                                if entries == 'data' and len(ch_dicts[entries]) > 0:
                                    is_there_something_to_remove += 1
                                    ch_dicts[entries] = []
                                elif entries == 'acquired':
                                    ch_dicts[entries] = 0
        logging.info('removed %s old dmm or triton data entries' % is_there_something_to_remove)


    def save_scan_par_to_db(self, iso):
        """
        will save all information in the scan_pars dict for the given isotope to the database.
        """
        self.add_global_infos_to_scan_pars(iso)
        scan_d = deepcopy(self.scan_pars[iso])
        # add_scan_dict_to_db will perform some changes on scan_d, therefore copy necessary
        trk_num, trk_lis = TildaTools.get_number_of_tracks_in_scan_dict(scan_d)
        for i in trk_lis:
            logging.debug('saving track ' + str(i) + ' dict is: ' +
                          str(scan_d['track' + str(i)]))
            logging.debug('measureVoltPars are: %s'
                          % scan_d['track' + str(i)]['measureVoltPars'])
            DbOp.add_scan_dict_to_db(self.database, scan_d, i, track_key='track' + str(i))

    ''' digital multimeter operations '''

    def init_dmm(self, type_str, addr_str, callback=False, start_config='periodic'):
        """
        initialize the dmm of given type and address.
        pass a callback to know when init is done.
        dmm should be in idle state after this.
        :param type_str: str, type of dmm
        :param addr_str: str, address of the given dmm
        :param callback: callback_bool, True will be emitted after init is done.
        :param start_config: str, name of a pre dinfed dmm setup, dft is periodic.
        """
        self.set_state(MainState.init_dmm, (type_str, addr_str, callback, start_config), only_if_idle=True)

    def _init_dmm(self, type_str, addr_str, callback, start_config):
        """ see init_dmm() """
        dmm_name = self.scan_main.prepare_dmm(type_str, addr_str)
        self.send_state()
        if start_config is not None and dmm_name is not None:
            self.scan_main.set_dmm_to_pre_config(dmm_name, start_config)
        if callback:
            callback.emit(True)
        self.set_state(MainState.idle)

    def deinit_dmm(self, dmm_name):
        self.set_state(MainState.deinit_dmm, dmm_name, only_if_idle=True)

    def _deinit_dmm(self, dmm_name):
        self.scan_main.de_init_dmm(dmm_name)
        self.set_state(MainState.idle)
        self.send_state()

    def config_and_arm_dmm(self, dmm_name, config_dict, reset_dmm):
        """
        configure the dmm via the config_dict and start a measurement, so the dmm starts to store values.
        :param dmm_name: str, name of dmm
        :param config_dict: dict, dictionary containing all parameters to configure the given dmm
        :param reset_dmm: bool, True if you want to reset the device before configuring
        """
        if self.get_dmm_status().get(dmm_name, False):  # only configure and arm device if active
            self.set_state(MainState.config_dmm, (dmm_name, config_dict, reset_dmm), only_if_idle=True)

    def _config_and_arm_dmm(self, dmm_name, config_dict, reset_dmm):
        """  see: config_and_arm_dmm() """
        self.scan_main.setup_dmm_and_arm(dmm_name, config_dict, reset_dmm)
        self.set_state(MainState.idle)
        self.send_state()

    def get_active_dmms(self):
        """
        function to return a dict of all active dmms
        see also self.get_dmm_status()
        :return: dict of tuples, {dmm_name: (type_str, address_str, state_str, last_readback, configPars_dict)}
        """
        return self.scan_main.get_active_dmms()

    def get_dmm_status(self):
        """
        this will get the status of all active dmms and return a dictionary.
        this encapsules the get_active_dmms funtion and removes some entries
        from their returend dict, so it can be used within self.send_state()
        :return: dict of dict, {dmm_name1: {'status': stat_str, 'lastReadback': (voltage_float, time_str)}}}
        """
        ret = {}
        for dmm_name, vals in self.get_active_dmms().items():
            type_str, address_str, state_str, last_readback, configPars_dict = vals
            ret[dmm_name] = {'status': state_str, 'lastReadback': last_readback}
        return ret

    def read_dmms(self, feed_to_pipe=False, reading_interval=timedelta(seconds=0)):
        """
        dmm's will be read when main is in idle state.
        Values are already measured by the dmm in advance and main only "fetches" those.
            -> should be a quick return of values.
            values are emitted via send_state()
            also values are emitted via the self.dmm_gui_callback, if there is a gui subscribed to it.
        :return: None or dict, dict will always contain at least one reading.
        """
        # maybe put timeout here
        if datetime.now() - self.last_dmm_reading_datetime > reading_interval:
            # return None
            self.last_dmm_reading_datetime = datetime.now()
            worth_sending = False
            readback = self.scan_main.read_multimeter('all', feed_to_pipe)
            if readback is not None:  # will be None if no dmms are active
                for dmm_name, vals in readback.items():
                    if vals is not None:  # will be None if no new readback is available
                        self.send_state()
                        worth_sending = True
                if self.dmm_gui_callback is not None and worth_sending:
                    # also send readback ot other guis that might be subscribed.
                    self.dmm_gui_callback.emit(readback)
            if worth_sending:
                return readback
            else:
                return None
        else:
            return None

    def request_dmm_config_pars(self, dmm_name):
        """
        request the config parameters from the dmm and return them to caller.
        :param dmm_name: str, name of dev
        :return: dict, raw_configuration parameters, with currently set values,
        """
        conf_dict = self.scan_main.request_config_pars(dmm_name)
        return conf_dict

    def request_dmm_available_preconfigs(self, dmm_name):
        """

        :param dmm_name: str, name of dev
        :return: list of strings with the names of the active configs.
        """
        configs = self.scan_main.request_dmm_available_preconfigs(dmm_name)
        return configs

    def dmm_gui_subscribe(self, callback):
        """
        here a gui can connect to the readbackvalues of the dmms.
        values will be emitted within read_dmms() when the main is in idle state.
        :param callback: callback_dict, here the last readings will be send.
        """
        self.dmm_gui_callback = callback

    def dmm_gui_unsubscribe(self):
        """
        unsubscribing from self.dmm_gui_callback
        """
        self.dmm_gui_callback = None

    def dmm_get_accuracy(self, dmm_name, config):
        """ get the accuracy tuple from the dmm with the given config """
        return self.scan_main.dmm_get_accuracy(dmm_name, config)

    ''' pulse pattern operations '''

    def ppg_init(self):
        """ this will initialize the fpga bitfile. """
        self.scan_main.ppg_init()

    def ppg_load_pattern(self, cmd_list):
        """ reset the ppg and load the list of cmds to it, then run it. """
        self.scan_main.ppg_run_with_list_of_commands(cmd_list)

    def ppg_stop(self, reset=False, deinit_ppg=False):
        """
        stops the pulse pattern generator from executing, delete pattern from memory if wanted by reset=True
        Always include a $stop command in order to have a defined stopping state for the outputs.
        This will NOT remove the bitfile from the fpga. Only when deinit is set True!
        """
        self.scan_main.ppg_stop(reset)
        if deinit_ppg:
            self.scan_main.ppg_deinit(False)

    def ppg_state_callback(self, callback_signal):
        """
        use this in order to connect a signal to the state changed function and
         emit the name of the state each time this is changed.
        :param callback_signal: pyqtboundsignal(str)
        """
        self.scan_main.ppg_state_callback_connect(callback_signal)

    def ppg_state_disconnect(self):
        print('disconnecting ...')
        self.scan_main.ppg_state_callback_disconnect()
        print('disconnected')

    ''' triton related'''

    def get_triton_log(self):
        triton_status = self.scan_main.get_triton_receivers()
        if triton_status != self.triton_status:
            self.triton_status = triton_status
            self.send_state()

    def triton_unsubscribe_all(self):
        self.set_state(MainState.triton_unsubscribe, only_if_idle=True)

    def _triton_unsubscribe_all(self):
        self.scan_main.stop_triton_listener(stop_dummy_dev=False, restart=True)
        self.set_state(MainState.idle)


    ''' send information / warnings to gui or so '''

    def send_info(self, info_str):
        """ send an info string via this signal """
        logging.info('sending the info: %s' % info_str)
        if isinstance(info_str, str):
            self.info_warning_string_main_signal.emit(info_str)
