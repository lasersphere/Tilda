"""

Created on '30.09.2015'

@author:'simkaufm'

"""

import logging
import os
from copy import deepcopy
from datetime import datetime

from PyQt5 import QtCore
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QApplication

import Application.Config as Cfg
import Service.DatabaseOperations.DatabaseOperations as DbOp
import Service.Scan.ScanDictionaryOperations as SdOp
import Service.Scan.draftScanParameters as Dft
from Application.Main.MainState import MainState
from Service.AnalysisAndDataHandling.DisplayData import DisplayData
from Service.Scan.ScanMain import ScanMain
from Service.SimpleCounter.SimpleCounter import SimpleCounterControl
from Service.TildaPassive.TildaPassiveControl import TildaPassiveControl


class Main(QtCore.QObject):
    # this will equal the number of completed steps in the active track:
    scan_prog_call_back_sig_pipeline = QtCore.pyqtSignal(int)
    # close_spec_display = QtCore.pyqtSignal(str)

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

        # pyqtSignal for sending the status to the gui, if there is one connected:
        self.main_ui_status_call_back_signal = None
        # pyqtSignal for sending the scan progress to the gui while scanning.
        self.scan_prog_call_back_sig_gui = None
        self.scan_prog_call_back_sig_pipeline.connect(self.update_scan_progress)

        self.scan_main = ScanMain()
        self.iso_scan_process = None
        self.scan_pars = {}  # {iso0: scan_dict, iso1: scan_dict} -> iso is unique
        self.scan_progress = {}  # {activeIso: str, activeTrackNum: int, completedTracks: list, nOfCompletedSteps: int}
        # nOfCompletedSteps is only for the active track!
        self.scan_start_time = None
        self.abort_scan = False
        self.halt_scan = False
        self.sequencer_status = None
        self.fpga_status = None

        self.tilda_passive_inst = None
        self.tilda_passive_status = -1
        self.tipa_status_callback_sig = None
        self.tipa_timeout_counter = 0

        self.dmm_gui_callback = None

        self.displayed_data = {}  # dict of all displayed files. complete filename is key.

        try:
            # pass
            self.work_dir_changed('D:/lala')
            # self.work_dir_changed('C:/temp108')
            # self.work_dir_changed('D:\Tilda_Debugging')
        except Exception as e:
            logging.error('while loading default location of db this happened:' + str(e))
        self.set_state(MainState.idle)
        self.autostart()

    """ cyclic function """

    def cyclic(self):
        """
        cyclic function called regularly by the QtTimer initiated in TildaStart.py
        This will control the main
        """
        if self.m_state[0] is MainState.idle:
            self.get_fpga_and_seq_state()
            self.read_dmms()
            self.work_on_next_job_during_idle()
            return True

        elif self.m_state[0] is MainState.starting_simple_counter:
            self._start_simple_counter(*self.m_state[1])
        elif self.m_state[0] is MainState.simple_counter_running:
            self.read_dmms()
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

        elif self.m_state[0] is MainState.load_track:
            self._load_track()
            self.get_fpga_and_seq_state()
        elif self.m_state[0] is MainState.scanning:
            self._scanning()
            self.get_fpga_and_seq_state()
            self.read_dmms(feed_to_pipe=True)
        elif self.m_state[0] is MainState.saving:
            self._stop_sequencer_and_save()

        elif self.m_state[0] is MainState.preparing_tilda_passiv:
            self._prepare_tilda_passive(*self.m_state[1])
        elif self.m_state[0] is MainState.tilda_passiv_running:
            self._tilda_passive_running()
        elif self.m_state[0] is MainState.closing_tilda_passiv:
            self._close_tilda_passive(self.m_state[1])

        elif self.m_state[0] is MainState.init_dmm:
            self._init_dmm(*self.m_state[1])
        elif self.m_state[0] is MainState.config_dmm:
            self._config_and_arm_dmm(*self.m_state[1])
        elif self.m_state[0] is MainState.request_dmm_config_pars:
            self._request_dmm_config_pars(*self.m_state[1])
        elif self.m_state[0] is MainState.deinit_dmm:
            self._deinit_dmm(self.m_state[1])

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
                logging.debug('changed state to %s', str(self.m_state[0].name))
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
            logging.debug('changed state to %s', str(self.m_state[0].name))
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
                'dmm_status': self.get_dmm_status()
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

    def load_spectra_to_main(self, file):
        """
        will be used for displaying a spectra.
        Later scan parameters from file can be loaded etc. to sum up more data etc.
        """
        try:
            if file in self.displayed_data.keys():
                print('already loaded')
                return None
            self.displayed_data[file] = DisplayData(file)
            self.displayed_data[file].con_close_event()
        except Exception as e:
            logging.error('Exception while loading file %s, exception is: %s' % (file, str(e)))

    def close_spectra_in_main(self, file):
        """
        call this to remove the corresponding file from the list of active files.
        """
        self.displayed_data.pop(file)

    def autostart(self):
        """
        this will be called during init. call this in order to load a device or so from startup.
        maybe this can be done via a text/xml file later on.
        """
        self.init_dmm('Ni4071', 'PXI1Slot5')
        # self.init_dmm('dummy', 'somewhere')

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
        scand_from_db = DbOp.extract_track_dict_from_db(self.database, iso, seq_type, next_track_num)
        if scand_from_db is not None:
            logging.debug('adding track' + str(next_track_num) + ' from database')
            logging.debug('scan dict is: ' + str(scand_from_db))
            scan_d[track_name] = scand_from_db[track_name]
        else:
            track_to_copy_from = 'track' + str(max(track_num_list))
            logging.debug('adding track' + str(next_track_num) + ' copying values from: ' + track_to_copy_from)
            scan_d[track_name] = deepcopy(scan_d[track_to_copy_from])
        tracks, track_num_list = SdOp.get_number_of_tracks_in_scan_dict(scan_d)
        scan_d['isotopeData']['nOfTracks'] = tracks

    def laser_freq_changed(self, laser_freq):
        """
        store the laser frequency in self.laserfreq and send the new status dict to subscribed GUIs.
        :param laser_freq: dbl, in cm-1
        """
        self.laserfreq = laser_freq
        self.send_state()

    def acc_volt_changed(self, acc_volt):
        """
        store the acceleration voltage in self.acc_voltage and send the new status dict to subscribed GUIs.
        :param acc_volt: dbl, in units of volt
        """
        self.acc_voltage = acc_volt
        self.send_state()

    """ file operations """

    def work_dir_changed(self, workdir_str):
        """
        Sets the working directory in which the main sqlite database is stored.
        """
        if workdir_str == '':  # answer of dialog when cancel is pressed
            return None
        try:
            self.working_directory = workdir_str
            self.database = workdir_str + '/' + os.path.split(workdir_str)[1] + '.sqlite'
            DbOp.createTildaDB(self.database)
            logging.debug('working directory has been set to: ' + str(workdir_str))
        except Exception as e:
            logging.error('while loading db from: ' + workdir_str + ' this happened:' + str(e))
            self.database = None
            self.working_directory = None
        finally:
            self.send_state()
            return self.working_directory

    """ sequencer operations """

    def start_scan(self, iso_name):
        """
        the given isotope scan dictionary will be completed with global informations, which are valid for all isotopes,
        such as:
        workingDirectory, version, measureVoltPars, laserFreq
        then the bitfile is loaded to the fpga and the first track is started for scanning.
        the state will therefor be changed to scanning
        :return: bool, True if scan started
        """
        try:
            if self.m_state[0] is MainState.idle:
                self.set_state(MainState.preparing_scan)
                self.abort_scan = False
                self.halt_scan = False
                self.scan_start_time = datetime.now()
                self.scan_progress['activeIso'] = iso_name
                self.scan_progress['completedTracks'] = []
                self.add_global_infos_to_scan_pars(iso_name)
                logging.debug('will scan: ' + iso_name + str(sorted(self.scan_pars[iso_name])))
                self.scan_main.prepare_scan(self.scan_pars[iso_name], self.scan_prog_call_back_sig_pipeline)
                self.set_state(MainState.load_track)
                return True
            else:
                logging.warning('could not start scan because state of main is ' + str(self.m_state[0].name))
                return False
        except Exception as e:
            print('error while starting scan: ', e)
            return False

    def add_global_infos_to_scan_pars(self, iso_name):
        """
        this will update the self.scan_pars[isoname] dict with the global informations.
        :param iso_name: str, name of the isotope
        :return: dict, completed dictionary
        """
        # self.scan_pars[iso_name]['measureVoltPars'] = SdOp.merge_dicts(
        #     self.scan_pars[iso_name]['measureVoltPars'], self.measure_voltage_pars)
        # # in self.measure_voltage pars, the pulselength and timeout is noted.
        # This is assumed globally for all isotopes for now.
        self.scan_pars[iso_name]['pipeInternals']['workingDirectory'] = self.working_directory
        self.scan_pars[iso_name]['isotopeData']['version'] = Cfg.version
        self.scan_pars[iso_name]['isotopeData']['laserFreq'] = self.laserfreq
        self.scan_pars[iso_name]['isotopeData']['accVolt'] = self.acc_voltage
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
            if self.scan_prog_call_back_sig_gui is not None:
                self.scan_prog_call_back_sig_gui.emit(progress_dict)

    def subscribe_to_scan_prog(self, callback_signal):
        """
        the scanProgressUi can subscribe via this function
        """
        self.scan_prog_call_back_sig_gui = callback_signal

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
        n_of_tracks, list_of_track_nums = SdOp.get_number_of_tracks_in_scan_dict(self.scan_pars[iso_name])
        try:
            active_track_num = min(set(list_of_track_nums) - set(self.scan_progress['completedTracks']))
            self.scan_progress['activeTrackNum'] = active_track_num
            track_index = list_of_track_nums.index(active_track_num)
            self.scan_main.prep_track_in_pipe(active_track_num, track_index)
            if self.scan_main.start_measurement(self.scan_pars[iso_name], active_track_num):
                self.set_state(MainState.scanning)
            else:
                logging.error('could not start scan on fpga')
                self.set_state(MainState.error)
        except ValueError:  # all tracks for this isotope are completed.
            # min(... ) will yield a value error when list of track_nums = self.scan_progress['completedTracks']
            self.scan_main.stop_measurement()
            self.set_state(MainState.idle)

    def _scanning(self):
        """
        will be called when in state 'scanning'
        will always feed data to the pipeline in scan_main.
        will change to 'load_track', when no data is available anymore
        AND the state is not measuring state anymore.
        """
        if self.abort_scan:  # abort the scan and return to idle state
            print('abort was pressed ', self.abort_scan)
            self.scan_main.abort_scan()
            self.set_state(MainState.saving)
        elif not self.scan_main.read_data():  # read_data() yields False if no Elements can be read from fpga
            if not self.scan_main.check_scanning():  # check if fpga is still in scanning state
                if self.halt_scan:  # scan was halted
                    self.halt_scan_func(False)  # set halt variabel to false afterwards
                    self.set_state(MainState.saving)
                else:  # normal exit after completion of each track
                    self.scan_progress['completedTracks'].append(self.scan_progress['activeTrackNum'])
                    # self.scan_main.stop_measurement(False)
                    # stop pipeline before starting with next track again, do not clear.
                    tracks, tr_l = SdOp.get_number_of_tracks_in_scan_dict(
                        self.scan_pars[self.scan_progress['activeIso']])
                    if len(self.scan_progress['completedTracks']) == tracks:
                        self.set_state(MainState.saving)
                    else:
                        self.set_state(MainState.load_track)

    def _stop_sequencer_and_save(self):
        """
        will be called in state 'saving'
        """
        logging.info('saving...')
        QApplication.setOverrideCursor(QCursor(QtCore.Qt.WaitCursor))  # ignore warning
        self.scan_main.stop_measurement()  # stop pipeline and clear
        QApplication.restoreOverrideCursor()  # ignore warning
        self.set_state(MainState.idle)

    """ simple counter """

    def start_simple_counter(self, act_pmt_list, datapoints, callback_sig):
        self.set_state(MainState.starting_simple_counter, (act_pmt_list, datapoints, callback_sig), only_if_idle=True)

    def _start_simple_counter(self, act_pmt_list, datapoints, callback_sig):
        if self.scan_main.sequencer is not None:
            self.scan_main.deinit_fpga()
        self.simple_counter_inst = SimpleCounterControl(act_pmt_list, datapoints, callback_sig)
        try:
            self.simple_counter_inst.run()
        except Exception as e:
            print('while starting the simple counter bitfile, this happened: ', str(e))
            print('don\'t worry, starting DUMMY Simple Counter now.')
            self.simple_counter_inst.run_dummy()
        finally:
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
        self.set_state(MainState.setting_power_supply, (power_supply, volt, call_back_signal), True)

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

    def add_new_iso_to_db(self, iso, seq_type):
        """
        add a new isotope of type seq, to the database
        """
        return DbOp.add_new_iso(self.database, iso, seq_type)

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

    def remove_iso_from_scan_pars(self, iso_seqtype):
        """
        this will remove the dictionary named 'iso_seqtype' from self.scan_pars
        """
        self.scan_pars.pop(iso_seqtype)
        logging.debug('scan_pars are: ' + str(self.scan_pars))

    def save_scan_par_to_db(self, iso):
        """
        will save all information in the scan_pars dict for the given isotope to the database.
        """
        self.add_global_infos_to_scan_pars(iso)
        scan_d = deepcopy(self.scan_pars[iso])
        # add_scan_dict_to_db will perform some changes on scan_d, therefore copy necessary
        trk_num, trk_lis = SdOp.get_number_of_tracks_in_scan_dict(scan_d)
        for i in trk_lis:
            logging.debug('saving track ' + str(i) + ' dict is: ' +
                          str(scan_d['track' + str(i)]))
            logging.debug('measureVoltPars are: %s' % scan_d['measureVoltPars'])
            DbOp.add_scan_dict_to_db(self.database, scan_d, i, track_key='track' + str(i))

    """ Tilda passive operations """

    def start_tilda_passive(self, n_of_bins, delay_10ns, raw_callback, status_callback, steps_scans_callback):
        self.tilda_passive_inst = TildaPassiveControl()
        iso_name = 'Ni_tipa'
        # tilda passive (tipa) works without the database! Database is for real sequencers only.
        self.scan_pars[iso_name] = self.tilda_passive_inst.tipa_get_default_scan_pars()
        self.scan_pars[iso_name]['measureVoltPars'] = self.measure_voltage_pars
        self.scan_pars[iso_name]['pipeInternals']['workingDirectory'] = self.working_directory
        self.scan_pars[iso_name]['isotopeData']['version'] = Cfg.version
        self.scan_pars[iso_name]['isotopeData']['laserFreq'] = self.laserfreq
        self.scan_pars[iso_name]['isotopeData']['accVolt'] = self.acc_voltage
        self.scan_pars[iso_name]['track0']['nOfBins'] = n_of_bins
        self.scan_pars[iso_name]['track0']['trigger']['trigDelay10ns'] = delay_10ns
        self.set_state(MainState.preparing_tilda_passiv, (raw_callback, steps_scans_callback))
        self.tipa_status_callback_sig = status_callback

    def stop_tilda_passive(self, silent=False):
        self.set_state(MainState.closing_tilda_passiv, silent)

    def _prepare_tilda_passive(self, raw_callback, steps_scans_callback):
        if self.scan_main.sequencer is not None:
            self.scan_main.deinit_fpga()
        iso_name = 'Ni_tipa'
        self.tilda_passive_inst.setup_tipa_ctrl(self.scan_pars[iso_name], raw_callback, steps_scans_callback)
        self.send_tipa_status(0)
        self.tilda_passive_inst.set_values(self.scan_pars[iso_name]['track0']['nOfBins'],
                                           self.scan_pars[iso_name]['track0']['trigger']['trigDelay10ns'])
        if self.tilda_passive_inst.start_scanning():
            self.tipa_timeout_counter = datetime.now()
            self.set_state(MainState.tilda_passiv_running)
        else:
            self.set_state(MainState.closing_tilda_passiv)

    def _tilda_passive_running(self):
        if self.tilda_passive_inst.read_data():
            self.tipa_timeout_counter = datetime.now()
            self.send_tipa_status()
        else:
            if (datetime.now() - self.tipa_timeout_counter).total_seconds() > 5:
                self.send_tipa_status(3)  # go to state 3 which is only a software state to indicate a timeout
            else:
                self.send_tipa_status()

    def _close_tilda_passive(self, silent):
        if self.tilda_passive_inst is not None:
            print('closing silently: ', silent)
            if not silent:
                self.send_tipa_status(-1)
                self.tipa_status_callback_sig = None
            self.tilda_passive_inst.stop()
            self.tilda_passive_inst = None
        self.set_state(MainState.idle)

    def send_tipa_status(self, maybe_new_status=None):
        if self.tipa_status_callback_sig is not None:
            if maybe_new_status is None:
                maybe_new_status = self.tilda_passive_inst.read_tipa_status()
            if maybe_new_status != self.tilda_passive_status:
                self.tilda_passive_status = maybe_new_status
                self.tipa_status_callback_sig.emit(self.tilda_passive_status)

    ''' digital multimeter operations '''

    def init_dmm(self, type_str, addr_str, callback=False, start_periodic=True):
        """
        initialize the dmm of given type and address.
        pass a callback to know when init is done.
        dmm should be in idle state after this.
        :param type_str: str, type of dmm
        :param addr_str: str, address of the given dmm
        :param callback: callback_bool, True will be emitted after init is done.
        :param start_periodic: bool, True(dft) if you want the dmm to start reading values in a predefined interval.
        """
        self.set_state(MainState.init_dmm, (type_str, addr_str, callback, start_periodic), only_if_idle=True)

    def _init_dmm(self, type_str, addr_str, callback, start_periodic):
        """ see init_dmm() """
        dmm_name = self.scan_main.prepare_dmm(type_str, addr_str)
        self.send_state()
        self.set_state(MainState.idle)
        if callback:
            callback.emit(True)
        if start_periodic:
            self.scan_main.set_dmm_to_periodic_reading(dmm_name)

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

    def read_dmms(self, feed_to_pipe=False):
        """
        dmm's will be read when main is in idle state.
        Values are already measured by the dmm in advance and main only "fetches" those.
            -> should be a quick return of values.
            values are emitted via send_state()
            also values are emitted via the self.dmm_gui_callback, if there is a gui subscirbed to it.
        """
        # return None
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

    def request_dmm_config_pars(self, dmm_name):
        """
        request the config parameters from the dmm and return them to caller.
        :param dmm_name: str, name of dev
        :return: dict, raw_configuration parameters, with currently set values,
        """
        conf_dict = self.scan_main.request_config_pars(dmm_name)
        return conf_dict

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
