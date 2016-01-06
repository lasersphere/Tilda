"""

Created on '30.09.2015'

@author:'simkaufm'

"""

import logging
import os
from copy import deepcopy
from PyQt5 import QtCore


from Service.Scan.ScanMain import ScanMain
from Service.SimpleCounter.SimpleCounter import SimpleCounterControl
import Service.Scan.ScanDictionaryOperations as SdOp
import Service.Scan.draftScanParameters as Dft
import Service.DatabaseOperations.DatabaseOperations as DbOp
import Application.Config as Cfg


class Main(QtCore.QObject):
    # this will equal the number of completed steps in the active track:
    scan_prog_call_back_sig = QtCore.pyqtSignal(int)

    def __init__(self):
        super(Main, self).__init__()
        self.m_state = ('init', None)  # tuple (str, val), each state can be entered with a value
        self.database = None  # path of the sqlite3 database
        self.working_directory = None  # path of the working directory, containig the database etc.
        self.measure_voltage_pars = Dft.draftMeasureVoltPars
        # dict containing all parameters for the voltage measurement.
        # default is: draftMeasureVoltPars = {'measVoltPulseLength25ns': 400, 'measVoltTimeout10ns': 100}
        self.laserfreq = 0  # laser frequency in cm-1
        self.acc_voltage = 0  # acceleration voltage of the source in volts
        self.simple_counter_inst = None
        self.cmd_queue = None


        # pyqtSignal for sending the status to the gui, if there is one connected:
        self.main_ui_status_call_back_signal = None
        self.scan_prog_call_back_sig.connect(self.update_scan_progress)

        self.scan_main = ScanMain()
        self.iso_scan_process = None
        self.scan_pars = {}  # {iso0: scan_dict, iso1: scan_dict} -> iso is unique
        self.scan_progress = {}  # {activeIso: str, activeTrackNum: int, completedTracks: list, nOfCompletedSteps: int}
        self.abort_scan = False
        self.halt_scan = False

        try:
            self.work_dir_changed('E:/lala')
        except Exception as e:
            logging.error('while loading default location of db this happened:' + str(e))
        self.set_state('idle')

    """ cyclic function """

    def cyclic(self):
        """
        cyclic function called regularly by the QtTimer initiated in TildaStart.py
        This will control the main
        """
        if self.m_state[0] == 'simple_counter_running':
            self.simple_counter_inst.read_data()
        elif self.m_state[0] == 'stop_simple_counter':
            self.stop_simple_counter()
        elif self.m_state[0] == 'setting_power_supply':
            self._set_power_supply_voltage(*self.m_state[1])
        elif self.m_state[0] == 'reading_power_supply':
            self._power_supply_status(*self.m_state[1])
        elif self.m_state[0] == 'init_power_supplies':
            self._init_power_sups(self.m_state[1])
        elif self.m_state[0] == 'set_output_power_sup':
            self._set_power_sup_outp(*self.m_state[1])
        elif self.m_state[0] == 'starting_simple_counter':
            self._start_simple_counter(*self.m_state[1])
        elif self.m_state[0] == 'load_track':
            self._load_track()
        elif self.m_state[0] == 'scanning':
            self._scanning()

    """ main functions """

    def set_state(self, req_state, val=None, only_if_idle=False):
        """
        this will set the state of the main to req_state
        :return: bool, True if success
        """
        if only_if_idle:
            if self.m_state[0] == 'idle':
                self.m_state = req_state, val
                self.send_state()
                logging.debug('changed state to %s', self.m_state)
                return True
            else:
                logging.error('main is not in idle state, could not change state to: %s,\n current state is: %s',
                              req_state, self.m_state)
                return False
        else:
            self.m_state = req_state, val
            self.send_state()
            logging.debug('changed state to %s', self.m_state)
            return True

    def send_state(self):
        """
        if a gui is subscribed via a call back signal in self.main_ui_status_call_back_signal.
        This function will emit a status dictionary containing the following keys:
        status_dict keys: ['workdir', 'status', 'database', 'laserfreq', 'accvolt']
        """
        if self.main_ui_status_call_back_signal is not None:
            stat_dict = {
                'workdir': self.working_directory,
                'status': self.m_state[0],
                'database': self.database,
                'laserfreq': self.laserfreq,
                'accvolt': self.acc_voltage
            }
            self.main_ui_status_call_back_signal.emit(stat_dict)

    """ operations on self.scan_pars dictionary """

    def remove_track_from_scan_pars(self, iso, track):
        """
        remove a track from the given isotope dictionary.
        """
        self.scan_pars.get(iso).pop(track)

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
            return workdir_str
        except Exception as e:
            logging.error('while loading db from: ' + workdir_str + ' this happened:' + str(e))
            self.database = None
            self.working_directory = None
        finally:
            self.send_state()

    """ scanning """

    def start_scan(self, iso_name):
        """
        the given isotope scan dictionary will be completed with global informations, which are valid for all isotopes,
        such as:
        workingDirectory, version, measureVoltPars, laserFreq
        then the bitfile is loaded to the fpga and the first track is started for scanning.
        the state will therefor be changed to scanning
        """
        try:
            if self.m_state[0] == 'idle':
                self.set_state('preparing_scan')
                self.scan_progress['activeIso'] = iso_name
                self.scan_progress['completedTracks'] = []
                self.scan_pars[iso_name]['measureVoltPars'] = self.measure_voltage_pars
                self.scan_pars[iso_name]['pipeInternals']['workingDirectory'] = self.working_directory
                self.scan_pars[iso_name]['isotopeData']['version'] = Cfg.version
                self.scan_pars[iso_name]['isotopeData']['laserFreq'] = self.laserfreq
                logging.debug('will scan: ' + iso_name + str(sorted(self.scan_pars[iso_name])))
                self.scan_main.prepare_scan(self.scan_pars[iso_name], self.scan_prog_call_back_sig)  # change this to non blocking!
                self.set_state('load_track')
            else:
                logging.warning('could not start scan because state of main is ' + self.m_state[0])
        except Exception as e:
            print('error: ', e)

    def update_scan_progress(self, number_of_completed_steps=None):
        """
        will be updated from the pipeline via Qt callback signal.
        """
        if number_of_completed_steps is not None:
            self.scan_progress['nOfCompletedSteps'] = number_of_completed_steps
        print('scan progress is: \t', self.scan_progress)

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
                self.set_state('scanning')
            else:
                logging.error('could not start scan on fpga')
                self.set_state('error')
        except ValueError:  # all tracks for this isotope are completed.
            # min(... ) will yield a value error when list of track_nums = self.scan_progress['completedTracks']
            self.scan_main.stop_measurement()
            self.set_state('idle')

    def _scanning(self):
        """
        will be called when in state 'scanning'
        will always feed data to the pipeline in scan_main.
        will change to 'load_track', when no data is available anymore
        AND the state is not measuring state anymore.
        """
        if not self.scan_main.read_data():
            if not self.scan_main.check_scanning():
                # also comparing received steps with total steps would be acceptable, maybe change in future
                self.scan_progress['completedTracks'].append(self.scan_progress['activeTrackNum'])
                self.update_scan_progress()
                if self.halt_scan:
                    self.set_state('idle')
                    self.scan_main.stop_measurement()
                else:  # normal exit after completion of each track
                    self.set_state('load_track')
        elif self.abort_scan:  # abort the scan and return to idle state
            self.scan_main.abort_scan()
            self.scan_main.stop_measurement()
            self.set_state('idle')


    """ simple counter """

    def start_simple_counter(self, act_pmt_list, datapoints, callback_sig):
        self.set_state('starting_simple_counter', (act_pmt_list, datapoints, callback_sig), only_if_idle=True)

    def _start_simple_counter(self, act_pmt_list, datapoints, callback_sig):
        self.simple_counter_inst = SimpleCounterControl(act_pmt_list, datapoints, callback_sig)
        try:
            self.simple_counter_inst.run()
        except Exception as e:
            print('while starting the simple counter bitfile, this happened: ', str(e))
            print('don\'t worry, starting DUMMY Simple Counter now.')
            self.simple_counter_inst.run_dummy()
        finally:
            self.set_state('simple_counter_running')

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
        fpga_status = self.simple_counter_inst.stop()
        logging.debug('fpga status after deinit is: ' + str(fpga_status))
        self.set_state('idle')

    """ postaccleration power supply functions """

    def init_power_sups(self, call_back_signal=None):
        """
        initializes all power supplies and reads the status afterwards.
        only changes state, when in idle
        """
        self.set_state('init_power_supplies', call_back_signal, only_if_idle=True)

    def _init_power_sups(self, call_back_signal=None):
        """
        initializes all power supplies and reads the status afterwards.
        """
        self.scan_main.init_post_accel_pwr_supplies()
        self.set_state('reading_power_supply', ('all', call_back_signal))

    def set_power_supply_voltage(self, power_supply, volt, call_back_signal=None):
        """
        this will request a change in the state in order to set the requested voltage.
        power_supply -> self.requested_power_supply
        volt -> self.requested_voltage
        """
        self.set_state('setting_power_supply', (power_supply, volt, call_back_signal), True)

    def _set_power_supply_voltage(self, name, volt, call_back_signal=None):
        """
        this will actually call to the power supply
        will set the Output voltage of the desired power supply,
        as stated in self.requested_power_supply to the requested voltage
        """
        self.scan_main.set_post_accel_pwr_supply(name, volt)
        self.set_state('reading_power_supply', (name, call_back_signal))

    def set_power_sup_outp(self, name, outp, call_back_signal=None):
        """
        change state
        """
        self.set_state('set_output_power_sup', (name, outp, call_back_signal), True)

    def _set_power_sup_outp(self, name, outp, call_back_signal=None):
        """
        set the output
        """
        self.scan_main.set_post_accel_pwr_spply_output(name, outp)
        self.set_state('reading_power_supply', (name, call_back_signal))

    def power_supply_status(self, power_supply, call_back_sig):
        """
        returns a dict containing the status of the power supply,
        keys are: name, programmedVoltage, voltageSetTime, readBackVolt
        """
        self.set_state('reading_power_supply', (power_supply, call_back_sig), True)

    def _power_supply_status(self, name, call_back_sig=None):
        """
        connects to the requested power supply and writes the status of the given power supply into
        self.requested_power_supply_status
        """
        stat = self.scan_main.get_status_of_pwr_supply(name)
        if call_back_sig is not None:
            call_back_sig.emit(stat)
        self.set_state('idle')

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
        DbOp.add_new_iso(self.database, iso, seq_type)

    def add_iso_to_scan_pars(self, iso, seq_type):
        """
        connect to the database and add all tracks with given isotope and sequencer type
        to self.scan_pars.
        :return: str, key of new isotope.
        """
        scand = DbOp.extract_all_tracks_from_db(self.database, iso, seq_type)
        key = iso + '_' + seq_type
        self.scan_pars[key] = scand
        return key

    def save_scan_par_to_db(self, iso):
        """
        will save all information in the scan_pars dict for the given isotope to the database.
        """
        scan_d = self.scan_pars[iso]
        trk_num, trk_lis = SdOp.get_number_of_tracks_in_scan_dict(scan_d)
        for i in trk_lis:
            logging.debug('saving track ' + str(i) + ' dict is: ' +
                          str(scan_d['track' + str(i)]))
            DbOp.add_scan_dict_to_db(self.database, scan_d, i, track_key='track' + str(i))
