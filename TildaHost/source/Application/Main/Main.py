"""

Created on '30.09.2015'

@author:'simkaufm'

"""

from PyQt5 import QtWidgets
import sys
import logging
import os
import multiprocessing
import re
from copy import deepcopy, copy



from Service.Scan.ScanMain import ScanMain
from Service.SimpleCounter.SimpleCounter import SimpleCounterControl

import Service.Scan.ScanDictionaryOperations as SdOp
import Service.VoltageConversions.VoltageConversions as VCon
import Service.Scan.draftScanParameters as Dft
import Service.DatabaseOperations.DatabaseOperations as DbOp
import Application.Config as Cfg


class Main:
    def __init__(self):
        self.m_state = ('init', None)  # tuple (str, val)
        self.database = None  # path of the sqlite3 database
        self.working_directory = None
        self.measure_voltage_pars = Dft.draftMeasureVoltPars  # dict containing all parameters
        self.requested_power_supply_status = None
        self.active_power_supplies = {}
        # for the voltage measurement.
        self.simple_counter_inst = None
        self.cmd_queue = None
        self.seconds = 0
        self.scan_pars = {}  # {iso0: scan_dict, iso1: scan_dict} -> iso is unique

        self.scan_main = ScanMain()
        self.iso_scan_process = None

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
            logging.debug('reading simple counter data')
            self.simple_counter_inst.read_data()
        if self.m_state[0] == 'stop_simple_counter':
            self.stop_simple_counter()
        elif self.m_state[0] == 'setting_power_supply':
            self._set_power_supply_voltage(*self.m_state[1])
        elif self.m_state[0] == 'reading_power_supply':
            self._power_supply_status(self.m_state[1])
        elif self.m_state[0] == 'init_power_supplies':
            self._init_power_sups()
        elif self.m_state[0] == 'set_output_power_sup':
            self._set_power_sup_outp(*self.m_state[1])
        pass

    """ main functions """
    def set_state(self, req_state, val=None, only_if_idle=False):
        """
        this will set the state of the main to req_state
        :return: bool, True if success
        """
        if only_if_idle:
            if self.m_state[0] == 'idle':
                self.m_state = req_state, val
                logging.debug('changed state to %s', self.m_state)
                return True
            else:
                logging.error('main is not in idle state, could not change state to: %s,\n current state is: %s',
                              req_state, self.m_state)
                return False
        else:
            self.m_state = req_state, val
            logging.debug('changed state to %s', self.m_state)
            return True

    """ operations on self.scn_pars dictionary """
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

    """ scanning """
    def start_scan(self, one_scan_dict):
        """
        * merge the given scan dict with measureVoltPars, workingDirectory, nOfTracks and version
        """
        if self.m_state == 'idle':
            self.set_state('preparing_scan')
            one_scan_dict['measureVoltPars'] = SdOp.merge_dicts(one_scan_dict['measureVoltPars'],
                                                                self.measure_voltage_pars)
            one_scan_dict['pipeInternals']['workingDirectory'] = self.working_directory
            tracks, track_num_list = SdOp.get_number_of_tracks_in_scan_dict(one_scan_dict)
            one_scan_dict['isotopeData']['nOfTracks'] = tracks
            one_scan_dict['isotopeData']['version'] = Cfg.version
            logging.debug('will scan: ' + str(sorted(one_scan_dict)))
            self.scan_main.scan_one_isotope(one_scan_dict)  # change this to non blocking!
        else:
            logging.warning('could not start scan because state of main is ' + self.m_state[0])
    """ simple counter """
    def start_simple_counter(self, act_pmt_list, datapoints):
        self.simple_counter_inst = SimpleCounterControl(act_pmt_list, datapoints)
        try:
            self.simple_counter_inst.run()
        except Exception as e:
            print('while starting the simple counter bitfile, this happened: ', str(e))
            print('don\'t worry, starting dummy Sequencer now.')
            self.simple_counter_inst.run_dummy()
        finally:
            self.set_state('simple_counter_running')

    def stop_simple_counter(self):
        self.set_state('busy')
        fpga_status = self.simple_counter_inst.stop()
        logging.debug('fpga status after deinit is: ' + str(fpga_status))
        self.set_state('idle')

    """ postaccleration power supply functions """
    def init_power_sups(self):
        """
        only changes state, when in idle
        """
        self.active_power_supplies = {}
        self.set_state('init_power_supplies', only_if_idle=True)

    def _init_power_sups(self):
        """
        initializes all power supplies and stores it in self.active_power_supplies
        """
        self.active_power_supplies = self.scan_main.init_post_accel_pwr_supplies()
        self.set_state('idle')

    def set_power_supply_voltage(self, power_supply, volt):
        """
        this will request a change in the state in order to set the requested voltage.
        power_supply -> self.requested_power_supply
        volt -> self.requested_voltage
        """
        self.set_state('setting_power_supply', (power_supply, volt), True)

    def _set_power_supply_voltage(self, name, volt):
        """
        this will actually call to the power supply
        will set the Output voltage of the desired power supply,
        as stated in self.requested_power_supply to the requested voltage
        """
        self.scan_main.set_post_accel_pwr_supply(name, volt)
        self.set_state('idle')

    def set_power_sup_outp(self, name, outp):
        """
        change state
        """
        self.set_state('set_output_power_sup', (name, outp), True)

    def _set_power_sup_outp(self, name, outp):
        """
        set the output
        """
        self.scan_main.set_post_accel_pwr_spply_output(name, outp)
        self.set_state('idle')

    def power_supply_status(self, power_supply):
        """
        returns a dict containing the status of the power supply,
        keys are: name, programmedVoltage, voltageSetTime, readBackVolt
        """
        self.set_state('reading_power_supply', power_supply, True)

    def _power_supply_status(self, name):
        """
        connects to the requested power supply and writes the status of the given power supply into
        self.requested_power_supply_status
        """
        self.requested_power_supply_status = self.scan_main.get_status_of_pwr_supply(name)
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
