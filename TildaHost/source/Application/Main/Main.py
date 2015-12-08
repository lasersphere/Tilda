"""

Created on '30.09.2015'

@author:'simkaufm'

"""

from PyQt5 import QtWidgets
import sys
import logging
import os
import multiprocessing
import time
from copy import deepcopy



from Service.Scan.ScanMain import ScanMain
from Service.SimpleCounter.SimpleCounter import SimpleCounterControl

import Service.Scan.ScanDictionaryOperations as SdOp
import Service.Scan.draftScanParameters as Dft
import Service.DatabaseOperations.DatabaseOperations as DbOp
import Application.Config as Cfg


class Main:
    def __init__(self):
        self.m_state = 'init'
        self.database = None  # path of the sqlite3 database
        self.working_directory = None
        self.measure_voltage_pars = Dft.draftMeasureVoltPars  # dict containing all parameters
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
            logging.warning('could not start scan because state of main is ' + self.m_state)

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

    def set_power_supply_voltage(self, power_supply, volt):
        """
        will set the Output voltage of the desiredself
        power supply as stated in the track dictionary
        """
        self.scan_main.set_post_accel_pwr_supply(power_supply, volt)

    def power_supply_status_request(self, power_supply):
        """
        returns a dict containing the status of the power supply,
        keys are: name, programmedVoltage, voltageSetTime, readBackVolt
        """
        return self.scan_main.get_status_of_pwr_supply(power_supply)

    def set_state(self, req_state):
        """
        this will set the state of the main to req_state
        :return: str, state of main
        """
        self.m_state = req_state
        logging.debug('changed state to %s', self.m_state)
        return self.m_state

    def cyclic(self):
        """
        cyclic function called regularly by the QtTimer initiated in TildaStart.py
        This will control the main
        """
        if self.m_state == 'simple_counter_running':
            logging.debug('reading simple counter data')
            self.simple_counter_inst.read_data()
        if self.m_state == 'stop_simple_counter':
            self.stop_simple_counter()
        pass

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

    def remove_track_from_scan_pars(self, iso, track):
        """
        remove a track from the given isotope dictionary.
        """
        self.scan_pars.get(iso).pop(track)

    def save_scan_par_to_db(self, iso):
        """
        will save all information in the scan_pars dict for the given isootpe to the database.
        """
        scan_d = self.scan_pars[iso]
        trk_num, trk_lis = SdOp.get_number_of_tracks_in_scan_dict(scan_d)
        for i in trk_lis:
            logging.debug('saving track ' + str(i) + ' dict is: ' +
                          str(scan_d['track' + str(i)]))
            DbOp.add_scan_dict_to_db(self.database, scan_d, i, track_key='track' + str(i))

    def add_next_track_to_iso_in_scan_pars(self, iso):
        """
        this will look for iso in self.scan_pars and add a new track with lowest possible number.
        If there is a track with this number available in the database, load from there.
        Otherwise copy from another track.
        """
        logging.debug('adding track')
        scan_d = self.scan_pars.get(iso)
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
