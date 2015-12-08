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

        self.scan_main = ScanMain()
        self.iso_scan_process = None

        # remove this later:
        # self.work_dir_changed('E:\\blub')
        # self.work_dir_changed('C:\\temp')
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
        # self.seconds += 1
        # print(time.strftime("%H:%M:%S", time.gmtime(self.seconds)))
        if self.m_state == 'simple_counter_running':
            logging.debug('reading simple counter data')
            self.simple_counter_inst.read_data()
        if self.m_state == 'stop_simple_counter':
            self.stop_simple_counter()
        pass
