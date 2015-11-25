"""

Created on '30.09.2015'

@author:'simkaufm'

"""

from PyQt5 import QtWidgets
import sys
import logging
import os
import multiprocessing

from Interface.ScanControlUi.ScanControlUi import ScanControlUi
from Interface.VoltageMeasurementConfigUi.VoltMeasConfUi import VoltMeasConfUi
from Interface.PostAccControlUi.PostAccControlUi import PostAccControlUi
import Interface.SimpleCounter.SimpleCounterDialogUi as ScDial

from Service.Scan.ScanMain import ScanMain
from Service.SimpleCounter.SimpleCounter import SimpleCounterControl

import Service.Scan.ScanDictionaryOperations as SdOp
import Service.Scan.draftScanParameters as Dft
import Service.DatabaseOperations.DatabaseOperations as DbOp
import Application.Config as Cfg


class Main:
    def __init__(self):
        self.act_scan_wins = []  # list of active scan windows
        self.database = None  # path of the sqlite3 database
        self.working_directory = None
        self.post_acc_win = None
        self.measure_voltage_pars = Dft.draftMeasureVoltPars  # dict containing all parameters
        # for the voltage measurement.
        self.simple_counter_proc = None
        self.cmd_queue = None

        self.scan_main = ScanMain()
        self.iso_scan_process = None

        # remove this later:
        # self.work_dir_changed('E:\\blub')
        # self.work_dir_changed('C:\\temp')
        try:
            self.work_dir_changed('E:\\lala')
        except Exception as e:
            logging.error('while loading default location of db this happened:' + str(e))

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
        merge the given scan dict with measureVoltPars, workingDirectory, nOfTracks and version
        start new process and:
            setup all devices, including the FPGAs
            start the Pipeline
            read data from FPGA and feed it every "period" seconds to the pipeline.
            The Pipeline will take care of plotting and displaying.
        """
        one_scan_dict['measureVoltPars'] = SdOp.merge_dicts(one_scan_dict['measureVoltPars'],
                                                            self.measure_voltage_pars)
        one_scan_dict['pipeInternals']['workingDirectory'] = self.working_directory
        tracks, track_num_list = SdOp.get_number_of_tracks_in_scan_dict(one_scan_dict)
        one_scan_dict['isotopeData']['nOfTracks'] = tracks
        one_scan_dict['isotopeData']['version'] = Cfg.version
        logging.debug('will scan: ' + str(sorted(one_scan_dict)))
        self.iso_scan_process = multiprocessing.Process(
            target=self.scan_main.scan_one_isotope, args=(one_scan_dict,))
        self.iso_scan_process.start()
        # multiprocessing is used instead of threading due to QT,
        # because the plot window could not be held open in another Thread than the main thread

    def start_simple_counter(self):
        self.close_simple_counter_proc()
        self.open_simp_count_dial()
        self.cmd_queue = multiprocessing.JoinableQueue()
        act_pmt_list, datapoints = self.open_simp_count_dial()
        self.simple_counter_proc = SimpleCounterControl(act_pmt_list, datapoints, self.cmd_queue)
        try:
            self.simple_counter_proc.start()
        except Exception as e:
            print('while starting the process, this happened:', str(e))
        self.open_simple_counter_stop_win()
        self.stop_simple_counter()

    def stop_simple_counter(self):
        self.cmd_queue.put(False)
        self.close_simple_counter_proc()

    def close_simple_counter_proc(self):
        if self.simple_counter_proc is not None:
            try:
                self.simple_counter_proc.join()
                logging.info('simple counter process successfully joined')
            except Exception as e:
                logging.error('error while closing simple counter process:' + str(e))
            finally:
                self.simple_counter_proc = None

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

    """ opening and closing of GUI's """

    def open_work_dir_win(self):
        self.work_dir_changed(QtWidgets.QFileDialog.getExistingDirectory(QtWidgets.QFileDialog(),
            'choose working directory', os.path.expanduser('~')))
        return self.working_directory

    def open_scan_control_win(self):
        if self.working_directory is None:
            self.open_work_dir_win()
        self.act_scan_wins.append(ScanControlUi(self))

    def scan_control_win_closed(self, win_ref):
        self.act_scan_wins.remove(win_ref)

    def open_volt_meas_win(self):
        self.measure_voltage_pars['actWin'] = VoltMeasConfUi(self, self.measure_voltage_pars)

    def close_volt_meas_win(self):
        self.measure_voltage_pars.pop('actWin')

    def open_post_acc_win(self):
        self.post_acc_win = PostAccControlUi(self)

    def closed_post_acc_win(self):
        self.post_acc_win = None

    def open_simple_counter_stop_win(self):
        ScDial.open_dial()

    def open_simp_count_dial(self):
        # open window here in the future, that configures the pipeline.
        # self.simple_counter_settings = '{str(activePmtList): [0, 1], str(plotPoints): 600}'
        return [0, 1], 600
    # def open_simple_counter_stop_win(self):


    def close_main_win(self):
        for win in self.act_scan_wins:
            logging.debug('will close: ' + str(win))
            try:
                win.close()
            except Exception as e:
                logging.error(str(e))
        try:
            if self.post_acc_win is not None:
                self.post_acc_win.close()
        except Exception as e:
            logging.error('error while closing post acceleration win:' + str(e))
        try:
            if self.measure_voltage_pars.get('actWin') is not None:
                self.measure_voltage_pars['actWin'].close()
        except Exception as e:
            logging.error('error while closing voltage measurement win:' + str(e))
        try:
            self.iso_scan_process.join()
        except Exception as e:
            logging.error('error while closing pipe process:' + str(e))


