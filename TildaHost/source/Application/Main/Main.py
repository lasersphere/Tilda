"""

Created on '30.09.2015'

@author:'simkaufm'

"""

from PyQt5 import QtWidgets
import sys
import logging
import os
import threading

from Interface.MainUi.MainUi import MainUi
from Interface.ScanControlUi.ScanControlUi import ScanControlUi
from Interface.VoltageMeasurementConfigUi.VoltMeasConfUi import VoltMeasConfUi

from Service.Scan.ScanMain import ScanMain

import Service.Scan.ScanDictionaryOperations as SdOp
import Service.Scan.draftScanParameters as Dft
import Service.DatabaseOperations.DatabaseOperations as DbOp


class Main:
    def __init__(self):
        self.scanpars = []  # list of scanparameter dictionaries, like in Service.draftScanParameters.py in
                            #  the beginning only one item should be in the list.
        self.database = None  # path of the sqlite3 database
        self.working_directory = None
        self.measure_voltage_pars = Dft.draftMeasureVoltPars  # dict containing all parameters
        #  for the voltage measurement.

        self.scan_main = ScanMain()
        self.iso_scan_thread = None

        # remove this later:
        # self.work_dir_changed('E:\\blub')
        # self.work_dir_changed('C:\\temp')
        self.work_dir_changed('D:\\lala')

        self.mainUi = self.start_gui()

    def work_dir_changed(self, workdir_str):
        """
        Sets the working directory in which the main sqlite database is stored.
        """
        self.working_directory = workdir_str
        self.database = workdir_str + '/' + os.path.split(workdir_str)[1] + '.sqlite'
        DbOp.createTildaDB(self.database)
        logging.debug('working directory has been set to: ' + str(workdir_str))
        return workdir_str

    def start_scan(self, one_scan_dict):
        """
        setup all devices, including the FPGAs
        start the Pipeline
        read data from FPGA and feed it every "period" seconds to the pipeline.
        The Pipeline will take care of plotting and displaying.
        """
        one_scan_dict['measureVoltPars'] = SdOp.merge_dicts(one_scan_dict['measureVoltPars'],
                                                            self.measure_voltage_pars)
        one_scan_dict['pipeInternals']['workingDirectory'] = self.working_directory
        logging.debug('will scan: ' + str(sorted(one_scan_dict)))
        self.iso_scan_thread = threading.Thread(target=self.scan_main.scan_one_isotope, args=(one_scan_dict,))
        self.iso_scan_thread.start()

    def set_power_supply_voltage(self, power_supply, volt):
        """
        will set the Output voltage of the desired
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
    def start_gui(self):
        """
        starts the gui for the main window.
        """
        app = QtWidgets.QApplication(sys.argv)
        ui = MainUi(self)
        app.exec_()
        return ui

    def open_scan_control_win(self):
        self.scanpars.append(ScanControlUi(self))

    def scan_control_win_closed(self, win_ref):
        self.scanpars.remove(win_ref)

    def open_volt_meas_win(self):
        self.measure_voltage_pars['actWin'] = VoltMeasConfUi(self, self.measure_voltage_pars)

    def close_volt_meas_win(self):
        self.measure_voltage_pars.pop('actWin')
