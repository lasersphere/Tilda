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

from Service.Scan.ScanMain import ScanMain

import Service.Scan.ScanDictionaryOperations as SdOp
import Service.DatabaseOperations.DatabaseOperations as DbOp


class Main():
    def __init__(self):
        self.scanpars = []  # list of scanparameter dictionaries, like in Service.draftScanParameters.py in
                            #  the beginning only one item should be in the list.
        self.database = None  # path of the sqlite3 database
        self.working_directory = None
        self.measure_voltage_pars = {}  # dict containing all parameters for the voltage measurement.

        # remove this later:
        self.work_dir_changed('E:\\blub')
        # self.work_dir_changed('C:\\temp')

        self.mainUi = self.start_gui()

    def start_gui(self):
        """
        starts the gui for the main window.
        """
        app = QtWidgets.QApplication(sys.argv)
        ui = MainUi(self)
        app.exec_()
        return ui

    def work_dir_changed(self, workdir_str):
        """
        Sets the working directory in which the main sqlite database is stored.
        """
        self.working_directory = workdir_str
        self.database = workdir_str + '/' + os.path.split(workdir_str)[1] + '.sqlite'
        DbOp.createTildaDB(self.database)
        logging.debug('working directory has been set to: ' + str(workdir_str))
        return workdir_str

    def start_scan(self, scan_dict):
        """
        setup all devices, including the FPGAs
        start the Pipeline
        read data from FPGA and feed it every "period" seconds to the pipeline.
        The Pipeline will take care of plotting and displaying.
        """
        scan = ScanMain()
        tr = threading.Thread(target=scan.print_timeout, args=[100])
        tr.start()

    def open_scan_control_win(self):
        self.scanpars.append(ScanControlUi(self))
        print(self.scanpars)

    def scan_control_win_closed(self, win_ref):
        self.scanpars.remove(win_ref)
        print(self.scanpars)
