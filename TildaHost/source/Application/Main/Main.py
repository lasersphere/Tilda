"""

Created on '30.09.2015'

@author:'simkaufm'

"""

from PyQt5 import QtWidgets
import sys
import logging
import os

from Interface.MainUi.MainUi import MainUi
import Service.Scan.ScanDictionaryOperations as ScanDicOp
import Service.DatabaseOperations.DatabaseOperations as DbOp


class Main():
    def __init__(self):
        self.scanpars = []  # list of scanparameter dictionaries, like in Service.draftScanParameters.py in
                            #  the beginning only one item should be in the list.
        self.database = None  # path of the sqlite3 database
        self.working_directory = None
        self.measure_voltage_pars = {}  # dict containing all parameters for the voltage measurement.

        self.scanpars.append(ScanDicOp.init_empty_scan_dict())

        # remove this later:
        # self.work_dir_changed('D:\\blub')
        self.work_dir_changed('C:\\temp')

        self.mainUi = self.start_gui()

    def start_gui(self):
        app = QtWidgets.QApplication(sys.argv)
        ui = MainUi(self)
        app.exec_()
        return ui

    def work_dir_changed(self, workdir_str):
        self.working_directory = workdir_str
        self.database = workdir_str + '/' + os.path.split(workdir_str)[1] + '.sqlite'
        DbOp.createTildaDB(self.database)
        logging.debug('working directory has been set to: ' + str(workdir_str))
        return workdir_str