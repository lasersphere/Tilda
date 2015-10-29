"""

Created on '30.09.2015'

@author:'simkaufm'

"""

from PyQt5 import QtWidgets
import sys

from Interface.MainUi.MainUi import MainUi
import Service.Scan.ScanDictionaryOperations as ScanDicOp


class Main():
    def __init__(self):
        self.global_scanpars = {}  # dict for all scanparameters which are globally available for all other scandicts
        self.scanpars = []  # list of scanparameter dictionaries, like in Service.draftScanParameters.py in
                            #  the beginning only one item should be in the list.

        self.scanpars.append(ScanDicOp.init_empty_scan_dict())

        self.mainUi = self.start_gui()

    def start_gui(self):
        app = QtWidgets.QApplication(sys.argv)
        ui = MainUi(self)
        app.exec_()
        return ui

    def w_global_scanpars(self, key, val):
        self.global_scanpars[key] = val