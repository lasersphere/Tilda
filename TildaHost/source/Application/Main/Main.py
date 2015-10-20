"""

Created on '30.09.2015'

@author:'simkaufm'

"""

from PyQt5 import QtWidgets
import sys

from Interface.MainUi.MainUi import MainUi
import Service.Scan.ScanDictionaryOperations as ScanDicOp
import Application.Config as Cfg

class Main():
    def __init__(self):
        self.scanpars = []  # list of scanparameter dictionaries, like in Service.draftScanParameters.py in
                            #  the beginning only one item should be in the list.

        self.scanpars.append(ScanDicOp.init_empty_scan_dict(Cfg.version))


        self.mainUi = self.start_gui()


    def start_gui(self):
        app = QtWidgets.QApplication(sys.argv)
        ui = MainUi(self)
        app.exec_()
        return ui