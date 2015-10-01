"""

Created on '30.09.2015'

@author:'simkaufm'

"""

from PyQt5 import QtWidgets
import sys

from Interface.MainUi.MainUi import MainUi


class Main():
    def __init__(self):
        self.scanpars = []  # list of scanparameter dictionaries, like in Service.draftScanParameters.py

        self.mainUi = self.start_gui()


    def start_gui(self):
        app = QtWidgets.QApplication(sys.argv)
        ui = MainUi(self)
        app.exec_()
        return ui