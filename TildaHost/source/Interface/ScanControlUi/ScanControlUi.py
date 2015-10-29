"""

Created on '29.10.2015'

@author:'simkaufm'

"""

from Interface.ScanControlUi.Ui_ScanControl import Ui_MainWindowScanControl
from Interface.TrackParUi.TrackUi import TrackUi
from Interface.SetupIsotopeUi.SetupIsotopeUi import SetupIsotopeUi

from PyQt5 import QtWidgets

import logging
import sys

class ScanControlUi(QtWidgets.QMainWindow, Ui_MainWindowScanControl):
    def __init__(self, main):
        super(ScanControlUi, self).__init__()
        self.setupUi(self)

        self.actionGo.triggered.connect(self.go)
        self.actionSetup_Isotope.triggered.connect(self.setup_iso)
        self.actionAdd_Track.triggered.connect(self.add_track)
        self.actionSave_settings_to_database.triggered.connect(self.save_to_db)

        self.show()

        self.main = main

    def go(self):
        logging.debug('starting measurement')

    def add_track(self):
        logging.debug('adding track')

    def setup_iso(self):
        logging.debug('setting up isotope')
        self.iso = SetupIsotopeUi()


    def save_to_db(self):
        logging.debug('saving settings to database')


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    blub = ScanControlUi(None)
    app.exec_()