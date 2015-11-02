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

        self.buffer_scan_dict = {}

        self.actionGo.triggered.connect(self.go)
        self.actionSetup_Isotope.triggered.connect(self.setup_iso)
        self.actionAdd_Track.triggered.connect(self.add_track)
        self.actionSave_settings_to_database.triggered.connect(self.save_to_db)
        self.listWidget.doubleClicked.connect(self.work_on_existing_track)

        self.show()

        self.main = main

    def go(self):
        # pss on the buffered scandict and let it run.
        logging.debug('starting measurement')

    def add_track(self):
        # self.track_win = TrackUi(self.main, )
        logging.debug('adding track')

    def work_on_existing_track(self, val):
        track_name = self.listWidget.currentItem().text()
        track_number = int(track_name[-1])
        logging.debug('working on track' + str(track_number))
        self.trackwin = TrackUi(self, track_number, self.buffer_scan_dict[track_name])
        print(self.buffer_scan_dict)

    def update_track_list(self):
        if self.buffer_scan_dict:
            self.listWidget.clear()
            newitems = sorted([key for key, val in self.buffer_scan_dict.items() if key[:-1] == 'track'])
            print(newitems)
            self.listWidget.addItems(newitems)

    def setup_iso(self):
        logging.debug('setting up isotope')
        SetupIsotopeUi(self.main, self)
        self.update_track_list()


    def save_to_db(self):
        logging.debug('saving settings to database')


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    blub = ScanControlUi(None)
    app.exec_()