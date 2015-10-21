"""

Created on '07.05.2015'

@author:'simkaufm'

"""
from PyQt5 import QtWidgets


from Interface.MainUi.Ui_Main import Ui_TildaMainWindow
from Interface.VersionUi.VersionUi import VersionUi
from Interface.TrackParUi.TrackUi import TrackUi

import threading
import time
import logging
import os

class MainUi(QtWidgets.QMainWindow, Ui_TildaMainWindow):
    def __init__(self, main):
        super(MainUi, self).__init__()
        self.main = main

        self.setupUi(self)

        self.actionWorking_directory.triggered.connect(self.choose_working_dir)
        self.actionTracks.triggered.connect(self.open_track_win)
        self.actionVersion.triggered.connect(self.open_version_win)
        self.actionScan_Control.triggered.connect(self.open_scan_ctrl_win)
        self.show()

    def choose_working_dir(self):
        """ will open a modal file dialog and set all workingdirectories of the pipeline to the chosen folder """
        workdir = QtWidgets.QFileDialog.getExistingDirectory(self, 'choose working directory', os.path.expanduser('~'))
        for scd in self.main.scanpars:
            scd['pipeInternals']['workingDirectory'] = workdir
        logging.debug('working directory has been set to: ' + str(workdir) + '\n \n ' + str(self.main.scanpars))

    def open_track_win(self):
        self.trackWin = TrackUi(self.main, 0)

    def open_version_win(self):
        VersionUi()

    def open_scan_ctrl_win(self):
        self.thr = threading.Thread(target=self.timout)
        self.thr.start()

    def timout(self):
        t = 0
        while t < 100:
            print('sleeping...', t, 'scan pars:  ', self.main.scanpars)
            t += 1
            time.sleep(0.1)