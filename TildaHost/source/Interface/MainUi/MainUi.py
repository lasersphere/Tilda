"""

Created on '07.05.2015'

@author:'simkaufm'

"""



from Interface.MainUi.Ui_Main import Ui_TildaMainWindow
from Interface.VersionUi.VersionUi import VersionUi
from Interface.TrackParUi.TrackUi import TrackUi
from Interface.VoltageMeasurementConfigUi.VoltMeasConfUi import VoltMeasConfUi
from Interface.ScanControlUi.ScanControlUi import ScanControlUi
import Service.Scan.draftScanParameters as Dft

from copy import deepcopy
import threading
import time
import logging
import os
from PyQt5 import QtWidgets

class MainUi(QtWidgets.QMainWindow, Ui_TildaMainWindow):
    def __init__(self, main):
        super(MainUi, self).__init__()
        self.main = main

        self.setupUi(self)

        self.actionWorking_directory.triggered.connect(self.choose_working_dir)
        self.actionTracks.triggered.connect(self.open_track_win)
        self.actionVersion.triggered.connect(self.open_version_win)
        self.actionScan_Control.triggered.connect(self.open_scan_ctrl_win)
        self.actionVoltage_Measurement.triggered.connect(self.open_volt_meas_win)
        self.show()

    def choose_working_dir(self):
        """ will open a modal file dialog and set all workingdirectories of the pipeline to the chosen folder """
        workdir = QtWidgets.QFileDialog.getExistingDirectory(self, 'choose working directory', os.path.expanduser('~'))
        self.main.w_global_scanpars('workingDirectory', workdir)
        self.label_workdir_set.setText(str(workdir))

        logging.debug('working directory has been set to: ' + str(workdir))

    def open_track_win(self):
        # print(self.main.scanpars[0]['activeTrackPar'])
        # self.trackWin = TrackUi(self.main, 0, self.main.scanpars[0]['activeTrackPar'])
        pass

    def open_version_win(self):
        VersionUi()

    def open_scan_ctrl_win(self):
        self.scanwin = ScanControlUi(self.main)

    def open_volt_meas_win(self):
        self.voltwin = VoltMeasConfUi(self.main, Dft.draftMeasureVoltPars)