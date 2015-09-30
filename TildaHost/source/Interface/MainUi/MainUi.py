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

class MainUi(QtWidgets.QMainWindow, Ui_TildaMainWindow):
    def __init__(self):
        super(MainUi, self).__init__()

        self.setupUi(self)

        self.actionTracks.triggered.connect(self.open_track_win)
        self.actionVersion.triggered.connect(self.open_version_win)
        self.actionScan_Control.triggered.connect(self.open_scan_ctrl_win)
        self.show()


    def open_track_win(self):
        self.trackWin = TrackUi()
        print(self.trackWin.buffer_pars, type(self.trackWin))

    def open_version_win(self):
        VersionUi()

    def open_scan_ctrl_win(self):
        self.thr = threading.Thread(target=self.timout)
        self.thr.start()

    def timout(self):
        t = 0
        while t < 100:
            print('sleeping...', t)
            t += 1
            time.sleep(0.1)