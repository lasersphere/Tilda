"""

Created on '07.05.2015'

@author:'simkaufm'

"""
from PyQt5 import QtWidgets


from Interface.MainUi.Ui_Main import Ui_TildaMainWindow
from Interface.VersionUi.VersionUi import VersionUi
from Interface.TrackParUi.TrackUi import TrackUi

class MainUi(QtWidgets.QMainWindow, Ui_TildaMainWindow):
    def __init__(self):
        super(MainUi, self).__init__()

        self.ui = Ui_TildaMainWindow()
        self.ui.setupUi(self)

        self.ui.actionTracks.triggered.connect(self.open_track_win)
        self.ui.actionVersion.triggered.connect(self.open_version_win)
        self.show()


    def open_track_win(self):
        TrackUi()

    def open_version_win(self):
        VersionUi()