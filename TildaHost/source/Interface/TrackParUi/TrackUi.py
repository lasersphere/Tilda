"""

Created on '29.09.2015'

@author:'simkaufm'

"""
from PyQt5 import QtWidgets


from Interface.TrackParUi.Ui_TrackPar import Ui_MainWindowTrackPars

class TrackUi(QtWidgets.QMainWindow, Ui_MainWindowTrackPars):
    def __init__(self):
        super(TrackUi, self).__init__()

        self.ui = Ui_MainWindowTrackPars()
        self.ui.setupUi(self)

        self.show()
