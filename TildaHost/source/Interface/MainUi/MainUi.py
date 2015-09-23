"""

Created on '07.05.2015'

@author:'simkaufm'

"""
from PyQt5 import QtWidgets, QtCore
import threading


from Interface.MainUi.Ui_Main import Ui_TildaMainWindow
from Interface.VersionUi.VersionUi import VersionUi
# from Interface.VersionUi.Ui_Version import Ui_Version

class MainUi(QtWidgets.QMainWindow, Ui_TildaMainWindow):
    def __init__(self):
        super(MainUi, self).__init__()

        self.ui = Ui_TildaMainWindow()
        self.ui.setupUi(self)

        self.ui.actionTracks.triggered.connect(self.dostuff)
        self.ui.actionVersion.triggered.connect(self.open_version_win)
        self.show()


    def dostuff(self):
        print('Hello World')

    def open_version_win(self):
        ver = VersionUi()
        ver.exec()
        # dialog = QtWidgets.QDialog(self)
        # dialog.ver = Ui_Version()
        # dialog.ver.setupUi(dialog)
        # dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        # dialog.show()