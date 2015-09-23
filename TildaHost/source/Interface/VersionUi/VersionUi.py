"""

Created on '23.09.2015'

@author:'simkaufm'

"""

from PyQt5 import QtWidgets

from Interface.VersionUi.Ui_Version import Ui_Version
import Application.Config as tildaCfg


class VersionUi(QtWidgets.QMainWindow, Ui_Version):
    def __init__(self):
        super(VersionUi, self).__init__()

        self.ui = Ui_Version()
        self.ui.setupUi(self)

        self.ui.labelDate.setText(tildaCfg.versiondate)
        self.ui.labelVersion.setText(tildaCfg.version)

        self.show()