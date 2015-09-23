"""

Created on '23.09.2015'

@author:'simkaufm'

"""

from PyQt5 import QtWidgets
from PyQt5 import QtCore


from Interface.VersionUi.Ui_Version import Ui_Version
import Application.Config as tildaCfg


class VersionUi(QtWidgets.QDialog, Ui_Version):
    def __init__(self):
        QtWidgets.QDialog.__init__(self)

        self.setupUi(self)
        # self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.labelDate.setText(tildaCfg.versiondate)
        self.labelVersion.setText(tildaCfg.version)

        # self.exec_()
        # self.show()