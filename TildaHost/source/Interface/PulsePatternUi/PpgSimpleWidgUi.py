"""

Created on '18.01.2017'

@author:'simkaufm'

"""

from PyQt5 import QtWidgets, QtCore

from Interface.PulsePatternUi.Ui_PpgSimpleWidg import Ui_PpgSimpleWidg


class PpgSimpleWidgUi(QtWidgets.QWidget, Ui_PpgSimpleWidg):
    pulse_pattern_status = QtCore.pyqtSignal(str)

    def __init__(self, parent, cmd_list_callback_signal):
        super(PpgSimpleWidgUi, self).__init__(parent)
        self.setupUi(self)
        self.show()

        self.cmd_list_callback_signal = cmd_list_callback_signal
