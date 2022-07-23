"""
Created on 16.07.2022

@author: Patrick Mueller
"""

from copy import deepcopy
from PyQt5 import QtWidgets, QtCore
from Gui.Ui_HFMixingConfig import Ui_HFMixingConfig


class HFMixingConfigUi(QtWidgets.QWidget, Ui_HFMixingConfig):
    close_signal = QtCore.pyqtSignal()

    def __init__(self, db, iso, config):
        super(HFMixingConfigUi, self).__init__()
        self.setupUi(self)
        self.db = db
        self.iso = iso

        self.old_config = config
        self.config = config

        self.line_j_g.setText(str(self.iso.Jl))
        self.line_j_e.setText(str(self.iso.Ju))

        self.b_ok.clicked.connect(self.close)
        self.b_cancel.clicked.connect(self.revert_and_close)

    def set_config(self, config):
        pass

    def revert_and_close(self):
        self.config = self.old_config
        self.close_signal.emit()
        self.close()
