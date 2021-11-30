"""
Created on 2021-11-30

@author: lrenth

Module Description:
"""

from Interface.SimpleCounter.Ui_timeGating import Ui_timeGating
from PyQt5 import QtWidgets


class TGWidg(QtWidgets.QFrame, Ui_timeGating):
    def __init__(self, tg_dict):
        QtWidgets.QFrame.__init__(self)
        self.setupUi(self)
        if tg_dict is None:
            tg_dict = {}
        self.buffer_pars = tg_dict

        #self.lineEdit_mid_tof.textChanged.connect(self.mid_tof_set)

    def mid_tof_set(self, value):
        self.buffer_pars['tg']['mid_tof'] = value
