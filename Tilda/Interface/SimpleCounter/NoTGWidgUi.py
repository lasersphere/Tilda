"""
Created on 2021-11-30

@author: lrenth

Module Description: empty widget as a place holder if no time gate is needed
"""

from Interface.SimpleCounter.Ui_no_TG import Ui_no_tg_widg
from PyQt5 import QtWidgets


class NoTGWidg(QtWidgets.QFrame, Ui_no_tg_widg):
    def __init__(self):
        QtWidgets.QFrame.__init__(self)
        self.setupUi(self)
