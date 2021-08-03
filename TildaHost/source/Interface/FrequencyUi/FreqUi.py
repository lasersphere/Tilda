"""

Created on '14.02.2019'

@author:'fsommer'

"""

import ast
import functools
import logging
import os
import sys
from copy import deepcopy
from datetime import datetime
from time import sleep

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import Service.FileOperations.FolderAndFileHandling as FileHandler

import Application.Config as Cfg

from Interface.FrequencyUi.Ui_Frequency import Ui_Frequency
from Interface.FrequencyUi.AddFreqUi import AddFreqUi


class FreqUi(QtWidgets.QMainWindow, Ui_Frequency):
    main_ui_status_call_back_signal = QtCore.pyqtSignal(dict)

    def __init__(self, main_gui):
        super(FreqUi, self).__init__()

        self.setupUi(self)
        self.stored_window_title = 'Frequency'
        self.setWindowTitle(self.stored_window_title)
        self.main_gui = main_gui

        ''' Windows '''

        self.add_freq_win = None

        ''' push button functionality '''
        self.pb_add.clicked.connect(self.add_freq)
        self.pb_remove.clicked.connect(self.rem_sel_freq)
        self.pb_edit.clicked.connect(self.edit_freq)

        self.show()


    def add_freq(self):
        self.open_add_freq_win()

    def rem_sel_freq(self):
        return False

    def edit_freq(self):
        return False

    def accept(self):
        self.closeEvent()

    def reject(self):
        self.closeEvent()

    ''' open windows '''

    def open_add_freq_win(self):
        AddFreqUi(self)

    ''' window related '''

    def closeEvent(self, *args, **kwargs):
        """ overwrite the close event """
        if self.main_gui is not None:
            # tell main window that this window is closed.
            self.main_gui.close_freq_win()