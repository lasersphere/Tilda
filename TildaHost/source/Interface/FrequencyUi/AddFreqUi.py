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

from Interface.FrequencyUi.Ui_Add_Freq import Ui_Add_Freq


class AddFreqUi(QtWidgets.QDialog, Ui_Add_Freq):
    """
    Modal Dialog
    """
    #main_ui_status_call_back_signal = QtCore.pyqtSignal(dict)

    def __init__(self, freq_win):
        super(AddFreqUi, self).__init__()

        self.setupUi(self)
        #self.stored_window_title = 'AddFrequency'
        #self.setWindowTitle(self.stored_window_title)
        #self.main_gui = main_gui
        #self.main = Cfg._main_instance
        #self.freq_win = freq_win

        ''' Windows '''

        self.add_freq_win = None

        self.exec_()

