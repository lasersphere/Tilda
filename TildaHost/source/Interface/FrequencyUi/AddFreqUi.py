"""

Created on '07.08.2021'

@author:'lrenth'

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
    Dialog asking for a frequency name and frequency value in MHz
    """
    def __init__(self, freq_win, name, value):
        super(AddFreqUi, self).__init__()
        self.setupUi(self)
        self.parent_ui = freq_win

        """Buttons"""
        self.pb_add.clicked.connect(self.add)
        self.pb_cancel.clicked.connect(self.cancel)

        """Line Edit"""
        self.le_name.setText(name)
        self.le_value.setText(value)

        self.exec_()

    def add(self):
        freq_name = self.le_name.text()
        freq_val = self.le_value.text()
        try:
            interger = int(freq_val)
            self.parent_ui.freq_list[freq_name] = freq_val
            self.parent_ui.new_freq_name = freq_name
            print('added new Frequency')
            self.close()
        except ValueError:
            print('Frequency value needs to be an Integer!')

    def cancel(self):
        self.close()
