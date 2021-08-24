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
    def __init__(self, freq_win, name=None, value=None):
        super(AddFreqUi, self).__init__()
        self.setupUi(self)
        self.parent_ui = freq_win
        self.original_name = name

        """Buttons"""
        self.pb_add.clicked.connect(self.add)
        self.pb_cancel.clicked.connect(self.cancel)

        """Line Edit"""
        self.le_name.setText(name)
        self.le_value.setText(value)

        self.exec_()

    def add(self):
        freq_name = self.check_name_rules(self.le_name.text())  # check naming conventions
        freq_val = self.le_value.text()
        try:
            interger = int(freq_val)
            try:
                self.parent_ui.freq_dict.pop(self.original_name)  # remove old freq name from dictionary
            except AttributeError:
                logging.info('No Attribute %s, skip pop' % self.original_name)
            self.parent_ui.freq_dict[freq_name] = interger
            self.parent_ui.new_freq_name = freq_name
            print('added new Frequency')
            self.close()
        except ValueError:
            print('Frequency value needs to be an Integer!')

    def check_name_rules(self, name):
        """
        Implement a few checks to make sure the names are nice (and not empty)
        :param name: str: user-input name
        :return: str: changed name if necessary, else user-input
        """
        name0 = name  # remember original name
        # check for bad string parts.
        if name:
            name = name.replace(':', '')  # no colon allowed because its the separator!
            name = name.replace(' ', '_')  # no whitespace allowed (best practice)
            # more rules could be added here...
        else:
            # user forgot to give it a name. Use default name
            name = 'newfreq'
        # If the name is new or was changed, we should check that it does not conflict with existing entries
        if name != self.original_name:  # name was changed or is added new
            running_num = 1  # create a running number to attach to duplicates
            run_name = name
            while run_name in self.parent_ui.freq_dict.keys():
                run_name = name+'_{}'.format(running_num)
                running_num += 1
            name = run_name
        # if any changes were made: inform the user with a WARNING
        if name != name0:
            logging.warning('Frequency name bad or already exists: changed from {} to {}'.format(name0, name))

        return name  # return the new name (or old if it was fine)

    def cancel(self):
        self.close()
