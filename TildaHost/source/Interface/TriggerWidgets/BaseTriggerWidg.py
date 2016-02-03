"""
Created on 

@author: simkaufm

Module Description: Base Trigger widget containing all generic functions
"""

from PyQt5 import QtWidgets


class BaseTriggerWidgUi(QtWidgets.QFrame):
    def __init__(self, trigger_dict):
        QtWidgets.QFrame.__init__(self)
        self.type = None
        self.set_type()

        if trigger_dict is None:
            trigger_dict = {}

        self.buffer_pars = trigger_dict
        self.connect_labels()
        # self.set_vals_by_dict()  # should be called in the widgets themself

    def get_trig_pars(self):
        self.buffer_pars['type'] = self.type
        return self.buffer_pars

    """ generic functions to overwrite: """
    def set_type(self):
        pass

    def connect_labels(self):
        pass

    def set_vals_by_dict(self):
        pass