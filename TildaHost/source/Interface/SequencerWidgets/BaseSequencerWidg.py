"""
Created on 

@author: simkaufm

Module Description:
"""

import Service.Scan.draftScanParameters as DftSc

from PyQt5 import QtWidgets


class BaseSequencerWidgUi(QtWidgets.QFrame):
    def __init__(self, track_dict):
        QtWidgets.QFrame.__init__(self)
        self.type = None
        self.set_type()
        self.key_list = getattr(DftSc, self.type + '_list')

        self.buffer_pars = {key: val for key, val in track_dict.items() if key in self.key_list}
        self.connect_labels()
        self.set_vals_by_dict()

    """ generic functions to overwrite: """
    def get_seq_pars(self):
        return self.buffer_pars

    def set_type(self):
        pass

    def connect_labels(self):
        pass

    def set_vals_by_dict(self):
        pass