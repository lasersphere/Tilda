"""
Created on 

@author: simkaufm

Module Description:
"""

import Tilda.Service.Scan.draftScanParameters as DftSc
import Tilda.Service.Scan.ScanDictionaryOperations as SdOp

from PyQt5 import QtWidgets


class BaseSequencerWidgUi(QtWidgets.QFrame):
    def __init__(self, track_dict):
        QtWidgets.QFrame.__init__(self)
        self.type = None
        self.set_type()
        key_list = getattr(DftSc, self.type + '_list')

        from_track_dict = {key: val for key, val in track_dict.items() if key in key_list}

        self.buffer_pars = SdOp.merge_dicts(dict.fromkeys(key_list), from_track_dict)
        self.connect_labels()
        self.set_vals_by_dict()

    def get_seq_pars(self, start=-10, stop=10):
        return self.buffer_pars

    """ generic functions to overwrite: """
    def set_type(self):
        pass

    def connect_labels(self):
        pass

    def set_vals_by_dict(self):
        pass
