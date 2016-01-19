"""
Created on 

@author: simkaufm

Module Description:
"""


class BaseSequencerWidgUi:
    def __init__(self, track_dict):
        self.track_d = track_dict

    def set_gui_from_dict(self):
        f