"""
Created on 

@author: simkaufm

Module Description:  This will return the desired sequencer Widget
"""

from Interface.SequencerWidgets.ContSequencerWidgUi import ContSeqWidg
from Interface.SequencerWidgets.TRSWidgUi import TRSWidg
from Interface.SequencerWidgets.KepcoScanWidg import KepcoScanWidg


def find_sequencer_widget(seq_type, track_dict, main_gui):
    if seq_type == 'cs' or seq_type == 'csdummy':
        return ContSeqWidg(track_dict, main_gui)
    elif seq_type == 'trs' or seq_type == 'trsdummy':
        return TRSWidg(track_dict, main_gui)
    elif seq_type == 'kepco':
        return KepcoScanWidg(track_dict, main_gui)
    else:
        return None

