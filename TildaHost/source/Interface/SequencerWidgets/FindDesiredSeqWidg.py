"""
Created on 

@author: simkaufm

Module Description:  This will return the desired sequencer Widget
"""

from Interface.SequencerWidgets.ContSequencerWidgUi import ContSeqWidg
from Interface.SequencerWidgets.TRSWidgUi import TRSWidg


def find_sequencer_widget(seq_type, track_dict):
    if seq_type == 'cs' or seq_type == 'csdummy':
        return ContSeqWidg(track_dict)
    elif seq_type == 'trs':
        return TRSWidg(track_dict)
    else:
        return None

