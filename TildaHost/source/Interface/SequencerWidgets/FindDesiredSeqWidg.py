"""
Created on 

@author: simkaufm

Module Description:  This will return the desired sequencer Widget
"""

from Interface.SequencerWidgets.ContSequencerWidgUi import ContSeqWidg


def find_sequencer_widget(seq_type, track_dict):
    print(seq_type)
    if seq_type == 'cs' or seq_type == 'csdummy':
        return ContSeqWidg(track_dict)
    else:
        return None

