"""

Created on '13.11.2015'

@author:'simkaufm'


Module for returning an instance of the desired Sequencer
"""


def ret_seq_instance_of_type(seq_type):
    if seq_type == 'cs':
        from Driver.DataAcquisitionFpga.ContinousSequencer import ContinousSequencer as Cs
        return Cs()
    elif seq_type == 'trs':
        from Driver.DataAcquisitionFpga.TimeResolvedSequencer import TimeResolvedSequencer as Trs
        return Trs()
    else:
        return None
