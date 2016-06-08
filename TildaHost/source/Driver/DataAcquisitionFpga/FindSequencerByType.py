"""

Created on '13.11.2015'

@author:'simkaufm'


Module for returning an instance of the desired Sequencer
"""


def ret_seq_instance_of_type(seq_type):
    if seq_type == 'cs' or seq_type == 'kepco':
        from Driver.DataAcquisitionFpga.ContinousSequencer import ContinousSequencer as Cs
        return Cs()
    elif seq_type == 'trs':
        from Driver.DataAcquisitionFpga.TimeResolvedSequencer import TimeResolvedSequencer as Trs
        return Trs()
    elif seq_type == 'csdummy':
        from Driver.DataAcquisitionFpga.ContinousSequencerDummy import ContinousSequencer as CsDummy
        return CsDummy()
    elif seq_type == 'trsdummy':
        from Driver.DataAcquisitionFpga.TimeResolvedSequencerDummy import TimeResolvedSequencer as TrsDummy
        return TrsDummy()
    else:
        return None
