"""

Created on '07.05.2015'

@author:'simkaufm'

"""

from Driver.DataAcquisitionFpga.TimeResolvedSequencer import TimeResolvedSequencer

class Main():
    def __init__(self):
        self.trs = TimeResolvedSequencer()



maininst = Main()
print(maininst.trs.getSeqState())