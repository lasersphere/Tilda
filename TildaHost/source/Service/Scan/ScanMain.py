"""

Created on '19.05.2015'

@author:'simkaufm'

"""

from Driver.DataAcquisitionFpga.TimeResolvedSequencer import TimeResolvedSequencer

class ScanMain():
    def __init__(self):
        self.trs = TimeResolvedSequencer()

    def startMeasurement(self, scanpars):
        self.setHeinzinger(scanpars)

        self.measureOneTrack(scanpars)


    def measureOneTrack(self, scanpars):
        self.trs.measureTrack(scanpars)