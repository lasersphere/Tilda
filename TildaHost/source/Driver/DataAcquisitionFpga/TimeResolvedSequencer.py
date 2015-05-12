"""

Created on '07.05.2015'

@author:'simkaufm'

"""



"""
Module in  charge for loading and accessing the TimeResolvedSequencer
Access Via the NiFpgaUniversalInterfaceDll.dll
"""

from Driver.DataAcquisitionFpga.FPGAInterfaceHandling import FPGAInterfaceHandling
import Driver.DataAcquisitionFpga.TimeResolvedSequencerConfig as TrsCfg
import ctypes
import time


class TimeResolvedSequencer(FPGAInterfaceHandling):
    def __init__(self):
        super(TimeResolvedSequencer, self).__init__(TrsCfg.bitfilePath, TrsCfg.bitfileSignature, TrsCfg.fpgaResource)

    def getMCSState(self):
        self.ReadWrite(TrsCfg.MCSstate['ref'], TrsCfg.MCSstate['val'], TrsCfg.MCSstate['ctr'])
        return TrsCfg.MCSstate['val']

    def cmdByHost(self, cmd):
        self.ReadWrite(TrsCfg.cmdByHost['ref'], ctypes.c_uint(cmd), TrsCfg.cmdByHost['ctr'])
        return self.status

    def getSeqState(self):
        self.ReadWrite(TrsCfg.seqState['ref'], TrsCfg.seqState['val'], TrsCfg.seqState['ctr'])
        return TrsCfg.seqState['val']


# instanciate that bitch:
blub2 = TimeResolvedSequencer()
print(blub2.__init__())
time.sleep(0.01)
print(blub2.getSeqState())
print(blub2.status)
time.sleep(0.1)
print(blub2.cmdByHost(5))
print(blub2.getSeqState())
time.sleep(0.1)
print(blub2.getSeqState())
