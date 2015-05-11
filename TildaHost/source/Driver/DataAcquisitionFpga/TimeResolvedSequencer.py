"""

Created on '07.05.2015'

@author:'simkaufm'

"""



"""
Module in  charge for loading and accessing the TimeResolvedSequencer
Access Via the NiFpgaUniversalInterfaceDll.dll
"""

from Driver.DataAcquisitionFpga.FPGAInterfaceHandling import FPGAInterfaceHandling
import ctypes
import time

"""
Indicators, controls and fifos as gained from C-Api generator
Using CApiAnalyser.py yields:
"""

'''Bitfile Signature:'''
bitfileSignature = 'BF31570369009FA00617B7055FD697C8'
'''Indicators:'''
DACQuWriteTimeout = {'ref':0x8116, 'val':ctypes.c_bool(), 'ctr':False}
MCSQuWriteTimeout = {'ref':0x811A, 'val':ctypes.c_bool(), 'ctr':False}
MCSerrorcount = {'ref':0x8112, 'val':ctypes.c_byte(), 'ctr':False}
MCSstate = {'ref':0x8122, 'val':ctypes.c_uint(), 'ctr':False}
measVoltState = {'ref':0x811E, 'val':ctypes.c_uint(), 'ctr':False}
seqState = {'ref':0x8146, 'val':ctypes.c_uint(), 'ctr':False}
'''Controls:'''
VoltOrScaler = {'ref':0x810E, 'val':ctypes.c_bool(), 'ctr':True}
abort = {'ref':0x813E, 'val':ctypes.c_bool(), 'ctr':True}
halt = {'ref':0x813A, 'val':ctypes.c_bool(), 'ctr':True}
hostConfirmsHzOffsetIsSet = {'ref':0x8136, 'val':ctypes.c_bool(), 'ctr':True}
invertScan = {'ref':0x8162, 'val':ctypes.c_bool(), 'ctr':True}
timedOutWhileHandshake = {'ref':0x814A, 'val':ctypes.c_bool(), 'ctr':True}
MCSSelectTrigger = {'ref':0x8132, 'val':ctypes.c_ubyte(), 'ctr':True}
heinzingerControl = {'ref':0x815E, 'val':ctypes.c_ubyte(), 'ctr':True}
cmdByHost = {'ref':0x8142, 'val':ctypes.c_uint(), 'ctr':True}
waitAfterReset25nsTicks = {'ref':0x8156, 'val':ctypes.c_uint(), 'ctr':True}
waitForKepco25nsTicks = {'ref':0x815A, 'val':ctypes.c_uint(), 'ctr':True}
measVoltPulseLength25ns = {'ref':0x8150, 'val':ctypes.c_long(), 'ctr':True}
measVoltTimeout10ns = {'ref':0x814C, 'val':ctypes.c_long(), 'ctr':True}
nOfBunches = {'ref':0x8124, 'val':ctypes.c_long(), 'ctr':True}
nOfScans = {'ref':0x8164, 'val':ctypes.c_long(), 'ctr':True}
nOfSteps = {'ref':0x8168, 'val':ctypes.c_long(), 'ctr':True}
start = {'ref':0x816C, 'val':ctypes.c_long(), 'ctr':True}
stepSize = {'ref':0x8170, 'val':ctypes.c_long(), 'ctr':True}
delayticks = {'ref':0x812C, 'val':ctypes.c_ulong(), 'ctr':True}
nOfBins = {'ref':0x8128, 'val':ctypes.c_ulong(), 'ctr':True}


class TimeResolvedSequencer(FPGAInterfaceHandling):
    def __init__(self):
        self.trsBitfilePath='D:\\Workspace\\Eclipse\\Tilda\\TildaTarget\\bin\\TimeResolvedSequencer\\NiFpga_TRS.lvbitx'
        self.trsBitfileSignature = 'BF31570369009FA00617B7055FD697C8'
        self.trsFpgaRessource = 'Rio1'
        super(TimeResolvedSequencer, self).__init__(self.trsBitfilePath, self.trsBitfileSignature, self.trsFpgaRessource)

    def getMCSState(self):
        self.ReadWrite(MCSstate['ref'], MCSstate['val'], MCSstate['ctr'])
        return MCSstate['val']

    def cmdByHost(self, cmd):
        self.ReadWrite(cmdByHost['ref'], ctypes.c_uint(cmd), cmdByHost['ctr'])
        return self.status

    def getSeqState(self):
        self.ReadWrite(seqState['ref'], seqState['val'], seqState['ctr'])
        return seqState['val']


# instanciate that bitch:
print(0x8146)
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
