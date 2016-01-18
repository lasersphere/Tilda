"""

Created on '12.05.2015'

@author:'simkaufm'

"""

import ctypes
import enum

import Service.Scan.draftScanParameters as draftPars


"""
Indicators, controls and fifos as gained from C-Api generator
Using CApiAnalyser.py yields:
"""

'''Bitfile Signature:'''
bitfileSignature = 'F20BD1B0DAC34D01DB29C415C1C25648'
'''Bitfile Path:'''
bitfilePath = 'D:/Workspace/PyCharm/Tilda/TildaTarget/bin/TimeResolvedSequencer/NiFpga_TRS_DAF_104.lvbitx'
'''FPGA Resource:'''
fpgaResource = 'Rio1'
'''Indicators:'''
DACQuWriteTimeout = {'ref': 0x811E, 'val': ctypes.c_bool(), 'ctr': False}
MCSQuWriteTimeout = {'ref': 0x8122, 'val': ctypes.c_bool(), 'ctr': False}
MCSerrorcount = {'ref': 0x811A, 'val': ctypes.c_byte(), 'ctr': False}
postAccOffsetVoltState = {'ref': 0x8112, 'val': ctypes.c_ubyte(), 'ctr': False}
MCSstate = {'ref': 0x812A, 'val': ctypes.c_uint(), 'ctr': False}
measVoltState = {'ref': 0x8126, 'val': ctypes.c_uint(), 'ctr': False}
seqState = {'ref': 0x814E, 'val': ctypes.c_uint(), 'ctr': False}
'''Controls:'''
VoltOrScaler = {'ref': 0x810E, 'val': ctypes.c_bool(), 'ctr': True}
abort = {'ref': 0x8146, 'val': ctypes.c_bool(), 'ctr': True}
halt = {'ref': 0x8142, 'val': ctypes.c_bool(), 'ctr': True}
hostConfirmsHzOffsetIsSet = {'ref': 0x813E, 'val': ctypes.c_bool(), 'ctr': True}
invertScan = {'ref': 0x816A, 'val': ctypes.c_bool(), 'ctr': True}
timedOutWhileHandshake = {'ref': 0x8152, 'val': ctypes.c_bool(), 'ctr': True}
postAccOffsetVoltControl = {'ref': 0x8166, 'val': ctypes.c_ubyte(), 'ctr': True}
selectTrigger = {'ref': 0x813A, 'val': ctypes.c_ubyte(), 'ctr': True}
cmdByHost = {'ref': 0x814A, 'val': ctypes.c_uint(), 'ctr': True}
triggerTypes = {'ref': 0x8116, 'val': ctypes.c_uint(), 'ctr': True}
waitAfterReset25nsTicks = {'ref': 0x815E, 'val': ctypes.c_uint(), 'ctr': True}
waitForKepco25nsTicks = {'ref': 0x8162, 'val': ctypes.c_uint(), 'ctr': True}
dacStartRegister18Bit = {'ref': 0x8174, 'val': ctypes.c_long(), 'ctr': True}
dacStepSize18Bit = {'ref': 0x8178, 'val': ctypes.c_long(), 'ctr': True}
measVoltPulseLength25ns = {'ref': 0x8158, 'val': ctypes.c_long(), 'ctr': True}
measVoltTimeout10ns = {'ref': 0x8154, 'val': ctypes.c_long(), 'ctr': True}
nOfBunches = {'ref': 0x812C, 'val': ctypes.c_long(), 'ctr': True}
nOfScans = {'ref': 0x816C, 'val': ctypes.c_long(), 'ctr': True}
nOfSteps = {'ref': 0x8170, 'val': ctypes.c_long(), 'ctr': True}
trig_delay_10ns = {'ref': 0x8134, 'val': ctypes.c_ulong(), 'ctr': True}
nOfBins = {'ref': 0x8130, 'val': ctypes.c_ulong(), 'ctr': True}
'''TargetToHostFifos:'''
transferToHost = {'ref': 0, 'val': ctypes.c_ulong(), 'ctr': False}


"""
Hand filled Values, for example certain values for enums:
"""
seqStateDict = {'init': 0, 'idle': 1, 'measureOffset': 2, 'measureTrack': 3, 'measComplete': 4, 'error': 5}
transferToHostReqEle = 100000

dummyScanParameters = draftPars.draftTrackPars

seq_type = 'trs'

