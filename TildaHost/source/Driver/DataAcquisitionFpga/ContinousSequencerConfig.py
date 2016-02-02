"""

Created on '09.07.2015'

@author:'simkaufm'

"""
import ctypes

'''Bitfile Signature:'''
bitfileSignature = '59D10AD8995EF1076D75C53D3E54E379'
'''Bitfile Path:'''
bitfilePath = 'D:/Workspace/PyCharm/Tilda/TildaTarget/bin/ContinousSequencer/NiFpga_ContSeqV108.lvbitx'
'''FPGA Resource:'''
fpgaResource = 'Rio1'
'''Indicators:'''
DACQuWriteTimeout = {'ref': 0x8176, 'val': ctypes.c_bool(), 'ctr': False}
SPCtrQuWriteTimeout = {'ref': 0x8132, 'val': ctypes.c_bool(), 'ctr': False}
SPerrorCount = {'ref': 0x8126, 'val': ctypes.c_ubyte(), 'ctr': False}
postAccOffsetVoltState = {'ref': 0x811E, 'val': ctypes.c_ubyte(), 'ctr': False}
SPstate = {'ref': 0x812A, 'val': ctypes.c_uint(), 'ctr': False}
measVoltState = {'ref': 0x8172, 'val': ctypes.c_uint(), 'ctr': False}
seqState = {'ref': 0x815E, 'val': ctypes.c_uint(), 'ctr': False}
'''Controls:'''
VoltOrScaler = {'ref': 0x810E, 'val': ctypes.c_bool(), 'ctr': True}
abort = {'ref': 0x8166, 'val': ctypes.c_bool(), 'ctr': True}
halt = {'ref': 0x816A, 'val': ctypes.c_bool(), 'ctr': True}
hostConfirmsHzOffsetIsSet = {'ref': 0x816E, 'val': ctypes.c_bool(), 'ctr': True}
invertScan = {'ref': 0x8146, 'val': ctypes.c_bool(), 'ctr': True}
timedOutWhileHandshake = {'ref': 0x815A, 'val': ctypes.c_bool(), 'ctr': True}
postAccOffsetVoltControl = {'ref': 0x8122, 'val': ctypes.c_ubyte(), 'ctr': True}
selectTrigger = {'ref': 0x8116, 'val': ctypes.c_ubyte(), 'ctr': True}
cmdByHost = {'ref': 0x8162, 'val': ctypes.c_uint(), 'ctr': True}
triggerTypes = {'ref': 0x811A, 'val': ctypes.c_uint(), 'ctr': True}
waitAfterReset25nsTicks = {'ref': 0x814E, 'val': ctypes.c_uint(), 'ctr': True}
waitForKepco25nsTicks = {'ref': 0x814A, 'val': ctypes.c_uint(), 'ctr': True}
dacStartRegister18Bit = {'ref': 0x8138, 'val': ctypes.c_long(), 'ctr': True}
dacStepSize18Bit = {'ref': 0x8134, 'val': ctypes.c_long(), 'ctr': True}
measVoltPulseLength25ns = {'ref': 0x8150, 'val': ctypes.c_long(), 'ctr': True}
measVoltTimeout10ns = {'ref': 0x8154, 'val': ctypes.c_long(), 'ctr': True}
nOfScans = {'ref': 0x8140, 'val': ctypes.c_long(), 'ctr': True}
nOfSteps = {'ref': 0x813C, 'val': ctypes.c_long(), 'ctr': True}
trigDelay10ns = {'ref': 0x8110, 'val': ctypes.c_ulong(), 'ctr': True}
dwellTime10ns = {'ref': 0x812C, 'val': ctypes.c_ulong(), 'ctr': True}
'''TargetToHostFifos:'''
transferToHost = {'ref': 0, 'val': ctypes.c_ulong(), 'ctr': False}

'''hand filled values:'''
SPstateDict = {'idle': 0, 'scanning': 1, 'done': 2, 'error': 3}
seqStateDict = {'init': 0, 'idle': 1, 'measureOffset': 2, 'measureTrack': 3, 'measComplete': 4, 'error': 5}
postAccOffsetVoltStateDict = {'Kepco': 0, 'Heinzinger1': 1, 'Heinzinger2': 2, 'Heinzinger3': 3, 'loading': 4}

transferToHostReqEle = 100000

seq_type = 'cs'
