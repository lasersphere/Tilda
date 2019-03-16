"""

Created on '12.05.2015'

@author:'simkaufm'

"""

import ctypes
from os import path, pardir

import Service.FileOperations.FolderAndFileHandling as FileHandl
import Service.Scan.draftScanParameters as draftPars

fpga_cfg_root, fpga_cfg_dict = FileHandl.load_fpga_xml_config_file()
data_acq_cfg = fpga_cfg_dict['fpgas']['data_acquisition_fpga']
fpga_type = data_acq_cfg['fpga_type']
fpga_resource = data_acq_cfg['fpga_resource']

"""
Indicators, controls and fifos as gained from C-Api generator
Using CApiAnalyser.py yields:
"""

'''Bitfile Signature:'''
bitfileSignatures = {'PXI-7852R': 'F3E28A084C5C702750F53394D303D30F',
                     'PXI-7841R': '30733D581C7AE2660BF873489967793C'
                     }
bitfileSignature = bitfileSignatures[fpga_type]
'''Bitfile Path:'''
bitfilePaths = {'PXI-7852R': path.join(path.dirname(__file__), pardir, pardir, pardir, pardir,
                                       'TildaTarget/bin/TimeResolvedSequencer/NiFpga_TRS_DAF_242.lvbitx'),
                'PXI-7841R': path.join(path.dirname(__file__), pardir, pardir, pardir, pardir,
                                       'TildaTarget/bin/TimeResolvedSequencer/NiFpga_TRS_DAF_220_7842.lvbitx')
                }

bitfilePath = bitfilePaths[fpga_type]

'''FPGA Resource:'''
fpgaResource = fpga_resource

'''Indicators:'''
DACQuWriteTimeout = {'ref': 0x817E, 'val': ctypes.c_bool(), 'ctr': False}
MCSQuWriteTimeout = {'ref': 0x8182, 'val': ctypes.c_bool(), 'ctr': False}
internalDacAvailable = {'ref': 0x8112, 'val': ctypes.c_bool(), 'ctr': False}
MCSerrorcount = {'ref': 0x817A, 'val': ctypes.c_byte(), 'ctr': False}
postAccOffsetVoltState = {'ref': 0x8172, 'val': ctypes.c_ubyte(), 'ctr': False}
MCSstate = {'ref': 0x818A, 'val': ctypes.c_uint(), 'ctr': False}
OutBitsState = {'ref': 0x8156, 'val': ctypes.c_uint(), 'ctr': False}
measVoltState = {'ref': 0x8186, 'val': ctypes.c_uint(), 'ctr': False}
seqState = {'ref': 0x81AE, 'val': ctypes.c_uint(), 'ctr': False}
nOfCmdsOutbit0 = {'ref': 0x8150, 'val': ctypes.c_ulong(), 'ctr': False}
nOfCmdsOutbit1 = {'ref': 0x814C, 'val': ctypes.c_ulong(), 'ctr': False}
nOfCmdsOutbit2 = {'ref': 0x8148, 'val': ctypes.c_ulong(), 'ctr': False}
'''Controls:'''
VoltOrScaler = {'ref': 0x810E, 'val': ctypes.c_bool(), 'ctr': True}
abort = {'ref': 0x81A6, 'val': ctypes.c_bool(), 'ctr': True}
halt = {'ref': 0x81A2, 'val': ctypes.c_bool(), 'ctr': True}
hostConfirmsHzOffsetIsSet = {'ref': 0x819E, 'val': ctypes.c_bool(), 'ctr': True}
invertScan = {'ref': 0x81CA, 'val': ctypes.c_bool(), 'ctr': True}
pause = {'ref': 0x815E, 'val': ctypes.c_bool(), 'ctr': True}
scanDevSet = {'ref': 0x811A, 'val': ctypes.c_bool(), 'ctr': True}
softwareScanTrigger = {'ref': 0x813A, 'val': ctypes.c_bool(), 'ctr': True}
softwareStepTrigger = {'ref': 0x8126, 'val': ctypes.c_bool(), 'ctr': True}
softwareTrigger = {'ref': 0x815A, 'val': ctypes.c_bool(), 'ctr': True}
stopVoltMeas = {'ref': 0x8162, 'val': ctypes.c_bool(), 'ctr': True}
timedOutWhileHandshake = {'ref': 0x81B2, 'val': ctypes.c_bool(), 'ctr': True}
postAccOffsetVoltControl = {'ref': 0x81C6, 'val': ctypes.c_ubyte(), 'ctr': True}
scanTriggerEdge = {'ref': 0x813E, 'val': ctypes.c_ubyte(), 'ctr': True}
selectScanTrigger = {'ref': 0x8146, 'val': ctypes.c_ubyte(), 'ctr': True}
selectStepTrigger = {'ref': 0x8132, 'val': ctypes.c_ubyte(), 'ctr': True}
selectTrigger = {'ref': 0x819A, 'val': ctypes.c_ubyte(), 'ctr': True}
stepTriggerEdge = {'ref': 0x812A, 'val': ctypes.c_ubyte(), 'ctr': True}
triggerEdge = {'ref': 0x816E, 'val': ctypes.c_ubyte(), 'ctr': True}
ScanDevice = {'ref': 0x8116, 'val': ctypes.c_uint(), 'ctr': True}
cmdByHost = {'ref': 0x81AA, 'val': ctypes.c_uint(), 'ctr': True}
measVoltCompleteDest = {'ref': 0x8166, 'val': ctypes.c_uint(), 'ctr': True}
scanTriggerTypes = {'ref': 0x8142, 'val': ctypes.c_uint(), 'ctr': True}
stepTriggerTypes = {'ref': 0x812E, 'val': ctypes.c_uint(), 'ctr': True}
triggerTypes = {'ref': 0x8176, 'val': ctypes.c_uint(), 'ctr': True}
dacStartRegister18Bit = {'ref': 0x81D4, 'val': ctypes.c_long(), 'ctr': True}
dacStepSize18Bit = {'ref': 0x81D8, 'val': ctypes.c_long(), 'ctr': True}
measVoltPulseLength25ns = {'ref': 0x81B8, 'val': ctypes.c_long(), 'ctr': True}
measVoltTimeout10ns = {'ref': 0x81B4, 'val': ctypes.c_long(), 'ctr': True}
nOfBunches = {'ref': 0x818C, 'val': ctypes.c_long(), 'ctr': True}
nOfScans = {'ref': 0x81CC, 'val': ctypes.c_long(), 'ctr': True}
nOfSteps = {'ref': 0x81D0, 'val': ctypes.c_long(), 'ctr': True}
scanDevTimeout10ns = {'ref': 0x811C, 'val': ctypes.c_long(), 'ctr': True}
dac0VRegister = {'ref': 0x8168, 'val': ctypes.c_ulong(), 'ctr': True}
nOfBins = {'ref': 0x8190, 'val': ctypes.c_ulong(), 'ctr': True}
scanTrigDelay10ns = {'ref': 0x8134, 'val': ctypes.c_ulong(), 'ctr': True}
stepTrigDelay10ns = {'ref': 0x8120, 'val': ctypes.c_ulong(), 'ctr': True}
trigDelay10ns = {'ref': 0x8194, 'val': ctypes.c_ulong(), 'ctr': True}
waitAfterResetus = {'ref': 0x81BC, 'val': ctypes.c_ulong(), 'ctr': True}
waitForKepcous = {'ref': 0x81C0, 'val': ctypes.c_ulong(), 'ctr': True}
'''TargetToHostFifos:'''
transferToHost = {'ref': 0, 'val': ctypes.c_ulong(), 'ctr': False}
'''HostToTargetFifos:'''
OutbitsCMD = {'ref': 1, 'val': ctypes.c_ulong(), 'ctr': False}


"""
Hand filled Values, for example certain values for enums:
"""
seqStateDict = {'init': 0, 'idle': 1, 'measureOffset': 2, 'measureTrack': 3, 'measComplete': 4, 'error': 5}
measVoltCompleteDestStateDict = {'PXI_Trigger_4': 0, 'Con1_DIO30': 1,
                                 'Con1_DIO31': 2, 'PXI_Trigger_4_Con1_DIO30': 3,
                                 'PXI_Trigger_4_Con1_DIO31': 4, 'PXI_Trigger_4_Con1_DIO30_Con1_DIO31': 5,
                                 'Con1_DIO30_Con1_DIO31': 6, 'software': 7}

transferToHostReqEle = 10000000

dummyScanParameters = draftPars.draftTrackPars

seq_type = 'trs'

