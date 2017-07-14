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
bitfileSignatures = {'PXI-7852R': 'A53A64C1BCCD22CD7F401CF0BFC05CE7',
                     'PXI-7841R': 'E79B112B2137785AAF60D6C55F07D8F3'
                     }
bitfileSignature = bitfileSignatures[fpga_type]
'''Bitfile Path:'''
bitfilePaths = {'PXI-7852R': path.join(path.dirname(__file__), pardir, pardir, pardir, pardir,
                                       'TildaTarget/bin/TimeResolvedSequencer/NiFpga_TRS_DAF_207.lvbitx'),
                'PXI-7841R': path.join(path.dirname(__file__), pardir, pardir, pardir, pardir,
                                       'TildaTarget/bin/TimeResolvedSequencer/NiFpga_TRS_DAF_207_7841.lvbitx')
                }

bitfilePath = bitfilePaths[fpga_type]

'''FPGA Resource:'''
fpgaResource = fpga_resource
'''Indicators:'''
DACQuWriteTimeout = {'ref': 0x8146, 'val': ctypes.c_bool(), 'ctr': False}
MCSQuWriteTimeout = {'ref': 0x814A, 'val': ctypes.c_bool(), 'ctr': False}
MCSerrorcount = {'ref': 0x8142, 'val': ctypes.c_byte(), 'ctr': False}
postAccOffsetVoltState = {'ref': 0x813A, 'val': ctypes.c_ubyte(), 'ctr': False}
MCSstate = {'ref': 0x8152, 'val': ctypes.c_uint(), 'ctr': False}
OutBitsState = {'ref': 0x811E, 'val': ctypes.c_uint(), 'ctr': False}
measVoltState = {'ref': 0x814E, 'val': ctypes.c_uint(), 'ctr': False}
seqState = {'ref': 0x8176, 'val': ctypes.c_uint(), 'ctr': False}
nOfCmdsOutbit0 = {'ref': 0x8118, 'val': ctypes.c_ulong(), 'ctr': False}
nOfCmdsOutbit1 = {'ref': 0x8114, 'val': ctypes.c_ulong(), 'ctr': False}
nOfCmdsOutbit2 = {'ref': 0x8110, 'val': ctypes.c_ulong(), 'ctr': False}
'''Controls:'''
VoltOrScaler = {'ref': 0x810E, 'val': ctypes.c_bool(), 'ctr': True}
abort = {'ref': 0x816E, 'val': ctypes.c_bool(), 'ctr': True}
halt = {'ref': 0x816A, 'val': ctypes.c_bool(), 'ctr': True}
hostConfirmsHzOffsetIsSet = {'ref': 0x8166, 'val': ctypes.c_bool(), 'ctr': True}
invertScan = {'ref': 0x8192, 'val': ctypes.c_bool(), 'ctr': True}
pause = {'ref': 0x8126, 'val': ctypes.c_bool(), 'ctr': True}
softwareTrigger = {'ref': 0x8122, 'val': ctypes.c_bool(), 'ctr': True}
stopVoltMeas = {'ref': 0x812A, 'val': ctypes.c_bool(), 'ctr': True}
timedOutWhileHandshake = {'ref': 0x817A, 'val': ctypes.c_bool(), 'ctr': True}
postAccOffsetVoltControl = {'ref': 0x818E, 'val': ctypes.c_ubyte(), 'ctr': True}
selectTrigger = {'ref': 0x8162, 'val': ctypes.c_ubyte(), 'ctr': True}
triggerEdge = {'ref': 0x8136, 'val': ctypes.c_ubyte(), 'ctr': True}
cmdByHost = {'ref': 0x8172, 'val': ctypes.c_uint(), 'ctr': True}
measVoltCompleteDest = {'ref': 0x812E, 'val': ctypes.c_uint(), 'ctr': True}
triggerTypes = {'ref': 0x813E, 'val': ctypes.c_uint(), 'ctr': True}
dacStartRegister18Bit = {'ref': 0x819C, 'val': ctypes.c_long(), 'ctr': True}
dacStepSize18Bit = {'ref': 0x81A0, 'val': ctypes.c_long(), 'ctr': True}
measVoltPulseLength25ns = {'ref': 0x8180, 'val': ctypes.c_long(), 'ctr': True}
measVoltTimeout10ns = {'ref': 0x817C, 'val': ctypes.c_long(), 'ctr': True}
nOfBunches = {'ref': 0x8154, 'val': ctypes.c_long(), 'ctr': True}
nOfScans = {'ref': 0x8194, 'val': ctypes.c_long(), 'ctr': True}
nOfSteps = {'ref': 0x8198, 'val': ctypes.c_long(), 'ctr': True}
dac0VRegister = {'ref': 0x8130, 'val': ctypes.c_ulong(), 'ctr': True}
nOfBins = {'ref': 0x8158, 'val': ctypes.c_ulong(), 'ctr': True}
trigDelay10ns = {'ref': 0x815C, 'val': ctypes.c_ulong(), 'ctr': True}
waitAfterResetus = {'ref': 0x8184, 'val': ctypes.c_ulong(), 'ctr': True}
waitForKepcous = {'ref': 0x8188, 'val': ctypes.c_ulong(), 'ctr': True}
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

