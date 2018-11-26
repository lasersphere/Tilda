"""

Created on '09.07.2015'

@author:'simkaufm'

"""
import ctypes
from os import path, pardir

import Service.FileOperations.FolderAndFileHandling as FileHandl

fpga_cfg_root, fpga_cfg_dict = FileHandl.load_fpga_xml_config_file()
data_acq_cfg = fpga_cfg_dict['fpgas']['data_acquisition_fpga']
fpga_type = data_acq_cfg['fpga_type']
fpga_resource = data_acq_cfg['fpga_resource']

'''Bitfile Signature:'''
bitfileSignatures = {'PXI-7852R': '50181A46831391680B489E44F3DD9313',
                     'PXI-7841R': 'A7C562E7EF98A7C7A783D9C03235A8AE'
                     }
bitfileSignature = bitfileSignatures[fpga_type]
'''Bitfile Path:'''
bitfilePaths = {'PXI-7852R': path.join(path.dirname(__file__), pardir, pardir, pardir, pardir,
                                       'TildaTarget/bin/ContinousSequencer/NiFpga_ContSeqV220.lvbitx'),
                'PXI-7841R': path.join(path.dirname(__file__), pardir, pardir, pardir, pardir,
                                       'TildaTarget/bin/ContinousSequencer/NiFpga_ContSeqV220_7841.lvbitx')
                }
bitfilePath = bitfilePaths[fpga_type]
'''FPGA Resource:'''
fpgaResource = fpga_resource
'''Indicators:'''
DACQuWriteTimeout = {'ref': 0x81B2, 'val': ctypes.c_bool(), 'ctr': False}
SPCtrQuWriteTimeout = {'ref': 0x816E, 'val': ctypes.c_bool(), 'ctr': False}
SPerrorCount = {'ref': 0x8162, 'val': ctypes.c_ubyte(), 'ctr': False}
postAccOffsetVoltState = {'ref': 0x815A, 'val': ctypes.c_ubyte(), 'ctr': False}
OutBitsState = {'ref': 0x8132, 'val': ctypes.c_uint(), 'ctr': False}
SPstate = {'ref': 0x8166, 'val': ctypes.c_uint(), 'ctr': False}
measVoltState = {'ref': 0x81AE, 'val': ctypes.c_uint(), 'ctr': False}
seqState = {'ref': 0x819A, 'val': ctypes.c_uint(), 'ctr': False}
nOfCmdsOutbit0 = {'ref': 0x812C, 'val': ctypes.c_ulong(), 'ctr': False}
nOfCmdsOutbit1 = {'ref': 0x8128, 'val': ctypes.c_ulong(), 'ctr': False}
nOfCmdsOutbit2 = {'ref': 0x8124, 'val': ctypes.c_ulong(), 'ctr': False}
'''Controls:'''
VoltOrScaler = {'ref': 0x810E, 'val': ctypes.c_bool(), 'ctr': True}
abort = {'ref': 0x81A2, 'val': ctypes.c_bool(), 'ctr': True}
halt = {'ref': 0x81A6, 'val': ctypes.c_bool(), 'ctr': True}
hostConfirmsHzOffsetIsSet = {'ref': 0x81AA, 'val': ctypes.c_bool(), 'ctr': True}
invertScan = {'ref': 0x8182, 'val': ctypes.c_bool(), 'ctr': True}
pause = {'ref': 0x813A, 'val': ctypes.c_bool(), 'ctr': True}
softwareTrigger = {'ref': 0x8136, 'val': ctypes.c_bool(), 'ctr': True}
softwarescanTrigger = {'ref': 0x8112, 'val': ctypes.c_bool(), 'ctr': True}
stopVoltMeas = {'ref': 0x813E, 'val': ctypes.c_bool(), 'ctr': True}
timedOutWhileHandshake = {'ref': 0x8196, 'val': ctypes.c_bool(), 'ctr': True}
postAccOffsetVoltControl = {'ref': 0x815E, 'val': ctypes.c_ubyte(), 'ctr': True}
scanTriggerEdge = {'ref': 0x8116, 'val': ctypes.c_ubyte(), 'ctr': True}
selectScanTrigger = {'ref': 0x811E, 'val': ctypes.c_ubyte(), 'ctr': True}
selectTrigger = {'ref': 0x8152, 'val': ctypes.c_ubyte(), 'ctr': True}
triggerEdge = {'ref': 0x814A, 'val': ctypes.c_ubyte(), 'ctr': True}
cmdByHost = {'ref': 0x819E, 'val': ctypes.c_uint(), 'ctr': True}
measVoltCompleteDest = {'ref': 0x8142, 'val': ctypes.c_uint(), 'ctr': True}
scanTriggerTypes = {'ref': 0x8122, 'val': ctypes.c_uint(), 'ctr': True}
triggerTypes = {'ref': 0x8156, 'val': ctypes.c_uint(), 'ctr': True}
dacStartRegister18Bit = {'ref': 0x8174, 'val': ctypes.c_long(), 'ctr': True}
dacStepSize18Bit = {'ref': 0x8170, 'val': ctypes.c_long(), 'ctr': True}
measVoltPulseLength25ns = {'ref': 0x818C, 'val': ctypes.c_long(), 'ctr': True}
measVoltTimeout10ns = {'ref': 0x8190, 'val': ctypes.c_long(), 'ctr': True}
nOfScans = {'ref': 0x817C, 'val': ctypes.c_long(), 'ctr': True}
nOfSteps = {'ref': 0x8178, 'val': ctypes.c_long(), 'ctr': True}
dac0VRegister = {'ref': 0x8144, 'val': ctypes.c_ulong(), 'ctr': True}
dwellTime10ns = {'ref': 0x8168, 'val': ctypes.c_ulong(), 'ctr': True}
scanTrigDelay10ns = {'ref': 0x8118, 'val': ctypes.c_ulong(), 'ctr': True}
trigDelay10ns = {'ref': 0x814C, 'val': ctypes.c_ulong(), 'ctr': True}
waitAfterResetus = {'ref': 0x8188, 'val': ctypes.c_ulong(), 'ctr': True}
waitForKepcous = {'ref': 0x8184, 'val': ctypes.c_ulong(), 'ctr': True}
'''TargetToHostFifos:'''
transferToHost = {'ref': 0, 'val': ctypes.c_ulong(), 'ctr': False}
'''HostToTargetFifos:'''
OutbitsCMD = {'ref': 1, 'val': ctypes.c_ulong(), 'ctr': False}


'''hand filled values:'''
SPstateDict = {'idle': 0, 'scanning': 1, 'done': 2, 'error': 3}
seqStateDict = {'init': 0, 'idle': 1, 'measureOffset': 2, 'measureTrack': 3, 'measComplete': 4, 'error': 5}
postAccOffsetVoltStateDict = {'Kepco': 0, 'Heinzinger1': 1, 'Heinzinger2': 2, 'Heinzinger3': 3, 'loading': 4}
measVoltCompleteDestStateDict = {'PXI_Trigger_4': 0, 'Con1_DIO30': 1,
                                 'Con1_DIO31': 2, 'PXI_Trigger_4_Con1_DIO30': 3,
                                 'PXI_Trigger_4_Con1_DIO31': 4, 'PXI_Trigger_4_Con1_DIO30_Con1_DIO31': 5,
                                 'Con1_DIO30_Con1_DIO31': 6, 'software': 7}

transferToHostReqEle = 100000

seq_type = 'cs'
