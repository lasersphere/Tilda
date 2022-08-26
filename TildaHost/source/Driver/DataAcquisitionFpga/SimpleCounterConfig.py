"""
Created on 

@author: simkaufm

Module Description: automatically created with the CApiAnalyser
"""

import ctypes
from os import path, pardir

import Service.FileOperations.FolderAndFileHandling as FileHandl

fpga_cfg_root, fpga_cfg_dict = FileHandl.load_fpga_xml_config_file()
data_acq_cfg = fpga_cfg_dict['fpgas']['data_acquisition_fpga']
fpga_type = data_acq_cfg['fpga_type']
fpga_resource = data_acq_cfg['fpga_resource']

'''Bitfile Signature:'''
bitfileSignatures = {
    'PXI-7852R': 'C2105F6862AC5608FFE4B53B62D561B4',
    'PXI-7841R': 'E87F746B4D75FAC83D98D9825FB513AE'
                     }
bitfileSignature = bitfileSignatures[fpga_type]
'''Bitfile Path:'''
bitfilePaths = {
    'PXI-7852R': path.join(path.dirname(__file__), pardir, pardir, pardir, pardir,
                           'TildaTarget/bin/SimpleCounter/NiFpga_SimpleCounterV262.lvbitx'),
    'PXI-7841R': path.join(path.dirname(__file__), pardir, pardir, pardir, pardir,
                           'TildaTarget/bin/SimpleCounter/NiFpga_SimpleCounter_7841_v200.lvbitx')
                }
bitfilePath = bitfilePaths[fpga_type]
'''FPGA Resource:'''
fpgaResource = 'Rio1'
'''Indicators:'''
postAccOffsetVoltState = {'ref': 0x813E, 'val': ctypes.c_ubyte(), 'ctr': False}
DacState = {'ref': 0x812A, 'val': ctypes.c_uint(), 'ctr': False}
actDACRegister = {'ref': 0x812C, 'val': ctypes.c_ulong(), 'ctr': False}
'''Controls:'''
StartSignal = {'ref': 0x8112, 'val': ctypes.c_bool(), 'ctr': True}
softwareTrigger = {'ref': 0x811A, 'val': ctypes.c_bool(), 'ctr': True}
postAccOffsetVoltControl = {'ref': 0x813A, 'val': ctypes.c_ubyte(), 'ctr': True}
selectTrigger = {'ref': 0x8126, 'val': ctypes.c_ubyte(), 'ctr': True}
triggerEdge = {'ref': 0x811E, 'val': ctypes.c_ubyte(), 'ctr': True}
DacStateCmdByHost = {'ref': 0x8136, 'val': ctypes.c_uint(), 'ctr': True}
triggerTypes = {'ref': 0x810E, 'val': ctypes.c_uint(), 'ctr': True}
dwellTime10ns = {'ref': 0x8114, 'val': ctypes.c_ulong(), 'ctr': True}
setDACRegister = {'ref': 0x8130, 'val': ctypes.c_ulong(), 'ctr': True}
trigDelay10ns = {'ref': 0x8120, 'val': ctypes.c_ulong(), 'ctr': True}
'''TargetToHostFifos:'''
transferToHost = {'ref': 0, 'val': ctypes.c_ulong(), 'ctr': False}
'''HostToTargetFifos:'''


''' Hand filled '''
transferToHostReqEle = 50000

dacState = {'init': 0, 'idle': 1, 'setVolt': 2, 'error': 3}
postAccOffsetVoltStateDict = {'Kepco': 0, 'Heinzinger1': 1, 'Heinzinger2': 2, 'Heinzinger3': 3, 'loading': 4}

seq_type = 'smplCnt'
