"""
Created on 

@author: simkaufm

Module Description: automatically created with the CApiAnalyser
"""

import ctypes
from os import path, pardir

import Tilda.Service.FileOperations.FolderAndFileHandling as FileHandle

fpga_cfg_root, fpga_cfg_dict = FileHandle.load_fpga_xml_config_file()
data_acq_cfg = fpga_cfg_dict['fpgas']['data_acquisition_fpga']
fpga_type = data_acq_cfg['fpga_type']
fpga_resource = data_acq_cfg['fpga_resource']

'''Bitfile Signature:'''
bitfileSignatures = {
    'PXI-7852R': 'C2BD7B62D4E20875CAA99C070B9A034C',
    'PXI-7841R': 'E87F746B4D75FAC83D98D9825FB513AE'
                     }
bitfileSignature = bitfileSignatures[fpga_type]
'''Bitfile Path:'''
bitfilePaths = {
    'PXI-7852R': path.join(path.dirname(__file__), pardir, pardir,
                           'TildaTarget/bin/SimpleCounter/NiFpga_SimpleCounterV251.lvbitx'),
    'PXI-7841R': path.join(path.dirname(__file__), pardir, pardir,
                           'TildaTarget/bin/SimpleCounter/NiFpga_SimpleCounter_7841_v200.lvbitx')
                }
bitfilePath = bitfilePaths[fpga_type]
'''FPGA Resource:'''
fpgaResource = fpga_resource
'''Indicators:'''
postAccOffsetVoltState = {'ref': 0x8122, 'val': ctypes.c_ubyte(), 'ctr': False}
DacState = {'ref': 0x8112, 'val': ctypes.c_uint(), 'ctr': False}
actDACRegister = {'ref': 0x8114, 'val': ctypes.c_ulong(), 'ctr': False}
'''Controls:'''
postAccOffsetVoltControl = {'ref': 0x811E, 'val': ctypes.c_ubyte(), 'ctr': True}
DacStateCmdByHost = {'ref': 0x811A, 'val': ctypes.c_uint(), 'ctr': True}
setDACRegister = {'ref': 0x810C, 'val': ctypes.c_ulong(), 'ctr': True}
'''TargetToHostFifos:'''
transferToHost = {'ref': 0, 'val': ctypes.c_ulong(), 'ctr': False}


''' Hand filled '''
transferToHostReqEle = 50000

dacState = {'init': 0, 'idle': 1, 'setVolt': 2, 'error': 3}
postAccOffsetVoltStateDict = {'Kepco': 0, 'Heinzinger1': 1, 'Heinzinger2': 2, 'Heinzinger3': 3, 'loading': 4}
