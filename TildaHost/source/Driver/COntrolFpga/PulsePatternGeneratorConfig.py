"""

Created on '12.01.2017'

@author:'simkaufm'

"""
import ctypes
from os import path, pardir

import Service.FileOperations.FolderAndFileHandling as FileHandl

fpga_cfg_root, fpga_cfg_dict = FileHandl.load_fpga_xml_config_file()
control_cfg = fpga_cfg_dict['fpgas']['control_fpga']
fpga_type = control_cfg['fpga_type']
fpga_resource = control_cfg['fpga_resource']

'''Bitfile Signature:'''
bitfileSignatures = {'PXI-7852R': 'A9FB914D11C1403B432A70F3CC22D307'
                     }
bitfileSignature = bitfileSignatures[fpga_type]
'''Bitfile Path:'''
bitfilePaths = {'PXI-7852R': path.join(path.dirname(__file__), pardir, pardir, pardir, pardir,
                                       'TildaTarget/bin/PulsePatternGenerator/NiFpga_TiTaProj_COF_PPG_100MHz_V135.lvbitx')
                }
bitfilePath = bitfilePaths[fpga_type]
'''FPGA Resource:'''
fpgaResource = fpga_resource
'''Indicators:'''
fifo_empty = {'ref': 0x814E, 'val': ctypes.c_bool(), 'ctr': False}
start_sctl = {'ref': 0x810E, 'val': ctypes.c_bool(), 'ctr': False}
stop_sctl = {'ref': 0x8156, 'val': ctypes.c_bool(), 'ctr': False}
error_code = {'ref': 0x813A, 'val': ctypes.c_ubyte(), 'ctr': False}
state = {'ref': 0x813E, 'val': ctypes.c_uint(), 'ctr': False}
elements_loaded = {'ref': 0x8120, 'val': ctypes.c_long(), 'ctr': False}
revision = {'ref': 0x8118, 'val': ctypes.c_long(), 'ctr': False}
ticks_per_us = {'ref': 0x8114, 'val': ctypes.c_long(), 'ctr': False}
number_of_cmds = {'ref': 0x8140, 'val': ctypes.c_ulong(), 'ctr': False}
stop_addr = {'ref': 0x8144, 'val': ctypes.c_ulong(), 'ctr': False}
'''Controls:'''
continuous = {'ref': 0x8152, 'val': ctypes.c_bool(), 'ctr': True}
load = {'ref': 0x812E, 'val': ctypes.c_bool(), 'ctr': True}
query = {'ref': 0x811E, 'val': ctypes.c_bool(), 'ctr': True}
replace = {'ref': 0x8126, 'val': ctypes.c_bool(), 'ctr': True}
reset = {'ref': 0x8132, 'val': ctypes.c_bool(), 'ctr': True}
run = {'ref': 0x812A, 'val': ctypes.c_bool(), 'ctr': True}
stop = {'ref': 0x814A, 'val': ctypes.c_bool(), 'ctr': True}
useJump = {'ref': 0x8112, 'val': ctypes.c_bool(), 'ctr': True}
mem_addr = {'ref': 0x8134, 'val': ctypes.c_ulong(), 'ctr': True}
'''TargetToHostFifos:'''
DMA_down = {'ref': 1, 'val': ctypes.c_ulong(), 'ctr': False}
'''HostToTargetFifos:'''
DMA_up = {'ref': 0, 'val': ctypes.c_ulong(), 'ctr': False}

''' hand filled values: '''
transferToHostReqEle = 100000

ppg_state_dict = {'idle': 0, 'reset': 1, 'query': 2,
                  'load': 3, 'run': 4, 'stopped': 5}
