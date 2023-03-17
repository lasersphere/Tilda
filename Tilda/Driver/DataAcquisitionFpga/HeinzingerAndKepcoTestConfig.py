__author__ = 'noertert'

from os import path, pardir
import ctypes

'''Bitfile Signature:'''
bitfileSignature = '07925F2E9ECF4BF922BD7AEDD0A076D3'
'''Bitfile Path:'''
bitfilePath = path.join(path.dirname(__file__), pardir, pardir,
                        'TildaTarget/bin/HeinzingerAndKepcoTest/NiFpga_HeinzingerAndKepcoTest_v102.lvbitx')
'''FPGA Resource:'''
fpgaResource = 'Rio1'
'''Indicators:'''
postAccOffsetVoltState = {'ref': 0x8112, 'val': ctypes.c_ubyte(), 'ctr': False}
DacState = {'ref': 0x811A, 'val': ctypes.c_uint(), 'ctr': False}
actDACRegister = {'ref': 0x811C, 'val': ctypes.c_ulong(), 'ctr': False}
'''Controls:'''
postAccOffsetVoltControl = {'ref': 0x8116, 'val': ctypes.c_ubyte(), 'ctr': True}
DacStateCmdByHost = {'ref': 0x8122, 'val': ctypes.c_uint(), 'ctr': True}
setDACRegister = {'ref': 0x810C, 'val': ctypes.c_ulong(), 'ctr': True}

'''Zustandsnummerierung Labview'''
hsbDict = {'Kepco': 0, 'Heinzinger1': 1, 'Heinzinger2': 2, 'Heinzinger3': 3, 'loading': 4}
dacStatesDict = {'init': 0, 'idle': 1, 'setVolt': 2, 'error': 3}
