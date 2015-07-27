__author__ = 'noertert'

import ctypes

'''Bitfile Signature:'''
bitfileSignature = '2702394610028D5D0D33227747280129'
'''Bitfile Path:'''
bitfilePath = 'D:\\Workspace\\PyCharm\\Tilda\\TildaTarget\\bin\\HeinzingerAndKepcoTest\\NiFpga_HeinzingerAndKepcoTest_v101.lvbitx'
'''FPGA Resource:'''
fpgaResource = 'Rio1'
'''Indicators:'''
DacState = {'ref': 0x8112, 'val': ctypes.c_uint(), 'ctr': False}
actDACRegister = {'ref': 0x8114, 'val': ctypes.c_ulong(), 'ctr': False}
'''Controls:'''
heinzingerControl = {'ref': 0x811E, 'val': ctypes.c_ubyte(), 'ctr': True}
DacStateCmdByHost = {'ref': 0x811A, 'val': ctypes.c_uint(), 'ctr': True}
setDACRegister = {'ref': 0x810C, 'val': ctypes.c_ulong(), 'ctr': True}

'''Zustandsnummerierung Labview'''
hsbDict = {'Kepco': 0, 'Heinzinger1': 1, 'Heinzinger2': 2, 'Heinzinger3': 3, 'loading': 4}
dacStatesDict = {'init': 0, 'idle': 1, 'setVolt': 2, 'error': 3}
