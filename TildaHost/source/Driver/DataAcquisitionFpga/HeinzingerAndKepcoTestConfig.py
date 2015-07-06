__author__ = 'noertert'

import ctypes

'''Bitfile Signature:'''
bitfileSignature = '65983986BDA85857BB806DE8DBC8A8B6'
'''Bitfile Path:'''
bitfilePath = 'D:\\Workspace\\PyCharm\\Tilda\\TildaTarget\\bin\\TimeResolvedSequencer\\NiFpga_HeinzingerAndKepcoTest.lvbitx'
'''FPGA Resource:'''
fpgaResource = 'Rio1'
'''Indicators:'''
DacState = {'ref': 0x8112, 'val': ctypes.c_uint(), 'ctr': False}
actDACRegister = {'ref': 0x8114, 'val': ctypes.c_ulong(), 'ctr': False}
'''Controls:'''
heinzingerControl = {'ref': 0x811E, 'val': ctypes.c_ubyte(), 'ctr': True}
DacStateCmdByHost = {'ref': 0x811A, 'val': ctypes.c_uint(), 'ctr': True}
setDACRegister = {'ref': 0x810C, 'val': ctypes.c_ulong(), 'ctr': True}
'''TargetToHostFifos:'''


'''Zustandsnummerierung Labview'''
hsbDict = {'Kepco': 0, 'Heinzinger0': 1, 'Heinzinger1': 2, 'Heinzinger2': 3}
dacStatesDict = {'init': 0, 'idle': 1, 'setVolt': 2, 'error': 3}
