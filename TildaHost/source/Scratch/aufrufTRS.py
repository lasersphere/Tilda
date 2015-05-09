"""

Created on '08.05.2015'

@author:'simkaufm'

"""
'''
Created on 08.08.2014

@author: skaufmann
'''
import ctypes


dll = ctypes.CDLL('D:\Workspace\Eclipse\Tilda\TildaHost\\binary\TRS.dll')

'''
first using TRWwrapper.c commands
'''

# 
# status = dll.init()
# print('Status: ' + str(status))
# session = dll.openFPGA()
# print('aktuelle Session: ' + str(session))
#  
# status = dll.runFPGA(session)
# print('running FPGA, Status is:' + str(status))
 


'''
now just using the standard NiFpga.c/.h Commands!
'''
NiFpga_TRS_IndicatorU16_seqState = {'ref': 0x8146, 'val': ctypes.c_uint()}
NiFpga_TRS_ControlU16_cmdByHost= {'ref': 0x8142, 'val': ctypes.c_uint()}
status = dll.NiFpga_Initialize()
print(status) 
dllPath = ctypes.create_string_buffer(b'D:\\Workspace\\Eclipse\\Tilda\\TildaTarget\\bin\\TimeResolvedSequencer\\NiFpga_TRS.lvbitx')
dllSign = ctypes.create_string_buffer(b'BF31570369009FA00617B7055FD697C8')
resource = ctypes.create_string_buffer(b'Rio1')
session = ctypes.c_ulong()
print(session)
status = dll.NiFpga_Open(dllPath, dllSign, resource, 1, ctypes.byref(session))
print(status)
print(session)
status = dll.NiFpga_Run(session, 0)
print(status)
status = dll.NiFpga_ReadU16(session, NiFpga_TRS_IndicatorU16_seqState['ref'], ctypes.byref(NiFpga_TRS_IndicatorU16_seqState['val']))
print(status)
print('seqState:' + str(NiFpga_TRS_IndicatorU16_seqState['val']))
status = dll.NiFpga_WriteU16(session, NiFpga_TRS_ControlU16_cmdByHost['ref'], 4)
print(status)
status = dll.NiFpga_ReadU16(session, NiFpga_TRS_IndicatorU16_seqState['ref'], ctypes.byref(NiFpga_TRS_IndicatorU16_seqState['val']))
print(status)
print(NiFpga_TRS_IndicatorU16_seqState['val'])
status = dll.NiFpga_ReadU16(session, NiFpga_TRS_IndicatorU16_seqState['ref'], ctypes.byref(NiFpga_TRS_IndicatorU16_seqState['val']))
print(status)
print(NiFpga_TRS_IndicatorU16_seqState['val'])
