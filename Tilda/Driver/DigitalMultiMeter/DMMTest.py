"""

Created on '24.05.2016'

@author:'simkaufm'


Module for a quick test with the Multimeter

"""

import ctypes

dll_path = '../../Binary/nidmm_32.dll'
dev_str = 'PXI1Slot5'
dev_name = ctypes.create_string_buffer(dev_str.encode('utf-8'))
print(type(dev_name))

dll = ctypes.WinDLL(dll_path)
session = ctypes.c_uint32(0)
# vi_true = ctypes.c_int16(1)
vi_true = ctypes.c_bool(True)
vi_false = ctypes.c_int16(0)
error_buf_size = ctypes.c_int32(1024)
ret_backlog = ctypes.c_int32()
ret_acqstate = ctypes.c_int16()
# ctypes.c_dou

error_buf = ctypes.create_string_buffer('buffer'.encode('utf-8'), 256)
dll.niDMM_init(dev_name, vi_true, vi_true, ctypes.byref(session))
dll.niDMM_ReadStatus(session, ctypes.byref(ret_backlog), ctypes.byref(ret_acqstate))

print(ret_backlog.value, ret_acqstate.value)
# print(session)

dll.niDMM_close(session)
