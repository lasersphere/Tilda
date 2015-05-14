'''
Created on 20.11.2014

@author: noertert
'''
import ctypes
print('Hello World!')

DACQuWriteTimeout = {'ref': 0x8116, 'val': ctypes.c_bool(), 'ctr': False}

def func(dict):
    print(dict['val'].value)
    print(dict['ref'])
    
func(DACQuWriteTimeout)


print(len((ctypes.c_ulong * 0)()))
print(int('00000010000000000000', 2))
print(int('00000000000001000000', 2))