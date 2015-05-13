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

print([i for i in range(1)])

print((ctypes.c_ulong * 1)()[0])
print(ctypes.c_long())
print(int(0).value < 1)