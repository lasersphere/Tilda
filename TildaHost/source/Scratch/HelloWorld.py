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