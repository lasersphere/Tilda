"""
Created on 

@author: simkaufm

Module Description:
"""

import ctypes

dll_path = 'C:\\Program Files (x86)\\Sirah\\Matisse Commander\\MC Utility.dll'

dll = ctypes.WinDLL(dll_path)
print(dll.signal_handler())

# ah fuck it ;)
# use pyvisa instead
