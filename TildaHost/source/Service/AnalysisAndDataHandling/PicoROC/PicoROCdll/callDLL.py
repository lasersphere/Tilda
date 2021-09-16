#!/usr/bin/env python3

import ctypes
import os
import time
#os.add_dll_directory(os.getcwd())

PicoROC = ctypes.CDLL('C:\\Users\\MBissell\\picoroc\\PicoROCdll\\PicoROCdll.dll')

PicoROC.attachSharedMemory()

fileName = "PythonIsReallyCool"
s = fileName.encode('utf-8')

PicoROC.stepScan.restype = None
PicoROC.startScan.restype = None
PicoROC.startScan.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_char_p]
PicoROC.stepScan.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_int]

PicoROC.startScan(0,5.0,0,s)
time.sleep(25)
PicoROC.stepScan(1,6,0)
time.sleep(25)
PicoROC.stepScan(2,7,0)
time.sleep(25)
PicoROC.stepScan(3,8,0)
time.sleep(25)
PicoROC.stepScan(4,15,0)
time.sleep(25)
PicoROC.stopScan()
PicoROC.detachSharedMemory()