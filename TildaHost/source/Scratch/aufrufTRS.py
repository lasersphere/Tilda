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

status = dll.init()
print('Status: ' + str(status))
session = dll.openFPGA()
print('aktuelle Session: ' + str(session))

status = dll.runFPGA(session)
print('running FPGA, Status is:' + str(status))

