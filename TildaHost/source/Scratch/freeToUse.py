"""

Created on '22.06.2015'

@author:'simkaufm'

"""
from functools import wraps
from threading import Thread, Event, Lock
from datetime import datetime
import logging, time
import numpy as np
import threading
import time
import datetime
# import pickle
# import Service.FolderAndFileHandling as fileh
# import os
# import Service.draftScanParameters as draftScan
import Service.Formating as form

# path = 'D:\\Workspace\\Testdata'
# # print(path)
# file = 'D:\Workspace\Testdata\\raw\\20150622_185322_trs_simontium_27_track0_0.raw'
# # print(fileh.nameFile(path,'raw', '20150622_183238_trs_', 'simontium_27'))
# # data = pickle.load(open(file, 'rb'))
# # print(data, type(data), data[0])
#
# # for nfile in os.listdir(os.path.split(file)[0]):
# #     print(nfile)
# #     if nfile.endswith('.raw'):
# #         print(nfile)
# #         print(fileh.loadPickle(os.path.join(path, 'raw', nfile)))
#
# # print(fileh.createXmlFileOneIsotope(draftScan.draftScanDict))
#
# ele = fileh.loadXml('D:\Wordkspace\Testdata\sums\\20150623_114059_trs_sum_simontium_27.xml')
#
#
# dicti = draftScan.draftTrackPars
# form.xmlAddCompleteTrack(ele, dicti, 1)

# print(512 >> 2)
# print(8192 >> 2)
#
# class printer():
#     def __init__(self):
#         print('hi There')
#
#     def druck(self):
#         return 'Hello World!'

# volt = 5.000058
# print(volt)
# bla = form.get24BitInputForVoltage(volt)
# print(format(bla, '024b'))
# blub = form.getVoltageFrom24Bit(bla)
# print(blub)
# print(volt-blub)
# print(20/(2 ** 18))

# def hello():
#     print(datetime.datetime.now())
#     lala = input('Type here')
#     print(lala)
#     threading.Timer(1.0, hello).start()
# hello()
#
# _timer = Event()
# interval = 1
# _timer.set()
#
#
#
# def periodic():
#     te = input('hier tippen: ')
#     print(te)
#
# def run():
#     while interval > 0:
#         periodic()
#         if _timer.wait(interval):
#             _timer.clear()
#
#
# _thread = Thread(target= run)
# _thread.start()


x1 = np.arange(9.0).reshape((3, 3))
x2 = np.arange(3.0)
print(x1, x2)
print(np.add(x1, x2))