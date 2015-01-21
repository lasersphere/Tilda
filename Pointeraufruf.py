'''
Created on 08.08.2014

@author: simkaufm
'''
import ctypes
import time

import source.Base.Formating as form

dll = ctypes.CDLL('D:\Workspace\Eclipse\Tilda\TildaTarget\SimpleCounter\SimpleCounter.dll')




status = dll._init()
print('Status: ' + str(status))
session = dll._openFPGA()
print('aktuelle Session: ' + str(session))
 
status = dll._runFPGA(session)
print('running FPGA, Status is:' + str(status))
time.sleep(0.045)
nOfEle = dll.DMAnOfEle(session)
print('Number of Elements in the Queue: ' + str(nOfEle))
time.sleep(0.045)
status = dll._stop(session)
print('Loop stopped, Status is:' + str(status))
print('Number of Elements still in the Queue: ' + str( dll.DMAnOfEle(session)))

start = time.clock()

# trying it with pointer now
# Numbers for pmts greater 7 dont make any sense, not working. 

pDmaCts = ctypes.cast((ctypes.c_ulong * 1)(), ctypes.POINTER(ctypes.c_ulong))

dll.readDMA(session, nOfEle, pDmaCts)
print(form.headunfold(pDmaCts[16]))


dll.readDMA(session, nOfEle, pDmaCts)
# print([form.headunfold(pDmaCts[i]) for i in range(nOfEle)])



# time.sleep(0.045)

# nOfEle = dll.DMAnOfEle(session)
# print('Number of Elements still in the Queue: ' + str(nOfEle))
# 
# # dll.readDMA(session, nOfEle, ctypes.byref(dmaCts))
# print('here we go')
# print(dmaCts)
# print(dmaCts[0], '{0:032b}'.format(dmaCts[0]))
# print(dmaCts[1], '{0:032b}'.format(dmaCts[1]))
# print(dmaCts[2], '{0:032b}'.format(dmaCts[2]))
# print(dmaCts[15], '{0:032b}'.format(dmaCts[15]))
# 
# dll.readDMA(session, nOfEle, ctypes.byref(dmaCts))
# 
# print('here we go again')
# print(dmaCts)
# print(dmaCts[0], '{0:032b}'.format(dmaCts[0]))
# print(dmaCts[1], '{0:032b}'.format(dmaCts[1]))
# print(dmaCts[2], '{0:032b}'.format(dmaCts[2]))
# print(dmaCts[15], '{0:032b}'.format(dmaCts[15]))


# while True:

#     try:
# #         print(bin(dmaCts[0:nOfEle]))
#         for i in range(0, nOfEle, 1):
#             print(bin(dmaCts[i]))
# #             print('PMT: ' + str(bin(dmaCts[i])[2:5]) + '\t cts/20ms: ' + str(bin(dmaCts[i])[7:]))
#         else: 
#             True  
#     except:
#         True
#            
status = dll._fpgaexit(session)

print('Exiting Program, Status: ' + str(status))

end = time.clock()

print('execution time:')
print (end - start)

# print('Repititionsrate auf  ' + str(rate) +' ms gesetzt, Running ... Status: '+str(status))
#  
# # print(dll._checkTemp(session))
# while True:
#     try:
#         inp = input('Befehl eingeben:')
#         if inp == "go":
#             status = dll._LoopGo(session)
#             print("starting next Loop, Status:" + str(status))
#         elif inp == "cts":         
#             actCts = dll._actCts(session)
#             print('cts:  ' + str(actCts))
#         elif inp == "write":
#             status = dll._writeDMA(session) 
#             print('Writing DMA, Status: ' + str(status))
#             mess = dll._measTime(session)
#             print('gelaufene Mess Zeit in Ticks:' + str(mess) )
#             messtinS = mess*25*10**-9
#             print('gelaufene Mess Zeit in s:' + str( round(messtinS, 2) ) )
#             print('expected cts:' + str( round( 1/(rate*2*10**-3)*messtinS, 0 ) ))
#         elif inp == 'read':
#             nOfele = dll.DMAnOfEle(session)
#             print(str(nOfele) + '  Elements in Queue')
#             dll.readDMA(session, nOfele, ctypes.byref(dmaCts))
#             for i in range(0, nOfele, 2):
#                 print('DMA cts:  ' + str(dmaCts[i]))
#                 print('Measurement time:  ' + str(dmaCts[i+1]))
#         elif inp == 'status':
#             print('Status:  ' + str(dll.getStatus()))
#         elif inp == "exit":
#             break
#              
#         else: 
#             True  
#     except:
#         True
#  


