'''
Created on 08.08.2014

@author: simkaufm
'''
import ctypes

dll = ctypes.CDLL('D:\Workspace\CAPITests\DMABSPTest\Debug\Example.dll')

# test2 = (ctypes.c_int*3)()
# print(ctypes.byref(test2))
# dll.blub(ctypes.byref(test2))
# print(test2[0])

dmaCts = (ctypes.c_int*2)()
print(ctypes.byref(dmaCts))
status = dll._init()
print('Status: ' + str(status))
session = dll._openFPGA()
print('aktuelle Session: ' + str(session))
rate = dll._rate(session, 100)
status = dll._runFPGA(session)
print('Repititionsrate auf  ' + str(rate) +' ms gesetzt, Running ... Status: '+str(status))
 
# print(dll._checkTemp(session))
while True:
    try:
        inp = input('Befehl eingeben:')
        if inp == "go":
            status = dll._LoopGo(session)
            print("starting next Loop, Status:" + str(status))
        elif inp == "cts":         
            actCts = dll._actCts(session)
            print('cts:  ' + str(actCts))
        elif inp == "write":
            status = dll._writeDMA(session) 
            print('Writing DMA, Status: ' + str(status))
            mess = dll._measTime(session)
            print('gelaufene Mess Zeit in Ticks:' + str(mess) )
            messtinS = mess*25*10**-9
            print('gelaufene Mess Zeit in s:' + str( round(messtinS, 2) ) )
            print('expected cts:' + str( round( 1/(rate*2*10**-3)*messtinS, 0 ) ))
        elif inp == 'read':
            nOfele = dll.DMAnOfEle(session)
            print(str(nOfele) + '  Elements in Queue')
            dll.readDMA(session, nOfele, ctypes.byref(dmaCts))
            for i in range(0, nOfele, 2):
                print('DMA cts:  ' + str(dmaCts[i]))
                print('Measurement time:  ' + str(dmaCts[i+1]))
        elif inp == 'status':
            print('Status:  ' + str(dll.getStatus()))
        elif inp == "exit":
            break
             
        else: 
            True  
    except:
        True
 
status = dll._fpgaexit(session)
print('Exiting Program, Status: ' + str(status))

