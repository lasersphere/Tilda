"""
Created on 21.01.2015

@author: skaufmann
"""

import ctypes
import time

import Scratch.Formating as form


dll = ctypes.CDLL('D:\Workspace\Eclipse\Tilda\TildaTarget\SimpleCounter\SimpleCounter.dll')

status = dll._init()
print('Status: ' + str(status))
session = dll._openFPGA()
print('aktuelle Session: ' + str(session))
 
status = dll._runFPGA(session)
print('running FPGA, Status is:' + str(status))
time.sleep(0.1)
nOfEle = dll.DMAnOfEle(session)
print('Number of Elements in the Queue: ' + str(nOfEle))
time.sleep(0.1)
status = dll._stop(session)
print('Loop stopped, Status is:' + str(status))
print('Number of Elements still in the Queue: ' + str( dll.DMAnOfEle(session)))

start = time.clock()

# Problem, da groesse vom Array festgelegt werden muss!?
#  Auszulesende Anzahl muss aber so oder so jedes mal festgelegt werden
# bei jedem neuen auslesen wird an den Anfang des Arrays geschrieben. 
# Append ist so nicht moeglich und Daten muessen in anderes Array kopiert werden.

dmaCtsFull = []
dmaCts = (ctypes.c_ulong * nOfEle)()
  
dll.readDMA(session, nOfEle, ctypes.byref(dmaCts))
dmaCtsFull = dmaCtsFull + [form.headunfold(dmaCts[i]) for i in range(nOfEle)]
print(dmaCtsFull)

dll.readDMA(session, nOfEle, ctypes.byref(dmaCts))
dmaCtsFull = dmaCtsFull + [form.headunfold(dmaCts[i]) for i in range(nOfEle)]
print(dmaCtsFull)

status = dll._fpgaexit(session)

print('Exiting Program, Status: ' + str(status))

end = time.clock()

print('execution time:')
print (end - start)

