"""

Created on '30.06.2015'

@author:'simkaufm'

"""

"""
module for testing Heinzinger Switch Box and the DAC/Kepco
"""

import Driver.Heinzinger.Heinzinger as hz
import Driver.Heinzinger.HeinzingerCfg as hzCfg
import Driver.DataAcquisitionFpga.HeinzingerAndKepcoTest as fpgaDAC
import Service.Formating as form
import Scratch.freeToUse as ft

# hz0 = hz.Heinzinger(hzCfg.comportHeinzinger0) #start Heinzinger 0
# hz1 = hz.Heinzinger(hzCfg.comportHeinzinger1) #start Heinzinger 1
fpga = fpgaDAC.HsbAndDac()
bla = ft.printer()

def readHeinzinger():
    retDict = {'hz0': 0, 'hz1': 0}
    hz0Volt = hz0.getVoltage()
    hz1Volt = hz1.getVoltage()
    retDict.update(hz0=hz0Volt, hz1=hz1Volt)
    return retDict

def readFpga():
    retDict = {'DacState': fpga.readDacState(),
               'actDacReg': fpga.readActDacReg(),
               'fpgaState': fpga.checkFpgaStatus()}
    return retDict

def setDacVolt(volt):
    regVal = form.get24BitInputForVoltage(volt)
    fpga.setDacRegister(regVal)

print('type in q to quit')
eingabe = ''
while eingabe != 'q':
    eingabe = input('Enter Commandstring: ')
    if eingabe == 'hr':
        print(readHeinzinger())
    elif eingabe == 'fr':
        print(readFpga())
    elif eingabe == 'fv':
        volt = input('please enter desired voltage: ')
        setDacVolt(float(volt))
    elif eingabe == 'hv':
        hznumber = input('enter Heinzinger Number: ')
        volt = input('please enter desired voltage: ')
        if hznumber == 0:
            hz0.setVoltage(volt)
        elif hznumber == 1:
            hz1.setVoltage(volt)
    elif eingabe == 'hout':
        anaus = input('Output an oder Aus? 0/1:' )
        hz0.setOutput(bool(anaus))
        hz1.setOutput(bool(anaus))
    else:
        print('Command ' + eingabe + ' not a standard command entering CLI mode:')
        try:
            neueeingabe = input('Enter Command (CLI-Mode): ')
            print(eval(neueeingabe))
        except:
            print('Command ' + neueeingabe + ' not accepted')
print(eingabe)