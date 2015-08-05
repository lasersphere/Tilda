"""

Created on '30.06.2015'

@author:'simkaufm'

"""

"""
module for testing Heinzinger Switch Box and the DAC/Kepco
"""

import Driver.Heinzinger.Heinzinger as hz
import Driver.Heinzinger.HeinzingerCfg as hzCfg
import Driver.DataAcquisitionFpga.HeinzingerAndKepcoTestConfig as hatCfg
import Driver.DataAcquisitionFpga.HeinzingerAndKepcoTest as fpgaDAC
import Service.Formating as form
import time
import logging
import sys

logging.basicConfig(level=getattr(logging, 'DEBUG'), format='%(message)s', stream=sys.stdout)


hz1 = hz.Heinzinger(hzCfg.comportHeinzinger0) #dacStartRegister18Bit Heinzinger 1
hz2 = hz.Heinzinger(hzCfg.comportHeinzinger1) #dacStartRegister18Bit Heinzinger 2
fpga = fpgaDAC.HsbAndDac()



def readHeinzinger():
    retDict = {'hz1': 0, 'hz2': 0}
    try:
        hz1Volt = hz1.getVoltage()
    except:
        hz1Volt = None
    try:
        hz2Volt = hz2.getVoltage()
    except:
        hz2Volt = None
    retDict.update(hz1=hz1Volt, hz2=hz2Volt)
    return retDict

def readFpga():
    actDaqReg = fpga.readActDacReg()
    retDict = {'DacState': fpga.readDacState(),
               'actDacReg': actDaqReg,
               'actVolt': form.getVoltageFrom24Bit(actDaqReg),
               'fpgaState': fpga.checkFpgaStatus()}
    return retDict

def setDacVolt(volt):
    timeout = 0
    if volt < -10:
        volt = -10
    elif volt > 10:
        volt = 10
    regVal = form.get24BitInputForVoltage(volt)
    fpga.setDacRegister(regVal)
    fpga.setDacState('setVolt')
    while fpga.readDacState() != hatCfg.dacStatesDict.get('idle') and timeout < 100:
        time.sleep(0.01)
        timeout += 1
    fpga.setDacState(fpga.readDacState())
    return regVal


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
        sta = setDacVolt(float(volt))
        print('voltage set, voltage is: ' + str(sta) + ' DAC reg or ' + str(form.getVoltageFrom24Bit(sta)) + ' Volt')
    elif eingabe == 'hv':
        hznumber = input('enter Heinzinger Number (1 or 2): ')
        try:
            hznumber = int(hznumber)
            if hznumber in [1, 2]:
                volt = input('please enter desired voltage: ')
                try:
                    volt = float(volt)
                    if 0.0 <= volt <= 10000.0:
                        if int(hznumber) == 1:
                            hz1.setVoltage(float(volt))
                        elif int(hznumber) == 2:
                            hz2.setVoltage(float(volt))
                    else:
                        print(str(volt), ' is out of range')
                except Exception as ex:
                    print(str(ex))
        except Exception as excep:
            print(str(excep))
    elif eingabe == 'hout':
        try:
            anaus = int(input('Output an oder Aus? 0/1:' ))
            hz1.setOutput(bool(anaus))
            hz2.setOutput(bool(anaus))
        except Exception as excep:
            print(str(excep))
    elif eingabe == 'hsb':
        hsbdevice = input('please enter name of desired output(Kepco, Heinzinger1, Heinzinger2, Heinzinger3) or number in list: ')
        fpga.setHsb(hsbdevice)
    elif eingabe == 'cli':
        print('Command ' + eingabe + ' ...  entering CLI mode:')
        neueeingabe = ''
        try:
            neueeingabe = input('Enter Command (CLI-Mode): ')
            print(eval(neueeingabe))
        except:
            print('Command ' + neueeingabe + ' not accepted')
try:
    hz1.deinit()
except Exception as e:
    logging.debug(str(e))
try:
    hz2.deinit()
except Exception as e:
    logging.debug(str(e))
print(fpga.DeInitFpga())