"""

Created on '19.05.2015'

@author:'simkaufm'

"""

import logging
import time

import serial

import Tilda.Driver.Heinzinger.HeinzingerCfg as hzCfg


class Heinzinger():
    def __init__(self, com, name='Heinzinger'):
        self.max_readback_time = 1  # time in seconds
        self.errorcount = 0
        self.name = name
        self.outp = False
        self.setCur = 0
        self.maxVolt = hzCfg.maxVolt
        self.setVolt = 0
        self.time_of_last_volt_set = None
        self.sleepAfterSend = 0.05
        logging.info('connecting to %s on com port: %s' % (self.name, str(com)))
        self.idn = ''
        self.ser = serial.Serial(port=com - 1, baudrate=9600, timeout=0.1,
                                 parity='N', stopbits=1, bytesize=8, xonxoff=True,
                                 rtscts=False)

        try:
            self.reset()
            self.idn = str(self.serWrite('*IDN?', True))
            if self.idn != str(None):
                logging.info('%s initialized on Com: %s' % (str(self.idn), str(com)))
                self.setAverage(1)
                self.setOutput(True)
                # self.setVoltage(0)   # not absolutely necessary
                self.setCurrent(hzCfg.currentWhenTurnedOn)
        except OSError:
            self.errorcount += 1
            logging.error('error occurred in %s, error count is: %s' % (self.name, str(self.errorcount)))

    def reset(self):
        """
        reset the devices interface, when the serial connection still can be established.
        """
        self.serWrite('*RST')

    def deinit(self):
        """
        deinitialize the heinzinger
        :return: int, Errorcount which is the number of Errors that occured during operation.
        The Errorcount is raised when serial connection fails.
        """
        self.setVoltage(0)
        self.setOutput(False)
        self.ser.close()
        logging.debug(str(self.errorcount) + ' Errors occured')
        return self.errorcount

    '''set Values'''
    def setVoltage(self, volt):
        """
        sets the ouput voltage, if volt <= maxVolt in Config
        :param volt: float, 3 Digits of precision
        :return: float, the voltage that has ben sent via serial
        """
        logging.info('%s setting Volt: ' % (self.name, str(volt)))
        if abs(volt) <= self.maxVolt:
            self.setVolt = round(float(volt), 3)
        self.serWrite('SOUR:VOLT ' + str(abs(self.setVolt)))
        self.time_of_last_volt_set = time.strftime('%d/%m/%y %H:%M:%S')
        return self.setVolt

    def setCurrent(self, curr):
            """
            sets the Current
            :param curr: float, 3 digits of precision
            :return: float, the set Current
            """
            self.setCur = round(float(curr), 3) #heinzinger needs float
            self.serWrite('SOUR:CURR ' + str(self.setCur))
            return self.setCur

    def setOutput(self, out):
        """
        Turn Output on or Off
        :param out: bool, True for output on, Fale, for Output Off
        :return: bool, the send Output
        """
        self.outp = out
        if self.outp:
            self.serWrite('OUTP ON')
        else:
            self.serWrite('OUTP OFF')
        return self.outp

    def setAverage(self, aver):
        """
        Sets the Average of the Voltage measurements of the Heinzinger.
        Each measurement should need something like 320 ms.
        """
        logging.debug('Setting Average of Voltage measurement to: ' + str(aver))
        return self.serWrite('AVER ' + str(aver))

    '''read values'''
    def getProgrammedVolt(self):
        """
        get the programmed Voltage.
        Each measurement should need something like 320 ms.
        """
        self.setVolt = round(float(self.serWrite('VOLT?', True, 350)), 3)
        return self.setVolt

    def getVoltage(self):
        """
        gets the Voltage which the Heinzinger measures.
        :return: float, the measured Voltage which Heinzinger thinks it has.
        """
        readback = self.serWrite('MEASure:VOLTage?', True)
        logging.debug('readback of Voltage is: ' + str(readback))
        volt = round(float(readback), 3)
        return volt

    def getCurrent(self):
        """
        gets the Current the Heinzinger thinks it applies
        :return: float
        """
        return round(float(self.serWrite('MEASure:CURRent?', True)), 3)

    def get_status(self):
        """
        returns a dict containing the status of the power supply,
        keys are: name, programmedVoltage, voltageSetTime, readBackVolt, output, com
        """
        status = {}
        status['name'] = self.name
        status['programmedVoltage'] = self.setVolt
        status['voltageSetTime'] = self.time_of_last_volt_set
        status['readBackVolt'] = self.getVoltage()  # takes roughly 320 ms
        status['output'] = self.outp
        status['com'] = self.ser.getPort()
        return status

    '''serial interface'''
    def serWrite(self, cmdstr, readback=False, sleepAfterSend=None):
        """
        Function for the serial communication
        :param cmdstr: str, Command String
        :param readback: bool, True if readback is wanted, False, if readback is not wanted
        :return: str, either readback or error
        """
        if sleepAfterSend == None:
            sleepAfterSend = self.sleepAfterSend
        #lock the thread, so that only one method accesses the serial write command at a time
        try:
            self.ser.write(str.encode(cmdstr + '\r\n'))
            time.sleep(sleepAfterSend)
            if readback:
                ret = self.ser.readline()
                time.sleep(sleepAfterSend)
                readbackTimeout = 0
                while ret == b'' and readbackTimeout < self.max_readback_time:
                    ret = self.ser.readline()
                    time.sleep(sleepAfterSend)
                    readbackTimeout += sleepAfterSend
                if ret == b'':
                    logging.debug('Readback timedout after ... tries: ' + str(readbackTimeout))
                    return None
                else:
                    return ret
            else:
                return str.encode(cmdstr +'\r\n')
        except:
            self.errorcount = self.errorcount + 1
            logging.error('error in writing serial in Heinzinger')
            return False