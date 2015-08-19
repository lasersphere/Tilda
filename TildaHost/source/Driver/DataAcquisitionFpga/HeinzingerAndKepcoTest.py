__author__ = 'noertert'


from Driver.DataAcquisitionFpga.FPGAInterfaceHandling import FPGAInterfaceHandling
import Driver.DataAcquisitionFpga.HeinzingerAndKepcoTestConfig as hat
import logging
import time

class HsbAndDac(FPGAInterfaceHandling):

    def __init__(self):
        self.setDacReg = 0
        self.actDacReg = 0
        self.dacState = -1
        super(HsbAndDac, self).__init__(hat.bitfilePath, hat.bitfileSignature,
                                                        hat.fpgaResource)

# '''read indicators'''
    def readDacState(self):
        self.dacState = self.ReadWrite(hat.DacState).value
        return self.dacState

    def readActDacReg(self):
        self.actDacReg = self.ReadWrite(hat.actDACRegister).value
        return self.actDacReg

# '''write controls'''
    def setHsb(self, deviceStr):
        """
        function to control which Heinzinger/Kepco is set by the Heinzinger Switch Box (HSB)
        :param deviceStr: str, naming the desired device as stated in HeinzingerAndKepcoTestConfig.py
        :return: fpga state
        """
        deviceNumber = None
        try:
            deviceNumber = int(deviceStr)
        except:
            try:
                deviceNumber = hat.hsbDict(deviceStr)
            except KeyError:
                logging.debug('key: ' + deviceStr + ' not found.')
        if deviceNumber != None:
            self.ReadWrite(hat.postAccOffsetVoltControl, deviceNumber)
        read = self.ReadWrite(hat.postAccOffsetVoltState).value
        while deviceNumber != read:
            read = self.ReadWrite(hat.postAccOffsetVoltState).value
            print(read)
            time.sleep(0.25)
        return self.checkFpgaStatus()

    def setDacState(self, stateStr):
        """
        function to set the state of the DAC
        :param stateStr: str, naming the desired state
        :return: fpga state
        """
        try:
            stateNumber = int(stateStr)
        except:
            stateNumber = hat.dacStatesDict.get(stateStr)
        if stateNumber != None:
            self.ReadWrite(hat.DacStateCmdByHost, stateNumber)
        return self.checkFpgaStatus()

    def setDacRegister(self, regVal):
        """
        function to set the value of the DAC register
        :param regVal: int
        :return: fpga state
        """
        self.ReadWrite(hat.setDACRegister, regVal)
        return self.checkFpgaStatus()