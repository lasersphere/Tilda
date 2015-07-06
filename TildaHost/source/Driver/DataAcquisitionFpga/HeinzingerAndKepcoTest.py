__author__ = 'noertert'


from Driver.DataAcquisitionFpga.FPGAInterfaceHandling import FPGAInterfaceHandling
import Driver.DataAcquisitionFpga.HeinzingerAndKepcoTestConfig as hat

class HsbAndDac(FPGAInterfaceHandling):

    def __init__(self):
        self.setDacReg = 0
        self.actDacReg = 0
        self.fpgaInst = super(HsbAndDac, self).__init__(hat.bitfilePath, hat.bitfileSignature,
                                                        hat.fpgaResource)

# '''read indicators'''
    def readDacState(self):
        return self.ReadWrite(hat.DacState).value
    def readActDacReg(self):
        return self.ReadWrite(hat.actDACRegister).value

# '''write controls'''
    def setHsb(self, deviceStr):
        """
        function to control which Heinzinger/Kepco is set by the Heinzinger Switch Box (HSB)
        :param deviceStr: str, naming the desired device
        :return: fpga state
        """
        deviceNumber = hat.hsbDict.get(deviceStr)
        if deviceNumber != None:
            self.fpgaInst.ReadWrite(hat.heinzingerControl, deviceNumber)
        return self.checkFpgaStatus()

    def setDacState(self, stateStr):
        """
        function to set the state of the DAC
        :param stateStr: str, naming the desired state
        :return: fpga state
        """
        stateNumber = hat.dacStatesDict.get(stateStr)
        if stateNumber != None:
            self.fpgaInst.ReadWrite(hat.DacStateCmdByHost, stateNumber)
        return self.checkFpgaStatus()

    def setDacRegister(self, regVal):
        """
        function to set the value of the DAC register
        :param regVal: int
        :return: fpga state
        """
        self.ReadWrite(hat.setDACRegister, regVal)
        return self.checkFpgaStatus()