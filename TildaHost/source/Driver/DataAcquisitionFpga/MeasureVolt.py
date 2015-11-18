"""

Created on '08.07.2015'

@author:'simkaufm'

"""

from Driver.DataAcquisitionFpga.FPGAInterfaceHandling import FPGAInterfaceHandling

class MeasureVolt(FPGAInterfaceHandling):
    """
    class for operation of the voltage Measurement parts of the FPGA
    """

    '''read Indicators'''
    def getmeasVoltState(self):
        """
        gets the state of Voltage measurement Statemachine
        :return:int, state of Voltage measurement Statemachine
        """
        return self.ReadWrite(self.config.measVoltState).value



    '''write Indicators'''
    def setmeasVoltParameters(self, measVoltPars):
        """
        Writes all values needed for the Voltage Measurement state machine to the fpga ui
        :param measVoltPars: dictionary, containing all necessary infos for Voltage measurement. These are:
        measVoltPulseLength25ns: long, Pulselength of the Trigger Pulse on PXI_Trig4 and CH
        measVoltTimeout10ns: long, timeout until which a response from the DMM must occur.
        :return: True if self.status == self.statusSuccess, else False
        """
        self.ReadWrite(self.config.measVoltPulseLength25ns, measVoltPars['measVoltPulseLength25ns'])
        self.ReadWrite(self.config.measVoltTimeout10ns, measVoltPars['measVoltTimeout10ns'])
        return self.checkFpgaStatus()
