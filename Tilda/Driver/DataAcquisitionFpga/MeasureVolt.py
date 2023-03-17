"""

Created on '08.07.2015'

@author:'simkaufm'

"""
import logging

from Tilda.Driver.DataAcquisitionFpga.FPGAInterfaceHandling import FPGAInterfaceHandling


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
        states are:
            'PXI_Trigger_4': 0, 'Con1_DIO30': 1,
            'Con1_DIO31': 2, 'PXI_Trigger_4_Con1_DIO30': 3,
            'PXI_Trigger_4_Con1_DIO31': 4, 'PXI_Trigger_4_Con1_DIO30_Con1_DIO31': 5,
            'Con1_DIO30_Con1_DIO31': 6,
        :return: True if self.status == self.statusSuccess, else False
        """
        self.ReadWrite(self.config.measVoltPulseLength25ns, measVoltPars['measVoltPulseLength25ns'])
        # will be 7 for software trigger and than the measurement will be stopped from the host
        meas_volt_complete_state = measVoltPars.get('measurementCompleteDestination',
                                                    'PXI_Trigger_4_Con1_DIO30_Con1_DIO31')
        meas_volt_complete_state_num = self.config.measVoltCompleteDestStateDict.get(meas_volt_complete_state, 0)
        self.ReadWrite(self.config.measVoltCompleteDest, meas_volt_complete_state_num)
        logging.info(
            'set the measurement complete destination to state enum: %s <-> %s' % (meas_volt_complete_state, meas_volt_complete_state_num))
        if meas_volt_complete_state == 'software':  # software triggering!
            # when software triggering is used, use a delay of 0 so, the measurement does not time out.
            self.ReadWrite(self.config.measVoltTimeout10ns, 0)
            logging.debug('setting the timeout for the voltage measurement to zero'
                          ' in order to have this triggered by software and it does bnot timeout.')
        else:
            self.ReadWrite(self.config.measVoltTimeout10ns, measVoltPars['measVoltTimeout10ns'])

        return self.checkFpgaStatus()

    def set_stopVoltMeas(self, stop_bool):
        """
        method to set the stopVoltMeas control, at anytime.
        This will stop the waiting loop which usually would wait until
        the voltmeters returned a 'measurement complete' TTL signal to the defined DIOS or
        the loop runs into a timeout.
        After this loop is stopped the fpga emits a handshake to its main loop and proceeds with further operation.
        Usually it will be faster to let the hardware talk to each other (e.g. during a kepco scan),
        but sometimes (e.g. no hardware feedback possible) a software stop might be the preferred method.
        :param stop_bool: bool
        :return: True if self.status == self.statusSuccess, else False
        """
        self.ReadWrite(self.config.stopVoltMeas, stop_bool)
        return self.checkFpgaStatus()
