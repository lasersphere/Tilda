"""

Created on '09.07.2015'

@author:'simkaufm'

"""

from Driver.DataAcquisitionFpga.SequencerCommon import Sequencer
from Driver.DataAcquisitionFpga.MeasureVolt import MeasureVolt
import Driver.DataAcquisitionFpga.ContinousSequencerConfig as CsCfg
import Service.Formating as Form

import logging
import time
import numpy as np

class ContinousSequencer(Sequencer, MeasureVolt):
    def __init__(self):
        """
        Initiates a fpga object using the init in FPGAInterfaceHandling
        """
        self.config = CsCfg
        super(Sequencer, self).__init__(self.config.bitfilePath, self.config.bitfileSignature,
                                        self.config.fpgaResource, dummy=True)
        self.type = 'csdummy'
        self.scanpars = None
        self.random_data = []
        self.status = CsCfg.seqStateDict['init']

    '''read Indicators'''

    def getDACQuWriteTimeout(self):
        """
        function to check the DACQuWriteTimeout indicator which indicates if the DAC has timed out while
        writing to the Target-to-Host Fifo
        :return: bool, True if timedout
        """
        return self.ReadWrite(self.config.DACQuWriteTimeout).value

    def getSPCtrQuWriteTimeout(self):
        """
        :return: bool, timeout indicator of the Simple Counter trying to write to the DMAQueue
        """
        return self.ReadWrite(self.config.SPCtrQuWriteTimeout).value

    def getSPerrorCount(self):
        """
        :return: int, the errorCount of the simpleCounter module on the fpga
        """
        return self.ReadWrite(self.config.SPerrorCount).value

    def getSPState(self):
        """
        :return:int, state of SimpleCounter Module
        """
        return self.ReadWrite(self.config.SPstate).value

    '''set Controls'''

    def setDwellTime(self, scanParsDict, track_num):
        """
        set the dwell time for the continous sequencer.
        """
        track_name = 'track' + str(track_num)
        self.ReadWrite(self.config.dwellTime10ns, int(scanParsDict[track_name]['dwellTime10ns']))
        return self.checkFpgaStatus()

    def setAllContSeqPars(self, scanpars, track_num):
        """
        Set all Scanparameters, needed for the continousSequencer
        :param scanpars: dict, containing all scanparameters
        :return: bool, if success
        """
        track_name = 'track' + str(track_num)
        if self.changeSeqState(self.config.seqStateDict['idle']):
            if (self.setDwellTime(scanpars, track_num) and
                    self.setmeasVoltParameters(scanpars['measureVoltPars']) and
                    self.setTrackParameters(scanpars[track_name]) and
                    self.selectKepcoOrScalerScan(scanpars['isotopeData']['type'])):
                return self.checkFpgaStatus()
        return False

    '''perform measurements:'''

    def measureOffset(self, scanpars, track_num):
        """
        set all scanparameters at the fpga and go into the measure Offset state.
        What the Fpga does then to measure the Offset is:
         set DAC to 0V
         set HeinzingerSwitchBox to the desired Heinzinger.
         send a pulse to the DMM
         wait until timeout/feedback from DMM
         done
         changed to state 'measComplete'
        Note: not included in Version 1 !
        :return:bool, True if successfully changed State
        """
        if self.setAllContSeqPars(scanpars, track_num):
            return self.changeSeqState(self.config.seqStateDict['measureOffset'])

    def measureTrack(self, scanpars, track_num):
        """
        set all scanparameters at the fpga and go into the measure Track state.
        Fpga will then measure one track independently from host and will finish either in
        'measComplete' or in 'error' state.
        In parallel, host has to read the data from the host sided buffer in parallel.
        :return:bool, True if successfully changed State
        """
        self.status = CsCfg.seqStateDict['measureTrack']
        print('measureing track: ' + str(track_num) +
                      '\nscanparameter are:' + str(scanpars))
        self.data_builder(scanpars, track_num)
        return True

    ''' overwriting interface functions here '''

    def getData(self):
        """
        nOfEle = int, number of Read Elements,
        newDataArray = numpy Array containing all data that was read
        elemRemainInFifo = int, number of Elements still in FifoBuffer
        :return:
        """
        result = {'nOfEle': 0, 'newData': None, 'elemRemainInFifo': 0}
        result['elemRemainInFifo'] = len(self.random_data)
        datapoints = 0
        if result['elemRemainInFifo'] > 0:
            datapoints = min(10, result['elemRemainInFifo'])
            result['newData'] = np.array(self.random_data[0:datapoints])
            result['nOfEle'] = datapoints
            result['elemRemainInFifo'] = len(self.random_data) - datapoints
        self.random_data = [i for j, i in enumerate(self.random_data)
                            if j not in range(datapoints)]
        if result['elemRemainInFifo'] == 0:
            self.status = CsCfg.seqStateDict['measComplete']
        return result

    def data_builder(self, scanpars, track_num):
        """
        build data for one track. Countervalue = Num_pmt + (num_of_step + 1)
        """
        track_ind, track_name = scanpars['pipeInternals']['activeTrackNumber']
        trackd = scanpars[track_name]
        x_axis = Form.create_x_axis_from_scand_dict(scanpars)[track_ind]
        num_of_steps = trackd['nOfSteps'] * trackd['nOfScans']
        x_axis = [Form.add_header_to23_bit(x << 2, 3, 0, 1) for x in x_axis]
        complete_lis = []
        scans = 0
        while scans < trackd['nOfScans']:
            scans += 1
            j = 0
            while j < trackd['nOfSteps']:
                complete_lis.append(x_axis[j])
                j += 1
                i = 0
                while i < 8:
                    # append 8 pmt count events
                    complete_lis.append(Form.add_header_to23_bit(i + j, 2, i, 1))
                    i += 1
        self.random_data = complete_lis

    def getSeqState(self):
        return self.status

