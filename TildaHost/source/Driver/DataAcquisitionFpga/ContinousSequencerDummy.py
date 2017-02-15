"""

Created on '09.07.2015'

@author:'simkaufm'

"""

import ctypes

import numpy as np

import Driver.DataAcquisitionFpga.ContinousSequencerConfig as CsCfg
import Service.Formating as Form
from Driver.DataAcquisitionFpga.MeasureVolt import MeasureVolt
from Driver.DataAcquisitionFpga.SequencerCommon import Sequencer


class ContinousSequencer(Sequencer, MeasureVolt):
    def __init__(self):
        """
        Dummy Continous Sequencer
        """
        self.config = CsCfg
        super(Sequencer, self).__init__(self.config.bitfilePath, self.config.bitfileSignature,
                                        self.config.fpgaResource, dummy=True)
        self.type = 'csdummy'
        self.artificial_build_data = []
        self.status = CsCfg.seqStateDict['init']
        self.session = ctypes.c_ulong(0)
        self.status = 0

    '''read Indicators'''

    def getDACQuWriteTimeout(self):
        """
        always False in dummy Mode
        :return: bool, True if timedout
        """
        return False

    def getSPCtrQuWriteTimeout(self):
        """
        always False in dummy Mode
        :return: bool, timeout indicator of the Simple Counter trying to write to the DMAQueue
        """
        return False

    def getSPerrorCount(self):
        """
        always 0 in dummy mode
        :return: int, the errorCount of the simpleCounter module on the fpga
        """
        return 0

    def getSPState(self):
        """
        always 0 in dummy mode
        :return:int, state of SimpleCounter Module
        """
        return 0

    '''set Controls'''

    def setDwellTime(self, scanParsDict, track_num):
        """
        always True in dummy Mode
        """
        return True

    def setAllContSeqPars(self, scanpars, track_num):
        """
        Set all Scanparameters, needed for the continousSequencer
        :param scanpars: dict, containing all scanparameters
        :return: bool, if success
        always True in dummy Mode
        """
        return True

    def setPostAccelerationControlState(self, desiredState, blocking=True):
        """
        will set the PostAccelerationControl State, so one can chose which PowerSupply will be used.
        :return: int, the current State of the Control Box
        """
        pass

    def getPostAccelerationControlStateIsDone(self, desired_state):
        """
        call this to check if the state of the hsb is already the desired one.
        :param desired_state: int, the desired state of the box
        :return: tuple, (bool_True_if_success, int_current_state, int_desired_state)
        """
        return True, desired_state, desired_state

    '''perform measurements:'''

    def measureOffset(self, scanpars, track_num):
        """
        set all scanparameters at the fpga and go into the measure Offset state.
         set DAC to 0V
        :return:bool, True if successfully changed State
        """
        return True

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

    def data_builder(self, scanpars, track_num):
        """
        build data for one track and stroe it in self.artificial_build_data:
        Countervalue = Num_pmt + (num_of_step + 1)
        """
        track_ind, track_name = scanpars['pipeInternals'].get('activeTrackNumber', (0, 'track0'))
        trackd = scanpars[track_name]
        x_axis = Form.create_x_axis_from_scand_dict(scanpars)[track_ind]
        num_of_steps = trackd['nOfSteps'] * trackd['nOfScans']
        x_axis = [Form.add_header_to23_bit(x << 2, 3, 0, 1) for x in x_axis]
        complete_lis = []
        scans = 0
        while scans < trackd['nOfScans']:
            complete_lis.append(Form.add_header_to23_bit(2, 4, 0, 1))  # means scan started
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
                    if i >= 8:
                        complete_lis.append(Form.add_header_to23_bit(1, int(b'0100', 2), 0, 1)) # step complete
        self.artificial_build_data = complete_lis

    ''' overwriting interface functions here '''

    def getData(self):
        """
        nOfEle = int, number of Read Elements,
        newDataArray = numpy Array containing all data that was read
        elemRemainInFifo = int, number of Elements still in FifoBuffer
        :return:
        """
        result = {'nOfEle': 0, 'newData': None, 'elemRemainInFifo': 0}
        result['elemRemainInFifo'] = len(self.artificial_build_data)
        max_read_data = 10
        datapoints = 0
        if result['elemRemainInFifo'] > 0:
            datapoints = min(max_read_data, result['elemRemainInFifo'])
            result['newData'] = np.array(self.artificial_build_data[0:datapoints])
            result['nOfEle'] = datapoints
            result['elemRemainInFifo'] = len(self.artificial_build_data) - datapoints
        self.artificial_build_data = [i for j, i in enumerate(self.artificial_build_data)
                                      if j not in range(datapoints)]
        if result['elemRemainInFifo'] == 0:
            self.status = CsCfg.seqStateDict['measComplete']
        return result

    def getSeqState(self):
        return self.status

    def abort(self):
        self.artificial_build_data = []
        return True

    def halt(self, val):
        return True

    def DeInitFpga(self, finalize_com=False):
        return True

    def FinalizeFPGACom(self):
        return True