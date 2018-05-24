"""
Created on 

@author: simkaufm

Module Description: Dummy module for the time resolved sequencer, when there is no fpga at hand.
"""

import ctypes
from datetime import datetime
import logging

import numpy as np

import Driver.DataAcquisitionFpga.TimeResolvedSequencerConfig as TrsCfg
import Service.Formating as Form
from Driver.DataAcquisitionFpga.MeasureVolt import MeasureVolt
from Driver.DataAcquisitionFpga.SequencerCommon import Sequencer


class TimeResolvedSequencer(Sequencer, MeasureVolt):
    def __init__(self):
        """
        Dummy TimeResolvedSequencer should behave more or less
        just like the original one but not communicate with any hardware
        """

        self.config = TrsCfg
        super(Sequencer, self).__init__(self.config.bitfilePath, self.config.bitfileSignature,
                                        self.config.fpgaResource, dummy=True)
        self.type = 'trsdummy'
        self.artificial_build_data = []
        self.status = TrsCfg.seqStateDict['init']
        self.session = ctypes.c_ulong(0)
        self.status = 0

    '''read Indicators:'''

    def getMCSState(self):
        """
        always 0 in dummy mode
        :return:int, state of MultiChannelScaler
        """
        return 0

    def getDACQuWriteTimeout(self):
        """
        always False in dummy Mode
        :return: bool, True if timedout
        """
        return False

    def getErrorCount(self):
        """
        gets the ErrorCount which represents how many errors have occured for the MultiChannelScaler
        for example each time abort is pressed, ErrorCount is raised by 1
        :return:int, MCSerrorcount
        """
        return 0

    '''set Controls'''

    def setMCSParameters(self, scanpars, track_name):
        """
        always True in dummy Mode
        """
        return True

    def setAllScanParameters(self, scanpars, track_num):
        """
        Use the dictionary format of act_scan_wins, to set all parameters at once.
         Therefore Sequencer must be in idle state
        :param scanpars: dictionary, containing all scanparameters
        :return: bool, True if successful
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

    def measureOffset(self, scanpars, track_num, pre_post_scan_meas_str):
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
                always True in dummy Mode

        """
        self.status = TrsCfg.seqStateDict['measureTrack']
        logging.info('measuring track: ' + str(track_num) +
                     '\nscanparameters are:' + str(scanpars))
        self.data_builder(scanpars, track_num)
        return True

    def data_builder(self, scanpars, track_num):
        """
        build artificial data for one track.
        """
        track_ind, track_name = scanpars['pipeInternals']['activeTrackNumber']
        trackd = scanpars[track_name]
        logging.debug('starting to build artificial data for dummy trs for track %s' % track_name)
        logging.debug('num of steps: %s num of bins: %s num of bunches: %s num of scans: %s ' %
                      (trackd['nOfSteps'], trackd['nOfBins'], trackd['nOfBunches'], trackd['nOfScans']))
        one_scan = self.build_one_scan(scanpars, trackd, track_ind)
        one_scan_inverted = self.build_one_scan(scanpars, trackd, track_ind, True)
        complete_lis = []
        for i in range(trackd['nOfScans']):
            complete_lis.append(Form.add_header_to23_bit(2, 4, 0, 1))  # means scan started
            complete_lis += one_scan if i % 2 == 0 or not trackd['invertScan'] else one_scan_inverted
            # if scan is inverted, use the data for an inverted scan on every odd scan number

        logging.debug('artificial data for dummy trs completed')
        self.artificial_build_data = complete_lis

    def build_one_scan(self, scanpars, trackd, track_ind, inverted=False):
        """ build data for one scan """
        step = trackd['nOfSteps'] - 1 if inverted else 0
        count_time_dif = trackd['nOfBins'] // trackd['nOfSteps']
        x_axis = Form.create_x_axis_from_scand_dict(scanpars)[track_ind]
        x_axis = [Form.add_header_to23_bit(x << 2, 3, 0, 1) for x in x_axis]
        one_scan = []
        while step < trackd['nOfSteps'] and not inverted or step >= 0 and inverted:
            # count steps upwards if not inverted, otherwise count downwards
            one_scan.append(int(x_axis[step]))  # start with dac information
            if step % 2 == 1:
                bunch = 0  # set bunch to 0 in order to create cts below
            else:  # no scaler entries for all even step numbers
                bunch = trackd['nOfBunches']
                for bun in range(bunch):  # add as meany bunch complete infos as needed to complete this step
                    one_scan.append(Form.add_header_to23_bit(3, 4, 0, 1))  # means new bunch
                one_scan.append(Form.add_header_to23_bit(1, int(b'0100', 2), 0, 1))  # step complete
            while bunch < trackd['nOfBunches']:  # only for uneven steps
                one_scan.append(Form.add_header_to23_bit(3, 4, 0, 1))  # means new bunch
                bunch += 1
                scaler03 = 2 ** 4 - 1  # easier for debugging, all pmt have a count
                scaler47 = 2 ** 4 - 1  # easier for debugging, all pmt have a count
                [one_scan.append(
                    Form.add_header_to23_bit(count_time_dif * i, scaler03, scaler47, 0))
                 for i in range(step)]  # add pmt events with time difference count_time_dif * stepIndex
                if bunch >= trackd['nOfBunches']:
                    # step complete, will be send after all bunches are completed
                    one_scan.append(Form.add_header_to23_bit(1, int(b'0100', 2), 0, 1))
            step += -1 if inverted else 1
        return one_scan

    ''' overwriting interface functions here '''

    def getData(self):
        """
        nOfEle = int, number of Read Elements,
        newDataArray = numpy Array containing all data that was read
        elemRemainInFifo = int, number of Elements still in FifoBuffer
        :return:
        """
        st = datetime.now()
        result = {'nOfEle': 0, 'newData': None, 'elemRemainInFifo': 0}
        result['elemRemainInFifo'] = len(self.artificial_build_data)
        max_read_data = 2000
        n_of_read_data = 0
        if result['elemRemainInFifo'] > 0:
            n_of_read_data = min(max_read_data, result['elemRemainInFifo'])
            result['newData'] = np.array(self.artificial_build_data[0:n_of_read_data])
            result['nOfEle'] = n_of_read_data
            result['elemRemainInFifo'] = len(self.artificial_build_data) - n_of_read_data
        self.artificial_build_data = self.artificial_build_data[n_of_read_data:]
        if result['elemRemainInFifo'] == 0:
            self.status = TrsCfg.seqStateDict['measComplete']
        elapsed = datetime.now() - st
        # logging.debug('reading from dummy trs sequencer took %.1f ms ' % (elapsed.total_seconds() * 1000))
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

    def set_stopVoltMeas(self, stop_bool):
        return True

    def read_outbits_state(self):
        return 0, 'dummy'

    def read_outbits_number_of_cmds(self):
        return 0, 0, 0

# scanp = DftSc.draftScanDict
# test = TimeResolvedSequencer()
# test.data_builder(scanp, 0)
# print(len(test.artificial_build_data))
# print(list(map(Form.split_32b_data, test.artificial_build_data)))
