"""
Created on 

@author: simkaufm

Module Description: Dummy module for the time resolved sequencer, when there is no fpga at hand.
"""

import numpy as np


from Driver.DataAcquisitionFpga.MeasureVolt import MeasureVolt
from Driver.DataAcquisitionFpga.SequencerCommon import Sequencer
import Driver.DataAcquisitionFpga.TimeResolvedSequencerConfig as TrsCfg
import Service.Formating as Form


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
                always True in dummy Mode

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
        print('measureing track: ' + str(track_num) +
              '\nscanparameters are:' + str(scanpars))
        self.data_builder(scanpars, track_num)
        return True

    def data_builder(self, scanpars, track_num):
        """
        build artificial data for one track.
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
            step = 0
            while step < trackd['nOfSteps']:
                complete_lis.append(int(x_axis[step]))
                step += 1
                bunch = 0
                while bunch < trackd['nOfBunches']:
                    bunch += 1
                    time = 0  # scans - 1
                    while time < trackd['nOfBins']:
                        scaler03 = max(min(int((time / trackd['nOfBins']) * (2 ** 4)), (2 ** 4) - 1), 1)
                        scaler47 = max(min(int((time / trackd['nOfBins']) * (2 ** 4)), (2 ** 4) - 1), 1)
                        scaler03 = 2 ** 5 - 1  # easier for debugging
                        scaler47 = 2 ** 5 - 1  # easier for debugging
                        complete_lis.append(Form.add_header_to23_bit(time, scaler03, scaler47, 0))
                        time += 10  # gives event pattern in 10 ns steps
                        if time >= trackd['nOfBins']:
                            # step complete, will be send after each bunch!
                            complete_lis.append(Form.add_header_to23_bit(1, int(b'0100', 2), 0, 1))
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
        max_read_data = 50
        n_of_read_data = 0
        if result['elemRemainInFifo'] > 0:
            n_of_read_data = min(max_read_data, result['elemRemainInFifo'])
            result['newData'] = np.array(self.artificial_build_data[0:n_of_read_data])
            result['nOfEle'] = n_of_read_data
            result['elemRemainInFifo'] = len(self.artificial_build_data) - n_of_read_data
        self.artificial_build_data = [i for j, i in enumerate(self.artificial_build_data)
                                      if j not in range(n_of_read_data)]
        if result['elemRemainInFifo'] == 0:
            self.status = TrsCfg.seqStateDict['measComplete']
        return result

    def getSeqState(self):
        return self.status

    def abort(self):
        self.artificial_build_data = []
        return True

    def halt(self, val):
        return True

    def DeInitFpga(self):
        return True

# scanp = DftSc.draftScanDict
# test = TimeResolvedSequencer()
# test.data_builder(scanp, 0)
# print(len(test.artificial_build_data))
# print(list(map(Form.split_32b_data, test.artificial_build_data)))
