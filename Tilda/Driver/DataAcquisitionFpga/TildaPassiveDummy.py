"""
Created on 12.04.16

@author: simkaufm

Module Description:

Module for controling the bitfile of TildaPassive.
This bitfile is foreseen for DAQ in parallel to the Master Control Program (MCP)

"""

import ctypes

import numpy as np

import Tilda.Driver.DataAcquisitionFpga.TildaPassiveConfig as TpCfg
import Tilda.Service.Formating as Form
from Tilda.Driver.DataAcquisitionFpga.FPGAInterfaceHandling import FPGAInterfaceHandling


class TildaPassiveDummy(FPGAInterfaceHandling):
    def __init__(self):
        self.type = 'tipadummy'
        bit_path = TpCfg.bitfilePath
        bit_sig = TpCfg.bitfileSignature
        res = TpCfg.fpgaResource
        super(TildaPassiveDummy, self).__init__(bit_path, bit_sig, res, dummy=True)
        self.artificial_build_data = []
        self.status = 0
        self.session = ctypes.c_ulong(0)
        self.dummy_status = 0
        self.scan_pars = TpCfg.draft_tipa_scan_pars

    """ overwrite  of FPGAInTerfacehandling """

    def DeInitFpga(self, finalize_com=False):
        return True

    def FinalizeFPGACom(self):
        return True

    """ normal functions to overwrite: """

    def read_data_from_fifo(self):
        """
        :return: {nOfEle,  newData, elemRemainInFifo}
        nOfEle = int, number of Read Elements, newData = numpy.ndarray containing all data that was read
               elemRemainInFifo = int, number of Elements still in FifoBuffer
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
        return result

    def read_tilda_passive_status(self):
        """
        read the status of the statemachine within tilda passive.
        states are:
        tilda_passive_states = {'idle': 0, 'scanning': 1, 'error': 2}
        :return number of the current state.
        """
        return self.dummy_status

    def set_tilda_passive_status(self, status_num):
        """
        sets the status of the statemachine.
        states are: tilda_passive_states = {'idle': 0, 'scanning': 1, 'error': 2}
            idle can be left due to cmd in TpCfg.TildaPassiveStateCtrl
                either to scanning or error
            scanning can only be left by an timeout of the DMA Queue.
            error can only be left by command.

            -> bitfile must be reloaded for every scan!
            -> stop bitfile to stop acquisition.
        """
        if status_num == 1:
            self.data_builder(self.scan_pars)
        self.dummy_status = status_num
        return True

    def set_bin_num(self, bins_10ns):
        """
        Sets the number of bins that will, be acquired after one trigger.
        Width of one bin is one tick, so 10ns.
        :return: True if everything is fine, else warning
        """
        self.scan_pars['track0']['nOfBins'] = bins_10ns
        return True

    def set_delay(self, delay_10ns):
        """
        set the delay/10ns relative to the falling edge in Dio24 of Controller 1
        (currently Ch24 on TTL-Linedriver 1)
        :return True if everything is fine, else warning
        """
        self.scan_pars['track0']['trigger']['trigDelay10ns'] = delay_10ns
        return True

    """ data buidling for dummy mode: """
    def data_builder(self, scanpars):
        """
        build artificial data for one track.
        """
        track_ind, track_name = scanpars['pipeInternals']['activeTrackNumber']
        trackd = scanpars[track_name]
        step_complete = int('01000000100000000000000000000001', 2)
        new_scan = int('01000000100000000000000000000010', 2)
        bunch_complete = int('01000000100000000000000000000011', 2)
        complete_lis = []
        scans = 0
        while scans < trackd['nOfScans']:
            complete_lis.append(new_scan)
            scans += 1
            step = 0
            while step < trackd['nOfSteps']:
                step += 1
                bunch = 0
                while bunch < trackd['nOfBunches']:
                    bunch += 1
                    time = 0
                    while time < trackd['nOfBins']:
                        scaler03 = 2 ** 5 - 1  # easier for debugging
                        scaler47 = 2 ** 5 - 1  # easier for debugging
                        complete_lis.append(Form.add_header_to23_bit(time, scaler03, scaler47, 0))
                        time += 10  # gives event pattern in 10 ns steps
                    complete_lis.append(bunch_complete)
                complete_lis.append(step_complete)
        self.artificial_build_data = complete_lis
