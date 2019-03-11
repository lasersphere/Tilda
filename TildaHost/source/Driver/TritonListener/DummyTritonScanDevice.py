"""
Created on 01.03.2019

@author: simkaufm

Module Description:  Dummy Triton DummyScanDevice used for Testing
copied from Triton Base/DraftScanDeviceTest.py:17. But must not kept up to date necessarily only for testing maybe.
see there on how to test etc.

"""

import Pyro4
import sys
import socket
import logging
import time


from Driver.TritonListener.TritonScanDeviceBase import ScanDeviceBase


class DummyScanDevice(ScanDeviceBase):
    '''
    draft device with minimal functionality
    '''

    '''Called when added'''

    def on(self, cfg):
        '''Setting necessary attributes, interval defaults to 0:'''

        self.type = 'DummyScanDevice'

        self.per_calls = 0
        self.start_step_units = self.possible_start_step_unit.frequency_mhz

        '''Resolving and subscribing to other devices:'''
        # self.dev = self.resolveName('Name')
        # self.subscribe(dev)

    '''Called when removed'''

    def off(self):
        pass

    '''Called regularly, running in separate thread'''

    def periodic(self):
        pass

    '''Called by subscriptors'''

    def receive(self, dev, t, ch, val):
        if ch == 'periodicCalls' and val % 2 == 0:
            print('%s rcd an even per call from %s' % (self.name, dev))
        # elif ch == 'scanSetup':
        #     self.setup_scan(*val)
        # elif ch == 'nextStep':
        #     self.setup_next_step()

    '''Called when settings are loaded, vals contains setting dictionary'''

    def load(self, vals):
        pass

    '''Send current status on this command'''

    def emit(self):
        pass

    ''' Device specific '''
    def set_step_in_dev(self, step_num):
        """
        do some thing in the hardware
        :param step_num: int, step ind
        :return:
        """
        logging.debug('%s was told to set step number: %d which has a value of %s' % (
            self.name, step_num, self.sc_one_scan_vals[step_num]
        ))
        sleep_t = 1.0
        logging.debug('will sleep now for %.2f s' % sleep_t)
        time.sleep(sleep_t)
        logging.debug('ok, step is set continue whatever you are doing')

    def set_pre_scan_measurement_setpoint(self, set_val):
        logging.debug('---------------- %s will set now %.2f %s' % (self.name, set_val, self.start_step_units.value))
        time.sleep(1)
        logging.debug('---------------- %s has set  %.2f %s' % (self.name, set_val, self.start_step_units.value))
