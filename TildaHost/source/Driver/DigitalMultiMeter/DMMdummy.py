"""

Created on '30.05.2016'

@author:'simkaufm'

Description:

Module representing a dummy digital multimeter with all required public functions.

"""
import datetime
import logging
from copy import deepcopy
from enum import Enum

import numpy as np


class DMMdummyPreConfigs(Enum):
    initial = {
        'range': 10.0,
        'resolution': 7.5,
        'triggerCount': 5,
        'sampleCount': 5,
        'autoZero': -1,
        'triggerSource': 'pxi_trig_3',
        'sampleInterval': -1,
        'powerLineFrequency': 50.0,
        'triggerDelay_s': 0,
        'triggerSlope': 'rising',
        'measurementCompleteDestination': 'Con1_DIO30',
        'highInputResistanceTrue': True,
        'assignment': 'offset',
        'accuracy': (None, None),
        'preConfName': 'initial'
    }
    periodic = {
        'range': 10.0,
        'resolution': 6.5,
        'triggerCount': 0,
        'sampleCount': 0,
        'autoZero': -1,
        'triggerSource': 'eins',
        'sampleInterval': -1,
        'powerLineFrequency': 50.0,
        'triggerDelay_s': 0,
        'triggerSlope': 'rising',
        'measurementCompleteDestination': 'Con1_DIO30',
        'highInputResistanceTrue': True,
        'assignment': 'offset',
        'accuracy': (None, None),
        'preConfName': 'periodic'
    }
    pre_scan = {
        'range': 10.0,
        'resolution': 6.5,
        'triggerCount': 0,
        'sampleCount': 10,
        'autoZero': -1,
        'triggerSource': 'softw_trigger',
        'sampleInterval': -1,
        'powerLineFrequency': 50.0,
        'triggerDelay_s': 0,
        'triggerSlope': 'rising',
        'measurementCompleteDestination': 'Con1_DIO30',
        'highInputResistanceTrue': True,
        'assignment': 'offset',
        'accuracy': (None, None),
        'preConfName': 'pre_scan'
    }
    kepco = {
        'range': 10.0,
        'resolution': 6.5,
        'triggerCount': 0,
        'sampleCount': 0,
        'autoZero': -1,
        'triggerSource': 'softw_trigger',
        'sampleInterval': -1,
        'powerLineFrequency': 50.0,
        'triggerDelay_s': 0,
        'triggerSlope': 'rising',
        'measurementCompleteDestination': 'Con1_DIO30',
        'highInputResistanceTrue': True,
        'assignment': 'offset',
        'accuracy': (None, None),
        'preConfName': 'kepco'
    }


class DMMdummy:
    def __init__(self, address_str='YourPC'):
        self.type = 'dummy'
        self.address = address_str
        self.name = self.type + '_' + address_str
        self.state = 'initialized'
        self.last_readback = None
        self.accuracy = 10 ** -4  # uncertainty for this dmm. Can be dependent on the range etc.
        self.accuracy_range = 10 ** -4
        self.last_fetch_datetime = datetime.datetime.now()
        self.allowed_fetch_time = datetime.timedelta(milliseconds=500)

        # default config dictionary for this type of DMM:
        self.pre_configs = DMMdummyPreConfigs
        self.selected_pre_config_name = self.pre_configs.periodic.name
        self.config_dict = self.pre_configs.periodic.value
        self.get_accuracy()
        logging.info('%s initialized' % self.name)

    ''' deinit and init '''

    def init(self, dev_name):
        session_num = 0
        self.state = 'initialized'
        return session_num

    def de_init_dmm(self):
        pass

    ''' config measurement '''

    def config_measurement(self, dmm_range, resolution):
        pass

    def config_multi_point_meas(self, trig_count, sample_count, sample_trig, sample_interval):
        pass

    def set_input_resistance(self, highResistanceTrue=True):
        pass

    def set_range(self, range_val):
        pass

    ''' Trigger '''

    def config_trigger(self, trig_src, trig_delay):
        pass

    def config_trigger_slope(self, trig_slope):
        pass

    def config_meas_complete_dest(self, meas_compl_des):
        pass

    def send_software_trigger(self):
        pass

    ''' Measurement '''

    def initiate_measurement(self):
        self.state = 'measuring'
        pass

    def fetch_multiple_meas(self, num_to_read, max_time=-1):
        if num_to_read == -1:
            num_to_read = 2
        ret_val = 1
        elapsed_since_laste_fetch = datetime.datetime.now() - self.last_fetch_datetime
        if elapsed_since_laste_fetch >= self.allowed_fetch_time:
            self.last_fetch_datetime = datetime.datetime.now()
            ret = np.full(num_to_read, ret_val, dtype=np.double)
            t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # take last element out of array and make a tuple with timestamp:
            self.last_readback = (round(ret[-1], 8), t)
        else:
            ret = np.zeros(0, dtype=np.double)
        return ret


    def abort_meas(self):
        self.state = 'aborted'
        pass

    def set_to_pre_conf_setting(self, pre_conf_name):
        """
        this will set and arm the dmm for a pre configured setting.
        :param pre_conf_name: str, name of the setting
        :return:
        """
        if pre_conf_name in self.pre_configs.__members__:
            self.selected_pre_config_name = pre_conf_name
            config_dict = self.pre_configs[pre_conf_name].value
            config_dict['assignment'] = self.config_dict.get('assignment', 'offset')
            self.load_from_config_dict(config_dict, False)
            self.initiate_measurement()
            logging.info('%s dmm loaded with preconfig: %s ' % (self.name, pre_conf_name))
        else:
            logging.error(
                'error: could not set the preconfiguration: %s in dmm: %s, because the config does not exist'
                % (pre_conf_name, self.name))

    ''' self calibration '''

    def self_calibration(self):
        pass

    ''' loading '''
    def load_from_config_dict(self, config_dict, reset_dev):
        self.config_dict = deepcopy(config_dict)
        self.get_accuracy()
        logging.info('dummy dmm named: %s' % self.name)
        logging.info('resetting_dev: %s' % str(reset_dev))
        logging.info('dummy dmm loaded with: %s' % str(config_dict))

    ''' emitting config pars '''
    def emit_config_pars(self):
        """
        function to return all needed parameters for the configuration dictionary and its values.
        This will also be used to automatically generate a gui.
        Use the indicator_or_control_bool to determine if this is only meant for displaying or also for editing.
        True for control
        :return:dict, tuples:
         (name, indicator_or_control_bool, type, certain_value_list, current_value_in_config_dict)
        """
        config_dict = {
            'range': ('range', True, float, [-3.0, -2.0, -1.0, 0.1, 1.0, 10.0, 100.0, 1000.0], self.config_dict['range']),
            'resolution': ('resolution', True, float, [3.5, 4.5, 5.5, 6.5, 7.5], self.config_dict['resolution']),
            'triggerCount': ('#trigger events', True, int, range(0, 100000, 1), self.config_dict['triggerCount']),
            'sampleCount': ('#samples', True, int, range(0, 10000, 1), self.config_dict['sampleCount']),
            'autoZero': ('auto zero', True, int, [-1, 0, 1, 2], self.config_dict['autoZero']),
            'triggerSource': ('trigger source', True, str,
                              ['eins', 'zwei'], self.config_dict['triggerSource']),
            'sampleInterval': ('sample Interval [s]', True, float,
                               [-1.0] + [i / 10 for i in range(0, 1000)], self.config_dict['sampleInterval']),
            'powerLineFrequency': ('power line frequency [Hz]', True, float,
                                   [50.0, 60.0], self.config_dict['powerLineFrequency']),
            'triggerDelay_s': ('trigger delay [s]', True, float,
                               [-2.0, -1.0] + [i / 10 for i in range(0, 1490)], self.config_dict['triggerDelay_s']),
            'triggerSlope': ('trigger slope', True, str, ['falling', 'rising'], self.config_dict['triggerSlope']),
            'measurementCompleteDestination': ('measurement compl. dest.', True, str,
                                               ['Con1_DIO30', 'Con1_DIO31', 'software'],
                                               self.config_dict['measurementCompleteDestination']),
            'highInputResistanceTrue': ('high input resistance', True, bool, [False, True]
                                        , self.config_dict['highInputResistanceTrue']),
            'accuracy': ('accuracy (reading, range)', False, tuple, [], self.config_dict['accuracy']),
            'assignment': ('assignment', True, str, ['offset', 'accVolt'], self.config_dict['assignment']),
            'preConfName': ('pre config name', False, str, [], self.selected_pre_config_name)
        }
        return config_dict

    ''' error '''
    def get_accuracy(self, config_dict=None):
        """ write the error to self.config_dict['accuracy']"""
        if config_dict is None:
            config_dict = self.config_dict
        dmm_range = config_dict.get('range', 10)
        acc_tuple = (10 ** -4, dmm_range * 10 ** -3)
        config_dict['accuracy'] = acc_tuple
        return acc_tuple


# dmm = DMMdummy()
# dmm.set_to_pre_conf_setting('periodic')