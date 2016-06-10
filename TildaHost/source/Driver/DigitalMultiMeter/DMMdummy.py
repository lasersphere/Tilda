"""

Created on '30.05.2016'

@author:'simkaufm'

Description:

Module representing a dummy digital multimeter with all required public functions.

"""
import numpy as np
import datetime
from copy import deepcopy


class DMMdummy:
    def __init__(self, address_str='YourPC'):
        self.type = 'dummy'
        self.address = address_str
        self.name = self.type + '_' + address_str
        self.state = 'initialized'
        self.last_readback = None
        self.accuracy = 10 ** -4  # uncertainty for this dmm. Can be dependent on the range etc.
        self.accuracy_range = 10 ** -4

        # default config dictionary for this type of DMM:
        self.config_dict = {
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
            'measurementCompleteDestination': 'pxi_trig_4',
            'highInputResistanceTrue': True,
            'accuracy': (None, None)
        }
        self.get_accuracy()
        print(self.name, ' initialized')

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

    ''' Measurement '''

    def initiate_measurement(self):
        self.state = 'measuring'
        pass

    def fetch_multiple_meas(self, num_to_read, max_time=-1):
        if num_to_read == -1:
            num_to_read = 5
        ret = np.full(num_to_read, 1.0, dtype=np.double)
        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # take last element out of array and make a tuple with timestamp:
        self.last_readback = (round(ret[-1], 8), t)
        return ret

    def abort_meas(self):
        self.state = 'aborted'
        pass

    ''' self calibration '''

    def self_calibration(self):
        pass

    ''' loading '''
    def load_from_config_dict(self, config_dict, reset_dev):
        self.config_dict = deepcopy(config_dict)
        print('dummy dmm named: ', self.name)
        print('resetting_dev: ', reset_dev)
        print('dummy dmm loaded with: ', config_dict)

    ''' emitting config pars '''
    def emit_config_pars(self):
        """
        function to return all needed parameters for the configuration dictionary and its values.
        This will also be used to automatically generate a gui.
        Use the indicator_or_control_bool to determine if this is only meant for displaying or also for editing.
        True for control
        :return:dict, tuples:
         (name, indicator_or_control_bool, type, certain_value_list)
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
                                               ['eins', 'zwei'],
                                               self.config_dict['measurementCompleteDestination']),
            'highInputResistanceTrue': ('high input resistance', True, bool, [False, True]
                                        , self.config_dict['highInputResistanceTrue']),
            'accuracy': ('accuracy (reading, range)', False, tuple, [], self.config_dict['accuracy'])
        }
        return config_dict

    ''' error '''
    def get_accuracy(self):
        """ write the error to self.config_dict['accuracy']"""
        acc_tuple = (10 ** -4, 10 ** -3)
        self.config_dict['accuracy'] = acc_tuple
        return acc_tuple