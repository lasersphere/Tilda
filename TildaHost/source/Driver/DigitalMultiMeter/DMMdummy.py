"""

Created on '30.05.2016'

@author:'simkaufm'

Description:

Module representing a dummy digital multimeter with all required public functions.

"""
import numpy as np


class DMMdummy:
    def __init__(self, address_str='YourPC'):
        self.name = 'dummyDMM_' + address_str

    ''' deinit and init '''

    def init(self, dev_name):
        return 0

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
        pass

    def fetch_multiple_meas(self, num_to_read):
        return np.full(num_to_read, 1.0, dtype=np.double)

    def abort_meas(self):
        pass

    ''' self calibration '''

    def self_calibration(self):
        pass

    ''' loading '''

    def load_from_config_dict(self, config_dict, reset_dev):
        pass

    ''' emitting config pars '''

    def emit_config_pars(self):
        """
        function to return all needed parameters for the configruation dictionary and its values.
        :return:dict, tuples:
         (name, type, certain_value_list)
        """
        config_dict = {
            'range': ('range', float, [-3.0, -2.0, -1.0, 0.1, 1.0, 10.0, 100.0, 1000.0]),
            'resolution': ('resolution', float, [3.5, 4.5, 5.5, 6.5, 7.5]),
            'triggerCount': ('#trigger events', int, range(0, 100000, 1)),
            'sampleCount': ('#samples', int, range(0, 10000, 1)),
            'autoZero': ('auto zero', int, [-1, 0, 1, 2]),
            'triggerSource': ('trigger source', str, [i for i in ['eins', 'zwei']]),
            'sampleInterval': ('sample Interval [s]', float, [-1.0] + [i / 10 for i in range(0, 1000)]),
            'powerLineFrequency': ('power line frequency [Hz]', float, [50.0, 60.0]),
            'triggerDelay_s': ('trigger delay [s]', float, [-2.0, -1.0] + [i / 10 for i in range(0, 1490)]),
            'triggerSlope': ('trigger slope', str, ['falling', 'rising']),
            'measurementCompleteDestination': ('measurement compl. dest.', str, [i for i in ['eins', 'zwei']]),
            'highInputResistanceTrue': ('high input resistance', bool, [False, True])
        }
        return config_dict
