"""

Created on '05.07.2016'

@author:'simkaufm'

Description:

Module representing a dummy digital multimeter with all required public functions.

"""
import socket
import time
from copy import deepcopy
from enum import Enum

import numpy as np
import serial


class AgilentPreConfigs(Enum):
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
            'measurementCompleteDestination': 'pxi_trig_4',
            'highInputResistanceTrue': True,
            'assignment': 'offset',
            'accuracy': (None, None)
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
            'measurementCompleteDestination': 'zwei',
            'highInputResistanceTrue': True,
            'assignment': 'offset',
            'accuracy': (None, None)
        }
    pre_scan = {
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
            'measurementCompleteDestination': 'zwei',
            'highInputResistanceTrue': True,
            'assignment': 'offset',
            'accuracy': (None, None)
        }


class Agilent34461A:
    def __init__(self, address_str='YourPC'):
        self.connection_type = None  # either 'socket' or 'serial'
        self.connection = None  # storage for the connection to the device either serial or ethernet
        self.sleepAfterSend = 0.005  # time the program blocks before trying to read back.
        self.buffersize = 1024

        self.type = 'Agilent34461A'
        self.address = address_str
        self.name = self.type + '_' + address_str
        self.state = 'initialized'
        self.last_readback = None
        self.accuracy = 10 ** -4  # uncertainty for this dmm. Can be dependent on the range etc.
        self.accuracy_range = 10 ** -4

        # default config dictionary for this type of DMM:
        self.config_dict = AgilentPreConfigs.initial.value
        self.get_accuracy()
        print(self.name, ' initialized')

    ''' connection: '''
    def establish_connection(self, addr):
        if isinstance(str, addr):  # its an ip string
            self.connection_type = 'socket'
            self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.connection.connect(addr)
        else:  # it should be a comport number
            self.connection_type = 'serial'
            self.connection = serial.Serial(port=int(addr) - 1)

    def send_command(self, cmd_str, read_back=False):
        if self.connection_type == 'socket':
            self.connection.send(str.encode(cmd_str + ' \n'))
            time.sleep(self.sleepAfterSend)
            if read_back:
                self.connection.recv(self.buffersize)


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
        """
        Reads and erases all measurements from reading memory up to the specified <max_readings>. The measurements
        are read and erased from the reading memory starting with the oldest measurement first.
        Parameter Typical Return
        1 to 2,000,000 (readings)
        Default is all readings in memory.
        (none)
        Read and remove the three oldest readings:
        R? 3
        Typical Response: #247-4.98748741E-01,-4.35163427E-01,-7.41859188E-01
        The "#2" means that the next 2 digits indicate how many characters are in the returned memory string.
        These two digits are the "47" after the "#2". Therefore, the remainder of the string is 47 digits long:
        -4.98748741E-01,-4.35163427E-01,-7.41859188E-01
        l The R? and DATA:REMove? queries can be used during a long series of readings to periodically remove
        readings from memory that would normally cause the reading memory to overflow. R? does not wait
        for all readings to complete. It sends the readings that are complete at the time the instrument
        receives the command. Use Read? or Fetch? if you want the instrument to wait until all readings are
        complete before sending readings.
        l If you do not specify a value for <max_readings>, all measurements are read and erased.
        l No error is generated if the reading memory contains less readings than requested. In this case, all
        available readings in memory are read and deleted.
        l The number of readings returned may be less than that requested depending on the amount of reading
        memory in your instrument. You can store up to 1,000 measurements in the reading memory of
        the 34460A, 10,000 measurements on the 34461A, 50,000 measurements on the 34465A/70A
        (without the MEM option), or 2,000,000 measurements on the 34465A/70A (with the MEM option). If
        reading memory overflows, new measurements overwrite the oldest measurements stored; the most
        recent measurements are always preserved. No error is generated, but the Reading Mem Ovfl bit (bit
        14) is set in the Questionable Data Register's condition register (see Status System Introduction).
        l The instrument clears all measurements from reading memory when the measurement configuration
        changes, or when any of these commands are executed: INITiate, MEASure:<function>?, READ?,
        *RST, SYSTem:PRESet
        :param num_to_read:
        :param max_time:
        :return:
        """
        ret = self.send_command('R? %s' %num_to_read, True)
        ret_length = int(ret[2: ret[1] + 2])
        ret = np.fromstring(ret[4:],)
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
        if pre_conf_name in DMMdummyPreConfigs.__members__:
            config_dict = DMMdummyPreConfigs[pre_conf_name].value
            config_dict['assignment'] = self.config_dict.get('assignment', 'offset')
            self.load_from_config_dict(config_dict, False)
            self.initiate_measurement()
        else:
            print(
                'error: could not set the preconfiguration: %s in dmm: %s, because the config does not exist'
                % (pre_conf_name, self.name))

    ''' self calibration '''

    def self_calibration(self):
        pass

    ''' loading '''
    def load_from_config_dict(self, config_dict, reset_dev):
        self.config_dict = deepcopy(config_dict)
        self.get_accuracy()
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
                                               ['eins', 'zwei'],
                                               self.config_dict['measurementCompleteDestination']),
            'highInputResistanceTrue': ('high input resistance', True, bool, [False, True]
                                        , self.config_dict['highInputResistanceTrue']),
            'accuracy': ('accuracy (reading, range)', False, tuple, [], self.config_dict['accuracy']),
            'assignment': ('assignment', True, str, ['offset', 'accVolt'], self.config_dict['assignment'])
        }
        return config_dict

    ''' error '''
    def get_accuracy(self):
        """ write the error to self.config_dict['accuracy']"""
        acc_tuple = (10 ** -4, 10 ** -3)
        self.config_dict['accuracy'] = acc_tuple
        return acc_tuple


# dmm = DMMdummy()
# dmm.set_to_pre_conf_setting('periodic')