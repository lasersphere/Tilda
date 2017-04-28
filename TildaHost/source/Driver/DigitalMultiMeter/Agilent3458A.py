"""

Created on '30.05.2016'

@author:'simkaufm'

Description:

Module for the Agilent/Keysight 3458A 8.5 digit Multimeter

"""
import datetime
import time
from copy import deepcopy
from enum import Enum

import numpy as np
import visa

import TildaTools as TiTs


class Agilent3458aTriggerSources(Enum):
    # found in Chapter 6 page 257 of Manual
    auto = 'AUTO'  # Triggers whenever the multimeter is not busy
    external_fall_edg = 'EXT'  # only falling edge trigger availabel pulse length min 250ns
    single = 'SGL'  # Triggers once (upon receipt of TRIG SGL) then reverts to TRIG HOLD)
    hold = 'HOLD'  # Disables readings
    synchrone = 'SYN'  # Triggers when the multimeter's output buffer is empty,
    #  memory is off or empty, and the controller requests data.
    level = 'LEVEL'  # Triggers when the input signal reaches the voltage specified
    #  by the LEVEL command on the slope specified by the SLOPE command.
    line = 'LINE'  # Triggers on a zero crossing of the AC line voltage


class Agilent3458aPreConfigs(Enum):
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


class Agilent3458a:
    def __init__(self, reset=True, address_str='YourPC'):
        self.type = 'dummy'
        self.address = address_str
        self.name = self.type + '_' + address_str
        self.state = 'error'
        self.last_readback = None
        self.accuracy = 10 ** -4  # uncertainty for this dmm. Can be dependent on the range etc.
        self.accuracy_range = 10 ** -4
        self.res_man = visa.ResourceManager()
        self.gpib = None
        self.gpib_timeout_ms = 100
        self.state = 'initialized'

        # default config dictionary for this type of DMM:
        self.pre_configs = Agilent3458aPreConfigs
        self.selected_pre_config_name = self.pre_configs.periodic.name
        self.config_dict = self.pre_configs.periodic.value
        self.init(address_str, reset)
        self.get_accuracy()
        print(self.name, ' initialized')

    ''' deinit and init '''

    def init(self, addr, reset):
        """
        Reset will call 'PRESET NORM':

        Table 10: PRESET NORM State
            Command Description
            ACBAND 20,2E+6 AC bandwidth 20Hz - 2MHz
            AZERO ON Autozero enabled
            BEEP ON Beeper enabled
            DCV AUTO DC voltage measurements, autorange
            DELAY –1 Default delay
            DISP ON Display enabled
            FIXEDZ OFF Disable fixed input resistance
            FSOURCE ACV Frequency and period source is AC voltage
            INBUF OFF Disable input buffer
            LOCK OFF Keyboard enabled
            MATH OFF Disable real-time math
            MEM OFF Disable reading memory
            MFORMAT SREAL Single real reading memory format
            MMATH OFF Disable post-process math
            NDIG 6 Display 6.5 digits
            NPLC 1 1 power line cycle of integration time
            NRDGS 1,AUTO 1 reading per trigger, auto sample event
            OCOMP OFF Disable offset compensated ohms
            OFORMAT ASCII ASCII output format
            TARM AUTO Auto trigger arm event
            TIMER 1 1 second timer interval
            TRIG SYN Synchronous trigger event
        """
        session_num = 0
        self.gpib = self.res_man.open_resource(addr)
        self.gpib.timeout = self.gpib_timeout_ms
        if reset:
            self.gpib.write('PRESET NORM')
            # self.gpib.write('RESET')
        time.sleep(0.1)
        self.gpib.read_termination = '\r\n'

        self.state = 'initialized'
        return session_num

    def de_init_dmm(self):
        self.gpib.clear()
        self.gpib.close()

    ''' config measurement '''

    def config_measurement(self, dmm_range, resolution):
        """
        set to dc meas with range and resolution
        :param dmm_range: int, -1 for AUTO, [0.1, 1, 10, 100, 1000]
        :param resolution:
        :return:
        """
        if dmm_range == -1:
            dmm_range = 'AUTO'
        self.gpib.write('DCV %s, %s' % (dmm_range, resolution))

    def config_multi_point_meas(self, trig_count, sample_count, trig_source_enum, sample_interval):
        """
        configure dmm for multipoint reading
        :param trig_count:
        :param sample_count:
        :param sample_trig:
        :param sample_interval:
        :return:
        """
        self.config_dict['triggerCount'] = trig_count
        self.config_dict['sampleCount'] = sample_count
        self.config_dict['triggerSource'] = trig_source_enum.name
        self.config_dict['sampleInterval'] = sample_interval
        if trig_count:
            self.gpib.write('TARM SGL, %d' % trig_count)  # will arm trigger for trig_count times
        else:
            self.gpib.write('TARM AUTO')  # set to arm always
        self.gpib.write('NRDGS %d' % sample_count)  # number of readings per sample event
        self.gpib.write('TRIG %s' % trig_source_enum.value)
        if sample_interval:
            pass
            #TODO

    def set_input_resistance(self, highResistanceTrue=True):
        if highResistanceTrue:
            # >10 GOhm for ranges up to 10 V higehr have 10 MOhm
            self.gpib.write('FIXEDZ OFF')
        else:
            # 10 MOhm for all ranges
            self.gpib.write('FIXEDZ ON')

    def set_range(self, range_val):
        dmm_range = TiTs.find_closest_value_in_arr([-1, 0.1, 1.0, 10.0, 100.0], range_val)[1]
        if dmm_range == -1:
            dmm_range = 'AUTO'
        self.gpib.write('RANGE %s' % dmm_range)
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
        available = self.gpib.query('MCOUNT?')
        print('avaiable:', available)
        try:
            available = int(available)
        except Exception as e:
            print(e)
        if available > 0:
            if num_to_read == -1:
                num_to_read = available
            self.gpib.write('TRIG HOLD')
            ret = self.gpib.query('RMEM 1,%d' % num_to_read)
            #TODO get trig source from config dict
            # self.gpib.write('MEM FIFO;TRIG %s' % self.config_dict[])
            self.gpib.write('MEM FIFO;TRIG EXT')
            # typical response:
            # ret = '-1.500911593E+00,-1.500911355E+00,-1.500910163E+00,-1.500910640E+00,-1.500911236E+00,-1.500909567E+00'
            # ret_val = 1
            # ret = np.full(num_to_read, ret_val, dtype=np.double)
            # t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # # take last element out of array and make a tuple with timestamp:
            # self.last_readback = (round(ret[-1], 8), t)
            return ret
        else:
            return -1

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
            print('%s dmm loaded with preconfig: %s ' % (self.name, pre_conf_name))
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
        acc_tuple = (10 ** -4, 10 ** -3)
        config_dict['accuracy'] = acc_tuple
        return acc_tuple

if __name__ == '__main__':
    try:
        dev = Agilent3458a(True, 'GPIB0::22::INSTR')
        # print(dev.gpib.query('ERR?'))
        # print('write ok')
        # print(dev.gpib.query('FUNC DCV 10, 0.0001'))
        dev.gpib.write('DCV 10, 0.0001')
        dev.gpib.write('TRIG EXT')
        dev.gpib.write('MEM FIFO')
        while True:
            time.sleep(5)
            reading_time = datetime.datetime.now()
            try:
                ret = dev.fetch_multiple_meas(-1)
                if ret != -1:
                    print(reading_time, ' : ', ret)
            except Exception as e:
                print(e)
    except Exception as e:
        print(e)