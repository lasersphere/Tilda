"""

Created on '05.07.2016'

@author:'simkaufm'

Description:

Module representing a dummy digital multimeter with all required public functions.

"""
import logging
import socket
import threading
import time
from copy import deepcopy
from enum import Enum, unique

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

@unique
class AgilentTriggerSources(Enum):
    """
    Immediate The trigger signal is always present. When you place the instrument in the "wait-for-trigger"
                state, the trigger is issued immediately.
    BUS  The instrument is triggered by *TRG over the remote interface once the DMM is in the "wait-fortrigger"
            state.
    External The instrument accepts hardware triggers applied to the rear-panel Ext Trig input and takes the
        specified number of measurements (SAMPle:COUNt), each time a TTL pulse specified by
        OUTPut:TRIGger:SLOPe is received. If the instrument receives an external trigger before it is
        ready, it buffers one trigger.
    """
    immediate = 'IMM'
    bus = 'BUS'
    external = 'EXT'


class Agilent:
    def __init__(self, reset=False, address_str='YourPC', type_num='34461A'):
        self.connection_type = None  # either 'socket' or 'serial'
        self.connection = None  # storage for the connection to the device either serial or ethernet
        self.sleepAfterSend = 1  # time the program blocks before trying to read back.
        self.buffersize = 1024
        self.con_end_of_trans = b'\r\n'
        self.lock = threading.Lock()

        self.type = 'Agilent'
        self.type_num = type_num
        self.address = address_str
        self.name = self.type + '_' + address_str
        self.state = 'initialized'
        self.last_readback = None
        self.accuracy = 10 ** -4  # uncertainty for this dmm. Can be dependent on the range etc.
        self.accuracy_range = 10 ** -4

        # default config dictionary for this type of DMM:
        self.config_dict = AgilentPreConfigs.initial.value
        self.get_accuracy()
        self.init(address_str, reset_dev=reset)
        # self.establish_connection(address_str)
        print(self.name, ' initialized')

    ''' connection: '''
    def establish_connection(self, addr):
        self.address = addr
        try:
            if 'com' in addr:  # its a comport string
                comport = int(addr.replace('com', '')) - 1
                self.connection_type = 'serial'
                self.connection = serial.Serial(port=comport, baudrate=9600, timeout=0.05,
                                                dsrdtr=True, parity=serial.PARITY_EVEN,
                                                stopbits=serial.STOPBITS_TWO, bytesize=serial.SEVENBITS)
                init_read = self.send_command('SYST:ERR?', True)
                print('serial connection established and error at init is: ', init_read)
                return True
            else:  # its an ipstring
                self.connection_type = 'socket'
                self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.connection.connect(addr)
                return True
        except Exception as e:
            logging.error('could not esatblish connection for %s, error is: %s' % (self.name, e))
            self.connection = None
            self.connection_type = None
            return False

    def send_command(self, cmd_str, read_back=False, delay=None, to_float=False):
        ret = None
        if self.connection_type == 'socket':
            self.connection.send(str.encode(cmd_str + '\n'))
            time.sleep(self.sleepAfterSend)
            if read_back:
                ret = self.connection.recv(self.buffersize)
        elif self.connection_type == 'serial':
            if delay is None:
                delay = self.sleepAfterSend
            if self.lock.acquire(timeout=5):
                print(self.name + ' sending comand: ' + str(cmd_str))
                # self.connection.flushInput()
                # self.connection.flushOutput()
                self.connection.write(str.encode(cmd_str + '\n'))
                time.sleep(delay)
                if read_back:
                    ret = self._ser_readline(delay)
                if to_float:
                    ret = self.convert_to_float(ret)
                self.lock.release()
        return ret

    def _ser_readline(self, delay):
        retries = 0
        ret = b''
        while True:
            newreadback = self.connection.read(1)
            if newreadback:
                ret += newreadback
                if self.con_end_of_trans in ret:
                    # transmission complete
                    ret = ret[:-len(self.con_end_of_trans)]
                    break
            else:  # nothing to transmit yet
                if retries > 10:
                    break
                time.sleep(delay)
                retries += 1
                print('retries: ', retries)
        return ret

    def convert_to_float(self, byte_str, prec=2, default_float=-1.0):
        try:
            if byte_str:
                fl_bytes = float(byte_str)
                fl_bytes = round(fl_bytes, prec)
                return fl_bytes
            else:
                return 0
        except Exception as e:
            logging.error('err', 'could not convert to float: ' + str(byte_str) + ' error is: ' + str(e))
            return default_float

    ''' deinit and init '''

    def init(self, addr_str, reset_dev):
        ret = None
        if self.establish_connection(addr_str):
            if reset_dev:
                self.send_command('*RST')
            ret = 0
            self.state = 'initialized'
        else:
            ret = -1
            self.state = 'error'
        return ret

    def de_init_dmm(self):
        pass

    ''' config measurement '''

    def config_measurement(self, dmm_range, range10V_res):
        """
        Sets all measurement parameters and trigger parameters to their default values for AC or DC voltage measurements.
        Also specifies the range and resolution.
        sets autorange if range is wrong
        :param dmm_range: int, range of dmm, [0.1, 1, 10, 100, 1000, -1 == Auto]
        :param range10V_res: float, resolution of the measurement in the 10V range
         which will here be set via the number of power line cycles:
         [1E-3, 1E-4, 3E-5, 1E-5, 3E-6, -1=Max=3E-6]
         from Manual p.452:
         ResFactor x Range = Resolution.
         PLC:       100     10   1   0.2    0.02
         ResFactor: 0.3ppm 1ppm 3ppm 10ppm 100ppm
         10VRangeRes: 3E-6  1E-5 3E-5  1E-4 1E-3
        :return:
        """
        range = self.set_range(dmm_range)
        res_fact = {0.02: 100E-6, 0.2: 10E-6, 1: 3E-6, 10: 1E-6, 100: 0.3E-6}
        nplc_from_res = {1e-3: '0.02', 1e-4: '0.2', 3e-5: '1', 1e-5: '10', 3e-6: '100', -1: '100'}
        self.send_command('VOLTage:DC:NPLCycles %s' % nplc_from_res.get(range10V_res, '100'))
        nplc = self.send_command('VOLT:DC:NPLC?', True, to_float=True)
        res = range * res_fact.get(nplc, -1)
        self.config_dict['range'] = range
        self.config_dict['resolution'] = res
        return range, nplc, res

    def config_multi_point_meas(self, trig_count, sample_count, trig_src_enum, sample_interval):
        """
        Configures the properties for multipoint measurements.
        :param trig_count: int, Selects the number of triggers that are accepted
         by the instrument before returning to the "idle" trigger state.
         1 to 1,000,000 (1x106) or continuous (INFinity). Default: 1. (34460A/61A)
         use -1 for continous triggering
        :param sample_count:int, Specifies the number of measurements (samples) the instrument takes per trigger
        1 to 1,000,000 (1x106). Default: 1. (34460A/61A)
        :param trig_src: enum, AgilentTriggerSources -> immediate, bus, external
        :param sample_interval: float, interval between follow up samples,
        max is 3600 s min depends on measurement time
        :return: trig_counts, sampl_cts, trig_source, sample_interval
        """
        if trig_count == -1:
            trig_count = 'INF'
        else:
            trig_count = max(trig_count, 1)
            trig_count = min(trig_count, 1e6)
        self.send_command('TRIG:COUN %s' % trig_count)
        trig_counts = self.send_command('TRIG:COUN?', True, to_float=True)

        sample_count = max(sample_count, 1)
        sample_count = min(sample_count, 1e6)
        self.send_command('SAMP:COUN %s' % sample_count)
        sampl_cts = self.send_command('SAMP:COUN?', True, to_float=True)

        self.send_command('TRIG:SOUR %s' % trig_src_enum.value)
        trig_source = self.send_command('TRIG:SOUR?', True)

        if self.type_num in ['34461A']:
            self.send_command('SAMP:TIM %s' % sample_interval)  # not for agilent 34401A
            sample_interval = self.send_command('SAMP:TIM?', True)

        return trig_counts, sampl_cts, trig_source, sample_interval

    def set_input_resistance(self, highResistanceTrue=True):
        """
        Disables or enables automatic input impedance mode for DC voltage and ratio measurements.
        :param highResistanceTrue: bool,
            False: the input impedance for DC voltage measurements is fixed at 10 MΩ for all ranges to minimize
                    noise pickup.
            True: the input impedance for DC voltage measurements varies by range. It is set to "HI-Z" (>10 GΩ) for
                    the 100 mV, 1 V, and 10 V ranges to reduce the effects of measurement loading errors on these lower
                    ranges. The 100 V and 1000 V ranges remain at a 10 MΩ input impedance.
        :return:
        """
        auto_imp = None
        if self.type_num in ['34461A']:
            on_off = 'ON' if highResistanceTrue else 'OFF'
            self.send_command('VOLT:DC:IMP:AUTO %s' % on_off)
            auto_imp = self.send_command('VOLT:DC:IMP:AUTO?', True)
        elif self.type_num in ['34401A']:
            on_off = 'ON' if highResistanceTrue else 'OFF'
            self.send_command('INP:IMP:AUTO %s' % on_off)
            auto_imp = self.send_command('INP:IMP:AUTO?', True)
        return auto_imp

    def set_range(self, range_val):
        ranges = {0.1: '0.1', 1: '1', 10: '10', 100: '100', 1000: '1000', -1: 'AUTO'}
        sel_range = ranges.get(range_val, 'AUTO')
        if sel_range != 'AUTO':
            self.send_command('VOLT:DC:RANG %s' % sel_range)
            set_range = self.send_command('VOLT:DC:RANG?', True, to_float=True)
        else:
            self.send_command('VOLT:DC:RANG:AUTO ON')
            set_range = -1 if self.send_command('VOLT:DC:RANG:AUTO?', True, to_float=True) == 1 else 0
        return set_range

    ''' Trigger '''

    def config_trigger(self, trig_src_enum, trig_delay):
        """
        Selects the trigger source for measurements.
        Sets the delay between the trigger signal and the first measurement. This may be useful in applications
        where you want to allow the input to settle before taking a measurement or for pacing a burst of measurements.
        :param trig_src_enum: enum, AgilentTriggerSources -> immediate, bus, external
        :param trig_delay: float, 0s to 3600s in 1µs steps
        :return: trig_source_read, trig_del_read
        """
        trig_delay = min(3600, trig_delay)
        trig_delay = max(0, trig_delay)
        self.send_command('TRIG:SOUR %s; DEL %s' % (trig_src_enum.value, trig_delay))
        trig_source_read = self.send_command('TRIG:SOUR?', True)
        trig_del_read = self.send_command('TRIG:DEL?', True)
        return trig_source_read, trig_del_read

    def config_trigger_slope(self, trig_slope):
        """
        Selects whether the instrument uses the rising edge (POS) or the falling edge (NEG) of the trigger signal on
        the rear-panel Ext Trig BNC connector when external triggering is selected;
        :param trig_slope: str, 'falling' or 'rising'
        :return: trig_slope_set
        """
        if self.type_num in ['34401A']:  # slope cannot be changed for 34401A, always active low (p. 83)
            return 'falling'
        trig_slope = 'NEG' if trig_slope == 'falling' else 'POS'
        self.send_command('TRIG:SLOP %s' % trig_slope)
        trig_slope_read = self.send_command('TRIG:SLOP?', True)
        return trig_slope_read

    def config_meas_complete_dest(self, meas_compl_des):
        pass

    def config_meas_complete_slope(self, meas_compl_slope_str):
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
        if pre_conf_name in AgilentPreConfigs.__members__:
            config_dict = AgilentPreConfigs[pre_conf_name].value
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
        error_dict_2y = {0.1: (0.0115, 0.0065), 1: (0.0105, 0.001),
                         10: (0.01, 0.0005), 100: (0.011, 0.0006),
                         1000: (0.011, 0.001)}
        dmm_range = self.config_dict['range']
        reading_accuracy_float, range_accuracy_float = error_dict_2y.get(dmm_range)
        reading_accuracy_float *= 10 ** -2  # percent
        range_accuracy_float *= 10 ** -2  # percent
        range_accuracy_float *= dmm_range
        acc_tpl = (reading_accuracy_float, range_accuracy_float)

        self.config_dict['accuracy'] = acc_tpl
        return acc_tpl


# dmm = DMMdummy()
# dmm.set_to_pre_conf_setting('periodic')
if __name__ == "__main__":
    dmm = Agilent(False, 'com1', '34401A')
    # print(dmm.send_command('*IDN?', True))
    # print(dmm.send_command('SYST:ERR?', True))
    # print(dmm.set_range(100))
    # print(dmm.config_measurement(100, 3e-6))
    # print(dmm.config_multi_point_meas(1, 1, AgilentTriggerSources.immediate, 5))
    # print(dmm.set_input_resistance(True))
    # print(dmm.config_trigger(AgilentTriggerSources.external, 0.5))
    print(dmm.config_trigger_slope('falling'))
