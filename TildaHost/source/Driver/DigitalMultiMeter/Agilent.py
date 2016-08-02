"""

Created on '05.07.2016'

@author:'simkaufm'

Description:

Module representing a dummy digital multimeter with all required public functions.

"""
import datetime
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
            'resolution': 3e-6,
            'triggerCount': -1,
            'sampleCount': 1,
            'autoZero': 'ONCE',
            'triggerSource': 'bus',
            'triggerDelay_s': 0,
            'triggerSlope': 'rising',
            'highInputResistanceTrue': True,
            'assignment': 'offset',
            'accuracy': (None, None)
        }
    periodic = {
            'range': 10.0,
            'resolution': 1e-5,
            'triggerCount': -1,
            'sampleCount': 1,
            'autoZero': 'ONCE',
            'triggerSource': 'immediate',
            'triggerDelay_s': 0,
            'triggerSlope': 'rising',
            'highInputResistanceTrue': True,
            'assignment': 'offset',
            'accuracy': (None, None)
        }
    pre_scan = {
            'range': 10.0,
            'resolution': 3e-6,
            'triggerCount': 511,
            'sampleCount': 1,
            'autoZero': 'ONCE',
            'triggerSource': 'bus',
            'triggerDelay_s': 0,
            'triggerSlope': 'rising',
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
        self.last_readback_len = 0  # is used to not emit a measurement twice.
        self.connection_type = None  # either 'socket' or 'serial'
        self.connection = None  # storage for the connection to the device either serial or ethernet
        self.sleepAfterSend = 0.5  # time the program blocks before trying to read back.
        self.buffersize = 1024
        self.con_end_of_trans = b'\r\n'
        self.lock = threading.Lock()

        self.type = 'Agilent' + '_' + type_num
        self.type_num = type_num
        self.address = address_str
        self.name = self.type + '_' + address_str
        self.state = 'initialized'
        self.last_readback = None
        self.accuracy = 10 ** -4  # uncertainty for this dmm. Can be dependent on the range etc.
        self.accuracy_range = 10 ** -4

        # default config dictionary for this type of DMM:
        self.config_dict = deepcopy(AgilentPreConfigs.initial.value)
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
                self.connection.settimeout(0.3)
                ipport = 5024  # fixed for 34461A
                self.connection.connect((addr, ipport))
                print(self.connection.recv(self.buffersize))
                print(self.connection.recv(self.buffersize))
                self.send_command('*CLS')
                return True
        except Exception as e:
            logging.error('could not establish connection for %s, error is: %s' % (self.name, e))
            self.connection = None
            self.connection_type = None
            return False

    def send_command(self, cmd_str, read_back=False, delay=None, to_float=False):
        ret = None
        if self.connection_type == 'socket':
            if read_back:
                self._flush_socket()
            cmd = str.encode(cmd_str + '\r\n')
            # print(self.name + ' sending comand: ' + str(cmd_str))
            self.connection.send(cmd)
            time.sleep(self.sleepAfterSend)
            if read_back:
                ret = self.connection.recv(self.buffersize)
                ret = ret[len(cmd):-len(b'\r\n34461A> ')]
                # the send cmd is still in the readback and the device sends b'\r\n34461A> '
                # everytime after return was pressed
                if ret and to_float:
                    ret = self.convert_to_float(ret)
        elif self.connection_type == 'serial':
            if delay is None:
                delay = self.sleepAfterSend
            if self.lock.acquire(timeout=5):
                # print(self.name + ' sending comand: ' + str(cmd_str))
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

    def _flush_socket(self):
        """
        the buffer on the serial port seems to holds all previously send commands
        and therefore those need to be flushed afterwards.
        """
        try:
            while self.connection.recv(1):
                pass
        except Exception as e:  # timed out
            # print("flushing buffer yielded in, ", e)
            pass

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

    def reset_dev(self):
        self.send_command('*RST')

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
        :return: range, nplc, res
        """
        range, err = self.set_range(dmm_range)
        res_fact = {0.02: 100E-6, 0.2: 10E-6, 1: 3E-6, 10: 1E-6, 100: 0.3E-6}
        nplc_from_res = {1e-3: '0.02', 1e-4: '0.2', 3e-5: '1', 1e-5: '10', 3e-6: '100', -1: '100'}
        nplc = nplc_from_res.get(range10V_res, '100')
        self.send_command('VOLTage:DC:NPLCycles %s' % nplc)
        # nplc = self.send_command('VOLT:DC:NPLC?', True, to_float=True)
        res = range * res_fact.get(float(nplc), -1)
        self.config_dict['range'] = range
        self.config_dict['resolution'] = res
        dev_err = self.get_dev_error()
        return range, nplc, res, dev_err

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
        sample_interval_read = sample_interval
        if self.type_num in ['34461A']:
            if trig_count == -1:
                trig_count = 'INF'
            else:
                trig_count = max(trig_count, 1)
                trig_count = min(trig_count, 1e6)
        elif self.type_num in ['34401A']:
            if trig_count == -1:
                trig_count = (511 // sample_count)  #  34401A can only hold up to 512 elements
            else:
                trig_count = max(trig_count, 1)
                trig_count = min(trig_count, 511)
        self.send_command('TRIG:COUN %s' % trig_count)
        trig_count_read = trig_count
        # trig_count_read = self.send_command('TRIG:COUN?', True, to_float=True)

        sample_count = max(sample_count, 1)
        sample_count = min(sample_count, 1e6)
        self.send_command('SAMP:COUN %s' % sample_count)
        sample_count_read = sample_count
        # sample_count_read = self.send_command('SAMP:COUN?', True, to_float=True)

        self.send_command('TRIG:SOUR %s' % trig_src_enum.value)
        trig_source_read = trig_src_enum.value
        # trig_source_read = self.send_command('TRIG:SOUR?', True)

        if self.type_num in ['34461A']:
            pass  # currently yields error: '-113,"Undefined header"
            # self.send_command('SAMPle:TIMer %s' % sample_interval)  # not for agilent 34401A
            # sample_interval_read = self.send_command('SAMPle:TIMer?', True)

        self.config_dict['triggerCount'] = -1 if trig_count == 'INF' else trig_count
        self.config_dict['sampleCount'] = sample_count
        self.config_dict['triggerSource'] = trig_src_enum.name
        self.config_dict['sampleInterval'] = sample_interval

        dev_err = self.get_dev_error()

        return trig_count_read, sample_count_read, trig_source_read, sample_interval_read, dev_err

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
            auto_imp = on_off
            # auto_imp = self.send_command('VOLT:DC:IMP:AUTO?', True)
        elif self.type_num in ['34401A']:
            on_off = 'ON' if highResistanceTrue else 'OFF'
            self.send_command('INP:IMP:AUTO %s' % on_off)
            auto_imp = on_off
            # auto_imp = self.send_command('INP:IMP:AUTO?', True)
        self.config_dict['highInputResistanceTrue'] = highResistanceTrue
        dev_err = self.get_dev_error()
        return auto_imp, dev_err

    def set_range(self, range_val):
        ranges = {0.1: '0.1', 1: '1', 10: '10', 100: '100', 1000: '1000', -1: 'AUTO'}
        sel_range = ranges.get(range_val, 'AUTO')
        set_range = float(sel_range)
        if sel_range != 'AUTO':
            self.send_command('VOLT:DC:RANG %s' % sel_range)
            # set_range = self.send_command('VOLT:DC:RANG?', True, to_float=True)
        else:
            self.send_command('VOLT:DC:RANG:AUTO ON')
            # set_range = -1 if self.send_command('VOLT:DC:RANG:AUTO?', True, to_float=True) == 1 else 0
        self.config_dict['range'] = set_range
        dev_err = self.get_dev_error()
        return set_range, dev_err

    def config_auto_zero(self, auto_zero_mode):
        """
        Disables or enables the autozero mode for DC voltage and ratio measurements.
        :param auto_zero_mode: str, OFF|ON|ONCE
        ON (default): the DMM internally measures the offset following each measurement. It then subtracts
        that measurement from the preceding reading. This prevents offset voltages present on the DMM’s
        input circuitry from affecting measurement accuracy.
        OFF: the instrument uses the last measured zero measurement and subtracts it from each measurement.
        It takes a new zero measurement each time you change the function, range or integration
        time.
        ONCE: the instrument takes one zero measurement and sets autozero OFF. The zero measurement
        taken is used for all subsequent measurements until the next change to the function, range or integration
        time. If the specified integration time is less than 1 PLC, the zero measurement is taken at 1 PLC
        to optimize noise rejection. Subsequent measurements are taken at the specified fast (< 1 PLC) integration
        time.
        """
        auto_zero_mode = auto_zero_mode if auto_zero_mode in ['OFF', 'ON', 'ONCE'] else 'ON'
        self.config_dict['autoZero'] = auto_zero_mode
        zero_auto = auto_zero_mode
        if self.type_num in ['34461A']:
            self.send_command('VOLT:DC:ZERO:AUTO %s' % auto_zero_mode)
            # zero_auto = self.send_command('VOLT:DC:ZERO:AUTO?', True)
        elif self.type_num in ['34401A']:
            self.send_command('ZERO:AUTO %s' % auto_zero_mode)
            # zero_auto = self.send_command('ZERO:AUTO?', True)
        dev_err = self.get_dev_error()
        return zero_auto, dev_err

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
        trig_source_read = trig_src_enum.value
        trig_del_read = trig_delay
        # trig_source_read = self.send_command('TRIG:SOUR?', True)
        # trig_del_read = self.send_command('TRIG:DEL?', True)
        self.config_dict['triggerSource'] = trig_src_enum.name
        self.config_dict['triggerDelay_s'] = trig_delay
        dev_err = self.get_dev_error()
        return trig_source_read, trig_del_read, dev_err

    def config_trigger_slope(self, trig_slope):
        """
        Selects whether the instrument uses the rising edge (POS) or the falling edge (NEG) of the trigger signal on
        the rear-panel Ext Trig BNC connector when external triggering is selected;
        :param trig_slope: str, 'falling' or 'rising'
        :return: trig_slope_set
        """
        self.config_dict['triggerSlope'] = trig_slope
        if self.type_num in ['34401A']:  # slope cannot be changed for 34401A, always active low (p. 83/42)
            return 'falling'
        trig_slope = 'NEG' if trig_slope == 'falling' else 'POS'
        self.send_command('TRIG:SLOP %s' % trig_slope)
        trig_slope_read = trig_slope
        # trig_slope_read = self.send_command('TRIG:SLOP?', True)
        dev_err = self.get_dev_error()

        return trig_slope_read, dev_err

    def config_meas_complete_slope(self, meas_compl_slope_str):
        """
        Selects the slope of the voltmeter complete output signal on the rear-panel VM Comp BNC connector
        slope cannot be changed for 34401A, always active low (p. 83)
        :param meas_compl_slope_str: str, falling or rising, will put rising if something else.
        :return: trig_slope_read
        """
        if self.type_num in ['34401A']:  # slope cannot be changed for 34401A, always active low (p. 83/42)
            return 'falling'
        trig_slope = 'NEG' if meas_compl_slope_str == 'falling' else 'POS'
        self.send_command('OUTP:TRIG:SLOP %s' % trig_slope)
        trig_slope_read = trig_slope
        # trig_slope_read = self.send_command('OUTP:TRIG:SLOP?', True)
        dev_err = self.get_dev_error()

        return trig_slope_read, dev_err

    def send_software_trigger(self):
        """
        Triggers the instrument if TRIGger:SOURce BUS is selected.
        :return: None
        """
        self.send_command('*TRG')
        dev_err = self.get_dev_error()
        return dev_err

    ''' Measurement '''

    def initiate_measurement(self):
        """
        Changes the state of the triggering system from "idle" to "wait-for-trigger", and clears the previous set of
        measurements from reading memory. Measurements begin when the specified trigger conditions are satisfied
        following the receipt of INITiate.
        :return:
        """
        self.send_command('INIT')
        self.state = 'measuring'
        dev_err = self.get_dev_error()

        return dev_err

    def fetch_multiple_meas(self, num_to_read):
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
        start_time = datetime.datetime.now()
        if self.type_num in ['34401A']:
            # Transfer readings stored in the multimeter’s internal memory by the
            # INITiate command to the multimeter’s output buffer where you can
            # read them into your bus controller.
            data_points = self.send_command('DATA:POINts?', True, to_float=True)
            # data_points = 1
            if data_points > 0:
                ret = self.send_command('FETC?', True)
                if data_points >= int(self.config_dict.get('triggerCount', 1) * 0.8):
                    # measurement is 80% completed, need to load a new measurement
                    # might be dangerous to miss a trigger while restarting
                    self.abort_meas()
                    self.initiate_measurement()
                    self.last_readback_len = 0
                if ret:
                    compl = np.fromstring(ret, sep=',')
                    ret = deepcopy(compl)[self.last_readback_len:]
                    self.last_readback_len = compl.size
                    print('complete data:', compl)
                    print('complee data size: ', compl.size)
                    print('newdata: ', ret)
                else:
                    return np.zeros(0, dtype=np.double)
            else:
                return np.zeros(0, dtype=np.double)
        else:
            self.connection.settimeout(0.0005)
            if num_to_read < 0:
                ret = self.send_command('R?', True, delay=0.0005)
            else:
                ret = self.send_command('R? %s' % num_to_read, True, delay=0.0005)
            self.connection.settimeout(0.3)
            if ret:
                if int(ret[2:3]):  # if no value is returned, this yields b'#10' -> int(ret[2:3]) = 0
                    ret = np.fromstring(ret[2 + int(ret[1:2]):], sep=',')
                else:
                    return np.zeros(0, dtype=np.double)
            else:
                return np.zeros(0, dtype=np.double)
        if ret.any():
            t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # take last element out of array and make a tuple with timestamp:
            self.last_readback = (round(ret[-1], 8), t)
        dev_err = self.get_dev_error()
        stop_time = datetime.datetime.now()
        reading_time = stop_time - start_time
        print('reading from dev took %s seconds' % (reading_time.microseconds * 10 ** -6))
        return ret

    def abort_meas(self):
        if self.type_num in ['34401A']:
            self.send_command('\x03')  # see p. 160 \x03 = <Ctrl-C>
        else:
            self.send_command('ABORt')
        self.state = 'aborted'
        dev_err = self.get_dev_error()

        return dev_err

    def set_to_pre_conf_setting(self, pre_conf_name):
        """
        this will set and arm the dmm for a pre configured setting.
        :param pre_conf_name: str, name of the setting
        :return:
        """
        if pre_conf_name in AgilentPreConfigs.__members__:
            config_dict = deepcopy(AgilentPreConfigs[pre_conf_name].value)
            # config_dict['assignment'] = self.config_dict.get('assignment', 'offset')
            print('setting %s to these preconfigured settings: %s %s' %
                  (self.name, pre_conf_name, config_dict))
            self.load_from_config_dict(config_dict, False)
            return self.initiate_measurement()
        else:
            print(
                'error: could not set the preconfiguration: %s in dmm: %s, because the config does not exist'
                % (pre_conf_name, self.name))

    ''' self calibration '''

    def self_calibration(self):
        pass

    ''' loading '''
    def load_from_config_dict(self, config_dict, reset_dev):
        try:
            print('setting up %s with the following config %s' % (self.name, config_dict))
            self.abort_meas()
            self.config_dict = deepcopy(config_dict)
            if reset_dev:
                self.reset_dev()
            dmm_range = int(float(config_dict.get('range')))  # combobox will return a string
            resolution = float(config_dict.get('resolution'))
            self.config_measurement(dmm_range, resolution)
            trig_count = config_dict.get('triggerCount')
            sample_count = config_dict.get('sampleCount')
            trig_src_enum = AgilentTriggerSources[config_dict.get('triggerSource')]
            sample_interval = 1  # config_dict.get('sampleInterval')  sample interval currently not used
            self.config_multi_point_meas(trig_count, sample_count, trig_src_enum, sample_interval)
            auto_z = config_dict.get('autoZero')
            self.config_auto_zero(auto_z)
            trig_delay = config_dict.get('triggerDelay_s', 0)
            self.config_trigger(trig_src_enum, trig_delay)
            trig_slope = config_dict.get('triggerSlope')
            self.config_trigger_slope(trig_slope)
            self.config_meas_complete_slope('rising')
            greater_10_g_ohm = config_dict.get('highInputResistanceTrue')
            self.set_input_resistance(greater_10_g_ohm)
            self.get_accuracy()
            # just to be sure this is included:
            self.config_dict['assignment'] = self.config_dict.get('assignment', 'offset')
            dev_err = self.get_dev_error()
            return dev_err
        except Exception as e:
            dev_err = self.get_dev_error()
            print('Exception while loading config to Agilent: ', e)

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
        trig_slope = ['falling'] if self.type_num in ['34401A'] else ['falling', 'rising']
        trig_events = [-1] + list(range(1, 512)) if self.type_num in ['34401A'] else [-1] + list(range(1, 100, 1))
        config_dict = {
            'range': ('range', True, str, ['-1.0', '0.1', '1.0', '10.0', '100.0', '1000.0'],
                      str(self.config_dict['range'])),
            'resolution': ('resolution', True, str, ['1e-03', '1e-04', '3e-05', '1e-05', '3e-06'],
                           str(self.config_dict['resolution'])),
            'triggerCount': ('#trigger events', True, int, trig_events,
                             self.config_dict['triggerCount']),
            'sampleCount': ('#samples', True, int, range(0, 100, 1), self.config_dict['sampleCount']),
            'autoZero': ('auto zero', True, str, ['ON', 'ONCE', 'OFF'], self.config_dict['autoZero']),
            'triggerSource': ('trigger source', True, str,
                              [i.name for i in AgilentTriggerSources], self.config_dict['triggerSource']),
            'triggerDelay_s': ('trigger delay [s]', True, float,
                               [i / 10 for i in range(0, 1490)], self.config_dict['triggerDelay_s']),
            'triggerSlope': ('trigger slope', True, str, trig_slope, self.config_dict['triggerSlope']),
            'highInputResistanceTrue': ('high input resistance', True, bool, [False, True]
                                        , self.config_dict['highInputResistanceTrue']),
            'accuracy': ('accuracy (reading, range)', False, tuple, [], self.config_dict['accuracy']),
            'assignment': ('assignment', True, str, ['offset', 'accVolt'], self.config_dict['assignment'])
            }
        return config_dict

    ''' error '''
    def get_dev_error(self):
        """
        ask device for present error
        :return (int, str), (errornum, complete error string
        """
        error = self.send_command('SYST:ERR?', True)
        if error:
            error_num = int(error.split(sep=b',')[0])
            if error_num != 0:
                print('%s yields the following error: %s' % (self.name, error))
            return error_num, error
        else:
            return 0, ''

    def get_accuracy(self):
        """
        write the error to self.config_dict['accuracy']

        accuracy as in the manual for the 34401A (p. 216)
        Also valid for 34461A (see datasheet p. 9)
        """
        error_dict_2y = {0.1: (0.005, 0.0035), 1: (0.004, 0.0007),
                         10: (0.0035, 0.0005), 100: (0.0045, 0.0006),
                         1000: (0.0045, 0.001)}
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
# if __name__ == "__main__":
#     dmm = Agilent(False, '137.138.135.94', '34461A')
#     dmm.set_to_pre_conf_setting(AgilentPreConfigs.periodic.name)
#     dmm.abort_meas()
#     dmm.fetch_multiple_meas(-1)
#     print(dmm.load_from_config_dict({'resolution': '3e-5',
#                                'autoZero': 'ON',
#                                'accuracy': (3.5000000000000004e-05, 5e-05),
#                                'triggerCount': -1,
#                                'triggerDelay_s': 0,
#                                'range': '10.0',
#                                'highInputResistanceTrue': True,
#                                'sampleCount': 1,
#                                'triggerSource': 'immediate',
#                                'assignment': 'accVolt',
#                                'triggerSlope': 'rising'}, False))
#
#     [print(dmm.set_to_pre_conf_setting(i)) for i in AgilentPreConfigs.__members__]
#     print(dmm.load_from_config_dict({'resolution': '3e-6',
#                                      'autoZero': 'ON',
#                                      'accuracy': (3.5000000000000004e-05, 5e-05),
#                                      'triggerCount': -1,
#                                      'triggerDelay_s': 0,
#                                      'range': '10.0',
#                                      'highInputResistanceTrue': True,
#                                      'sampleCount': 1,
#                                      'triggerSource': 'bus',
#                                      'assignment': 'accVolt',
#                                      'triggerSlope': 'rising'}, False))
#     # print(dmm.send_command('*IDN?', True))
#     print(dmm.abort_meas())  # needs to be called before changes are made
#
#     # print(dmm.reset_dev())
#     # print(dmm.send_command('SYST:ERR?', True))
#     # print('range ', dmm.set_range(10))
#     # dmm.send_command('DATA:FEED RDG_STORE, "CALCulate"')
#     # print('range, nplc, res ', dmm.config_measurement(10, 3e-6))
#     # # print(dmm.send_command('SYST:ERR?', True))
#     # print(dmm.send_command('TRIG:COUN INF'))
#     # print(dmm.send_command('TRIG:COUN?', True))
#     # print(dmm.send_command('TRIG:SOUR?', True))
#     # print(dmm.send_command('*LRN?', True))
#     # dmm.send_software_trigger()
#     # print(dmm.get_dev_error())
#
#     # print(format(int(dmm.send_command('STAT:OPER:COND?', True, to_float=True)), '016b'))
#
#     # print('trig_counts, sampl_cts, trig_source, sample_interval ',
#     #       dmm.config_multi_point_meas(-1, 1, AgilentTriggerSources.bus, 0.5))
#     # print(format(int(dmm.send_command('STAT:OPER:COND?', True, to_float=True)), '016b'))
#
#     # print(dmm.config_auto_zero('OFF'))
#     # print(dmm.send_command('SYST:ERR?', True))
#     # print(dmm.set_input_resistance(True))
#     # print('trig_source_read, trig_del_read', dmm.config_trigger(AgilentTriggerSources.external, 0))
#     # print(dmm.config_trigger_slope('rising'))
#     # print(dmm.abort_meas())
#     # # print(dmm.send_command('SYST:ERR?', True))
#     # print(dmm.send_command('SYST:ERR?', True))
#     #
#     # print(dmm.initiate_measurement())
#     # print(dmm.send_command('SYST:ERR?', True))
#     # time.sleep(1)
#     # print(dmm.send_software_trigger())
#     # time.sleep(2)
#     # print(dmm.fetch_multiple_meas(-1))
#     # # print(dmm.send_command('SYST:ERR?', True))
#     # # print(dmm.send_command('SYST:ERR?', True))
#     # dmm.set_to_pre_conf_setting('periodic')
#     dmm.set_to_pre_conf_setting('initial')
#     # print(dmm.emit_config_pars())
#     x = 0
#     while x < 100:
#         start = datetime.datetime.now()
#         dmm.send_software_trigger()
#         dmm.send_software_trigger()
#         dmm.send_software_trigger()
#         dmm.send_software_trigger()
#         dmm.send_software_trigger()
#         dmm.send_software_trigger()
#         dmm.send_software_trigger()
#         dmm.send_software_trigger()
#         time.sleep(1)
#         stop = datetime.datetime.now()
#         dif = stop - start
#         print('time for sending 8 softw. triggers and 1 second delay: ', dif.seconds)
#         print(dmm.fetch_multiple_meas(-1))
#         x += 1
#     print(dmm.send_command('SYST:ERR?', True))
#     print(dmm.send_command('*ESR?', True, to_float=True))
