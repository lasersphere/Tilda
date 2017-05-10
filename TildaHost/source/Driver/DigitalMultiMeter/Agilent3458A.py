"""

Created on '30.05.2016'

@author:'simkaufm'

Description:

Module for the Agilent/Keysight 3458A 8.5 digit Multimeter

"""
import datetime
import logging
import time
from enum import Enum

import numpy as np
import visa


class Agilent3458aTriggerSources(Enum):
    # found in Chapter 6 page 257 of Manual
    auto = ('AUTO', 1)  # Triggers whenever the multimeter is not busy
    external_fall_edg = ('EXT', 2)  # only falling edge trigger availabel pulse length min 250ns
    single = ('SGL', 3)  # Triggers once (upon receipt of TRIG SGL) then reverts to TRIG HOLD)
    hold = ('HOLD', 4)  # Disables readings
    synchronus = ('SYN', 5)  # Triggers when the multimeter's output buffer is empty,
    #  memory is off or empty, and the controller requests data.
    level = ('LEVEL', 7)  # Triggers when the input signal reaches the voltage specified
    #  by the LEVEL command on the slope specified by the SLOPE command.
    line = ('LINE', 8)  # Triggers on a zero crossing of the AC line voltage


class Agilent3458aPreConfigs(Enum):
    initial = {
        'range': 'AUTO',
        'resolution': '0.0001',
        'triggerCount': -1,
        'sampleCount': 1,
        'triggerSource': Agilent3458aTriggerSources.auto.name,
        'sampleInterval': -1,
        'triggerDelay_s': 0,
        'highInputResistanceTrue': True,
        'assignment': 'offset',
        'accuracy': (None, None),
        'measurementCompleteDestination': 'software',
        'preConfName': 'initial'
    }
    periodic = {
        'range': '10',
        'resolution': '0.01',
        'triggerCount': -1,
        'sampleCount': 1,
        'triggerSource': Agilent3458aTriggerSources.auto.name,
        'sampleInterval': 0.5,
        'triggerDelay_s': 0,
        'highInputResistanceTrue': True,
        'assignment': 'offset',
        'accuracy': (None, None),
        'measurementCompleteDestination': 'Con1_DIO30',
        'preConfName': 'periodic'
    }
    pre_scan = {
        'range': '10',
        'resolution': '0.0001',
        'triggerCount': 1,
        'sampleCount': 20,
        'triggerSource': Agilent3458aTriggerSources.hold.name,
        'sampleInterval': -1,
        'triggerDelay_s': 0,
        'highInputResistanceTrue': True,
        'assignment': 'offset',
        'accuracy': (None, None),
        'measurementCompleteDestination': 'software',
        'preConfName': 'pre_scan'
    }
    kepco = {
        'range': '1000',
        'resolution': '0.0001',
        'triggerCount': -1,
        'sampleCount': 1,
        'triggerSource': Agilent3458aTriggerSources.external_fall_edg.name,
        'sampleInterval': -1,
        'triggerDelay_s': 0,
        'highInputResistanceTrue': True,
        'assignment': 'offset',
        'accuracy': (None, None),
        'measurementCompleteDestination': 'Con1_DIO30',
        'preConfName': 'kepco'
    }


class Agilent3458A:
    def __init__(self, reset=True, address_str='YourPC'):
        self.type = 'Agilent_3458A'
        self.address = address_str.replace('.', ':')  # colons not allowed in name but needed for gpib
        # self.address = 'GPIB0::' + address_str + '::INSTR'
        self.name = self.type + '_' + address_str
        self.state = 'error'
        # self.last_readback = None
        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.last_readback = (0, t)
        self.res_man = visa.ResourceManager()
        self.gpib = None
        self.gpib_timeout_ms = 100
        self.state = 'initialized'

        self.stored_send_cmd = ''

        self.trig_src_enum = Agilent3458aTriggerSources

        # default config dictionary for this type of DMM:
        self.pre_configs = Agilent3458aPreConfigs
        self.selected_pre_config_name = self.pre_configs.initial.name
        self.config_dict = self.pre_configs.initial.value
        self.init(self.address, reset)
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
        try:
            self.gpib = self.res_man.open_resource(addr)
            self.gpib.timeout = self.gpib_timeout_ms
            if reset:
                self.send_command('PRESET NORM')
                self.config_dict = self.pre_configs.initial.value
            time.sleep(0.1)
            self.gpib.read_termination = '\r\n'
            self.state = 'initialized'
        except Exception as e:
            print('error: while establishing gpib connection, %s yielded the error: %s' % (self.name, e))
            self.gpib = None  # this will cause a dummy mode and all sends will just be printed.
            self.state = 'connection error'

    def de_init_dmm(self):
        if self.gpib is not None:
            self.gpib.clear()
            self.gpib.close()

    ''' GPIB communication '''
    def send_command(self, cmd_str, as_query=False, postpone_send=False, default_return=None):
        """
        this encapsules the gpib.write/query commands.
        it is possible to postpone sending a command and later send all commands via the bus at once.
        :param default_return: val, which will be returned when only writing or no gpib available
        :param cmd_str: str, command for the dmm
        :param as_query: bool, if True the function will return the query
        :param postpone_send: bool, if True, the command will not be send vie the bus,
            but will be appended to the already existing command queue and
            will be send as soon as a send_command without postpone is called.
            querries are always send directly! (together with whatever commands where stored before!)
        :return:
            is query: str, return of query
            else: None
        """
        if postpone_send and not as_query:  # cannot postpone query
            if self.stored_send_cmd != '':
                self.stored_send_cmd += ';' + cmd_str
            else:
                self.stored_send_cmd += cmd_str
            return None
        else:
            if self.stored_send_cmd != '':
                cmd_str = self.stored_send_cmd + ';' + cmd_str
                self.stored_send_cmd = ''
            if self.gpib is not None:
                if as_query:
                    return self.gpib.query(cmd_str)
                else:
                    self.gpib.write(cmd_str)
                    return default_return
            else:
                logging.debug('%s sending command: ' % self.name + cmd_str)
                return default_return


    ''' config measurement '''

    def config_measurement(self, dmm_range, resolution, postpone_send=False):
        """
        set to dc meas with range and resolution
        :param dmm_range: str, ['AUTO', '0.1', '1', '10', '100', '1000']
        :param resolution: str, [0.1, 0.01, 0.001, 0.0001, 0.00001]
        :return: None
        """
        self.config_dict['range'] = dmm_range
        self.config_dict['resolution'] = resolution  # str, [0.1, 0.01, 0.001, 0.0001, 0.00001]
        self.send_command('DCV %s, %s' % (dmm_range, resolution), postpone_send=postpone_send)

    def config_multi_point_meas(self, trig_count, trig_delay_s, trig_source_enum,
                                sample_count, sample_interval_s=0, postpone_send=False):
        """
        configure dmm for multipoint reading
        :param sample_interval_s: float, 1/max ssampling rate to 6000 s in 100ns incr.
        :param trig_delay_s: float, 1E-7 (100 ns) to 6000 seconds in 100ns increments,
        The DELAY command allows you to specify a time interval that is inserted
        between the trigger event and the first sample event.
        use delay=0 for automatic = shortest possible delay (default)
        :param trig_count: int, 0 - 2.1E+9. Specifying 0 or 1 with the SGL event has the
        same effect as using the default value (1): the trigger is armed once and then reverts
        to the HOLD state (disabled).
        :param sample_count: int, 1 to 16777215, Number of Readings. Designates the number of readings taken per trigger and
        the event (sample event) that initiates each reading.
        :param trig_source_enum: enum, one of the Agilent3458aTriggerSources Enums
        :return: None
        """
        self.config_dict['triggerCount'] = trig_count  # int, 0 - 2.1E+9
        self.config_dict['sampleCount'] = sample_count  # int, 1 to 16777215
        self.config_dict['sampleInterval'] = sample_interval_s  # float
        if trig_count > 0:
            self.send_command('TARM SGL, %d' % trig_count, postpone_send=True)  # will arm trigger for trig_count times
        else:  # always arm trigger again and again
            self.send_command('TARM AUTO', postpone_send=True)  # set to arm always
        if sample_interval_s > 0:
            self.send_command('TIMER %s' % sample_interval_s)
            self.send_command('NRDGS %d, TIMER' % sample_count, postpone_send=True)  # number of readings per sample event
        else:
            self.send_command('NRDGS %d' % sample_count, postpone_send=True)  # number of readings per sample event
        self.config_trigger(trig_source_enum, trig_delay_s, postpone_send=postpone_send)

    def set_input_resistance(self, highResistanceTrue=True, postpone_send=False):
        """
        multimeter can be set to high input resistance in the ranges up to 10V
        :param highResistanceTrue: bool, True for high input resistance (range<10V)
        """
        self.config_dict['highInputResistanceTrue'] = highResistanceTrue
        if highResistanceTrue:
            # >10 GOhm for ranges up to 10 V higher have 10 MOhm
            self.send_command('FIXEDZ OFF', postpone_send=postpone_send)
        else:
            # 10 MOhm for all ranges
            self.send_command('FIXEDZ ON', postpone_send=postpone_send)

    def set_range(self, dmm_range, postpone_send=False):
        """
        set the range of the dmm
        :param range_val: str, ['AUTO', '0.1', '1', '10', '100', '1000']
        :param postpone_send: bool, postpone sending of cmd if wanted
        :return: None
        """
        self.config_dict['range'] = dmm_range
        self.send_command('RANGE %s' % dmm_range, postpone_send=postpone_send)

    ''' Trigger '''

    def hold_trigger(self):
        """ set trigger to hold in order to not start measureing while configuring """
        self.config_trigger(self.trig_src_enum.hold, 0)

    def config_trigger(self, trig_source_enum, trig_delay_s, postpone_send=False):
        """
        configure the trigger and the delay to the trigger
        :param trig_source_enum: enum, one of the Agilent3458aTriggerSources Enums
        :param trig_delay_s: float, 1E-7 (100 ns) to 6000 seconds in 100ns increments,
        The DELAY command allows you to specify a time interval that is inserted
        between the trigger event and the first sample event.
        use delay=0 for automatic = shortest possible delay (default)
        :param postpone_send: bool, postpone sending of cmd if wanted
        :return:
        """
        self.config_dict['triggerSource'] = trig_source_enum.name  # str
        self.config_dict['triggerDelay_s'] = trig_delay_s  # float
        self.send_command('TRIG %s' % trig_source_enum.value[0], postpone_send=True)
        self.send_command('DELAY %s' % trig_delay_s, postpone_send=postpone_send)

    # def config_trigger_slope(self, trig_slope):
    #     # ALWAYS FALLING EDGE! CANNOT BE CHANGED!
    #     pass

    def config_meas_complete_dest(self, meas_compl_des):
        if meas_compl_des in ['Con1_DIO30', 'Con1_DIO31', 'software']:
            self.config_dict['measurementCompleteDestination'] = meas_compl_des

    def config_meas_compl_slope(self, positive=True, postpone_send=False):
        """
        External Output. Specifies the event that will generate a signal on the rear
        panel Ext Out connector (EXTOUT signal). This command also specifies the
        polarity of the EXTOUT signal.
        Event = Reading complete (1μs pulse after each reading)
        :param positiv: bool, True high going TTL else low going TTL
        :return:
        """
        slope = 'POS' if positive else 'NEG'
        self.send_command('EXTOUT RCOMP %s' % slope, postpone_send=postpone_send)

    def send_software_trigger(self):
        """
        Send a software trigger by calling TRIG SGL, be sure to setup number of readings first etc.
        Will return to TRIG HOLD afterwards.
        p. 257 from manual:

            Triggers once (upon receipt of TRIG SGL) then
            reverts to TRIG HOLD)

            For all measurements except sub-sampling (see Chapter 5), the trigger event
            operates along with the trigger arm event (TARM command) and the sample
            event (NRDGS command). (The trigger event and the sample event are ignored
            for sub-sampling.) To make a measurement, the trigger arm event must occur
            first, followed by the trigger event, and finally the sample event. The trigger
            event does not initiate a measurement. It merely enables a measurement, making
            it possible for a measurement to take place. The measurement is initiated when
            the sample event (NRDGS or SWEEP command) occurs. Refer to "Triggering
            Measurements" in Chapter 4 for an in-depth discussion of the interaction of the
            various events for most measurement functions. Refer to Chapter 5 for
            information on sub-sampling.
        """
        reads = self.config_dict['sampleCount']
        self.send_command('INBUF ON;TARM AUTO;NRDGS %d,AUTO;TRIG SGL' % reads)
        # input Buffer needs to be turned on! Because Trig sgl otherwise holds the bus

    ''' Measurement '''

    def initiate_measurement(self):
        """
        The 3458A seems not have a function similiar to this.
        but this is called from dmmControl upstream.
        """
        self.state = 'measuring'

    def fetch_multiple_meas(self, num_to_read=-1):
        """
        fetch all available values from the Multimeter.
        Calling values from the Memory unfortunately will stop the memory,
        therefore it must be restarted after reading.
        While reading and restarting the memory, all triggers will go to the buffer and
        the dmm will continue measuring after this.
        :parameter num_to_read: int, unused, because all values are read
        :return: -1 for failure, numpy array
        """
        ret = ''
        available = self.send_command('MCOUNT?', as_query=True, default_return='6')
        try:
            available = int(available)
        except Exception as fail:
            print('error, reading from %s failed to convert: %s, error is: %s' % (self.name, available, fail))
            return np.array([])
        if available > 0:
            # typical response:
            typ_ret = '-1.500911593E+00,-1.500911355E+00,-1.500910163E+00, ' \
                      '-1.500910640E+00,-1.500911236E+00,-1.500909567E+00'
            num_to_read = available
            # nice way but might loose trigger
            # self.send_command('TBUFF ON;TRIG HOLD', postpone_send=True)
            # self.gpib.write('TRIG HOLD')
            # ret = self.send_command('RMEM 1,%d' % num_to_read, as_query=True, default_return=typ_ret)
            # trig_src = self.trig_src_enum[self.config_dict['triggerSource']].value[0]
            # self.send_command('MEM FIFO;TRIG %s' % trig_src)
            # in order not to loose triggers, read directly by polling from bus
            for i in range(0, num_to_read):
                ret += self.send_command('', as_query=True, default_return=typ_ret) + ','
            try:
                ret = np.fromstring(ret, sep=',')
                t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # take last element out of array and make a tuple with timestamp:
                self.last_readback = (round(ret[-1], 8), t)
            except Exception as fail:
                print('error, reading from %s failed to convert: %s, error is: %s' % (self.name, ret, fail))
                return np.array([])
            return ret
        else:
            return np.array([])

    def abort_meas(self):
        """
        This will abort reading. The only function doing this seems to be RESET, so this is called followed by calling PRESET NORM
        :return: None
        """
        self.send_command('RESET')
        self.send_command('PRESET NORM')
        self.state = 'aborted'

    def set_to_pre_conf_setting(self, pre_conf_name):
        """
        this will set and arm the dmm for a pre configured setting.
        :param pre_conf_name: str, name of the setting
        :return: None
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
        """ maybe someone wants to implement this in the future... """
        pass

    ''' loading '''
    def load_from_config_dict(self, config_dict, reset_dev):
        """
        setup the dmm from config dict.
        :param config_dict:
        :param reset_dev:
        :return:
        """
        try:
            self.hold_trigger()

            meas_compl_dest = config_dict['measurementCompleteDestination']
            self.config_meas_complete_dest(meas_compl_dest)

            dmm_range = config_dict['range']  # str, ['AUTO', '0.1', '1', '10', '100', '1000']
            dmm_res = config_dict['resolution']  # str, [0.1, 0.01, 0.001, 0.0001, 0.00001]
            self.config_measurement(dmm_range, dmm_res, postpone_send=True)

            self.set_input_resistance(config_dict['highInputResistanceTrue'], postpone_send=True)   # bool

            trig_count = config_dict['triggerCount']  # int, 0 - 2.1E+9
            trig_del_s = config_dict['triggerDelay_s']  # 1E-7 (100 ns) to 6000 seconds in 100ns increments
            trig_src_enum = self.trig_src_enum[config_dict['triggerSource']]
            sample_count = config_dict['sampleCount']  # int, 1 to 16777215
            sample_interval_s = config_dict['sampleInterval']  # float
            self.config_multi_point_meas(trig_count, trig_del_s, trig_src_enum, sample_count, sample_interval_s,
                                         postpone_send=True)
            if reset_dev:
                self.send_command('MEM FIFO', postpone_send=True)
            else:
                self.send_command('MEM CONT', postpone_send=True)
            self.config_meas_compl_slope(positive=True)  # always positive TTL

            self.get_accuracy()
            print('%s dmm loaded with: ' % self.name, config_dict)
        except Exception as eexc:
            print('error: %s failed to setup from config dict: %s error is: %s' % (self.name, config_dict, eexc))

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
        # print(self.name, 'emitting config pars!')
        config_dict = {
            'range': ('range', True, str, ['AUTO', '0.1', '1', '10', '100', '1000'], self.config_dict['range']),
            'resolution': ('resolution', True, str, ['0.1', '0.01', '0.001', '0.0001', '0.00001'],
                           self.config_dict['resolution']),
            'triggerCount': ('#trigger events', True, int, range(-1, 100000, 1), self.config_dict['triggerCount']),
            'sampleCount': ('#samples', True, int, range(1, 10000, 1), self.config_dict['sampleCount']),
            'triggerSource': ('trigger source', True, str,
                              [i.name for i in self.trig_src_enum], self.config_dict['triggerSource']),
            'sampleInterval': ('sample Interval / s', True, float,
                               [-1.0] + [i / 10 for i in range(0, 1000)], self.config_dict['sampleInterval']),
            'triggerDelay_s': ('trigger delay / s', True, float,
                               [i / 100 for i in range(0, 60000)], self.config_dict['triggerDelay_s']),
            'highInputResistanceTrue': ('high input resistance', True, bool, [False, True]
                                        , self.config_dict['highInputResistanceTrue']),
            'accuracy': ('accuracy (reading, range)', False, tuple, [], self.config_dict['accuracy']),
            'assignment': ('assignment', True, str, ['offset', 'accVolt'], self.config_dict['assignment']),
            'preConfName': ('pre config name', False, str, [], self.config_dict['preConfName']),
            'measurementCompleteDestination': ('measurement compl. dest.', True, str,
                                               ['Con1_DIO30', 'Con1_DIO31', 'software'],
                                               self.config_dict['measurementCompleteDestination']),
        }
        return config_dict

    ''' error '''
    def get_accuracy(self, config_dict=None):
        """
        function to return the accuracy for the current configuration
        the error of the read voltage should be:
        reading +/- (reading * reading_accuracy_float + range_accuracy_float)

        accuracy for ranges and readings from Manual for 2 Year

        :return: tpl, (reading_accuracy_float, range_accuracy_float)
        """
        # ('reading_err', 'range_err')
        if config_dict is None:
            config_dict = self.config_dict
        dmm_range = config_dict.get('range', '10')
        error_dict_2y = {
            'AUTO': (14, 3),  # for autorange assume biggest error
            '0.1': (14, 3),
            '1': (14, 0.3),
            '10': (14, 0.05),
            '100': (14, 0.3),
            '1000': (14, 0.1),
        }
        reading_accuracy_float, range_accuracy_float = error_dict_2y.get(dmm_range, (14, 3))
        reading_accuracy_float *= 10 ** -6  # ppm
        range_accuracy_float *= 10 ** -6  # ppm
        # again assume biggest error for auto range:
        range_accuracy_float *= 1000 if dmm_range == 'AUTO' else float(dmm_range)
        acc_tpl = (reading_accuracy_float, range_accuracy_float)
        config_dict['accuracy'] = acc_tpl
        return acc_tpl

    def get_dev_err(self):
        """
        Error String Query. The ERRSTR? command reads the least significant set bit
        in either the error register or the auxiliary error register and then clears the bit.
        The ERRSTR? command returns two responses separated by a comma. The first
        response is an error number (100 series = error register; 200 series = auxiliary
        error register) and the second response is a message (string) explaining the error.
        :return: tuple, (error num, error str)
        """
        ret = ''
        try:
            ret = self.send_command('ERRSTR?', as_query=True)
            ret = ret.split(',')
            ret[0] = int(ret[0])
            if ret[0] != 0:
                print('error: digital multimeter: %s yields the error: %s' % (self.name, ret[1]))
            return ret
        except Exception as errrr:
            print('error: while polling the error from digital multimeter: %s'
                  ' the following error occured: %s' % (self.name, errrr))
            print('-> return was: ', ret)
            return [0, 'i dont know, something went wrong']


if __name__ == '__main__':
    try:
        dev = Agilent3458A(True, 'GPIB0..22..INSTR')

        dev.set_to_pre_conf_setting('pre_scan')

        while input('send software trigger') != 'q':
            print('sending software trigger')
            dev.send_software_trigger()
            # pass

        print('measured', dev.fetch_multiple_meas(-1))

        # # print(dev.gpib.query('ERR?'))
        # # print('write ok')
        # # print(dev.gpib.query('FUNC DCV 10, 0.0001'))
        # # TODO test this
        # dev.gpib.write('DCV 10, 0.0001')
        #
        # #niceway, but losses trigger:
        # dev.gpib.write('TRIG EXT')
        # dev.gpib.write('MEM FIFO')
        # input('first triggers ok?')
        # # dev.gpib.write('TBUFF ON;TRIG HOLD')
        # # dev.gpib.write('TBUFF ON')
        # # num_to_read = int(dev.gpib.query('MCOUNT?'))
        # # print('num_to_read: ', num_to_read)
        # # ret = dev.gpib.query('RMEM 1,%d' % num_to_read)
        # # print('ret: ', ret)
        # # input('now trigger again!')
        # # dev.gpib.write('MEM FIFO;TRIG EXT')
        # # time.sleep(5)  # sleep for catching up with the triggers
        # # num_to_read = int(dev.gpib.query('MCOUNT?'))
        # # if num_to_read:
        # #     print('num_to_read: ', num_to_read)
        # #     ret = dev.gpib.query('RMEM 1,%d' % num_to_read)
        # #     print('ret: ', ret)
        # # else:
        # #     print('nothing to read, not triggered again.')
        # start = datetime.datetime.now()
        # ret = dev.fetch_multiple_meas(-1)
        # print('ret: ', ret)
        # stopp = datetime.datetime.now()
        # read_time = stopp - start
        # print('read_time: ', read_time)   # 157 ms for 50 vals
        #
        #
        # # # ugly alternative:
        # # ret = ''
        # # dev.gpib.write('TRIG EXT')
        # # dev.gpib.write('MEM FIFO')
        # # input('first triggers ok?')
        # # num_to_read = int(dev.gpib.query('MCOUNT?'))
        # # print('num_to_read: ', num_to_read)
        # # for i in range(0, num_to_read):
        # #     ret += dev.send_command('', as_query=True) + ','
        # # print('ret: ', ret)
        # # input('now trigger again!')
        # # # dev.gpib.write('MEM FIFO;TRIG EXT')
        # # time.sleep(5)  # sleep for catching up with the triggers
        # # num_to_read = int(dev.gpib.query('MCOUNT?'))
        # # print('num_to_read: ', num_to_read)
        # # ret = ''
        # # start = datetime.datetime.now()
        # # for i in range(0, num_to_read):
        # #     ret += dev.send_command('', as_query=True) + ','
        # # print('ret: ', ret)
        # # stopp = datetime.datetime.now()
        # # read_time = stopp - start
        # # print('read_time: ', read_time)   # 150 ms for 50 vals


    except Exception as e:
        print(e)


        # TODO test software triggering! -> NO
        # TODO test periodic reading  -> NO
        # TODO test Kepco Scan  -> No