"""

Created on '30.05.2016'

@author:'felixsommer'

Description:

Module representing the Agilent M918x digital multimeter with all required public functions.

Note:
    Due to the hardware limitation of only 80 Trigger counts the dmm is not easily
    suitable for Kepco scans with hardware feedback.

    Just Restarting the measurement when 80 Triggers have ben
    collected would be too risky to loose triggers in between.
    Software triggered kepco scans would probably work,
    but since they are not favorable anyhow implementation does not seem to make sense.
    Just don't perform Kepco scans with this multimeter.
"""
import ctypes
import datetime
import logging
import time
from copy import deepcopy
from enum import Enum, unique
from os import path, pardir

import numpy as np

@unique
class AgM918xTriggerSources(Enum):
    """
    Will be used with the funciton AgM918x_TriggerSetSource
    from AgM918x.h :
        #define AGM918X_VAL_TRIGGER_SOURCE_IMMEDIATE                0
        #define AGM918X_VAL_TRIGGER_SOURCE_EXTERNAL                 1
        #define AGM918X_VAL_TRIGGER_SOURCE_INTERNAL                 2
        #define AGM918X_VAL_TRIGGER_SOURCE_TTL1                     3
        #define AGM918X_VAL_TRIGGER_SOURCE_TTL2                     4
        #define AGM918X_VAL_TRIGGER_SOURCE_TTL3                     5
        #define AGM918X_VAL_TRIGGER_SOURCE_TTL4                     6
        #define AGM918X_VAL_TRIGGER_SOURCE_TTL5                     7
        #define AGM918X_VAL_TRIGGER_SOURCE_TTL6                     8
        #define AGM918X_VAL_TRIGGER_SOURCE_PXI_STAR                 9

        AGM918X_VAL_TRIGGER_SOURCE_IMMEDIATE
            The DMM exits the Wait-For-Trigger state immediately after entering.
            It does not wait for a trigger of any kind. This source only applies when the Acquistion Model is Trigger.

        AGM918X_VAL_TRIGGER_SOURCE_EXTERNAL
            The DMM exits the Wait-For-Trigger state when a trigger occurs on the external trigger input.

        AGM918X_VAL_TRIGGER_SOURCE_INTERNAL
            The DMM exits the Wait-For-Trigger state when the input signal passes through
            the programmed analog level with the programmed slope

        AGM918X_VAL_TRIGGER_SOURCE_TTL1
            The DMM exits the Wait-For-Trigger state when it receives a trigger on TTL1.

    """
    immediate = 0
    external = 1
    internal = 2
    ttl_1 = 3
    ttl_2 = 4
    ttl_3 = 5
    ttl_4 = 6
    ttl_5 = 7
    ttl_6 = 8
    pxi_star = 9


class AgM918xMeasCompleteLoc(Enum):
    """
    Specifies the destination of the DMM Measurement Complete (MC) signal.
    :param meas_compl_dest: int, This signal is issued when the DMM completes a single measurement.
     This signal is commonly referred to as Voltmeter Complete.

    None (-1) No destination specified.
    External (2) Routes the measurement complete signal to the external connector (Pin 2).
    -> External = 'Con1_DIO30' or 'Con1_DIO31'
    TTL 0 (111) Routes the measurement complete signal to TTL0.
    TTL 1 (112) Routes the measurement complete signal to TTL1.
    TTL 2 (113) Routes the measurement complete signal to TTL2.
    TTL 3 (114) Routes the measurement complete signal to TTL3.
    TTL 4 (115) Routes the measurement complete signal to TTL4.
    TTL 5 (116) Routes the measurement complete signal to TTL5.
    TTL 6 (117) Routes the measurement complete signal to TTL6.
    TTL 7 (118) Routes the measurement complete signal to TTL7.
    ECL 0 (119) Routes the measurement complete signal to ECL0.
    ECL 1 (120) Routes the measurement complete signal to ECL1.
    PXI STAR (131) Routes the measurement complete signal to the PXI Star trigger bus.
    RTSI 0 (140) Routes the measurement complete signal to RTSI1.
    RTSI 1 (141) Routes the measurement complete signal to RTSI2.
    RTSI 2 (142) Routes the measurement complete signal to RTSI3.
    RTSI 3 (143) Routes the measurement complete signal to RTSI4.
    RTSI 4 (144) Routes the measurement complete signal to RTSI5.
    RTSI 5 (145) Routes the measurement complete signal to RTSI6.
    RTSI 6 (146) Routes the measurement complete signal to RTSI7.
    """
    undefined = -1
    Con1_DIO30 = 2  # this means external
    Con1_DIO31 = 2  # this means external
    PXI_Trigger_0 = 111
    PXI_Trigger_1 = 112
    PXI_Trigger_2 = 113
    PXI_Trigger_3 = 114
    PXI_Trigger_4 = 115
    PXI_Trigger_5 = 116
    PXI_Trigger_6 = 117
    PXI_Trigger_7 = 118
    ECL_Trigger_0 = 119
    ECL_Trigger_1 = 120
    PXI_STAR = 131
    RTSI_TRIGGER_0 = 140
    RTSI_TRIGGER_1 = 141
    RTSI_TRIGGER_2 = 142
    RTSI_TRIGGER_3 = 143
    RTSI_TRIGGER_4 = 144
    RTSI_TRIGGER_5 = 145
    RTSI_TRIGGER_6 = 146
    software = 0


class AgilentM918xPreConfigs(Enum):
    periodic = {
        'range': '20.0',
        'resolution': '0.00048',
        'sampleCount': 20,
        'triggerSource': AgM918xTriggerSources.immediate.name,
        'powerLineFrequency': '50.0',
        'triggerDelay_s': 0,
        'triggerSlope': 'rising',
        'measurementCompleteDestination': AgM918xMeasCompleteLoc.software.name,
        'highInputResistanceTrue': True,
        'assignment': 'offset',
        'accuracy': (None, None),
        'preConfName': 'periodic'
    }
    pre_scan = {
        'range': '20.0',
        'resolution': '0.00001',
        'sampleCount': 20,
        'triggerSource': AgM918xTriggerSources.immediate.name,
        'powerLineFrequency': '50.0',
        'triggerDelay_s': 0,
        'triggerSlope': 'rising',
        'measurementCompleteDestination': AgM918xMeasCompleteLoc.software.name,
        'highInputResistanceTrue': True,
        'assignment': 'offset',
        'accuracy': (None, None),
        'preConfName': 'pre_scan'
    }


class AgilentM918x:
    """
    Class for accessing the Agilent M918x digital Multimeter.
    """
    def __init__(self, reset=True, address_str='PXI6..15..INSTR', pwr_line_freq='50'):
        dll_path = path.normpath(path.join(path.dirname(__file__), pardir, pardir, pardir, 'binary\\AgM918x.dll'))
        print(dll_path)
        #dll_path = 'C:\\Program Files (x86)\\IVI Foundation\\IVI\\Bin\\AgM918x.dll'

        self.type = 'Agilent_M918x'
        self.state = 'None'
        self.address = address_str.replace('.', ':')  # colons not allowed in name but needed for gpib
        self.name = self.type + '_' + address_str
        # default config dictionary for this type of DMM:

        self.pre_configs = AgilentM918xPreConfigs
        self.selected_pre_config_name = self.pre_configs.periodic.name
        self.config_dict = self.pre_configs.periodic.value
        self.get_accuracy()

        self.last_readback = None  # tuple, (voltage_float, time_str)
        self.already_read = False  # buffer on device cannot be cleared
        # -> use this variable if you have read from buffer =>
        #  dev is in idle state. reset variable when initializing measurement!
        self.stored_data = np.array([])

        self.dll = ctypes.WinDLL(dll_path)
        self.session = ctypes.c_uint32(0)

        stat = self.init(self.address, reset_dev=reset)
        if stat < 0:  # if init fails, start simulation
            self.get_error_message(stat, 'while initializing AgM918x: ')
            self.de_init_dmm()
            print('starting simulation now')
            stat = self.init_with_option(self.address, "Simulate=True, DriverSetup=Model:AgM9183A")
            self.get_error_message(stat)
        self.config_power_line_freq(pwr_line_freq)
        self.set_to_pre_conf_setting(self.pre_configs.periodic.name)


        print(self.name, ' initialized, status is: %s, session is: %s' % (stat, self.session))


    ''' init and close '''

    def init(self, dev_name, id_query=True, reset_dev=True):
        """
        Creates a new session to the instrument
        :return: the status
        """
        dev_name = ctypes.create_string_buffer(dev_name.encode('utf-8'))
        ret = self.dll.AgM918x_init(dev_name, id_query, reset_dev, ctypes.byref(self.session))
        if ret >= 0:
            self.state = 'initialized'
        else:
            self.state = 'error %s %s' % (self.get_error_message(ret, comment=''))
        return ret

    def init_with_option(self, dev_name, options_string, id_query=True, reset_dev=True):
        """
        initialize the dmm with options.
        e.g. "Simulate=True, DriverSetup=Model:M9183A" to simulate a pxi device
        :param dev_name: str, resource name of the dev
        :param options_string: str, can be used to pass the options
        :param id_query: bool
        :param reset_dev: bool
        :return: the status
        """
        dev_name = ctypes.create_string_buffer(dev_name.encode('utf-8'))
        options_string = ctypes.create_string_buffer(options_string.encode('utf-8'))
        ret = self.dll.AgM918x_InitWithOptions(
            dev_name, id_query, reset_dev, options_string, ctypes.byref(self.session))
        if ret > 0:
            self.state = 'initialized'
        else:
            self.state = 'error %s %s' % (self.get_error_message(ret, comment=''))
        return ret

    def de_init_dmm(self):
        """ Closes the current session to the instrument. """
        if self.get_initiated():
            self.abort_meas()
        self.dll.AgM918x_close(self.session)
        self.session = ctypes.c_ulong(0)

    ''' Configure '''

    def config_measurement(self, dmm_range, resolution, func=1):
        """
        Configures the common properties of the measurement.
        named configure measurement digits in quick ref.
        :param dmm_range: str, ['0.2', '2.0', '20.0', '200.0', '300.0']
        :param resolution: str, The measurement resolution in absolute units.
        :param func: int, DC volts, AC volts and so on
        1: DC_VOLTS
        2: AC_VOLTS
        3: DC_CURRENT
        4: AC_CURRENT
        5: 2_WIRE_RES
        101: 4_WIRE_RES
        104: FREQ
        105: PERIOD
        106: AC_PLUS_DC_VOLTS
        107: AC_PLUS_DC_CURRENT
        108: TEMPERATURE
        """
        self.config_dict['range'] = dmm_range
        self.config_dict['resolution'] = resolution
        func = ctypes.c_int32(func)
        dmm_range = ctypes.c_double(float(dmm_range))
        res = ctypes.c_double(float(resolution))
        self.get_error_message(self.dll.AgM918x_ConfigureMeasurement(self.session, func, dmm_range, res))

    def config_dcvoltage_measurement(self, dcv_range, dcv_res):
        """
        Configures all instrument settings necessary to measure DC Voltage, given the parameters of Range, Resolution
        and AutoRange. If auto range is enabled, then the Range parameter specifies the initial range.
        :param dcv_range:   str, ['0.2', '2.0', '20.0', '200.0', '300.0']
        :param dcv_res:     str, ['3.5', '4.5', '5.5', '6.5']
                            The measurement resolution in units of Volts. The Resolution parameter is divided by the
                            absolute value of the Range parameter to produce a dimensionless number of measurement
                            counts which is used to select the integration time of the analog-to-digital converter.
        autorange must be False for Multi Sample mode and will therefore not be user-defined
        """
        dcv_range = ctypes.c_double(float(dcv_range))
        dcv_res = ctypes.c_double(float(dcv_res))
        autorange = ctypes.c_bool(False)
        self.get_error_message(
            self.dll.AgM918x_DCVoltConfigureAll(self.session, dcv_range, dcv_res, autorange))

    def config_multi_point_meas(self, sample_count, trig_src_enum, trig_delay_s, trig_slope):
        """
        Configures the trigger source and the number of counts before the dmm will return to idle.

        :param sample_count: int32, actually will be the number of triggers (followed by one sample each)
        until the dmm returns to idle state (only in idle state values can be fetched!)

        :param trig_src_enum: enum, specifies the sample trigger source to use.
         Enum defined in AgM918x TriggerSources class

        :param trig_delay_s: float, delay of the trigger

        :param trigger_slope: str, 'falling' or 'rising'

        """
        self.config_dict['sampleCount'] = sample_count
        sample_count = ctypes.c_int32(sample_count)
        self.get_error_message(self.dll.AgM918x_TriggerSetAcquisitionModel(self.session, ctypes.c_int32(0)))  # trigger model
        self.get_error_message(self.dll.AgM918x_TriggerSetCount(self.session, sample_count))
        self.config_trigger(trig_src_enum, trig_delay_s, trig_slope)

    def set_input_resistance(self, greater_10_g_ohm=True):
        """
        function to set the input resistance of the AgM918x
        :param greater_10_g_ohm: bool,
            True if you want an input resistance higher than 10GOhm
            False if you want 10MOhm input resistance

         NOTE: >10 GOhm only supported until 2V range!
        """
        self.config_dict['highInputResistanceTrue'] = greater_10_g_ohm
        NIDMM_VAL_1_MEGAOHM = 1000000.0
        NIDMM_VAL_10_MEGAOHM = 10000000.0
        NIDMM_VAL_GREATER_THAN_10_GIGAOHM = 10000000000.0
        NIDMM_VAL_RESISTANCE_NA = 0.0
        NIDMM_ATTR_INPUT_RESISTANCE_id = 29
        resistance = ctypes.c_double(NIDMM_VAL_10_MEGAOHM)
        if greater_10_g_ohm:
            resistance = ctypes.c_double(NIDMM_VAL_GREATER_THAN_10_GIGAOHM)
        self.set_property_node(resistance, NIDMM_ATTR_INPUT_RESISTANCE_id)

    def get_range(self):
        """
        read the currently used range.
        :return: dbl, value for the range.
        valid ranges:
        """
        val = self.read_property_node(ctypes.c_double(), 2,
                                      attr_base_str='IVI_CLASS_PUBLIC_ATTR_BASE')
        return str(val)

    def get_resolution(self):
        val = self.read_property_node(ctypes.c_double(), 8, attr_base_str='IVI_CLASS_ATTR_BASE')
        return val

    def set_range(self, range_val):
        """
        function for only setting the range of the dmm
        :param range_val: dbl, value for the range.
        valid ranges:
        """
        self.config_dict['range'] = range_val
        self.set_property_node(ctypes.c_double(float(range_val)), 2,
                               attr_base_str='IVI_CLASS_PUBLIC_ATTR_BASE')

    ''' Measurement Options'''

    def config_power_line_freq(self, pwr_line_freq):
        """
        Specifies the powerline frequency.
        :param pwr_line_freq: str, '50' or '60' Hz
        """
        self.config_dict['powerLineFrequency'] = pwr_line_freq
        self.get_error_message(
            self.dll.AgM918x_ConfigurePowerLineFrequency(self.session, ctypes.c_double(float(pwr_line_freq))))

    ''' Trigger '''

    def config_trigger(self, trig_src_enum, trig_delay_s, trigger_slope):
        """
        Configures the DMM trigger source and trigger delay.
        :param trig_src_enum: enum, specifies the sample trigger source to use.
         Enum defined in AgM918xTriggerSources class.

        :param trig_delay_s: dbl, The length of time between when the DMM receives the trigger
        and when it takes a measurement (in seconds).

        :param trigger_slope: str, 'falling' or 'rising'
        """
        self.config_dict['triggerSource'] = trig_src_enum.name
        self.config_dict['triggerDelay_s'] = trig_delay_s
        self.get_error_message(self.dll.AgM918x_TriggerSetSource(self.session, ctypes.c_int32(trig_src_enum.value)))
        self.get_error_message(self.dll.AgM918x_TriggerSetDelay(self.session, ctypes.c_double(trig_delay_s)))
        self.config_trigger_slope(trigger_slope)

    def config_trigger_slope(self, trig_slope):
        """
        Sets the Trigger Slope property to either rising edge (positive) or falling edge (negative) polarity.
        :param trig_slope: str, 'falling' or 'rising'
        Rising Edge: 0
        Falling Edge: 1 (default)
        """
        self.config_dict['triggerSlope'] = trig_slope
        slope_int = 1 if trig_slope == 'falling' else 0
        self.get_error_message(self.dll.AgM918x_ConfigureTriggerSlope(self.session, ctypes.c_int32(slope_int)))

    def config_meas_complete_dest(self, meas_compl_dest_enum):
        """
        Specifies the destination of the DMM Measurement Complete (MC) signal.
        :param meas_compl_dest: enum, as defined in AgM918xMeasCompleteLoc class.
        """
        self.config_dict['measurementCompleteDestination'] = meas_compl_dest_enum.name
        if meas_compl_dest_enum != AgM918xMeasCompleteLoc.software:
            self.get_error_message(
                self.dll.AgM918x_ConfigureMeasCompleteDest(self.session, ctypes.c_int32(meas_compl_dest_enum.value)))

    def send_software_trigger(self):
        """
        Software Trigger seems to be not supported by AgM918x. There is no AgM918x_SendSoftwareTrigger
        -> therefore just initiate measurement with the previously configured settings (trigger = immediate)
        """
        logging.debug(self.name + ' sending software trigger')
        self.abort_meas()
        self.initiate_measurement()

    ''' Measurement '''

    def initiate_measurement(self):
        """
        Initiates an acquisition.
        After you call this VI, the DMM leaves the Idle state and enters the Wait-For-Trigger state.
        If trigger is set to Immediate mode, the DMM begins acquiring measurement data.
        Use fetch_single_meas(), AgM918x Fetch Multi Point, or
        AgM918x Fetch Waveform to retrieve the measurement data.
        But they will only return a value when the dmm is back in the idle state!
        Abort will delete all measurements.
        """
        # print(self.name, '  initiating measurement')
        if self.get_initiated():
            print('WARNING! trying to initialize %s again!' % self.name)
        # logging.debug(self.name + '  initiating measurement now!')
        err_code, err_msg = self.get_error_message(self.dll.AgM918x_Initiate(self.session))
        # print(err_msg)
        self.already_read = False
        self.state = 'measuring' if err_code >= 0 else 'error'
        time.sleep(0.3)

    def fetch_single_meas(self, max_time_ms=-1):
        """
        Returns the value from a previously initiated measurement.
        You must call initiate_measurement() before calling this function.
        Will only return a value when dmm is back in idle state!
        :param max_time_ms: int, specifies the maximum time allowed for this VI to complete in milliseconds.
         0-86400000 allowed.
         -1 means Auto
        :return: dbl, measurement value
        """
        max_time_ms = ctypes.c_int32(max_time_ms)
        read = ctypes.c_double()
        self.dll.AgM918x_Fetch(self.session, max_time_ms, ctypes.byref(read))
        return read.value

    def fetch_multiple_meas(self, num_to_read=-1, max_time_ms=10):
        """
        Returns an array of values from a previously initiated measurement
        when the dmm is back in idle state.
        The number of measurements the DMM makes is determined by the values you specify
        for the Sample Count parameters of AgM918x Configure Multi Point.
        You must call initiate_measurement() to initiate a measurement before calling this function
        It will only return a value when the dmm is back in idle though!
        Aborting the measurement will delete all readings in storage!

        :param max_time_ms: int, specifies the maximum time allowed for this function to complete in milliseconds.
        0-86400000 allowed.
        -1 means Auto

        :param num_to_read: int, specifies the number of measurements to acquire. Is limited to 80.
        num_to_read = -1 will read number of samples as stored in self.config_dict.get('sampleCount')
        :return: numpy array with all values
        """
        ret = np.array([])
        start_time = datetime.datetime.now()
        while self.get_initiated():
            now = datetime.datetime.now()
            elapsed_time_ms = (now - start_time).total_seconds() * 1000
            if elapsed_time_ms >= max_time_ms:
                # when still measuring after max_time_ms
                # do not call AgM918x_FetchMultiPoint then, because this will result in an error
                # but return an empty array
                # print(self.name, 'timedout during reading')
                return ret
            pass
        if not self.already_read:  # only return values once!
            # if max_time_ms == -1:
            #     max_time_ms = 100  # set default to 100 ms
            max_time = ctypes.c_int32(max_time_ms)
            if num_to_read < 1:  # if -1 or 0 take sampleCount as number to read
                num_to_read = self.config_dict.get('sampleCount')
            # print("Called fetch_multiple_meas() function. num_to_read is: " + str(num_to_read))
            array_of_values = ctypes.c_double * num_to_read
            array_of_values = array_of_values()
            number_to_read = ctypes.c_int32(num_to_read)
            number_of_read_elements = ctypes.c_int32()
            self.dll.AgM918x_FetchMultiPoint(
                self.session, max_time, number_to_read, array_of_values, ctypes.byref(number_of_read_elements))
            ret = np.ctypeslib.as_array(array_of_values)[0:number_of_read_elements.value]
            if ret.any():
                self.already_read = True
                t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # take last element out of array and make a tuple with timestamp:
                self.last_readback = (round(ret[-1], 8), t)
            if self.config_dict['preConfName'] == 'periodic':
                self.initiate_measurement()  # restart the measurement for periodic measurements
        return ret

    def abort_meas(self, force=False):
        """
        Aborts a previously initiated measurement and returns the DMM to the idle state.
        """
        if self.get_initiated() or force:
            logging.debug(self.name + ' aborting measurement')
            self.get_error_message(self.dll.AgM918x_Abort(self.session))
        self.state = 'aborted'

    def set_to_pre_conf_setting(self, pre_conf_name):
        """
        this will configure the dmm for a pre configured setting.
        be sure to initiate measurement afterwards!
        :param pre_conf_name: str, name of the setting
        :return:
        """
        if pre_conf_name in self.pre_configs.__members__:
            self.selected_pre_config_name = pre_conf_name
            config_dict = self.pre_configs[pre_conf_name].value
            config_dict['assignment'] = self.config_dict.get('assignment', 'offset')
            self.load_from_config_dict(config_dict, False)
            print('%s dmm loaded with preconfig: %s ' % (self.name, pre_conf_name))
        else:
            print(
                'error: could not set the preconfiguration: %s in dmm: %s, because the config does not exist'
                % (pre_conf_name, self.name))

    ''' low level acquisition '''

    def get_error_message(self, input, comment='AgM918x yields error:'):
        """
        Takes the error cluster returned by the VIs, interprets it, and returns it as a user-readable string.
        :return: int, str
        """
        ret_status = ctypes.c_int32(input)
        ret_message = ctypes.create_string_buffer("".encode('utf-8'), 256)
        self.dll.AgM918x_error_message(self.session, ret_status, ret_message)
        if ret_status.value < 0:
            print(
                comment + ' errorcode is: ' + str(ret_status.value) +
                '\n error message is: ' + ret_message.value.decode('utf-8'))
        return ret_status.value, ret_message.value.decode('utf-8')

    def get_initiated(self):
        """
        Returns true if the DMM is currently measuring.
        :return: bool
        """
        ret = False
        ret = ctypes.c_bool(ret)
        self.get_error_message(self.dll.AgM918x_MeasurementGetInitiated(self.session, ctypes.byref(ret)))
        self.state = 'measuring' if ret else 'idle'
        return ret

    '''Utility'''
    def reset_dev(self):
        """
        Resets the instrument to a known state and sends initialization commands to the instrument.
        The initialization commands set instrument settings to the state
        necessary for the operation of the instrument driver.
        :return:
        """
        logging.debug('Resetting DMM: ' + self.name)
        self.dll.AgM918x_reset(self.session)

    ''' loading '''

    def load_from_config_dict(self, config_dict, reset_dev):
        """
        function to load all parameters stored in a config dict to the device.
        Available values can be found by calling emit_config_pars()
        :param config_dict: dict, containing all required values
        :param reset_dev: bool, true for resetting the DMM on startup.
        :return:
        """
        try:
            # without abort it 'works'
            self.abort_meas()
            self.config_dict = deepcopy(config_dict)

            if reset_dev:
                self.reset_dev()
            dmm_range = config_dict.get('range')
            resolution = config_dict.get('resolution')
            self.config_measurement(dmm_range, resolution, func=1)
            self.config_dcvoltage_measurement(dmm_range, resolution)

            sample_count = config_dict.get('sampleCount')
            trig_src_enum = AgM918xTriggerSources[config_dict.get('triggerSource')]
            trig_delay_s = config_dict.get('triggerDelay_s', 0)
            trig_slope = config_dict.get('triggerSlope')
            self.config_multi_point_meas(sample_count, trig_src_enum, trig_delay_s, trig_slope)

            pwr_line_freq = config_dict.get('powerLineFrequency')
            self.config_power_line_freq(pwr_line_freq)

            meas_compl_dest_enum = AgM918xMeasCompleteLoc[config_dict.get('measurementCompleteDestination')]
            self.config_meas_complete_dest(meas_compl_dest_enum)

            greater_10_g_ohm = config_dict.get('highInputResistanceTrue')
            self.set_input_resistance(greater_10_g_ohm)

            self.get_accuracy()

            # just to be sure this is included:
            self.config_dict['assignment'] = self.config_dict.get('assignment', 'offset')
        except Exception as e:
            print('error, Exception while loading config to %s: ' % self.name, e)
            print('config dict was:')
            import json
            print(
                json.dumps(self.config_dict, sort_keys=True, indent=4))

    ''' property Node operations: '''

    def read_property_node(self, property_type, attr_id, name_str="", attr_base_str='IVI_SPECIFIC_ATTR_BASE'):
        """
        read an attribute with a given id and a given type. Both can be found in the AgM918x.h File if necessary.
        :param attr_base_str: str, name of the IVI ATTR Base
        :param property_type: ctypes.type
        :param name_str: Some attributes are unique for each channel. For these, pass the name of the channel.
         Other attributes are unique for each switch. Pass VI_NULL or an empty string for this parameter.
          The default value is an empty string.
        :param attr_id: int, id of the desired attribute. Can be found in the AgM918x.h or IviDmm.h file
        :return: value of the readback, type depends on input
        """
        ivi_attr_base = 1000000
        bases = {'IVI_INHERENT_ATTR_BASE': 50000 + ivi_attr_base,
                 'IVI_SPECIFIC_ATTR_BASE': 150000 + ivi_attr_base,
                 'IVI_LXISYNC_ATTR_BASE': 950000 + ivi_attr_base,
                 'IVI_CLASS_ATTR_BASE': 250000 + ivi_attr_base}
        # from header files:
        #AgM918x.h
        #define IVI_ATTR_BASE                 1000000
        #define IVI_INHERENT_ATTR_BASE        (IVI_ATTR_BASE +  50000)
        # /* base for inherent capability attributes */
        #define IVI_CLASS_ATTR_BASE           (IVI_ATTR_BASE + 250000)
        # /* base for IVI-defined class attributes */
        #define IVI_LXISYNC_ATTR_BASE         (IVI_ATTR_BASE + 950000)
        # /* base for IviLxiSync attributes */
        #define IVI_SPECIFIC_ATTR_BASE        (IVI_ATTR_BASE + 150000)
        # /* base for attributes of specific drivers */
        attr_base = bases[attr_base_str]
        attr_id += attr_base
        attr_id = ctypes.c_int32(attr_id)
        name = ctypes.create_string_buffer(name_str.encode('utf-8'), 256)
        if isinstance(property_type, ctypes.c_int32):
            read_back = ctypes.c_int32()
            self.dll.AgM918x_GetAttributeViInt32(self.session, name, attr_id, ctypes.byref(read_back))
            return read_back.value
        elif isinstance(property_type, ctypes.c_double):
            read_back = ctypes.c_double()
            self.dll.AgM918x_GetAttributeViReal64(self.session, name, attr_id, ctypes.byref(read_back))
            return read_back.value
        elif isinstance(property_type, ctypes.c_char):
            read_back = ctypes.create_string_buffer('', 256)
            self.dll.Ag918x_GetAttributeViString(self.session, name, attr_id, ctypes.byref(read_back))
            return read_back.value.decode('utf-8')
        elif isinstance(property_type, ctypes.c_bool):
            read_back = ctypes.c_bool()
            self.dll.Ag918x_GetAttributeViBoolean(self.session, name, attr_id, ctypes.byref(read_back))
            return read_back.value

    def set_property_node(self, set_val, attr_id, name_str="", attr_base_str='IVI_SPECIFIC_ATTR_BASE'):
        """
        set an attribute with a given id and a given type. Both can be found in the AgM918x.h file if necessary.
        :param set_val: ctypes.c_double etc. object, with the desired value
        :param attr_id: int, id of the desired attribute. Can be found in the AgM918x.h file
        :param name_str: Some attributes are unique for each channel. For these, pass the name of the channel.
         Other attributes are unique for each switch. Pass VI_NULL or an empty string for this parameter.
          The default value is an empty string.
        :param attr_base_str: str, name of the IVI ATTR Base
        """
        ivi_attr_base = 1000000
        bases = {'IVI_INHERENT_ATTR_BASE': 50000 + ivi_attr_base,
                 'IVI_SPECIFIC_ATTR_BASE': 150000 + ivi_attr_base,
                 'IVI_LXISYNC_ATTR_BASE': 950000 + ivi_attr_base,
                 'IVI_CLASS_ATTR_BASE': 250000 + ivi_attr_base}
        # from header files:
        #AgM918x.h
        #define IVI_ATTR_BASE                 1000000
        #define IVI_INHERENT_ATTR_BASE        (IVI_ATTR_BASE +  50000)
        # /* base for inherent capability attributes */
        #define IVI_CLASS_ATTR_BASE           (IVI_ATTR_BASE + 250000)
        # /* base for IVI-defined class attributes */
        #define IVI_LXISYNC_ATTR_BASE         (IVI_ATTR_BASE + 950000)
        # /* base for IviLxiSync attributes */
        #define IVI_SPECIFIC_ATTR_BASE        (IVI_ATTR_BASE + 150000)
        # /* base for attributes of specific drivers */

        attr_base = bases[attr_base_str]
        attr_id += attr_base
        attr_id = ctypes.c_int32(attr_id)
        name = ctypes.create_string_buffer(name_str.encode('utf-8'), 256)
        if isinstance(set_val, ctypes.c_int32):
            self.dll.AgM918x_SetAttributeViInt32(self.session, name, attr_id, set_val)
        elif isinstance(set_val, ctypes.c_double):
            self.dll.AgM918x_SetAttributeViReal64(self.session, name, attr_id, set_val)
        elif isinstance(set_val, ctypes.c_char):
            self.dll.AgM918x_SetAttributeViString(self.session, name, attr_id, set_val)
        elif isinstance(set_val, ctypes.c_bool):
            self.dll.AgM918x_SetAttributeViBoolean(self.session, name, attr_id, set_val)

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
            'range': ('range', True, str, ['0.2', '2.0', '20.0', '200.0', '300.0'], self.config_dict['range']),
            'resolution': ('resolution', True, str,
                           ['0.00001', '0.00004', '0.00008', '0.00048', '0.001', '0.005'],
                           self.config_dict['resolution']),
            'sampleCount': ('#samples', True, int, range(0, 80, 1), self.config_dict['sampleCount']),
            'triggerSource': ('trigger source', True, str,
                              [i.name for i in AgM918xTriggerSources], self.config_dict['triggerSource']),
            'powerLineFrequency': ('power line frequency / Hz', True, str,
                                   ['50.0', '60.0'], self.config_dict['powerLineFrequency']),
            'triggerDelay_s': ('trigger delay / s', True, float,
                               [i / 10 for i in range(0, 1490)], self.config_dict['triggerDelay_s']),
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
        """
        function to return the accuracy for the current configuration
        the error of the read voltage should be:
        reading +/- (reading * reading_accuracy_float + range_accuracy_float)
        :return: tpl, (reading_accuracy_float, range_accuracy_float)
        write the error to self.config_dict['accuracy']
        """
        if config_dict is None:
            config_dict = self.config_dict
        # from AgM918x specs:
        # 1 Year, 23°C, +/-1°C, values in ppm:
        error_dict_1y = {'0.2': (30, 5), '2.0': (20, 2), '20.0': (40, 6), '200.0': (30, 2), '300.0': (1300, 2)}
        dmm_range = config_dict['range']
        reading_accuracy_float, range_accuracy_float = error_dict_1y.get(dmm_range, (1300, 2))
        reading_accuracy_float *= 10 ** -6 #ppm
        range_accuracy_float *= 10 ** -6 #ppm
        range_accuracy_float *= float(dmm_range)
        acc_tpl = (reading_accuracy_float, range_accuracy_float)
        config_dict['accuracy'] = acc_tpl
        return acc_tpl


if __name__=='__main__':
    dmm = AgilentM918x()
    conf_dict = AgilentM918xPreConfigs.pre_scan.value
    ress = []
    for i in ['0.00001', '0.00005', '0.0001', '0.0005', '0.001', '0.005']:
        conf_dict['resolution'] = i
        dmm.load_from_config_dict(conf_dict, False)
        ress += dmm.get_resolution(),
        print(i, ress[-1])
    print(ress)
    dmm.initiate_measurement()
    dmm.send_software_trigger()
    # print(dmm.fetch_single_meas(10))
    # print(dmm.fetch_single_meas(10))
    # print(dmm.fetch_single_meas(10))
    while True:
        ret = dmm.fetch_multiple_meas(-1, -1)
        time.sleep(1)

        if ret.any():
            print(ret)
        else:
            print('nope')
    # print(dmm.fetch_multiple_meas(5, 100))
    # print(dmm.fetch_multiple_meas(5, 100))