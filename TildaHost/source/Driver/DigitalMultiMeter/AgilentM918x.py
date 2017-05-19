"""

Created on '30.05.2016'

@author:'felixsommer'

Description:

Module representing the Agilent M918x digital multimeter with all required public functions.

"""



import ctypes
import datetime
import logging
from copy import deepcopy
from enum import Enum, unique
from os import path, pardir

import numpy as np

@unique
class AgM918xTriggerSources(Enum):
    """
    Immediate (1) DMM does not wait for a trigger of any kind
    External (2) DMM waits for a trigger on the external trigger input. Positive Signal (3.5V-12V) between 7(pos) and 4(neg)
    Software (3) DMM waits for execution of Send Software Trigger function
    Interval (10) DMM waits for the length of time specified by the Sample Interval attribute to elapse
    TTL 0 (111) PXI Trigger Line 0
    TTL 1 (112) PXI Trigger Line 1
    TTL 2 (113) PXI Trigger Line 2
    TTL 3 (114) PXI Trigger Line 3
    TTL 4 (115) PXI Trigger Line 4
    TTL 5 (116) PXI Trigger Line 5
    TTL 6 (117) PXI Trigger Line 6
    TTL 7 (118) PXI Trigger Line 7
    ECL 0 (119) ECL Trigger Line 0
    ECL 1 (120) ECL Trigger Line 1
    PXI Star (131) PXI Star trigger line
    RTSI 0 (140) RTSI Trigger Line 0
    RTSI 1 (141) RTSI Trigger Line 1
    RTSI 2 (142) RTSI Trigger Line 2
    RTSI 3 (143) RTSI Trigger Line 3
    RTSI 4 (144) RTSI Trigger Line 4
    RTSI 5 (145) RTSI Trigger Line 5
    RTSI 6 (146) RTSI Trigger Line 6
    """
    immediate = 1
    external = 2
    softw_trig = 3
    interval = 10
    pxi_trig_0 = 111
    pxi_trig_1 = 112
    pxi_trig_2 = 113
    pxi_trig_3 = 114
    pxi_trig_4 = 115
    pxi_trig_5 = 116
    pxi_trig_6 = 117
    pxi_trig_7 = 118
    ecl_trig_0 = 119
    ecl_trig_1 = 120
    pxi_star_trig = 131
    rtsi_trig_0 = 140
    rtsi_trig_1 = 141
    rtsi_trig_2 = 142
    rtsi_trig_3 = 143
    rtsi_trig_4 = 144
    rtsi_trig_5 = 145
    rtsi_trig_6 = 146

@unique
class AgM918xMeasCompleteLoc(Enum):
    """
    Specifies the destination of the DMM Measurement Complete (MC) signal.
    :param meas_compl_dest: int, This signal is issued when the DMM completes a single measurement.
     This signal is commonly referred to as Voltmeter Complete.

    None (-1) No destination specified.
    External (2) Routes the measurement complete signal to the external connector (Pin 6?).
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
    external = 2
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
    initial = {
        'range': 20.0,
        'resolution': 6.5,
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
        'range': 20.0,
        'resolution': 6.5,
        'triggerCount': 5,
        'sampleCount': 5,
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
        'range': 20.0,
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
        'range': 20.0,
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


class AgilentM918x:
    """
    Class for accessing the Agilent M918x digital Multimeter.
    """
    def __init__(self, reset=True, address_str='PXI6::15::INSTR', pwr_line_freq=50):
        dll_path = path.join(path.dirname(__file__), pardir, pardir, pardir, 'binary\\AgM918x.dll')
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
        ''' Closes the current session to the instrument. '''
        if self.get_initiated() == True:
            self.abort_meas()
        self.dll.AgM918x_close(self.session)
        self.session = ctypes.c_ulong(0)
        pass

    ''' Configure '''

    def config_measurement(self, dmm_range, resolution, func=1):
        """
        Configures the common properties of the measurement.
        named configure measurement digits in quick ref.
        :param dmm_range: dbl, range 20.00, 200.00, etc.
        Positive values represent the absolute value of the maximum measurement expected.
        The driver coerces this value to the appropriate range for the instrument.
        Possible ranges are: 0.2, 2.0, 20.0, 200.0, 2000.0
        Auto range ON: -1.0
        :param resolution: dbl, resolution in digits
        3.5 (3.5000000E+0) Specifies 3.5 digits resolution.
        4.5 (4.500000E+0) Specifies 4.5 digits resolution.
        5.5 (5.500000E+0) Specifies 5.5 digits resolution.
        6.5 (6.500000E+0) Specifies 6.5 digits resolution.
        7.5 (7.500000E+0) Specifies 7.5 digits resolution.
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
        dmm_range = ctypes.c_double(dmm_range)
        res = ctypes.c_double(resolution)
        self.dll.AgM918x_ConfigureMeasurement(self.session, func, dmm_range, res)
        pass

    def config_dcvoltage_measurement(self, dcv_range, dcv_res):
        """
        Configures all instrument settings necessary to measure DC Voltage, given the parameters of Range, Resolution
        and AutoRange. If auto range is enabled, then the Range parameter specifies the initial range.
        :param dcv_range:   The measurement range in units of Volts. Positive values represent the absolute value of the
                            maximum DC voltage expected. The driver uses this value to select the appropriate range for
                            the measurement.
        :param dcv_res:     The measurement resolution in units of Volts. The Resolution parameter is divided by the
                            absolute value of the Range parameter to produce a dimensionless number of measurement
                            counts which is used to select the integration time of the analog-to-digital converter.
        autorange must be False for Multi Sample mode and will therefore not be user-defined
        """
        dcv_range = ctypes.c_double(dcv_range)
        dcv_res = ctypes.c_double(dcv_res)
        autorange = ctypes.c_bool(False)
        self.dll.AgM918x_DCVoltConfigureAll(self.session, dcv_range, dcv_res, autorange)

    def config_multi_point_meas(self, trig_count, sample_count, trig_src_enum, sample_interval):
        """
        Configures the properties for multipoint measurements.
        REMARK: (from documentation) This function is implemented by calling into the instrument-specific MultiSample Configure function.
        Therefore, the TriggerCount parameter must be 1, and the SampleTrigger parameter must specify Interval
        triggering. Also, the trigger source must be something other than Immediate.

        :param trig_count: int, must be 1
        :param sample_count: int32, sets the number of measurements the DMM makes
         in each measurement sequence initiated by a trigger.

        :param trig_src_enum: enum, specifies the sample trigger source to use.
         Enum defined in AgM918x TriggerSources class
        :param sample_trigger: int32, this parameter must specify Interval sampling (10)

        :param sample_interval: dbl, sets the amount of time in seconds the DMM waits between measurement cycles.
        Specify a sample interval to add settling time between measurement cycles or to decrease
        the measurement rate. Sample Interval only applies when the Sample Trigger is set to Interval.

        On the NI 4060, the Sample Interval value is used as the settling time. (Same on AgM918x??)
        When sample interval is set to 0, the DMM does not settle between measurement cycles.
        The NI 4065 and NI 4070/4071/4072 use the value specified in Sample Interval as additional delay.
        The default value (-1) ensures that the DMM settles for a recommended time.
        This is the same as using an Immediate trigger.
        """
        self.config_dict['triggerCount'] = trig_count
        self.config_dict['sampleCount'] = sample_count
        self.config_dict['triggerSource'] = trig_src_enum.name
        self.config_dict['sampleInterval'] = sample_interval
        trig_count = ctypes.c_int32(trig_count)
        sample_count = ctypes.c_int32(sample_count)
        sample_trig = ctypes.c_int32(trig_src_enum.value)
        sample_interval = ctypes.c_double(sample_interval)
        self.dll.AgM918x_ConfigureMultiPoint(self.session, trig_count, sample_count, sample_trig, sample_interval)

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
        valid ranges: 0.1, 1.0, 10.0, 100.0, 1000.0
        Auto range ON: -1.0
        Auto range OFF: -2.0
        Auto range ONCE: -3.0
        """
        val = self.read_property_node(ctypes.c_double(), 2,
                                      attr_base_str='IVI_CLASS_PUBLIC_ATTR_BASE')
        return val

    def set_range(self, range_val):
        """
        function for only setting the range of the dmm
        :param range_val: dbl, value for the range.
        valid ranges: 0.1, 1.0, 10.0, 100.0, 1000.0
        Auto range ON: -1.0
        Auto range OFF: -2.0
        Auto range ONCE: -3.0
        """
        self.config_dict['range'] = range_val
        self.set_property_node(ctypes.c_double(range_val), 2,
                               attr_base_str='IVI_CLASS_PUBLIC_ATTR_BASE')
        pass

    ''' Measurement Options'''

    def config_power_line_freq(self, pwr_line_freq):
        """
        Specifies the powerline frequency.
        :param pwr_line_freq: dbl, 50 or 60 Hz
        """
        self.config_dict['powerLineFrequency'] = pwr_line_freq
        self.dll.AgM918x_ConfigurePowerLineFrequency(self.session, ctypes.c_double(pwr_line_freq))

    def config_auto_zero(self, auto_zero_mode):
        """
        AgM918x seems to not support auto zero function.
        !!!
        Configures the DMM for Auto Zero. When Auto Zero is ON, the DMM internally disconnects the input
         and takes a zero reading. It then subtracts the zero reading from the measurement.
          This prevents offset voltages present on the input circuitry of the DMM from affecting
           measurement accuracy. When Auto Zero is OFF, the DMM does not compensate for zero reading offset.
        :param auto_zero_mode: int,
        Auto (default) (-1)  NI-DMM chooses the Auto Zero setting based on the configured function and resolution.
        Off (0) Disables Auto Zero.  Note  The NI 4065 does not support this setting.
        On (1) The DMM internally disconnects the input signal following each measurement and takes a zero reading.
            It then subtracts the zero reading from the preceding reading.
        Once (2) The DMM internally disconnects the input signal following each measurement and takes a zero reading.
            It then subtracts the zero reading from the preceding reading.
        """
        pass

    def config_adc_cal(self, adc_cal_mode):
        """
        AgM918x seems not to support adc cal function
        !!!
        Allows the DMM to compensate for gain drift since the last external or self-calibration.
         When ADC Calibration is ON, the DMM measures an internal reference to calculate the correct gain
          for the measurement. When ADC Calibration is OFF, the DMM does not compensate for changes to the gain.
        :param adc_cal_mode: int,
        ON (-1) The DMM measures an internal reference to calculate the correct gain for the measurement.
        OFF (0) The DMM does not compensate for changes to the gain.
        AUTO (1) The DMM enables or disables ADC calibration based on the configured function and resolution.
        """
        pass


    ''' Trigger '''

    def config_trigger(self, trig_src_enum, trig_delay):
        """
        Configures the DMM trigger source and trigger delay.
        :param trig_src_enum: enum, specifies the sample trigger source to use.
         Enum defined in AgM918xTriggerSources class.

        :param trig_delay: dbl, The length of time between when the DMM receives the trigger
        and when it takes a measurement (in seconds).
        By default, Trigger Delay is -1, which means the DMM waits an appropriate settling time
        before taking the measurement.

        valid values:
        50µs to 15.0 s (resolution is 1µs up to 65µs then 16µs)
        -1.0: auto delay ON
        -2.0: auto delay OFF
        """
        self.config_dict['triggerSource'] = trig_src_enum.name
        self.config_dict['triggerDelay_s'] = trig_delay
        self.dll.AgM918x_ConfigureTrigger(
            self.session, ctypes.c_int32(trig_src_enum.value), ctypes.c_double(trig_delay))
        pass

    def config_trigger_slope(self, trig_slope):
        """
        Sets the Trigger Slope property to either rising edge (positive) or falling edge (negative) polarity.
        :param trig_slope: int,
        Rising Edge: 0
        Falling Edge: 1 (default)
        """
        self.config_dict['triggerSlope'] = 'falling' if trig_slope else 'rising'
        self.dll.AgM918x_ConfigureTriggerSlope(self.session, ctypes.c_int32(trig_slope))

    def config_meas_complete_dest(self, meas_compl_dest_enum):
        """
        Specifies the destination of the DMM Measurement Complete (MC) signal.
        :param meas_compl_dest: enum, as defined in AgM918xMeasCompleteLoc class.
        """
        self.config_dict['measurementCompleteDestination'] = meas_compl_dest_enum.name
        if meas_compl_dest_enum != AgM918xMeasCompleteLoc.software:
            self.dll.AgM918x_ConfigureMeasCompleteDest(self.session, ctypes.c_int32(meas_compl_dest_enum.value))

    def send_software_trigger(self):
        """
        sends a command to trigger the DMM
        Function seems to be not supported by AgM918x. There is no AgM918x_SendSoftwareTrigger
        """
        pass

    ''' Actual Values '''



    ''' Measurement '''

    def initiate_measurement(self):
        """
        Initiates an acquisition.
         After you call this VI, the DMM leaves the Idle state and enters the Wait-For-Trigger state.
          If trigger is set to Immediate mode, the DMM begins acquiring measurement data.
          Use fetch_single_meas(), AgM918x Fetch Multi Point, or AgM918x Fetch Waveform to retrieve the measurement data.
        """
        self.get_error_message(self.dll.AgM918x_Initiate(self.session))
        #---SINCE READSTATUS() DOES NOT WORK, WE WILL EXCLUDE IT AND USE A SIMPLE CALL OF INITIATE---
        #tries = 0
        #max_tries = 10
        #while self.readstatus()[1] != 0 and tries <= max_tries:
        #    tries += 1
        #if self.readstatus()[1] == 0:
        #    logging.debug('successfully started measurement on AgM918x after %s tries' % tries)
        #else:
        #    logging.error('error: could not started measurement on AgM918x after %s tries' % tries)
        #    logging.error('error: status of AgM918x is: backlog: %s acquisition state: %s' % self.readstatus())
        self.state = 'measuring'

    def fetch_single_meas(self, max_time_ms=-1):
        """
        Returns the value from a previously initiated measurement.
        You must call initiate_measurement() before calling this function.
        :param max_time_ms: int, specifies the maximum time allowed for this VI to complete in milliseconds.
         0-86400000 allowed.
         -1 means Auto
        :return: dbl, measurement value
        """
        max_time_ms = ctypes.c_int32(max_time_ms)
        read = ctypes.c_double()
        self.dll.AgM918x_Fetch(self.session, max_time_ms, ctypes.byref(read))
        return read.value

    def fetch_multiple_meas(self, num_to_read=-1, max_time_ms=-1):
        """
         Returns an array of values from a previously initiated multipoint measurement.
         The number of measurements the DMM makes is determined by the values you specify
         for the Trigger Count and Sample Count parameters of AgM918x Configure Multi Point.
         You must call initiate_measurement() to initiate a measurement before calling this function
         :param max_time_ms: int, specifies the maximum time allowed for this VI to complete in milliseconds.
          0-86400000 allowed.
          -1 means Auto
         :param num_to_read: int, specifies the number of measurements to acquire.
         The maximum number of measurements for a finite acquisition is the (Trigger Count x Sample Count)
         parameters in config_multi_point_meas()
         For continuous acquisitions, up to 100,000 points can be returned at once.
         num_to_read = -1 will read all available data
         :return: numpy array with all values
         """
        max_time = ctypes.c_int32(max_time_ms)
        # ---SINCE READSTATUS() DOES NOT WORK, WE WILL EXCLUDE IT AND USE A SIMPLE CALL OF INITIATE---
        #BUT then num_to_read cannot be -1 as default read all, but must be set to the number ov values available
        #????How do we get the number of available readings???? For now assume its the same as in samplecount
        #if num_to_read < 0:  # read all available
        #    num_to_read = self.readstatus()[0]
        #if num_to_read == 0:
        #    return np.zeros(0, dtype=np.double)
        num_to_read = self.config_dict.get('sampleCount')
        print("Called fetch_multiple_meas() function. num_to_read is: " + str(num_to_read))
        array_of_values = ctypes.c_double * num_to_read
        array_of_values = array_of_values()
        number_to_read = ctypes.c_int32(num_to_read)
        number_of_read_elements = ctypes.c_int32()
        self.dll.AgM918x_FetchMultiPoint(
            self.session, max_time, number_to_read, array_of_values, ctypes.byref(number_of_read_elements))
        ret = np.ctypeslib.as_array(array_of_values)[0:number_of_read_elements.value]
        if ret.any():
            t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # take last element out of array and make a tuple with timestamp:
            self.last_readback = (round(ret[-1], 8), t)
        return ret

    def readstatus(self):
        """
        Readstatus seems not to be available for AgM918x.
        Either we have to work around it by using other functions (which?) or we remove all uses of readstatus.
        Should: Returns measurement backlog and acquisition status.
        :return: (backlog, acqusition_state)
          * Backlog specifies the number of measurements available to be read.
           If the backlog continues to increase, data is eventually overwritten, resulting in an error.
          * Acquisition State indicates status of the acquisition.
                0 Running
                1 Finished with Backlog
                2 Finished with no Backlog
                3 Paused
                4 No acquisition in progress
        """
        #Dummy implementation!!! REMOVE BEFORE USE
        ret_backlog = ctypes.c_int32(1)
        ret_acqstate = ctypes.c_int16(4)
        return ret_backlog.value, ret_acqstate.value

    def abort_meas(self):
        self.state = 'aborted'
        self.dll.AgM918x_Abort(self.session)
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
            logging.error(
                comment + ' errorcode is: ' + str(ret_status.value) +
                '\n error message is: ' + ret_message.value.decode('utf-8'))
        return ret_status.value, ret_message.value.decode('utf-8')

    def get_initiated(self):
        '''
        Returns true if the DMM is currently measuring.
        :return: bool
        '''
        ret = None
        ret = ctypes.c_bool(ret)
        self.dll.AgM918x_MeasurementGetInitiated(self.session, ctypes.byref(ret))
        return ret

    ''' self calibration '''

    def self_calibration(self):
        pass

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
            self.config_dict = deepcopy(config_dict)
            if reset_dev:
                self.reset_dev()
            dmm_range = config_dict.get('range')
            resolution = config_dict.get('resolution')
            self.config_measurement(dmm_range, resolution)
            self.config_dcvoltage_measurement(dmm_range, resolution)
            trig_count = config_dict.get('triggerCount')
            sample_count = config_dict.get('sampleCount')
            trig_src_enum = AgM918xTriggerSources[config_dict.get('triggerSource')]
            sample_interval = config_dict.get('sampleInterval')
            self.config_multi_point_meas(trig_count, sample_count, trig_src_enum, sample_interval)
            auto_z = config_dict.get('autoZero')
            self.config_auto_zero(auto_z)
            self.config_adc_cal(0)
            pwr_line_freq = config_dict.get('powerLineFrequency')
            self.config_power_line_freq(pwr_line_freq)
            trig_delay = config_dict.get('triggerDelay_s', -1.0)
            self.config_trigger(trig_src_enum, trig_delay)
            trig_slope = config_dict.get('triggerSlope')
            if trig_slope == 'falling':
                trig_slope = 1
            else:
                trig_slope = 0
            self.config_trigger_slope(trig_slope)
            self.config_sample_trigger_slope(trig_slope)
            meas_compl_dest_enum = AgM918xMeasCompleteLoc[config_dict.get('measurementCompleteDestination')]
            self.config_meas_complete_dest(meas_compl_dest_enum)
            self.config_meas_complete_slope(0)
            greater_10_g_ohm = config_dict.get('highInputResistanceTrue')
            self.set_input_resistance(greater_10_g_ohm)
            self.get_accuracy()
            # just to be sure this is included:
            self.config_dict['assignment'] = self.config_dict.get('assignment', 'offset')
        except Exception as e:
            print('Exception while loading config to Ni4071: ', e)

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
            'range': ('range', True, float, [-3.0, -2.0, -1.0, 0.2, 2.0, 20.0, 200.0, 300.0], self.config_dict['range']),
            'resolution': ('resolution', True, float, [3.5, 4.5, 5.5, 6.5], self.config_dict['resolution']),
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
        error_dict_1y = {0.2: (30, 5), 2: (20, 2), 20: (40, 6), 200: (30, 2), 2000: (1300, 2)}
        dmm_range = config_dict['range']
        reading_accuracy_float, range_accuracy_float = error_dict_1y.get(dmm_range)
        reading_accuracy_float *= 10 ** -6 #ppm
        range_accuracy_float *= 10 **-6 #ppm
        range_accuracy_float *= dmm_range
        acc_tpl = (reading_accuracy_float, range_accuracy_float)
        config_dict['accuracy'] = acc_tpl
        return acc_tpl



# dmm = DMMdummy()
# dmm.set_to_pre_conf_setting('periodic')