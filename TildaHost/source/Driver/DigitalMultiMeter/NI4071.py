"""

Created on '19.05.2015'

@author:'simkaufm'

"""
import os
import time
import numpy as np
import ctypes
import datetime


class Ni4071:
    """
    Class for accessing the Nationale Instruments 4071 digital Multimeter.
    """

    def __init__(self, reset=True, dev_name_str='PXI1Slot5', pwr_line_freq=50):
        dll_path = '..\\..\\..\\binary\\nidmm_32.dll'
        dev_name = ctypes.create_string_buffer(dev_name_str.encode('utf-8'))

        self.dll = ctypes.WinDLL(dll_path)
        self.session = ctypes.c_uint32(0)
        self.init(dev_name, reset_dev=reset)
        self.config_power_line_freq(pwr_line_freq)

    ''' Init and close '''

    def init(self, dev_name, id_query=True, reset_dev=True):
        """
        Creates a new session to the instrument
        :return: self.session
        """
        self.dll.niDMM_init(dev_name, id_query, reset_dev, ctypes.byref(self.session))
        return self.session

    def de_init_dmm(self):
        ''' Closes the current session to the instrument. '''
        self.dll.niDMM_close(self.session)

    ''' Configure '''

    def config_meas_digits(self, func=1, dmm_range=10.0, resolution=7.5):
        """
        Configures the common properties of the measurement.
        :param func: int, DC volts, AC volts and so on
        1: DC Volts
        2: AC Volts
        3: DC Current
        4: AC Current
        5: 2 Wire Res
        101: 4 Wire Res
        106: AC Plus DC Volts
        107: AC Plus DC Current
        104: Freq
        105: period
        108: temperature
        :param dmm_range: dbl, range 10.00, 100.00, etc.
        valid ranges: 0.1, 1.0, 10.0, 100.0, 1000.0
        Auto range ON: -1.0
        Auto range OFF: -2.0
        Auto range ONCE: -3.0
        :param resolution: dbl, resolution in digits
        3.5 (3.5000000E+0) Specifies 3.5 digits resolution.
        4.5 (4.500000E+0) Specifies 4.5 digits resolution.
        5.5 (5.500000E+0) Specifies 5.5 digits resolution.
        6.5 (6.500000E+0) Specifies 6.5 digits resolution.
        7.5 (7.500000E+0) Specifies 7.5 digits resolution.
        """
        func = ctypes.c_int32(func)
        dmm_range = ctypes.c_double(dmm_range)
        res = ctypes.c_double(resolution)
        self.dll.niDMM_ConfigureMeasurementDigits(self.session, func, dmm_range, res)

    def config_multi_point_meas(self, trig_count, sample_count, sample_trig, sample_interval):
        """
        Configures the properties for multipoint measurements.
        :param trig_count: int, nuzmber of triggers before returning to idle state
        :param sample_count: int32, sets the number of measurements the DMM makes
         in each measurement sequence initiated by a trigger.

        :param sample_trig: specifies the sample trigger source to use.
        Immediate (1) No trigger specified
        Interval (10) Interval trigger
        External (2) Pin 9 on the AUX Connector
        Software Trig (3) Configures the DMM to wait until niDMM Send Software Trigger is called.
        TTL 0 (111) PXI Trigger Line 0
        TTL 1 (112) PXI Trigger Line 1
        TTL 2 (113) PXI Trigger Line 2
        TTL 3 (114) PXI Trigger Line 3
        TTL 4 (115) PXI Trigger Line 4
        TTL 5 (116) PXI Trigger Line 5
        TTL 6 (117) PXI Trigger Line 6
        TTL 7 (118) PXI Trigger Line 7
        PXI Star (131) PXI Star trigger line
        LBR Trig 1 (1004) Local Bus Right Trigger Line 1 of PXI/SCXI combination chassis
        AUX Trig 1 (1001) Pin 3 on the AUX connector

        :param sample_interval: dbl, sets the amount of time in seconds the DMM waits between measurement cycles.
        Specify a sample interval to add settling time
        between measurement cycles or to decrease the measurement rate. Sample Interval only applies
        when the Sample Trigger is set to Interval. On the NI 4060, the Sample Interval value
        is used as the settling time. When sample interval is set to 0, the DMM does not settle
        between measurement cycles. The NI 4065 and NI 4070/4071/4072 use the value specified in
        Sample Interval as additional delay. The default value (-1) ensures that the DMM settles
        for a recommended time. This is the same as using an Immediate trigger.
        """
        trig_count = ctypes.c_int32(trig_count)
        sample_count = ctypes.c_int32(sample_count)
        sample_trig = ctypes.c_int32(sample_trig)
        sample_interval = ctypes.c_double(sample_interval)
        self.dll.niDMM_ConfigureMultiPoint(self.session, trig_count, sample_count, sample_trig, sample_interval)

    def conf_waveform_meas(self, func, wfm_range, rate, points):
        """
        Configures the NI 4070/4071/4072 for waveform acquisitions.
        :param func: int,
        WAVEFORM VOLTAGE (default) (1003) Waveform Voltage
        WAVEFORM CURRENT (1004) Waveform Current
        :param wfm_range: dbl,
        :param rate: dbl, specifies the rate of the acquisition in samples per second.
        The valid rate is 10.0 – 1,800,000 S/s. Values are coerced to the closest integer
        divisor of 1,800,000. The default value is 1,800,000.
        :param points: int, waveform points
        """
        func = ctypes.c_int32(func)
        wfm_range = ctypes.c_double(wfm_range)
        rate = ctypes.c_double(rate)
        points = ctypes.c_int32(points)
        self.dll.niDMM_ConfigureWaveformAcquisition(self.session, func, wfm_range, rate, points)

    ''' Measurement Options '''

    def config_power_line_freq(self, pwr_line_freq):
        """
        Specifies the powerline frequency.
        :param pwr_line_freq: dbl, 50 or 60 Hz
        """
        self.dll.niDMM_ConfigurePowerLineFrequency(self.session, ctypes.c_double(pwr_line_freq))

    def config_auto_zero(self, auto_zero_mode):
        """
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
        self.dll.niDMM_ConfigureAutoZeroMode(self.session, ctypes.c_int32(auto_zero_mode))

    def config_adc_cal(self, adc_cal_mode):
        """
        Allows the DMM to compensate for gain drift since the last external or self-calibration.
         When ADC Calibration is ON, the DMM measures an internal reference to calculate the correct gain
          for the measurement. When ADC Calibration is OFF, the DMM does not compensate for changes to the gain.
        :param adc_cal_mode: int,
        ON (-1) The DMM measures an internal reference to calculate the correct gain for the measurement.
        OFF (0) The DMM does not compensate for changes to the gain.
        AUTO (1) The DMM enables or disables ADC calibration based on the configured function and resolution.
        """
        self.dll.niDMM_ConfigureADCCalibration(self.session, ctypes.c_int32(adc_cal_mode))

    def config_offset_comp_ohms(self, off_comp_mode):
        """ Allows the DMM to compensate for voltage offsets in resistance measurements.
         When Offset Compensated Ohms is enabled, the DMM measures the resistance twice
          (once with the current source on and again with it turned off).
           Any voltage offset present in both measurements is cancelled out.
            Offset Compensated Ohms is useful when measuring resistance values less than 10 kOhms.
        OFF: 0
        ON: 1
        """
        self.dll.niDMM_ConfigureOffsetCompOhms(self.session, ctypes.c_int32(off_comp_mode))

    def config_ac_bandwith(self, min_freq, max_freq):
        """
        Configures the Min Frequency and Max Frequency properties that the DMM uses for AC measurements.
        :param min_freq: dbl, Minimum Frequency   Hz
        :param max_freq: dbl, Maximum Freq. Hz
        """
        self.dll.niDMM_ConfigureACBandwidth(self.session, ctypes.c_double(min_freq), ctypes.c_double(max_freq))

    def config_freq_volt_range(self, freq_volt):
        """
         Specifies the expected maximum amplitude of the input signal for frequency and period measurements.
        :param freq_volt: dbl,
        ranges: 0.1, 1.0, 10.0, 100.0, 1000.0 V
        or
        Auto Range ON: -1.0
        Auto Range OFF: -2.0
        """
        self.dll.niDMM_ConfigureFrequencyVoltageRange(self.session, ctypes.c_double(freq_volt))

    def config_current_source(self, src):
        """
        Configures the current source for diode measurements. The NI 4050 and NI 4060 are not supported.
        :param src: dbl, specifies the current source provided during diode measurements.
        valid values: 0.000001, 0.000010, 0.000100, 0.001000 A
        ranges: Diode 10 V at 1 µA / 10µA / 100µA; 4.0 V at 1 mA
        """
        self.dll.niDMM_ConfigureCurrentSource(self.session, ctypes.c_double(src))

    def config_wavefrom_coupling(self, coupling):
        """
        Configures instrument coupling for voltage waveforms.
        :param coupling: int, selects DC or AC coupling
        AC coupling: 0
        DC coupling: 1 (default)
        """
        self.dll.niDMM_ConfigureWaveformCoupling(self.session, ctypes.c_int32(coupling))

    ''' Triggers '''

    def config_trigger(self, trig_src, trig_delay):
        """
        Configures the DMM trigger source and trigger delay.
        :param trig_src: int, specifies the sample trigger source to use.
        Immediate (1) No trigger specified
        Interval (10) Interval trigger
        External (2) Pin 9 on the AUX Connector
        Software Trig (3) Configures the DMM to wait until niDMM Send Software Trigger is called.
        TTL 0 (111) PXI Trigger Line 0
        TTL 1 (112) PXI Trigger Line 1
        TTL 2 (113) PXI Trigger Line 2
        TTL 3 (114) PXI Trigger Line 3
        TTL 4 (115) PXI Trigger Line 4
        TTL 5 (116) PXI Trigger Line 5
        TTL 6 (117) PXI Trigger Line 6
        TTL 7 (118) PXI Trigger Line 7
        PXI Star (131) PXI Star trigger line
        LBR Trig 1 (1004) Local Bus Right Trigger Line 1 of PXI/SCXI combination chassis
        AUX Trig 1 (1001) Pin 3 on the AUX connector

        :param trig_delay: dbl, NI-DMM sets the Trigger Delay property to this value.
        By default, Trigger Delay is -1, which means the DMM waits an appropriate settling time
        before taking the measurement. The NI 4065 and NI 4070/4071/4072 use the value
        specified in Trigger Delay as ADDITIONAL settling time.

        valid values:
        0.0 to 149.0 s (resolution is 0.1 s)
        -1.0: auto delay ON
        -2.0: auto delay OFF
        """
        self.dll.niDMM_ConfigureTrigger(self.session, ctypes.c_int32(trig_src), ctypes.c_double(trig_delay))

    def send_software_trigger(self):
        """
        sends a command to trigger the DMM
        """
        self.dll.niDMM_SendSoftwareTrigger(self.session)

    def config_trigger_slope(self, trig_slope):
        """
        Sets the Trigger Slope property to either rising edge (positive) or falling edge (negative) polarity.
        :param trig_slope: int,
        Rising Edge: 0
        Falling Edge: 1 (default)
        """
        self.dll.niDMM_ConfigureTriggerSlope(self.session, ctypes.c_int32(trig_slope))

    def config_sample_trigger_slope(self, smpl_trig_slope):
        """
        Sets the Sample Trigger Slope property to either rising edge (positive) or falling edge (negative) polarity.
        (For multipoint acquisition)
        :param smpl_trig_slope: int,
        Rising Edge: 0
        Falling Edge: 1 (default)
        """
        self.dll.niDMM_ConfigureSampleTriggerSlope(self.session, ctypes.c_int32(smpl_trig_slope))

    def config_meas_complete_dest(self, meas_compl_dest):
        """
        Specifies the destination of the DMM Measurement Complete (MC) signal.
        :param meas_compl_dest: int, This signal is issued when the DMM completes a single measurement.
         This signal is commonly referred to as Voltmeter Complete.

        None (-1) No destination specified.
        External (2) Pin 6 on the AUX Connector
        TTL 0 (111) PXI Trigger Line 0
        TTL 1 (112) PXI Trigger Line 1
        TL 2 (113) PXI Trigger Line 2
        TTL 3 (114) PXI Trigger Line 3
        TL 4 (115) PXI Trigger Line 4
        TTL 5 (116) PXI Trigger Line 5
        TTL 6 (117) PXI Trigger Line 6
        TTL 7 (118) PXI Trigger Line 7
        LBR Trig 0 (1003) Local Bus Right Trigger Line 0 of PXI/SCXI combination chassis
        """
        self.dll.niDMM_ConfigureMeasCompleteDest(self.session, ctypes.c_int32(meas_compl_dest))

    def config_meas_complete_slope(self, meas_compl_slope):
        """
        Sets the Measurement Complete signal to either rising edge (positive) or falling edge (negative) polarity.
        :param meas_compl_slope: int,
        Rising Edge: 0
        Falling Edge: 1 (default)
        """
        self.dll.niDMM_ConfigureMeasCompleteSlope(self.session, ctypes.c_int32(meas_compl_slope))

    ''' Actual Values '''

    def get_auto_range_value(self):
        """
        Returns the actual range that the DMM is using, even when auto ranging is off.
        :return: dbl, auto range value
        """
        ret_val = ctypes.c_double()
        self.dll.niDMM_GetAutoRangeValue(self.session, ctypes.byref(ret_val))
        return ret_val.value

    def get_aperture_time_info(self):
        """
        Returns the aperture time and aperture time units.
        specifies the amount of time the DMM digitizes the input signal for a single measurement.
        This parameter does not include settling time.
        On the NI 4070/4071/4072, the minimum aperture time is 8.89 µs, and the maximum aperture time is 149 s.
        Any number of powerline cycles (PLCs) within the minimum and maximum ranges
        is allowed on the NI 4070/4071/4072.
        :return: (dbl, int), (aperture time, aperture time units seconds (0) or PLC (1))
        """
        apt_t = ctypes.c_double()
        apt_u = ctypes.c_int32()
        self.dll.niDMM_GetApertureTimeInfo(self.session, ctypes.byref(apt_t), ctypes.byref(apt_u))
        return apt_t.value, apt_u.value

    def get_meas_period(self):
        """
        Returns the Measurement Period, which is the amount of time it takes to complete
        one measurement with the current configuration.
        :return: dbl, returns the number of seconds it takes to make one measurement.
        The first measurement in a multipoint acquisition requires additional settling time.
        """
        meas_per = ctypes.c_double()
        self.dll.niDMM_GetMeasurementPeriod(self.session, ctypes.byref(meas_per))
        return meas_per.value

    ''' Acquisition '''

    def read_single_voltage(self, max_time_ms=-1):
        """
        Acquires a single measurement and returns the measured value.
        :param max_time_ms: int, specifies the maximum time allowed for this VI to complete in milliseconds.
        If the VI does not complete within this time interval, the VI returns the
        NIDMM_ERROR_MAX_TIME_EXCEEDED error code. This may happen if an external trigger has not been received,
        or if the specified timeout is not long enough for the acquisition to complete.
        The valid range is 0–86400000.
        The default value is TIME LIMIT AUTO (-1). The DMM calculates the timeout automatically.
        :return: dbl, voltage
        """
        max_time_ms = ctypes.c_int32(max_time_ms)
        read = ctypes.c_double()
        self.dll.niDMM_Read(self.session, max_time_ms, ctypes.byref(read))
        return read.value

    def read_multi_point(self, number_to_read, max_time=-1):
        """
        Acquires multiple measurements and returns an array of measured values.
        The number of measurements the DMM makes is determined by the values you specify
        for the Trigger Count and Sample Count parameters of niDMM Configure Multi Point.
        :param max_time: int, specifies the maximum time allowed for this VI to complete in milliseconds.
         0-86400000 allowed.
         -1 means Auto
        :param number_to_read: specifies the number of measurements to acquire.
        The maximum number of measurements for a finite acquisition is the (Trigger Count x Sample Count)
        parameters in niDMM Configure Multi Point.
        For continuous acquisitions, up to 100,000 points can be returned at once.
        The number of measurements can be a subset.
        The valid range is any positive ViInt32. The default value is 1.
        :return: numpy array with all values
        """
        max_time = ctypes.c_int32(max_time)
        array_of_values = ctypes.c_double * number_to_read
        array_of_values = array_of_values()
        number_to_read = ctypes.c_int32(number_to_read)
        number_of_read_elements = ctypes.c_int32()
        self.dll.niDMM_ReadMultiPoint(
            self.session, max_time, number_to_read, array_of_values, ctypes.byref(number_of_read_elements))
        return np.ctypeslib.as_array(array_of_values)[0:number_of_read_elements.value]

    def read_waveform(self, max_time=-1, num_to_read=1):
        """
        Acquires a waveform and returns data as an array of values or as a waveform data type.
        :param max_time: int, specifies the maximum time allowed for this VI to complete in milliseconds.
         0-86400000 allowed.
         -1 means Auto
        :param num_to_read: int, specifies the number of waveform points to return.
        :return: numpy array with all values
        """
        max_time = ctypes.c_int32(max_time)
        array_of_values = ctypes.c_double * num_to_read
        array_of_values = array_of_values()
        number_to_read = ctypes.c_int32(num_to_read)
        number_of_read_elements = ctypes.c_int32()
        self.dll.niDMM_ReadWaveform(
            self.session, max_time, number_to_read, array_of_values, ctypes.byref(number_of_read_elements))
        return np.ctypeslib.as_array(array_of_values)[0:number_of_read_elements.value]

    def check_over_range(self, measurement):
        """
        Takes a measurement value and determines if the value is a valid measurement
         or a value indicating that an overrange condition occurred.
        :param measurement: dbl, is the measured value returned from the DMM.
        Note  If an overrange condition occurs, the Measurement value
        contains an IEEE-defined NaN (Not a Number) value.
        :return: bool, True if overrange
        """
        ret_bool = ctypes.c_bool()
        self.dll.niDMM_IsOverRange(self.session, ctypes.c_double(measurement), ctypes.byref(ret_bool))
        return ret_bool.value

    ''' low-level Acquisition '''

    def initiate_measurement(self):
        """
        Initiates an acquisition.
         After you call this VI, the DMM leaves the Idle state and enters the Wait-For-Trigger state.
          If trigger is set to Immediate mode, the DMM begins acquiring measurement data.
          Use fetch_single_meas(), niDMM Fetch Multi Point, or niDMM Fetch Waveform to retrieve the measurement data.
        """
        self.dll.niDMM_Initiate(self.session)

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
        self.dll.niDMM_Fetch(self.session, max_time_ms, ctypes.byref(read))
        return read.value

    def fetch_multiple_meas(self, max_time_ms=-1, num_to_read=4):
        """
        Returns an array of values from a previously initiated multipoint measurement.
        The number of measurements the DMM makes is determined by the values you specify
        for the Trigger Count and Sample Count parameters of niDMM Configure Multi Point.
        You must call initiate_measurement() to initiate a measurement before calling this function
        :param max_time_ms: int, specifies the maximum time allowed for this VI to complete in milliseconds.
         0-86400000 allowed.
         -1 means Auto
        :param num_to_read: int, specifies the number of measurements to acquire.
        The maximum number of measurements for a finite acquisition is the (Trigger Count x Sample Count)
        parameters in config_multi_point_meas()
        For continuous acquisitions, up to 100,000 points can be returned at once.
        :return: numpy array with all values
        """
        max_time = ctypes.c_int32(max_time_ms)
        array_of_values = ctypes.c_double * num_to_read
        array_of_values = array_of_values()
        number_to_read = ctypes.c_int32(num_to_read)
        number_of_read_elements = ctypes.c_int32()
        self.dll.niDMM_FetchMultiPoint(
            self.session, max_time, number_to_read, array_of_values, ctypes.byref(number_of_read_elements))
        return np.ctypeslib.as_array(array_of_values)[0:number_of_read_elements.value]

    def fetch_waveform(self, max_time_ms=-1, num_to_read=1):
        """
        Returns an array of values from a previously initiated multipoint measurement.
        The number of measurements the DMM makes is determined by the values you specify
        for the Trigger Count and Sample Count parameters of niDMM Configure Multi Point.
        You must call initiate_measurement() to initiate a measurement before calling this function
        :param max_time_ms: int, specifies the maximum time allowed for this VI to complete in milliseconds.
         0-86400000 allowed.
         -1 means Auto
        :param num_to_read: int, specifies the number of measurements to acquire.
        The maximum number of measurements for a finite acquisition is the (Trigger Count x Sample Count)
        parameters in config_multi_point_meas()
        For continuous acquisitions, up to 100,000 points can be returned at once.
        :return: numpy array with all values
        """
        max_time = ctypes.c_int32(max_time_ms)
        array_of_values = ctypes.c_double * num_to_read
        array_of_values = array_of_values()
        number_to_read = ctypes.c_int32(num_to_read)
        number_of_read_elements = ctypes.c_int32()
        self.dll.niDMM_FetchWaveform(
            self.session, max_time, number_to_read, array_of_values, ctypes.byref(number_of_read_elements))
        return np.ctypeslib.as_array(array_of_values)[0:number_of_read_elements.value]

    def readstatus(self):
        """
        Returns measurement backlog and acquisition status.
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
        ret_backlog = ctypes.c_int32()
        ret_acqstate = ctypes.c_int16()
        self.dll.niDMM_ReadStatus(self.session, ctypes.byref(ret_backlog), ctypes.byref(ret_acqstate))
        return ret_backlog.value, ret_acqstate.value

    def abort_meas(self):
        """
        Aborts a previously initiated measurement and returns the DMM to the Idle state.
        """
        self.dll.niDMM_Abort(self.session)

    ''' Utility '''

    def reset_dev(self):
        """
        Resets the instrument to a known state and sends initialization commands to the instrument.
        The initialization commands set instrument settings to the state
        necessary for the operation of the instrument driver.
        :return:
        """
        self.dll.niDMM_reset(self.session)

    def self_test(self):
        """
        Performs a self test on the DMM to ensure that the DMM is functioning properly.
        Self test does not calibrate the DMM.
        This function calls reset_dev(), and any configurations previous to the call will be lost.
        All properties will be set to their default values after the call returns.
        :return: (int, str), (test_res_num, test_res_message):
            * test_res_num, contains the value returned from the instrument self test. Zero indicates success.
            * For the NI 4065 and NI 4070/4071/4072, the error code returned for a self-test failure
             is NIDMM_ERROR_SELF_TEST_FAILURE. This error code indicates that the DMM should be repaired.
        """
        test_res = ctypes.c_int16()
        test_message = ctypes.create_string_buffer("".encode('utf-8'), 256)
        self.dll.niDMM_self_test(self.session, ctypes.byref(test_res), test_message)
        return test_res.value, test_message.value.decode('utf-8')

    def revision_query(self):
        """
        Returns the revision numbers of the instrument driver and instrument firmware.
        :return: (str, str), (instr. driver rev, firmware rev.)
        """
        instr_driver_rev = ctypes.create_string_buffer("".encode('utf-8'), 256)
        firmware_rev = ctypes.create_string_buffer("".encode('utf-8'), 256)
        self.dll.niDMM_revision_query(self.session, instr_driver_rev, firmware_rev)
        return instr_driver_rev.value.decode('utf-8'), firmware_rev.value.decode('utf-8')

    def get_digits_of_precision(self):
        """
        Returns the digits of precision calculated from the range and resolution information
        specified in config_meas_digits()
        :return: dbl, digits
        """
        ret_digits = ctypes.c_double()
        self.dll.niDMM_GetDigitsOfPrecision(self.session, ctypes.byref(ret_digits))
        return ret_digits.value

    def get_error_message(self):
        """
        Takes the error cluster returned by the VIs, interprets it, and returns it as a user-readable string.
        :return: int, str
        """
        ret_status = ctypes.c_int32(0)
        ret_message = ctypes.create_string_buffer("".encode('utf-8'), 256)
        self.dll.niDMM_error_message(self.session, ret_status, ret_message)
        return ret_status.value, ret_message.value.decode('utf-8')

    ''' Calibration '''

    def self_calibration(self):
        """
        Executes the self-calibration routine to maintain measurement accuracy.

        Note  This VI calls niDMM Reset, and any configurations previous to the call will be lost.
        All properties will be set to their default values after the call returns.
        """
        self.dll.niDMM_SelfCal(self.session)

    def get_calibration_count(self, internal=True):
        """
        Returns the calibration count for the specified type of calibration.
        :param internal: bool, True for internal, False for external
        :return: int, number of calbrations of that type.
        """
        cal_type = ctypes.c_int32(1)
        if internal:
            cal_type = ctypes.c_int32(0)
        cal_count = ctypes.c_int32()
        self.dll.niDMM_GetCalCount(self.session, cal_type, ctypes.byref(cal_count))
        return cal_count.value

    def get_dmm_dev_temp(self):
        """
        Returns the current temperature of the NI 4070/4071/4072.
        :return: dbl, temperature in deg Celsius
        """
        reserved_str = ctypes.create_string_buffer("".encode('utf-8'))
        temp = ctypes.c_double()
        self.dll.niDMM_GetDevTemp(self.session, reserved_str, ctypes.byref(temp))
        return temp.value

    def get_last_cal_temp(self, internal=True):
        """
        Returns the temperature during the last calibration procedure on the NI 4070/4071/4072.
        :param internal: bool, True for internal, False for external
        :return: dbl, temperature in deg C
        """
        cal_type = ctypes.c_int32(1)
        if internal:
            cal_type = ctypes.c_int32(0)
        temp = ctypes.c_double()
        self.dll.niDMM_GetLastCalTemp(self.session, cal_type, ctypes.byref(temp))
        return temp.value

    def get_cal_date_and_time(self, internal=True):
        """
        Returns the temperature during the last calibration procedure on the NI 4070/4071/4072.
        :param internal: bool, True for internal, False for external
        :return: datestr, date
        """
        cal_type = ctypes.c_int32(1)
        if internal:
            cal_type = ctypes.c_int32(0)
        month = ctypes.c_int32()
        day = ctypes.c_int32()
        year = ctypes.c_int32()
        hour = ctypes.c_int32()
        minute = ctypes.c_int32()
        self.dll.niDMM_GetCalDateAndTime(self.session, cal_type,
                                         ctypes.byref(month), ctypes.byref(day), ctypes.byref(year),
                                         ctypes.byref(hour), ctypes.byref(minute))
        date = datetime.datetime(
            month=month.value, day=day.value, year=year.value, hour=hour.value, minute=minute.value)
        date_str = date.strftime('%Y-%m-%d %H:%m')
        return date_str, date

    ''' property Node operations: '''

    def read_property_node(self, property_type, attr_id, name_str="", attr_base_str='IVI_SPECIFIC_PUBLIC_ATTR_BASE'):
        """
        read an attribute with a given id and a given type. Both can be found in the nidmm.h File if necessary.
        :param attr_base_str: str, name of the IVI ATTR Base
        :param property_type: ctypes.type
        :param name_str: Some attributes are unique for each channel. For these, pass the name of the channel.
         Other attributes are unique for each switch. Pass VI_NULL or an empty string for this parameter.
          The default value is an empty string.
        :param attr_id: int, id of the desired attribute. Can be found in the nidmm.h File
        :return: value of the readback, type depends on input
        """
        ivi_attr_base = 1000000
        bases = {'IVI_SPECIFIC_PUBLIC_ATTR_BASE': 150000 + ivi_attr_base,
                 'IVI_SPECIFIC_PRIVATE_ATTR_BASE': 200000 + ivi_attr_base,
                 'IVI_CLASS_PUBLIC_ATTR_BASE': ivi_attr_base + 250000}
        # from header files:
        # nidmm.h:
        # define NIDMM_ATTR_BASE            IVI_SPECIFIC_PUBLIC_ATTR_BASE
        # define NIDMM_ATTR_PRIVATE_BASE             IVI_SPECIFIC_PRIVATE_ATTR_BASE
        # ivi.h:
        # define IVI_ATTR_BASE                   1000000
        # define IVI_SPECIFIC_PUBLIC_ATTR_BASE   (IVI_ATTR_BASE + 150000)
        # /* base for public attributes of specific drivers */
        # define IVI_SPECIFIC_PRIVATE_ATTR_BASE  (IVI_ATTR_BASE + 200000)
        #    /* base for private attributes of specific drivers */
        attr_base = bases[attr_base_str]
        attr_id += attr_base
        attr_id = ctypes.c_int32(attr_id)
        name = ctypes.create_string_buffer(name_str.encode('utf-8'), 256)
        if isinstance(property_type, ctypes.c_int32):
            read_back = ctypes.c_int32()
            self.dll.niDMM_GetAttributeViInt32(self.session, name, attr_id, ctypes.byref(read_back))
            return read_back.value
        elif isinstance(property_type, ctypes.c_double):
            read_back = ctypes.c_double()
            self.dll.niDMM_GetAttributeViReal64(self.session, name, attr_id, ctypes.byref(read_back))
            return read_back.value
        elif isinstance(property_type, ctypes.c_char):
            read_back = ctypes.create_string_buffer('', 256)
            self.dll.niDMM_GetAttributeViString(self.session, name, attr_id, ctypes.byref(read_back))
            return read_back.value.decode('utf-8')
        elif isinstance(property_type, ctypes.c_bool):
            read_back = ctypes.c_bool()
            self.dll.niDMM_GetAttributeViBoolean(self.session, name, attr_id, ctypes.byref(read_back))
            return read_back.value

    def set_property_node(self, set_val, attr_id, name_str="", attr_base_str='IVI_SPECIFIC_PUBLIC_ATTR_BASE'):
        """
        set an attribute with a given id and a given type. Both can be found in the nidmm.h File if necessary.
        :param set_val: ctypes.c_double etc. object, with the desired value
        :param attr_id: int, id of the desired attribute. Can be found in the nidmm.h File
        :param name_str: Some attributes are unique for each channel. For these, pass the name of the channel.
         Other attributes are unique for each switch. Pass VI_NULL or an empty string for this parameter.
          The default value is an empty string.
        :param attr_base_str: str, name of the IVI ATTR Base
        """
        ivi_attr_base = 1000000
        bases = {'IVI_SPECIFIC_PUBLIC_ATTR_BASE': 150000 + ivi_attr_base,
                 'IVI_SPECIFIC_PRIVATE_ATTR_BASE': 200000 + ivi_attr_base,
                 'IVI_CLASS_PUBLIC_ATTR_BASE': ivi_attr_base + 250000}
        # from header files:
        # nidmm.h:
        # define NIDMM_ATTR_BASE            IVI_SPECIFIC_PUBLIC_ATTR_BASE
        # define NIDMM_ATTR_PRIVATE_BASE             IVI_SPECIFIC_PRIVATE_ATTR_BASE
        # ivi.h:
        # define IVI_ATTR_BASE                   1000000
        # define IVI_SPECIFIC_PUBLIC_ATTR_BASE   (IVI_ATTR_BASE + 150000)
        # /* base for public attributes of specific drivers */
        # define IVI_SPECIFIC_PRIVATE_ATTR_BASE  (IVI_ATTR_BASE + 200000)
        #    /* base for private attributes of specific drivers */
        attr_base = bases[attr_base_str]
        attr_id += attr_base
        attr_id = ctypes.c_int32(attr_id)
        name = ctypes.create_string_buffer(name_str.encode('utf-8'), 256)
        if isinstance(set_val, ctypes.c_int32):
            self.dll.niDMM_SetAttributeViInt32(self.session, name, attr_id, set_val)
        elif isinstance(set_val, ctypes.c_double):
            self.dll.niDMM_SetAttributeViReal64(self.session, name, attr_id, set_val)
        elif isinstance(set_val, ctypes.c_char):
            self.dll.niDMM_SetAttributeViString(self.session, name, attr_id, set_val)
        elif isinstance(set_val, ctypes.c_bool):
            self.dll.niDMM_SetAttributeViBoolean(self.session, name, attr_id, set_val)

    ''' self written functions (e.g. encapsulating property nodes) '''

    def set_input_resistance(self, greater_10_g_ohm=True):
        """
        function to set the input resistance of the NI-4071
        :param greater_10_g_ohm: bool,
            True if you want an input resistance higher than 10GOhm
            False if you want 10MOhm input resistance

         NOTE: >10 GOhm only supported until 10V range!
        """
        NIDMM_VAL_1_MEGAOHM = 1000000.0
        NIDMM_VAL_10_MEGAOHM = 10000000.0
        NIDMM_VAL_GREATER_THAN_10_GIGAOHM = 10000000000.0
        NIDMM_VAL_RESISTANCE_NA = 0.0
        NIDMM_ATTR_INPUT_RESISTANCE_id = 29
        resistance = ctypes.c_double(NIDMM_VAL_10_MEGAOHM)
        if greater_10_g_ohm:
            resistance = ctypes.c_double(NIDMM_VAL_GREATER_THAN_10_GIGAOHM)
        self.set_property_node(resistance, NIDMM_ATTR_INPUT_RESISTANCE_id)

    def get_input_resistance(self):
        """
        function to get the currently used input resistance.
        :return: (dbl, str),
        (value of the input resistance as defined in nidmm.h, user readable output)
        """
        NIDMM_VAL_1_MEGAOHM = 1000000.0
        NIDMM_VAL_10_MEGAOHM = 10000000.0
        NIDMM_VAL_GREATER_THAN_10_GIGAOHM = 10000000000.0
        NIDMM_VAL_RESISTANCE_NA = 0.0
        NIDMM_ATTR_INPUT_RESISTANCE_id = 29
        resistance_vals = [NIDMM_VAL_1_MEGAOHM, NIDMM_VAL_10_MEGAOHM,
                           NIDMM_VAL_GREATER_THAN_10_GIGAOHM, NIDMM_VAL_RESISTANCE_NA]
        resistance_name = ['1_M_Ohm', '10_M_Ohm', '>10_G_Ohm', 'Not_Available']
        val = self.read_property_node(ctypes.c_double(), NIDMM_ATTR_INPUT_RESISTANCE_id)
        name = resistance_name[resistance_vals.index(val)]
        return val, name

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
        self.set_property_node(ctypes.c_double(range_val), 2,
                               attr_base_str='IVI_CLASS_PUBLIC_ATTR_BASE')


# there are more functions that can be found in the nidmm.h file,
# but those above were the ones in the quick reference and most important ones.
# how to start external calibration??  niDMM_InitExtCal  -> not in quick ref!


def test_multi():
    dmm.config_multi_point_meas(1, 4, 1, 0.1)
    print(dmm.read_multi_point(5))


def test_wfm():
    dmm.conf_waveform_meas(1003, 1.0, 1800000, 100)
    print(dmm.read_waveform(num_to_read=100))


def test_self_test():
    print(dmm.self_test())


if __name__ == "__main__":
    dmm = Ni4071()
    print(dmm.readstatus())
    i = 0
    # while i < 100:
    #     print(dmm.get_dmm_dev_temp())
    #     i += 1
    #     time.sleep(0.5)
    # test_multi()
    # test_wfm()
    # test_self_test()
    # print(dmm.revision_query())
    # print(dmm.get_error_message())
    # # dmm.self_calibration()
    # print(dmm.get_calibration_count())
    # print(dmm.get_cal_date_and_time())
    # print(dmm.get_last_cal_temp())
    print(dmm.get_range())
    dmm.config_meas_digits(1, 100.0, 7.5)
    print(dmm.get_range())

    print(dmm.readstatus())
    dmm.set_input_resistance(True)
    print(dmm.get_input_resistance())
    dmm.set_range(10.0)
    # print(dmm.read_single_voltage())
    dmm.set_input_resistance(True)
    # print(dmm.read_single_voltage())
    print(dmm.get_input_resistance())
    print(dmm.get_range())
    # print(dmm.read_single_voltage())
    dmm.de_init_dmm()
