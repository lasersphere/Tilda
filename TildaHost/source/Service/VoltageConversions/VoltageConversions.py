"""

Created on '26.10.2015'

@author:'simkaufm'

"""
import numpy as np

try:
    import Service.VoltageConversions.DAC_Calibration as AD5781Fit
except:
    print('No DAC calibration file found on your local harddrive!')
    print('please calibrate your DAC and create a calibration file as described as in:')
    print('TildaHost\\source\\Service\\VoltageConversion\\DacRegisterToVoltageFit.py')
    raise Exception


def get_18bit_from_voltage(voltage, dac_gauge_pars=AD5781Fit.dac_gauge_vals, ref_volt_neg=-10, ref_volt_pos=10):
    """
    function to return an 18-Bit Integer by putting in a voltage +\-10V in DBL
    :param voltage: dbl, desired Voltage
    :param ref_volt_neg/ref_volt_pos: dbl, value for the neg./pos. reference Voltage for the DAC
    :return: int, 18-Bit Code.
    """
    if voltage is None:
        return None
    if dac_gauge_pars is None:
        #  function as described in the AD5781 Manual
        b18 = (voltage - ref_volt_neg) * ((2 ** 18) - 1) / (ref_volt_pos - ref_volt_neg)  # from the manual
        b18 = int(round(b18))  # round is used to avoid the floor rounding of int().
    else:
        # linear function (V = slope * D + offset) with offset and slope from measurement
        b18 = int(round(((voltage - dac_gauge_pars[0])/dac_gauge_pars[1])))
    b18 = max(0, b18)
    b18 = min(b18, (2 ** 18) - 1)
    return b18


def get_18bit_stepsize(step_voltage, dac_gauge_pars=AD5781Fit.dac_gauge_vals):
    """
    function to get the StepSize in dac register integer form derived from a double Voltage
    :return ~ step_voltage/lsb
    """
    if step_voltage is None:
        return None
    lsb = 20 / ((2 ** 18) - 1)  # least significant bit in +/-10V 18Bit DAC
    if dac_gauge_pars is not None:
        lsb = dac_gauge_pars[1]  # lsb = slope from DAC-Scan
    b18 = int(round(step_voltage/lsb))
    b18 = max(-((2 ** 18) - 1), b18)
    b18 = min(b18, (2 ** 18) - 1)
    return b18


def get_stepsize_in_volt_from_18bit(voltage_18bit, dac_gauge_pars=AD5781Fit.dac_gauge_vals):
    """
    function to calculate the stepsize by a given 18bit dac register difference.
    :return ~ voltage_18b * lsb
    """
    if voltage_18bit is None:
        return None
    voltage_18bit = max(-((2 ** 18) - 1), voltage_18bit)
    voltage_18bit = min(voltage_18bit, (2 ** 18) - 1)
    lsb = 20 / ((2 ** 18) - 1)  # least significant bit in +/-10V 18Bit DAC
    if dac_gauge_pars is not None:
        lsb = dac_gauge_pars[1]  # lsb = slope from DAC-Scan
    volt = round(voltage_18bit * lsb, 6)
    return volt


def get_24bit_input_from_voltage(voltage, dac_gauge_pars=AD5781Fit.dac_gauge_vals,
                                 add_reg_add=True, loose_sign=False, ref_volt_neg=-10, ref_volt_pos=10):
    """
    function to return an 24-Bit Integer by putting in a voltage +\-10V in DBL
    :param voltage: dbl, desired Voltage
    :param ref_volt_neg/ref_volt_pos: dbl, value for the neg./pos. reference Voltage for the DAC
    :return: int, 24-Bit Code.
    """
    b18 = get_18bit_from_voltage(voltage, dac_gauge_pars, ref_volt_neg, ref_volt_pos)
    b24 = (int(b18) << 2)
    if add_reg_add:
        # adds the address of the DAC register to the bits
        b24 += int(2 ** 20)
    if loose_sign:
        b24 -= int(2 ** 19)
    return b24


def get_voltage_from_18bit(voltage_18bit, dac_gauge_pars=AD5781Fit.dac_gauge_vals, ref_volt_neg=-10, ref_volt_pos=10):
    """function from the manual of the AD5781"""
    if voltage_18bit is None:
        return None
    voltage_18bit = max(0, voltage_18bit)
    voltage_18bit = min(voltage_18bit, (2 ** 18) - 1)
    if dac_gauge_pars is None:
        # function as described in the AD5781 Manual
        voltfloat = (ref_volt_pos - ref_volt_neg) * voltage_18bit / ((2 ** 18) - 1) + ref_volt_neg
        voltfloat = round(voltfloat, 6)
    else:
        # linear function (V = slope * D + offset) with offset and slope from measurement
        voltfloat = voltage_18bit * dac_gauge_pars[1] + dac_gauge_pars[0]
        voltfloat = round(voltfloat, 6)
    return voltfloat


def get_voltage_from_24bit(voltage_24bit, dac_gauge_pars=AD5781Fit.dac_gauge_vals,
                           remove_add=True, ref_volt_neg=-10, ref_volt_pos=10):
    """
    function to get the output voltage of the DAC by the corresponding 24-Bit register input
    :param voltage_24bit: int, 24 bit, register entry of the DAC
    :param remove_add: bool, to determine if the integer has still the registry adress attached
    :param ref_volt_neg/P: dbl, +/- 10 V for the reference Voltage of the DAC
    :return: dbl, Voltage that will be applied.
    """
    v18bit = get_18bit_from_24bit_dac_reg(voltage_24bit, remove_add)
    voltfloat = get_voltage_from_18bit(v18bit, dac_gauge_pars, ref_volt_neg, ref_volt_pos)
    return voltfloat


def get_18bit_from_24bit_dac_reg(voltage_24bit, remove_address=True):
    """
    function to convert a 24Bit DAC Reg to 18Bit
    :param voltage_24bit: int, 24 Bit DAC Reg entry
    :param remove_address: bool, True if the Registry Address is still included
    :return: int, 18Bit DAC Reg value
    """
    if remove_address:
        voltage_24bit -= 2 ** 20
    v18bit = (voltage_24bit >> 2) & ((2 ** 18) - 1)
    return v18bit


def find_volt_in_array(voltage, volt_array, track_ind):
    """
    find the index of voltage in volt_array. If not existant, create.
    empty entries in volt_array must be (2 ** 30)
    :return: (int, np.array), index and VoltageArray
    """
    '''payload is 23-Bits, Bits 2 to 20 is the DAC register'''
    voltage = get_18bit_from_24bit_dac_reg(voltage, True)  # shift by 2 and delete higher parts of payload
    index = np.where(volt_array[track_ind] == voltage)
    if len(index[0]) == 0:
        # voltage not yet in array, put it at next empty position
        index = np.where(volt_array[track_ind] == (2 ** 30))[0][0]
    else:
        # voltage already in list, take the found index
        index = index[0][0]
    np.put(volt_array[track_ind], index, voltage)
    return index, volt_array


def calc_dac_stop_18bit(start, step, num_of_steps):
    """
    calculate the stop voltage by:
    stop = start + step * num_of_steps
    :return stop_18bit
    """
    stop = start + step * (num_of_steps - 1)
    return stop


def calc_step_size(start, stop, steps):
    """
    calculates the stepsize: (stop - start) / nOfSteps
    :return stepsize_18bit
    """
    try:
        dis = stop - start
        stepsize_18bit = int(round(dis / (steps - 1)))
    except ZeroDivisionError:
        stepsize_18bit = 0
    return stepsize_18bit


def calc_n_of_steps(start, stop, step_size):
    """
    calculates the number of steps: abs((stop - start) / stepSize)
    """
    try:
        dis = abs(stop - start) + abs(step_size)
        n_of_steps = int(round(dis / abs(step_size)))
    except ZeroDivisionError:
        n_of_steps = 0
    return n_of_steps
