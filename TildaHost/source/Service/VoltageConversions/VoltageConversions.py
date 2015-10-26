"""

Created on '26.10.2015'

@author:'simkaufm'

"""
import numpy as np


def get_18bit_from_voltage(voltage, ref_volt_neg=-10, ref_volt_pos=10):
    """
    function to return an 18-Bit Integer by putting in a voltage +\-10V in DBL
    as described in the manual of the AD5781
    :param voltage: dbl, desired Voltage
    :param ref_volt_neg/ref_volt_pos: dbl, value for the neg./pos. reference Voltage for the DAC
    :return: int, 18-Bit Code.
    """
    b18 = (voltage - ref_volt_neg) * ((2 ** 18) - 1) / (ref_volt_pos - ref_volt_neg)  # from the manual
    b18 = int(b18)
    return b18


def get_18bit_stepsize(step_voltage, ref_volt_neg=-10, ref_volt_pos=10):
    """
    function to get the StepSize in integer form derived from a double Voltage
    :param step_voltage: dbl, desired StepSize Voltage
    :param ref_volt_neg/ref_volt_pos: dbl, value for the neg./pos. reference Voltage for the DAC
    :return: int, 18-Bit Code
    """
    b18 = get_18bit_from_voltage(step_voltage, ref_volt_neg, ref_volt_pos) - int(2 ** 17)
    # must loose the 1 in the beginning.
    # b18 += 1  # needed?
    return b18


def get_24bit_input_from_voltage(voltage, add_reg_add=True, loose_sign=False, ref_volt_neg=-10, ref_volt_pos=10):
    """
    function to return an 24-Bit Integer by putting in a voltage +\-10V in DBL
    :param voltage: dbl, desired Voltage
    :param ref_volt_neg/ref_volt_pos: dbl, value for the neg./pos. reference Voltage for the DAC
    :return: int, 24-Bit Code.
    """
    b18 = get_18bit_from_voltage(voltage, ref_volt_neg, ref_volt_pos)
    b24 = (int(b18) << 2)
    if add_reg_add:
        # adds the address of the DAC register to the bits
        b24 += int(2 ** 20)
    if loose_sign:
        b24 -= int(2 ** 19)
    return b24


def get_voltage_from_24bit(voltage_24bit, remove_add=True, ref_volt_neg=-10, ref_volt_pos=10):
    """
    function to get the output voltage of the DAC by the corresponding 24-Bit register input
    :param voltage_24bit: int, 24 bit, register entry of the DAC
    :param remove_add: bool, to determine if the integer has still the registry adress attached
    :param ref_volt_neg/P: dbl, +/- 10 V for the reference Voltage of the DAC
    :return: dbl, Voltage that will be applied.
    """
    v18bit = get_18bit_from_24bit_dac_reg(voltage_24bit, remove_add)
    voltfloat = get_voltage_from_18bit(v18bit, ref_volt_neg, ref_volt_pos)
    return voltfloat


def get_voltage_from_18bit(voltage_18bit, ref_volt_neg=-10, ref_volt_pos=10):
    """function from the manual of the AD5781"""
    voltfloat = (ref_volt_pos - ref_volt_neg) * voltage_18bit / ((2 ** 18) - 1) + ref_volt_neg
    voltfloat = round(voltfloat, 6)
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


def find_volt_in_array(voltage, volt_array):
    """
    find the index of voltage in volt_array. If not existant, create.
    empty entries in volt_array must be (2 ** 30)
    :return: (int, np.array), index and VoltageArray
    """
    '''payload is 23-Bits, Bits 2 to 20 is the DAC register'''
    voltage = get_18bit_from_24bit_dac_reg(voltage, True)  # shift by 2 and delete higher parts of payload
    index = np.where(volt_array == voltage)
    if len(index[0]) == 0:
        # voltage not yet in array, put it at next empty position
        index = np.where(volt_array == (2 ** 30))[0][0]
    else:
        # voltage already in list, take the found index
        index = index[0][0]
    np.put(volt_array, index, voltage)
    return index, volt_array