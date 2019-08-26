"""

Created on '26.10.2015'

@author:'simkaufm'

"""
import numpy as np


def get_bits_from_voltage(voltage, dac_gauge_pars, ref_volt_neg=-10, ref_volt_pos=10):
    """
    function to return an 20-Bit Integer by putting in a voltage +\-10V in DBL
    :param voltage: dbl, desired Voltage
    :param ref_volt_neg/ref_volt_pos: dbl, value for the neg./pos. reference Voltage for the DAC
    :return: int, 20-Bit Code
    """
    if voltage is None:
        return None
    if dac_gauge_pars is None:
        #  function as described in the AD5781 Manual
        b20 = (voltage - ref_volt_neg) * ((2 ** 20) - 1) / (ref_volt_pos - ref_volt_neg)  # from the manual
        b20 = int(round(b20))  # round is used to avoid the floor rounding of int().
    else:
        # linear function (V = slope * D + offset) with offset and slope from measurement
        b20 = int(round(((voltage - dac_gauge_pars[0])/dac_gauge_pars[1])))
    b20 = max(0, b20)
    b20 = min(b20, (2 ** 20) - 1)
    return b20


def get_bit_stepsize(step_voltage, dac_gauge_pars):
    """
    function to get the StepSize in dac register integer form derived from a double Voltage
    :return ~ step_voltage/lsb
    """
    if step_voltage is None:
        return None
    lsb = 20 / ((2 ** 20) - 1)  # least significant bit in +/-10V 18Bit DAC
    if dac_gauge_pars is not None:
        lsb = dac_gauge_pars[1]  # lsb = slope from DAC-Scan
    b20 = int(round(step_voltage/lsb))
    b20 = max(-((2 ** 20) - 1), b20)
    b20 = min(b20, (2 ** 20) - 1)
    return b20


def get_stepsize_in_volt_from_bits(voltage_bits, dac_gauge_pars):
    """
    function to calculate the stepsize by a given 20bit dac register difference.
    :return ~ voltage_18b * lsb
    """
    if voltage_bits is None:
        return None
    voltage_bits = max(-((2 ** 20) - 1), voltage_bits)
    voltage_bits = min(voltage_bits, (2 ** 20) - 1)
    lsb = 20 / ((2 ** 20) - 1)  # least significant bit in +/-10V 20Bit DAC
    if dac_gauge_pars is not None:
        lsb = dac_gauge_pars[1]  # lsb = slope from DAC-Scan
    # volt = round(voltage_18bit * lsb, 8)
    volt = voltage_bits * lsb
    return volt


def get_24bit_input_from_voltage(voltage, dac_gauge_pars,
                                 add_reg_add=True, loose_sign=False, ref_volt_neg=-10, ref_volt_pos=10):
    """
    function to return an 24-Bit Integer by putting in a voltage +\-10V in DBL
    :param voltage: dbl, desired Voltage
    :param ref_volt_neg/ref_volt_pos: dbl, value for the neg./pos. reference Voltage for the DAC
    :return: int, 24-Bit Code.
    """
    b20 = get_bits_from_voltage(voltage, dac_gauge_pars, ref_volt_neg, ref_volt_pos)
    b24 = (int(b20))
    if add_reg_add:
        # adds the address of the DAC register to the bits
        b24 += int(2 ** 20)
    if loose_sign:
        b24 -= int(2 ** 19)#??? what is happening here
    return b24


def get_voltage_from_bits(voltage_20bit, dac_gauge_pars, ref_volt_neg=-10, ref_volt_pos=10):
    """function from the manual of the AD5781"""
    if voltage_20bit is None:
        return None
    voltage_20bit = max(0, voltage_20bit)
    voltage_20bit = min(voltage_20bit, (2 ** 20) - 1)
    if dac_gauge_pars is None:
        # function as described in the AD5781 Manual
        voltfloat = (ref_volt_pos - ref_volt_neg) * voltage_20bit / ((2 ** 20) - 1) + ref_volt_neg
        # voltfloat = round(voltfloat, 8)
    else:
        # linear function (V = slope * D + offset) with offset and slope from measurement
        voltfloat = voltage_20bit * dac_gauge_pars[1] + dac_gauge_pars[0]
        # voltfloat = round(voltfloat, 8)
    return voltfloat


def get_voltage_from_24bit(voltage_24bit, dac_gauge_pars,
                           remove_add=True, ref_volt_neg=-10, ref_volt_pos=10):
    """
    function to get the output voltage of the DAC by the corresponding 24-Bit register input
    :param voltage_24bit: int, 24 bit, register entry of the DAC
    :param remove_add: bool, to determine if the integer has still the registry adress attached
    :param ref_volt_neg/P: dbl, +/- 10 V for the reference Voltage of the DAC
    :return: dbl, Voltage that will be applied.
    """
    v20bit = get_20bit_from_24bit_dac_reg(voltage_24bit, remove_add)
    voltfloat = get_voltage_from_bits(v20bit, dac_gauge_pars, ref_volt_neg, ref_volt_pos)
    return voltfloat


def get_20bit_from_24bit_dac_reg(voltage_24bit, remove_address=True):
    """
    function to convert a 24Bit DAC Reg to 18Bit
    :param voltage_24bit: int, 24 Bit DAC Reg entry
    :param remove_address: bool, True if the Registry Address is still included
    :return: int, 20Bit DAC Reg value
    """
    if remove_address:
        voltage_24bit -= 2 ** 20  # redundant with the next step?
    v20bit = voltage_24bit & ((2 ** 20) - 1)
    return v20bit

def calc_step_size(start, stop, steps):
    """
    calculates the stepsize: (stop - start) / nOfSteps
    :return stepsize_18bit
    """
    try:
        dis = stop - start
        stepsize_18bit = int(dis / (steps - 1))
    except ZeroDivisionError:
        stepsize_18bit = 0
    # stepsize_18bit = max(-(2 ** 18 - 1), stepsize_18bit)
    # stepsize_18bit = min((2 ** 18 - 1), stepsize_18bit)
    return stepsize_18bit


def calc_n_of_steps(start, stop, step_size):
    """
    calculates the number of steps: abs((stop - start) / stepSize)
    """
    try:
        dis = abs(stop - start) + abs(step_size)
        n_of_steps = int(dis / abs(step_size))
    except ZeroDivisionError:
        n_of_steps = 0
    # n_of_steps = max(2, n_of_steps)
    # n_of_steps = min((2 ** 18 - 1), n_of_steps)
    return n_of_steps

if __name__=='__main__':
    print(get_stepsize_in_volt_from_bits(1))

# # testing the step_size calculation
# start = 0
# stop = 2 ** 18 - 1
# steps = []
# stop_dif = []
# step_size = []
# for num_of_steps in range(2, 10000):
#     step_18b = calc_step_size(start, stop, num_of_steps)
#     res_stop = start + (num_of_steps - 1) * step_18b
#     steps.append(num_of_steps)
#     step_size.append(step_size)
#     stop_dif.append(stop - res_stop)
#
#     # if res_stop > 2 ** 18 - 1:
#     #     print(res_stop, ' at %s steps' % num_of_steps)
#     # if res_stop != stop:
#     #     print('stop value was not reached, stop is: %s, difference is %s at num of steps %s and step_size %s' %
#     #           (res_stop, stop - res_stop, num_of_steps, step_18b))
#
# import matplotlib.pyplot as plt
#
# plt.plot(steps, stop_dif)
# plt.plot(steps, step_size)
# plt.show()


# # testing the num of steps calculation
# start = 0
# stop = 2 ** 18 - 1
# step_size_l = []
# stop_dif = []
# steps = []
# for step_size in range(2, 100000):
#     num_of_steps = calc_n_of_steps(start, stop, step_size)
#     res_stop = start + (num_of_steps - 1) * step_size
#     steps.append(num_of_steps)
#     step_size_l.append(step_size)
#     stop_dif.append(stop - res_stop)
#
# import matplotlib.pyplot as plt
#
# # step_size_l = [get_stepsize_in_volt_from_18bit(bit) for bit in step_size_l]
# plt.plot(step_size_l, stop_dif, label='stop-calc_stop')
# plt.plot(step_size_l, steps, label='num_of_steps')
# plt.legend()
# # plt.xlabel('step_size_V')
# plt.show()

