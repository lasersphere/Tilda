"""

Created on '26.10.2015'

@author:'simkaufm'

"""
import numpy as np
from Service.VoltageConversions.bitconverter import BitConverter
from Service.VoltageConversions.BitConverterAD5791 import AD5791BitConverter
from Service.VoltageConversions.BitConverterAD5781legacy import AD5781BitConverter

try:
    import Service.VoltageConversions.DAC_Calibration as DAC_calib
except:
    raise Exception(
        '\n------------------------------------------- \n'
        'No DAC calibration file found on your local harddrive found at:\n'
        'TildaHost\\source\\Service\\VoltageConversion\\DAC_Calibration.py.py\n'
        'please calibrate your DAC and create a calibration file as described as in:\n'
        'TildaHost\\source\\Service\\VoltageConversion\\DacRegisterToVoltageFit.py\n'
        '------------------------------------------- \n'
    )

bitconv = BitConverter()# this returns the dummy class which can't do anything, but from which the real implementations are derived.
if DAC_calib.dac_name.startswith("AD5781"):
    bitconv = AD5781BitConverter()
elif DAC_calib.dac_name == "AD5791":
    bitconv = AD5791BitConverter()
else:
    raise Exception(
        '\n------------------------------------------- \n'
        'Unknown DAC specified in calibration file!\n'
        '------------------------------------------- \n')


def get_max_value_in_bits():
    return bitconv.get_max_value_in_bits()

def get_bits_from_voltage(voltage, dac_gauge_pars=DAC_calib.dac_gauge_vals, ref_volt_neg=-10, ref_volt_pos=10):
    """
    function to return an 18-Bit Integer by putting in a voltage +\-10V in DBL
    :param voltage: dbl, desired Voltage
    :param ref_volt_neg/ref_volt_pos: dbl, value for the neg./pos. reference Voltage for the DAC
    :return: int, bit Code.
    """
    return bitconv.get_bits_from_voltage(voltage, dac_gauge_pars, ref_volt_neg, ref_volt_pos)


def get_stepsize_in_bits(step_voltage, dac_gauge_pars=DAC_calib.dac_gauge_vals):
    """
    function to get the StepSize in dac register integer form derived from a double Voltage
    :return ~ step_voltage/lsb
    """
    return bitconv.get_stepsize_in_bits(step_voltage, dac_gauge_pars)


def get_stepsize_in_volt_from_bits(voltage_18bit, dac_gauge_pars=DAC_calib.dac_gauge_vals):
    """
    function to calculate the stepsize by a given 18bit dac register difference.
    :return ~ voltage_18b * lsb
    """
    return bitconv.get_stepsize_in_volt_from_bits(voltage_18bit, dac_gauge_pars)


def get_24bit_input_from_voltage(voltage, dac_gauge_pars=DAC_calib.dac_gauge_vals,
                                 add_reg_add=True, loose_sign=False, ref_volt_neg=-10, ref_volt_pos=10):
    """
    function to return an 24-Bit Integer by putting in a voltage +\-10V in DBL
    :param voltage: dbl, desired Voltage
    :param ref_volt_neg/ref_volt_pos: dbl, value for the neg./pos. reference Voltage for the DAC
    :return: int, 24-Bit Code.
    """
    return bitconv.get_24bit_input_from_voltage(voltage, dac_gauge_pars, add_reg_add, loose_sign, ref_volt_neg,
                                                     ref_volt_pos)


def get_voltage_from_bits(voltage_18bit, dac_gauge_pars=DAC_calib.dac_gauge_vals, ref_volt_neg=-10, ref_volt_pos=10):
    """function from the manual of the AD5781"""
    return bitconv.get_voltage_from_bits(voltage_18bit, dac_gauge_pars, ref_volt_neg, ref_volt_pos)


def get_voltage_from_24bit(voltage_24bit, dac_gauge_pars=DAC_calib.dac_gauge_vals,
                           remove_add=True, ref_volt_neg=-10, ref_volt_pos=10):
    """
    function to get the output voltage of the DAC by the corresponding 24-Bit register input
    :param voltage_24bit: int, 24 bit, register entry of the DAC
    :param remove_add: bool, to determine if the integer has still the registry adress attached
    :param ref_volt_neg/P: dbl, +/- 10 V for the reference Voltage of the DAC
    :return: dbl, Voltage that will be applied.
    """
    return bitconv.get_voltage_from_24bit(voltage_24bit, dac_gauge_pars, remove_add, ref_volt_neg, ref_volt_pos)


def get_value_bits_from_24bit_dac_reg(voltage_24bit, remove_address=True):
    """
    function to convert a 24Bit DAC Reg to 18Bit
    :param voltage_24bit: int, 24 Bit DAC Reg entry
    :param remove_address: bool, True if the Registry Address is still included
    :return: int, 18Bit DAC Reg value
    """
    return bitconv.get_bits_from_24bit_dac_reg(voltage_24bit, remove_address)

def calc_step_size(start, stop, steps):
    """
    calculates the stepsize: (stop - start) / nOfSteps
    :return stepsize_18bit
    """
    return bitconv.calc_step_size(start,stop,steps)


def calc_n_of_steps(start, stop, step_size):
    """
    calculates the number of steps: abs((stop - start) / stepSize)
    """
    return bitconv.calc_n_of_steps(start,stop,step_size)



def find_volt_in_array(voltage, volt_array, track_ind):
    """
    find the index of voltage in volt_array. If not existant, create.
    empty entries in volt_array must be (2 ** 30)
    :return: (int, np.array), index and VoltageArray
    """
    '''payload is 23-Bits, Bits 2 to 20 is the DAC register'''
    voltage = get_value_bits_from_24bit_dac_reg(voltage, True)  # shift by 2 and delete higher parts of payload
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
    stop = int(start + step * (num_of_steps - 1))
    # stop = max(0, stop)
    # stop = min((2 ** 18 - 1), stop)
    return stop



if __name__ == '__main__':
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
