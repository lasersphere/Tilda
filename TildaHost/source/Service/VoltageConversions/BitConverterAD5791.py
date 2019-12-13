"""

Created on '26.10.2015'

@author:'simkaufm'

"""
import numpy as np
from Service.VoltageConversions.bitconverter import BitConverter


class AD5791BitConverter(BitConverter):

    def get_max_value_in_bits(self):
        return (2 ** 20) - 1

    def get_20bits_from_voltage(self, voltage, dac_gauge_pars, ref_volt_neg=-10, ref_volt_pos=10):
        return self.get_nbits_from_voltage(voltage, dac_gauge_pars, ref_volt_neg, ref_volt_pos)

    def get_nbits_from_voltage(self, voltage, dac_gauge_pars, ref_volt_neg=-10, ref_volt_pos=10):
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
            b20 = int(round(((voltage - dac_gauge_pars[0]) / dac_gauge_pars[1])))
        b20 = max(0, b20)
        b20 = min(b20, (2 ** 20) - 1)
        return b20

    def get_nbit_stepsize(self, step_voltage, dac_gauge_pars):
        """
        function to get the StepSize in dac register integer form derived from a double Voltage
        :return ~ step_voltage/lsb
        """
        if step_voltage is None:
            return None
        lsb = 20 / ((2 ** 20) - 1)  # least significant bit in +/-10V 18Bit DAC
        if dac_gauge_pars is not None:
            lsb = dac_gauge_pars[1]  # lsb = slope from DAC-Scan
        b20 = int(round(step_voltage / lsb))
        b20 = max(-((2 ** 20) - 1), b20)
        b20 = min(b20, (2 ** 20) - 1)
        return b20

    def get_20bit_stepsize(self, step_voltage, dac_gauge_pars):
        return self.get_nbit_stepsize(step_voltage, dac_gauge_pars)

    def get_stepsize_in_volt_from_bits(self, voltage_bits, dac_gauge_pars):
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

    def get_24bit_input_from_voltage(self, voltage, dac_gauge_pars,
                                     add_reg_add=True, loose_sign=False, ref_volt_neg=-10, ref_volt_pos=10):
        """
        function to return an 24-Bit Integer by putting in a voltage +\-10V in DBL
        :param voltage: dbl, desired Voltage
        :param ref_volt_neg/ref_volt_pos: dbl, value for the neg./pos. reference Voltage for the DAC
        :return: int, 24-Bit Code.
        """
        b20 = self.get_nbits_from_voltage(voltage, dac_gauge_pars, ref_volt_neg, ref_volt_pos)
        b24 = (int(b20))
        if add_reg_add:
            # adds the address of the DAC register to the bits
            b24 += int(2 ** 20)
        if loose_sign:
            b24 -= int(2 ** 19)  # ??? what is happening here
        return b24

    def get_voltage_from_bits(self, voltage_20bit, dac_gauge_pars, ref_volt_neg=-10, ref_volt_pos=10):
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

    def get_voltage_from_24bit(self, voltage_24bit, dac_gauge_pars,
                               remove_add=True, ref_volt_neg=-10, ref_volt_pos=10):
        """
        function to get the output voltage of the DAC by the corresponding 24-Bit register input
        :param voltage_24bit: int, 24 bit, register entry of the DAC
        :param remove_add: bool, to determine if the integer has still the registry adress attached
        :param ref_volt_neg/P: dbl, +/- 10 V for the reference Voltage of the DAC
        :return: dbl, Voltage that will be applied.
        """
        v20bit = self.get_bits_from_24bit_dac_reg(voltage_24bit, remove_add)
        voltfloat = self.get_voltage_from_bits(v20bit, dac_gauge_pars, ref_volt_neg, ref_volt_pos)
        return voltfloat

    def get_bits_from_24bit_dac_reg(self, voltage_24bit, remove_address=True):
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
