"""

Created on '26.10.2015'

@author:'simkaufm'

"""
import numpy as np
from Service.VoltageConversions.bitconverter import BitConverter


class AD5781BitConverter(BitConverter):

    def get_max_value_in_bits(self):
        return (2 ** 18) - 1

    def get_bits_from_voltage(self, voltage, dac_gauge_pars, ref_volt_neg=-10, ref_volt_pos=10):
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
            b18 = int(round(((voltage - dac_gauge_pars[0]) / dac_gauge_pars[1])))
        b18 = max(0, b18)
        b18 = min(b18, (2 ** 18) - 1)
        return b18

    def get_stepsize_in_bits(self, step_voltage, dac_gauge_pars):
        """
        function to get the StepSize in dac register integer form derived from a double Voltage
        :return ~ step_voltage/lsb
        """
        if step_voltage is None:
            return None
        lsb = 20 / ((2 ** 18) - 1)  # least significant bit in +/-10V 18Bit DAC
        if dac_gauge_pars is not None:
            lsb = dac_gauge_pars[1]  # lsb = slope from DAC-Scan
        b18 = int(round(step_voltage / lsb))
        b18 = max(-((2 ** 18) - 1), b18)
        b18 = min(b18, (2 ** 18) - 1)
        return b18

    def get_stepsize_in_volt_from_bits(self, voltage_18bit, dac_gauge_pars):
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
        # volt = round(voltage_18bit * lsb, 8)
        volt = voltage_18bit * lsb
        return volt

    def get_24bit_input_from_voltage(self, voltage, dac_gauge_pars,
                                     add_reg_add=True, loose_sign=False, ref_volt_neg=-10, ref_volt_pos=10):
        """
        function to return an 24-Bit Integer by putting in a voltage +\-10V in DBL
        :param voltage: dbl, desired Voltage
        :param ref_volt_neg/ref_volt_pos: dbl, value for the neg./pos. reference Voltage for the DAC
        :return: int, 24-Bit Code.
        """
        b18 = self.get_bits_from_voltage(voltage, dac_gauge_pars, ref_volt_neg, ref_volt_pos)
        b24 = (int(b18) << 2)
        if add_reg_add:
            # adds the address of the DAC register to the bits
            b24 += int(2 ** 20)
        if loose_sign:
            b24 -= int(2 ** 19)
        return b24

    def get_voltage_from_bits(self, voltage_18bit, dac_gauge_pars, ref_volt_neg=-10, ref_volt_pos=10):
        """function from the manual of the AD5781"""
        if voltage_18bit is None:
            return None
        voltage_18bit = max(0, voltage_18bit)
        voltage_18bit = min(voltage_18bit, (2 ** 18) - 1)
        if dac_gauge_pars is None:
            # function as described in the AD5781 Manual
            voltfloat = (ref_volt_pos - ref_volt_neg) * voltage_18bit / ((2 ** 18) - 1) + ref_volt_neg
            # voltfloat = round(voltfloat, 8)
        else:
            # linear function (V = slope * D + offset) with offset and slope from measurement
            voltfloat = voltage_18bit * dac_gauge_pars[1] + dac_gauge_pars[0]
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
        v18bit = self.get_bits_from_24bit_dac_reg(voltage_24bit, remove_add)
        voltfloat = self.get_voltage_from_bits(v18bit, dac_gauge_pars, ref_volt_neg, ref_volt_pos)
        return voltfloat

    def get_bits_from_24bit_dac_reg(self, voltage_24bit, remove_address=True):
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
