"""

Created on '26.10.2015'

@author:'simkaufm'


This is a sample file!
Create one of the very same type named 'DAC_Calibration.py' in this very folder
with the measured slope and offset of the DAC which you are using.
One can create it with this very settings and then update the slope and the offset after calibration.




Config File for the storage of the slope and the offset
 fitted from measurement of the DAC-Register of the AD5781 vs. Voltage read back with Agilent 3401A.

  V ^
    |        .
    |      .
    |    .
    |  .
    |.
    -----------------> DAC Reg (D)

        V = slope * D + offset

 The slope should be close to 76.29 µV (which is the least significant bit in an 18Bit +/-10V DAC)
 Offset should be the negative reference voltage (close to -10 V)
"""

offset = -9.993201  # [V] as a result of 175 measurements acquired on 20.10.2014/21.10.2014
delta_offset = 2.2 * 10 ** -6  # [V] as a result of 175 measurements acquired on 20.10.2014/21.10.2014

slope = 7.62534880235751 * 10 ** -5  # [V/Bit] as a result of 175 measurements acquired on 20.10.2014/21.10.2014
delta_slope = 1.02698558902023 * 10 ** -11  # [V/Bit] as a result of 175 measurements acquired on 20.10.2014/21.10.2014

dac_gauge_vals = (offset, slope)

# lsb = 20 / (2 ** 18)
# print(lsb)
