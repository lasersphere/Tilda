"""

Created on '22.06.2015'

@author:'simkaufm'

"""
import numpy as np
#
import Service.Formating as form
import Service.VoltageConversions.DacRegisterToVoltageFit as DacG
import ast
# import Service.Scan.draftScanParameters as dft
from Service.VoltageConversions.VoltageConversions import get_18bit_from_voltage, get_voltage_from_18bit
import Service.VoltageConversions.VoltageConversions as VCon
#

# n = 0
# for i in range((2 ** 18) - 1):
#     v = get_voltage_from_18bit(i, dac_gauge_pars=DacG.dac_gauge_vals)
#     j = get_18bit_from_voltage(v, dac_gauge_pars=DacG.dac_gauge_vals)
#     jv = get_voltage_from_18bit(j, dac_gauge_pars=DacG.dac_gauge_vals)
#     # j = form.get_18bit_stepsize(form.get_voltage_from_18bit(i) + 10)
#     if (i - j) != 0:
#         n += 1
#         print(i, '\t', v, '\t', j, '\t', jv, '\t', i - j, '\t', v - jv)
#
# print(n)


for stV in np.arange(-1, 0, 20/(2 ** 18)):
    s18 = VCon.get_18bit_stepsize(stV, dac_gauge_pars=DacG.dac_gauge_vals)
    s18V = VCon.get_stepsize_in_volt_from_18bit(s18, dac_gauge_pars=DacG.dac_gauge_vals)
    s18v18 = VCon.get_18bit_stepsize(s18V, dac_gauge_pars=DacG.dac_gauge_vals)
    print(stV, '\t', s18, '\t', s18V, '\t', s18v18, '\t', s18 - s18v18, '\t', format(s18, '018b'))