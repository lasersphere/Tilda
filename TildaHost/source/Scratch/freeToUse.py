"""

Created on '22.06.2015'

@author:'simkaufm'

"""
# import numpy as np
#
import Service.Formating as form
import Service.VoltageConversions.DacRegisterToVoltageFit as DacG
import ast
# import Service.Scan.draftScanParameters as dft
from Service.VoltageConversions.VoltageConversions import get_18bit_from_voltage, get_voltage_from_18bit
#
#

n = 0
for i in range(100):
    v = get_voltage_from_18bit(i, dac_gauge_pars=DacG.dac_gauge_vals)
    j = get_18bit_from_voltage(v, dac_gauge_pars=DacG.dac_gauge_vals)
    jv = get_voltage_from_18bit(j, dac_gauge_pars=DacG.dac_gauge_vals)
    # j = form.get_18bit_stepsize(form.get_voltage_from_18bit(i) + 10)
    if (i - j) != 0:
        n += 1
    print(i, '\t', v, '\t', j, '\t', jv, '\t', i - j, '\t', v - jv)

print(n)
