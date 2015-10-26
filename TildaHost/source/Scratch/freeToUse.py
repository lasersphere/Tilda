"""

Created on '22.06.2015'

@author:'simkaufm'

"""
import numpy as np

import Service.Formating as form
import Service.Scan.draftScanParameters as dft


for i in range(100):
    v = form.getVoltageFrom18Bit(i)
    j = form.get18BitInputForVoltage(v)
    jv = form.getVoltageFrom18Bit(j)
    # j = form.get18BitStepSize(form.getVoltageFrom18Bit(i) + 10)
    # if (i - j) != 0:
    print(i, '\t', v, '\t', j, '\t', jv, '\t', i - j, '\t', v - jv)

