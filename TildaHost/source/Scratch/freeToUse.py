"""

Created on '22.06.2015'

@author:'simkaufm'

"""
import numpy as np

import Service.Formating as form
import Service.Scan.draftScanParameters as dft


arr = form.createDefaultScalerArrayFromScanDict(dft.draftScanDict)
print(arr)
print(np.in1d(arr, (2 ** 30)))
