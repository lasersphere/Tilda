"""

Created on '21.08.2015'

@author:'simkaufm'

"""

import numpy as np
import Scratch.freetoUse2 as fT2
import time


proc, curve, data = fT2.processPlotter()
data.extend([1,5,2,4,3], _callSync='off')
curve.setData(y=data, _callSync='off')
for i in range(0, 100):
    print(i)
    if i % 5 == 0:
        dat = np.random.random(10)
        print('updating: ', dat)
        data.extend(dat, _callSync='off')
        curve.setData(y=data, _callSync='off')
    time.sleep(0.2)
