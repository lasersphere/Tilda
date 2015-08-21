"""

Created on '21.08.2015'

@author:'simkaufm'

"""

import numpy as np
import Scratch.freetoUse2 as fT2
import time


rpg, win, p1, p2, data, data2 = fT2.processPlotter()
# p1.plot(np.random.normal(size=100), pen=(255,0,0), name="Red curve")
# p2.plot(np.random.normal(size=110)+5, pen=(0,255,0), name="Blue curve")
# input('press')
# data.extend([1,5,2,4,3], _callSync='off')
# curve.setData(y=data, _callSync='off')
for i in range(0, 100):
    print(i)
    if i % 5 == 0:
        dat = np.random.random(10)
        print('plotting: ', dat)
        p1.plot(dat, pen=(255,0,0), name="Red curve")
        p2.plot(dat, pen=(255,0,0), name="Red curve")
        p2.plot(dat+5, pen=(0,255,0), name="Blue curve")
        # data.extend(dat, _callSync='off')
        # curve.setData(y=data, _callSync='off')
        # curve2.setData(g=data, _callSync='off')
    elif i == 49:
        p3 = win.addPlot()
        dat = np.random.random(10)
        p1.plot(dat+5, pen=(255,255,0), name="Red curve")

    time.sleep(0.02)
input('anything for exiting...')
