"""

Created on '20.08.2015'

@author:'simkaufm'

"""

import pyqtgraph as pg
import pyqtgraph.multiprocess as mp
from pyqtgraph.Qt import QtGui, QtCore
import time
import logging
import sys

logging.basicConfig(level=getattr(logging, 'INFO'), format='%(message)s', stream=sys.stdout)

def plot(*args):
    pass




app = pg.mkQApp()
print(app)

win = pg.GraphicsWindow()
p1 = win.addPlot()


if __name__ == '__main__':
    app = pg.mkQApp()
    print(app)

    win = pg.GraphicsWindow()
    p1 = win.addPlot()
    # import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()




# import os
# import pyqtgraph as pg
# import pyqtgraph.multiprocess as mp
#
#
# class pyqtgrTest():
#     def __init__(self):
#         pg.mkQApp()
#         # Create remote process with a plot window
#         self.proc = mp.QtProcess(debug=True)
#         # self.proc.debug = True
#         # print(self.proc.proxies)
#         # print(self.proc.exited)
#         self.rpg = self.proc._import('pyqtgraph')
#         self.plotwin = self.rpg.plot()
#         self.curve = self.plotwin.plot(pen='y')
#
#         # create an empty list in the remote process
#         self.data = self.proc.transfer([])
#         # Send new data to the remote process and plot it
#         # We use the special argument _callSync='off' because we do
#         # not want to wait for a return value.
#         self.data.extend([1, 5, 2, 4, 3], _callSync='off')
#         self.curve.setData(y=self.data, _callSync='off')
#         print(os.getpid())
#         # print(self.proc.exited)
#
#     def close(self):
#         # print('closing now, is it closed arleady:', self.proc.exited)
#         # print(self.proc.proxies)
#         # self.proc.deleteProxy(0)
#         # print(self.proc.proxies)
#         self.proc.close()
