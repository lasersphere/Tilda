"""

Created on '20.08.2015'

@author:'simkaufm'

"""

# -*- coding: utf-8 -*-
"""
This example demonstrates many of the 2D plotting capabilities
in pyqtgraph. All of the plots may be panned/scaled by dragging with
the left/right mouse buttons. Right click on any plot to show a context menu.
"""

# import initExample ## Add path to library (just for examples; you do not need this)


from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import time
import os
import pyqtgraph as pg
import pyqtgraph.multiprocess as mp


class pyqtgrTest():
    def __init__(self):
        pg.mkQApp()
        # Create remote process with a plot window
        self.proc = mp.QtProcess()
        self.proc.debug = True
        # print(self.proc.proxies)
        # print(self.proc.exited)
        self.rpg = self.proc._import('pyqtgraph')
        self.plotwin = self.rpg.plot()
        self.curve = self.plotwin.plot(pen='y')

        # create an empty list in the remote process
        self.data = self.proc.transfer([])
        # Send new data to the remote process and plot it
        # We use the special argument _callSync='off' because we do
        # not want to wait for a return value.
        self.data.extend([1, 5, 2, 4, 3], _callSync='off')
        self.curve.setData(y=self.data, _callSync='off')
        print(os.getpid())
        # print(self.proc.exited)

    def close(self):
        # print('closing now, is it closed arleady:', self.proc.exited)
        # print(self.proc.proxies)
        # self.proc.deleteProxy(0)
        # print(self.proc.proxies)
        self.proc.close()


blub = pyqtgrTest()
time.sleep(2)
# blub.close()


# Send new data to the remote process and plot it
# We use the special argument _callSync='off' because we do
# not want to wait for a return value.
# data.extend([1, 5, 2, 4, 3], _callSync='off')
# curve.setData(y=data, _callSync='off')

# win = rpg.GraphicsWindow()
# p1 = win.addPlot()
# p2 = win.addPlot()

# ## Start Qt event loop unless running in interactive mode or using pyside.
# if __name__ == '__main__':
#     import sys
#     if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#         QtGui.QApplication.instance().exec_()
