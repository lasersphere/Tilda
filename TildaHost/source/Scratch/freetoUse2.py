"""

Created on '21.08.2015'

@author:'simkaufm'

"""


"""

Created on '21.08.2015'

@author:'simkaufm'

"""

# -*- coding: utf-8 -*-
"""
This example demonstrates the use of RemoteGraphicsView to improve performance in
applications with heavy load. It works by starting a second process to handle
all graphics rendering, thus freeing up the main process to do its work.

In this example, the update() function is very expensive and is called frequently.
After update() generates a new set of data, it can either plot directly to a local
plot (bottom) or remotely via a RemoteGraphicsView (top), allowing speed comparison
between the two cases. IF you have a multi-core CPU, it should be obvious that the
remote case is much faster.
"""

# import initExample ## Add path to library (just for examples; you do not need this)
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.multiprocess as mp


def processPlotter():
    app = pg.mkQApp()

    # Create remote process with a plot window
    proc = mp.QtProcess()
    rpg = proc._import('pyqtgraph')
    print(rpg)
    win = rpg.GraphicsWindow()
    p1 = win.addPlot()
    win.nextRow()
    p2 = win.addPlot()
    data = proc.transfer([])
    data2 = proc.transfer([])
    return rpg, win, p1, p2, data, data2


# outer_pro = processPlotter()
# processPlotter()
# Send new data to the remote process and plot it
# We use the special argument _callSync='off' because we do
# # not want to wait for a return value.
# data.extend([1,5,2,4,3], _callSync='off')
# curve.setData(y=data, _callSync='off')
#
# # win = rpg.GraphicsWindow()
# # p1 = win.addPlot()
# # p2 = win.addPlot()
#
# input('press Enter...')
#
#
# data.extend([7,8,9,10,20], _callSync='off')
# curve.setData(y=data, _callSync='off')
#
# input('press Enter...')
## Start Qt event loop unless running in interactive mode or using pyside.
# if __name__ == '__main__':
#     import sys
#     if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#         QtGui.QApplication.instance().exec_()
#         print('hello')
#         proc.close()

