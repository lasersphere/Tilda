"""

Created on '22.06.2015'

@author:'simkaufm'

"""
import multiprocessing
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import sys
import os

def MakeGraph(conn=None):

    win = pg.GraphicsWindow(title = "test")
    win.resize(300, 300)

    p1 = win.addPlot(title = "test")

    curve = p1.plot(pen = 'y')
    timer = QtCore.QTimer()
    CurveData = []

    def Update():
        global CurveData
        try:
            # ConnData = conn.rcv()
            # ConnData = [float(i) for i in ConnData]
            CurveData = np.append(CurveData)  # , ConnData)
            curve.setData(CurveData)
        except EOFError:
            print("Graph connection closed\n")
            timer.stop()
            QtGui.QApplication.quit()

    timer.timeout.connect(Update)
    timer.start(50)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()


