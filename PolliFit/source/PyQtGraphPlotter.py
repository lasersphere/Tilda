"""

Created on '20.08.2015'

@author:'simkaufm'

"""


import logging
import sys

logging.basicConfig(level=getattr(logging, 'INFO'), format='%(message)s', stream=sys.stdout)

import pyqtgraph as pg
import pyqtgraph.multiprocess as mp


def init():
    app = pg.mkQApp()

    # Create remote process with a plot window
    proc = mp.QtProcess()
    rpg = proc._import('pyqtgraph')
    rpg.setConfigOption('background', 'w')
    win = rpg.GraphicsWindow()
    return proc, rpg, win

def addPlot(win, title):
    plt = win.addPlot(title=title)
    win.nextRow()
    return plt

def plot(plRef, *args):
    for a in args:
        plRef.plot(a[0], a[1])


# def processPlotter():
#
#     print(rpg)
#     win = rpg.GraphicsWindow()
#     p1 = win.addPlot()
#     win.nextRow()
#     p2 = win.addPlot()
#     data = proc.transfer([])
#     data2 = proc.transfer([])
#     return rpg, win, p1, p2, data, data2