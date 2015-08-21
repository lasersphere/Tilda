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

def plot(plRef, *args, **kwargs):
    if kwargs['clear']:
        plRef.clear()
    for a in args:
        for i, j in enumerate(a[1][0]):
            plRef.plot(a[0], a[1][:, i], pen=(i, len(a[1][0])))
