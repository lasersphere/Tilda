"""

Created on '20.08.2015'

@author:'simkaufm'

"""

import pyqtgraph as pg

def plot(*args):
    for a in args:
        pg.plot(a[0], a[1])
