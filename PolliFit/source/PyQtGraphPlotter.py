"""

Created on '20.08.2015'

@author:'simkaufm'

"""

import logging
import sys

import numpy as np

logging.basicConfig(level=getattr(logging, 'INFO'), format='%(message)s', stream=sys.stdout)
import pyqtgraph as pg

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


def plot_x_y(x, y):
    win = pg.plot(x=x, y=y)

    return win


def plot_spec_data(spec_data, sc, tr):
    x, y, err = spec_data.getArithSpec(sc, tr)
    return plot_x_y(x,y)


def create_image_view(x_label='line voltage', y_label='time'):
    plt_item = pg.PlotItem()
    imv_widget = pg.ImageView(view=plt_item)
    plt_item.invertY(False)
    plt_item.showAxis('top')
    plt_item.showAxis('right')
    plt_item.showLabel('bottom', False)
    plt_item.showLabel('right', False)
    plt_item.getAxis('right').setStyle(showValues=False)
    plt_item.getAxis('bottom').setStyle(showValues=False)
    plt_item.setLabel('left', y_label)
    plt_item.setLabel('top', x_label)
    colors = [
        (255, 255, 255),
        (0, 0, 255),
        (0, 255, 255),
        (0, 255, 0),
        (255, 255, 0),
        (255, 0, 0),
    ]
    color = pg.ColorMap(pos=np.linspace(0.0, 1.0, len(colors)), color=colors)
    imv_widget.setColorMap(color)
    return imv_widget, plt_item


def create_x_y_widget(do_not_show_label=['top', 'right'], x_label='line voltage', y_label='cts'):
    widg = pg.PlotWidget()
    plt_item = widg.getPlotItem()
    plt_item.showAxis('top')
    plt_item.showAxis('right')
    for ax in do_not_show_label:
        plt_item.showLabel(ax, False)
        plt_item.getAxis(ax).setStyle(showValues=False)
    plt_item.setLabel('left' if 'left' not in do_not_show_label else 'right', y_label)
    plt_item.setLabel('bottom' if 'bottom' not in do_not_show_label else 'top', x_label)
    return widg, plt_item


def create_viewbox():
    return pg.ViewBox()


def create_plot_data_item(x, y, pen='b'):
    return pg.PlotDataItem(x, y, pen=pen)


def create_axisitem(orientation):
    return pg.AxisItem(orientation)


def image(data):
    return pg.image(data)


def create_roi(pos, size):
    roi = pg.ROI(pos, size, pen=0.5)
    ## handles scaling horizontally around center
    roi.addScaleHandle([1, 0.5], [0.5, 0.5])
    roi.addScaleHandle([0, 0.5], [0.5, 0.5])

    ## handles scaling vertically from opposite edge
    roi.addScaleHandle([0.5, 0], [0.5, 1])
    roi.addScaleHandle([0.5, 1], [0.5, 0])

    ## handles scaling both vertically and horizontally
    roi.addScaleHandle([1, 1], [0, 0])
    roi.addScaleHandle([0, 0], [1, 1])
    return roi


def create_infinite_line(pos, angle=90, pen=0.5):
    inf_line = pg.InfiniteLine(pos, angle=angle, pen=pen)
    return inf_line


def start_examples():
    import pyqtgraph.examples
    pyqtgraph.examples.run()

# import sys
# from PyQt5 import QtWidgets
# import pyqtgraph as pg
#
#
# if __name__=='__main__':
#     app = QtWidgets.QApplication(sys.argv)
#     pg.plot(x=[0, 1, 2], y=[1, 3, 0])
#     status = app.exec_()
#     sys.exit(status)
from PyQt5 import QtWidgets

if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    start_examples()
    status = app.exec_()
    sys.exit(status)
