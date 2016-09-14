"""

Created on '20.08.2015'

@author:'simkaufm'

"""


import logging
import sys

logging.basicConfig(level=getattr(logging, 'INFO'), format='%(message)s', stream=sys.stdout)
import pyqtgraph as pg


def plot_x_y(x, y):
    win = pg.plot(x=x, y=y)

    return win


def plot_spec_data(spec_data, sc, tr):
    x, y, err = spec_data.getArithSpec(sc, tr)
    return plot_x_y(x,y)


def create_image_view():
    plt_item = pg.PlotItem()
    imv_widget = pg.ImageView(view=plt_item)
    plt_item.invertY(False)
    return imv_widget, plt_item


def create_x_y_widget():
    widg = pg.PlotWidget()
    plotitem = widg.getPlotItem()
    return widg, plotitem


def create_viewbox():
    return pg.ViewBox()


def create_plot_data_item(x, y, pen='b'):
    return pg.PlotDataItem(x, y, pen=pen)


def create_axisitem(orientation):
    return pg.AxisItem(orientation)


def image(data):
    return pg.image(data)


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
