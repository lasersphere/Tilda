"""

Created on '20.08.2015'

@author:'simkaufm'

"""

import functools
import logging
import sys

import numpy as np

logging.basicConfig(level=getattr(logging, 'INFO'), format='%(message)s', stream=sys.stdout)
import pyqtgraph as pg

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


def plot_x_y(x, y):
    """
    Create and return a :class:`PlotWindow <pyqtgraph.PlotWindow>`
    (this is just a window with :class:`PlotWidget <pyqtgraph.PlotWidget>` inside), plot data in it.
    Accepts a *title* argument to set the title of the window.
    All other arguments are used to plot data. (see :func:`PlotItem.plot() <pyqtgraph.PlotItem.plot>`)
    """
    win = pg.plot(x=x, y=y)

    return win


def plot_spec_data(spec_data, sc, tr, plot_item=None, pen='k'):
    x, y, err = spec_data.getArithSpec(sc, tr)
    if plot_item is None:
        return plot_x_y(x, y)
    else:
        plot_item.plot(x, y, pen=pen)


def create_image_view(x_label='line voltage', y_label='time [µs]'):
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
    if 'left' not in do_not_show_label or 'right' not in do_not_show_label:
        plt_item.setLabel('left' if 'left' not in do_not_show_label else 'right', y_label)
    if 'bottom' not in do_not_show_label or 'top' not in do_not_show_label:
        plt_item.setLabel('bottom' if 'bottom' not in do_not_show_label else 'top', x_label)
    return widg, plt_item


def create_plot_for_all_sc(target_layout, pmt_list, slot_for_mouse_move, max_rate, plot_sum=True):
    """
    this will add a pyqtgraph widget for each scaler and an additional one for the sum to the target layout.
    :param target_layout: QtLayout, here the widgets will be added
    :param pmt_list: list, containing the indices/numbers of the scalers, use specdata.active_pmt_list[tr]
    :return: list, of dicts, {'widget', 'proxy', 'vertLine', 'indList', 'pltDataItem', 'name', 'pltItem', 'fitLine'}
     with sum at the last position
    """
    return_list = []
    max_rate = max_rate
    for sc_ind, sc_name in enumerate(pmt_list):
        label_list = ['top', 'right', 'bottom']
        if not plot_sum:  # if the sum is not plotted, the last pmt must hold the label
            label_list = label_list if sc_ind < len(pmt_list) - 1 else ['top', 'right']
        widg, plt_item = create_x_y_widget(do_not_show_label=label_list, y_label='cts sc%s' % sc_name)
        plt_proxy = create_proxy(signal=plt_item.scene().sigMouseMoved,
                                 slot=functools.partial(slot_for_mouse_move, plt_item.vb, False),
                                 rate_limit=max_rate)
        if sc_ind:  # link to the plot before in list, not in first index(=0) of course
            plt_item.vb.setXLink(return_list[-1]['pltItem'].getViewBox())
        plt_data_item = plt_item.plot(pen='k')
        plt_inf_line = create_infinite_line(0, pen='r')
        plt_item.addItem(plt_inf_line)
        singl_dict = {
            'name': str(sc_name),
            'indList': [sc_ind],
            'widget': widg,
            'proxy': plt_proxy,
            'vertLine': plt_inf_line,
            'pltItem': plt_item,
            'pltDataItem': plt_data_item,
            'fitLine': None
        }
        return_list.append(singl_dict)
        target_layout.addWidget(widg)
    if plot_sum:
        sum_wid, sum_plt_item = create_x_y_widget(do_not_show_label=['top', 'right'], y_label='cts sum')
        sum_proxy = create_proxy(signal=sum_plt_item.scene().sigMouseMoved,
                                 slot=functools.partial(slot_for_mouse_move, sum_plt_item.vb, False),
                                 rate_limit=max_rate)
        sum_plt_data_item = sum_plt_item.plot(pen='b')
        sum_inf_line = create_infinite_line(0, pen='r')
        sum_plt_item.addItem(sum_inf_line)
        target_layout.addWidget(sum_wid)
        sum_plt_item.vb.setXLink(return_list[-1]['pltItem'].getViewBox())
        sum_dict = {
            'name': 'sum',
            'indList': range(0, len(pmt_list)),
            'widget': sum_wid,
            'proxy': sum_proxy,
            'vertLine': sum_inf_line,
            'pltItem': sum_plt_item,
            'pltDataItem': sum_plt_data_item,
            'fitLine': None
        }
        return_list.append(sum_dict)
    return return_list


def plot_all_sc(list_of_widgets_etc, spec_data, tr):
    # print('plotting all pmts in %s' % list_of_widgets_etc)
    for val in list_of_widgets_etc:
        sc = val['indList']
        plt_data_itm = val['pltDataItem']
        x, y, err = spec_data.getArithSpec(sc, tr)
        plt_data_itm.setData(x, y)


def create_viewbox():
    return pg.ViewBox()


def create_plot_data_item(x, y, pen='b'):
    return pg.PlotDataItem(x, y, pen=pen)


def create_axisitem(orientation):
    return pg.AxisItem(orientation)


def image(data):
    return pg.image(data)


def create_proxy(signal, slot, rate_limit=60):
    proxy = pg.SignalProxy(signal, rateLimit=rate_limit, slot=slot)
    return proxy


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


def create_text_item(text="", color=(200, 200, 200), html=None, anchor=(0, 0)):
    return pg.TextItem(text=text, color=color, html=html, anchor=anchor)

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

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    start_examples()
    status = app.exec_()
    sys.exit(status)
