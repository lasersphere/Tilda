"""
author: Simon Kaufmann

created on 06_05_16
"""

import functools
import random
import sys
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.lines import Line2D


class Window(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        # a figure instance to plot on
        self.figure = plt.figure(facecolor='white')

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Just some button connected to `plot` method
        self.button = QtWidgets.QPushButton('Plot')
        self.button.clicked.connect(self.plot)

        # set the layout
        self.data_len = 1000

        data10 = [random.random() * 10 for i in range(self.data_len)]
        data100 = [random.random() * 100 for i in range(self.data_len)]
        data = np.random.random_sample(self.data_len)
        data = [random.random() for i in range(self.data_len)]

        self.plots = {'pl0': {}}
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.button)
        v_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(v_layout)
        self.id = 0
        self.setLayout(layout)
        self.plots['pl0']['axes'] = self.figure.add_subplot(111)
        self.plots['pl0']['id'] = deepcopy(self.id)
        self.plots['pl0']['label'] = 'random1'
        self.plots['pl0']['data'] = data
        self.plots['pl0']['color'] = 'blue'

        self.add_par_plot(self.plots['pl0']['axes'], 'random10', data10, 'red')
        self.add_par_plot(self.plots['pl0']['axes'], 'random100', data100, 'green')
        self.add_par_plot(self.plots['pl0']['axes'], 'random1000', data100, 'orange')
        self.add_par_plot(self.plots['pl0']['axes'], 'random10000', data100, 'black')

        for key, val in sorted(self.plots.items()):
            self.plots[key]['statusLabel'] = QtWidgets.QLabel(self.plots[key]['label'])
            v_layout.addWidget(self.plots[key]['statusLabel'])
            self.plots[key]['short'] = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+%s' % self.plots[key]['id']), self)
            self.plots[key]['short'].activated.connect(functools.partial(self.active_zoom, self.plots[key]['id']))
            self.plots[key]['shortVis'] = QtWidgets.QShortcut(QtGui.QKeySequence('Alt+%s' % self.plots[key]['id']), self)
            self.plots[key]['shortVis'].activated.connect(functools.partial(self.toggle_visible, key))
            self.plots[key]['axes'].set_ylabel(self.plots[key]['label'])
            data = self.plots[key]['data']
            self.plots[key]['line'] = self.plots[key]['axes'].add_line(Line2D(range(0, len(data)), data))
            self.plots[key]['line'].set_color(self.plots[key]['color'])
            self.plots[key]['axes'].yaxis.label.set_color(self.plots[key]['line'].get_color())
            self.plots[key]['axes'].spines['right'].set_color(self.plots[key]['color'])
            self.plots[key]['axes'].tick_params(axis='y', colors=self.plots[key]['color'])
            self.plots[key]['axes'].autoscale_view()
        self.set_all_zoomable(True)

        self.short_all_zoom = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+a'), self)
        self.short_all_zoom.activated.connect(functools.partial(self.set_all_zoomable, True))

    def set_all_zoomable(self, zoom_bool):
        for key, val in self.plots.items():
            self.set_nav(key, zoom_bool)
    
    def add_par_plot(self, main_ax, label, initial_data, color):
        self.id += 1
        name = 'pl%s' % self.id 
        self.plots[name] = {}
        self.plots[name]['axes'] = main_ax.twinx()
        if self.id > 1:
            adj = 1 - self.id / 10
            print('adjusting position: ', adj)
            self.figure.subplots_adjust(right=adj)
            if self.id == 2:
                self.plots[name]['axes'].spines["right"].set_position(("axes", 1.2))
            else:
                self.plots[name]['axes'].spines["right"].set_position(("axes", 1.2 + (self.id - 2 ) * 0.2))
        self.plots[name]['id'] = deepcopy(self.id)
        self.plots[name]['label'] = label
        self.plots[name]['data'] = initial_data
        self.plots[name]['color'] = color
    
    def plot(self):
        ''' plot some random stuff '''
        # random data
        self.plots['pl0']['data'] = np.random.random_sample(self.data_len)
        self.plots['pl1']['data'] = np.random.random_sample(self.data_len) * 10
        self.plots['pl2']['data'] = np.random.random_sample(self.data_len) * 100
        start = time.clock()
        # create an axis
        for key, val in self.plots.items():
            self.plots[key]['line'].set_ydata(self.plots[key]['data'])
            self.plots[key]['axes'].relim()
            self.plots[key]['axes'].autoscale_view()

        # refresh canvas
        self.canvas.draw()
        print('plotting time:', time.clock()-start)

    def active_zoom(self, active):
        for key, val in self.plots.items():
            self.set_nav(key, False)
            if self.plots[key]['id'] == active:
                self.set_nav(key, True)

    def set_nav(self, key, nav_bool):
        if key in self.plots.keys():
            axes = self.plots[key]['axes']
            label_str = self.plots[key]['label']
            status_l = self.plots[key]['statusLabel']
            pl_id = self.plots[key]['id']
            color = self.plots[key]['color']
            if nav_bool:
                axes.set_navigate(True)
                status_l.setStyleSheet("QLabel { background-color : %s}" % color)
                status_l.setText(label_str + ' zooming'+ '  (Ctrl+%s)' % pl_id)
            else:
                axes.set_navigate(False)
                status_l.setStyleSheet("QLabel { background-color : lightgray}")
                status_l.setText(label_str + '  (Ctrl+%s)' % pl_id)

    def toggle_visible(self, key):
        if key in self.plots.keys():
            pl_id = self.plots[key]['id']
            label_str = self.plots[key]['label']
            axes = self.plots[key]['axes']
            vis = not axes.get_visible()
            axes.set_visible(vis)
            if vis:
                self.set_nav(key, axes.get_navigate())
            else:
                self.plots[key]['statusLabel'].setStyleSheet("QLabel { background-color : white}")
                self.plots[key]['statusLabel'].setText(label_str + '  (Alt+%s)' % pl_id)
            self.canvas.draw()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    main = Window()
    main.show()

    sys.exit(app.exec_())
