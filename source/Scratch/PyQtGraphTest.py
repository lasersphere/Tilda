"""

Created on '20.08.2015'

@author:'simkaufm'

"""

import sys

import numpy as np
from PyQt5 import QtWidgets

import PyQtGraphPlotter as pyGpl
from Measurement.XMLImporter import XMLImporter

file = 'E:/TildaDebugging/sums/dbug_trsdummy_377.xml'

spec_data = XMLImporter(file)
print(np.min(spec_data.x[0]), np.max(spec_data.x[0]))
pyGpl.start_examples()

if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)

    # win = pyGpl.plot_spec_data(spec_data, [0, 4], -1)
    x_range = (float(np.min(spec_data.x[0])), np.max(spec_data.x[0]))
    x_scale = np.mean(np.ediff1d(spec_data.x[0]))
    y_range = (np.min(spec_data.t[0]), np.max(spec_data.t[0]))
    y_scale = np.mean(np.ediff1d(spec_data.t[0]))
    win = QtWidgets.QMainWindow()
    cw = QtWidgets.QWidget()
    win.setCentralWidget(cw)
    win.resize(800, 600)
    layout = QtWidgets.QGridLayout()
    cw.setLayout(layout)

    imv_widget, plotitem = pyGpl.create_image_view()
    plotitem.setLabel('bottom', text='line voltage')
    imv_widget.setImage(spec_data.time_res[0][0], pos=[x_range[0] - abs(0.5 * x_scale),
                                                       y_range[0] - abs(0.5 * y_scale)],
                        scale=[x_scale, y_scale])

    plotitem.setAspectLocked(False)
    plotitem.setRange(xRange=x_range, yRange=y_range)

    print(plotitem.viewRange())

    proj_widg, proj_plt_itm = pyGpl.create_x_y_widget()
    proj_widg_t, proj_plt_itm_t = pyGpl.create_x_y_widget()
    proj_plt_itm.plot(spec_data.x[0], spec_data.cts[0][0], symbol='o')

    proj_plt_itm.setXLink(plotitem)
    # plotitem.setXLink(proj_plt_itm)
    # proj_plt_itm_t.setYLink(plotitem)
    plotitem.setYLink(proj_plt_itm_t)

    proj_plt_itm_t.plot(spec_data.t_proj[0][0], spec_data.t[0])

    layout.addWidget(imv_widget)
    layout.addWidget(proj_widg)
    layout.addWidget(proj_widg_t, 0, 1)
    win.show()
    status = app.exec()
    sys.exit(status)
