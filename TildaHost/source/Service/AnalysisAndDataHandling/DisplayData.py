"""
Created on 22/04/2016

@author: sikaufma

Module Description:

Use this class for displaying your data.

"""

from datetime import datetime

from PyQt5 import QtCore

import Application.Config as Cfg
import Service.AnalysisAndDataHandling.tildaPipeline as TP
from Interface.LiveDataPlottingUi.LiveDataPlottingUi import TRSLivePlotWindowUi
from Measurement.XMLImporter import XMLImporter as XmlImp


class DisplayData:
    def __init__(self, file, x_as_volt=False):
        self.pipe = None
        self.gui = None
        self.file = None
        self.fig = None
        self.spec = None
        self.x_as_volt = x_as_volt
        self.load_spectra(file)
        self.select_pipe()
        self.feed_loaded_spec()
        # self.clear_pipe()  # for now i don't want to save here.

    def load_spectra(self, file):
        self.file = file
        self.spec = XmlImp(file, x_as_volt=self.x_as_volt)

    def select_pipe(self):
        if self.spec.seq_type in ['trs', 'trsdummy', 'tipa', 'tipadummy']:
            print('loading time resolved spectrum: ', self.file)
            self.gui = TRSLivePlotWindowUi(self.file, self)
            self.pipe = TP.time_resolved_display(self.file, self.gui.callbacks)
            self.pipe.start()
        else:
            print('sorry, only resolved spectra currently supported')

    def feed_loaded_spec(self):
        start = datetime.now()
        self.pipe.feed(self.spec)
        stop = datetime.now()
        print('displaying data took: %s  seconds' % (stop - start))

    def clear_pipe(self):
        self.pipe.clear()

    def bring_to_focus(self):
        window = self.gui
        # this will remove minimized status
        # and restore window with keeping maximized/normal state
        window.setWindowState(window.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)

        # this will activate the window
        window.activateWindow()

    def close_live_plot_win(self):
        Cfg._main_instance.close_spectra_in_main(self.file)



# path = 'E:\\lala\\sums\\dummy_trsdummy_008.xml'
# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     # ui = TRSLivePlotWindowUi()
#     # ui.show()
#     disp_data = DisplayData(path, True)
#     app.exec()

    # print(disp_data.spec.cts)
    # print(disp_data.spec.t_proj)
