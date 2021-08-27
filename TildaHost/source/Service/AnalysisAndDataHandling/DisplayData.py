"""
Created on 22/04/2016

@author: sikaufma

Module Description:

Use this class for displaying your data in combination with a prestarted gui.
This module will run the pipeline etc. to display the data use a LiveDataPlottingUi as gui

"""

from datetime import datetime
import logging

from PyQt5 import QtCore

import Service.AnalysisAndDataHandling.tildaPipeline as TP
from Measurement.XMLImporter import XMLImporter as XmlImp


class DisplayData:
    def __init__(self, file, gui, x_as_volt=False, loaded_spec=None):
        self.pipe = None
        self.gui = gui
        self.file = None
        self.spec = None
        self.x_as_volt = x_as_volt
        self.load_spectra(file, loaded_spec)
        self.select_pipe()
        self.feed_loaded_spec(self.spec)
        self.gui.gate_data(None, True)
        self.gui.plots_not_updated_since_window_created = False

    def load_spectra(self, file, loaded_spec=None):
        self.file = file
        if loaded_spec is None:
            self.spec = XmlImp(file, x_as_volt=self.x_as_volt)
        else:
            self.spec = loaded_spec

    def select_pipe(self):
        callbacks = (None, None, None) if self.gui is None else self.gui.callbacks
        self.pipe = TP.time_resolved_display(self.file, callbacks)
        self.pipe.start()
        logging.info('pipeline started to display %s ' % self.file)

    def feed_loaded_spec(self, spec=None):
        start = datetime.now()
        if spec is None and self.spec is not None:
            self.pipe.feed(self.spec)
        else:
            self.pipe.feed(spec)
        stop = datetime.now()
        logging.info('displaying data took: %.1f ms' % ((stop - start).total_seconds() * 1000))

    def clear_pipe(self):
        self.pipe.clear()

    def bring_to_focus(self):
        window = self.gui
        # this will remove minimized status
        # and restore window with keeping maximized/normal state
        window.setWindowState(window.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)

        # this will activate the window
        window.activateWindow()

    def close_display_data(self):
        # self.clear_pipe()
        self.pipe.stop()
        del self.pipe
        del self.spec
        # self.pipe = None
        # self.spec = None
        logging.info('closed Displaydata of file %s' % self.file)



# path = 'E:\\lala\\sums\\dummy_trsdummy_008.xml'
# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     # ui = TRSLivePlotWindowUi()
#     # ui.show()
#     disp_data = DisplayData(path, True)
#     app.exec()

    # print(disp_data.spec.cts)
    # print(disp_data.spec.t_proj)
