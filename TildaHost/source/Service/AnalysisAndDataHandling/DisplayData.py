"""
Created on 22/04/2016

@author: sikaufma

Module Description:

Use this class for displaying your data.

"""


import Service.AnalysisAndDataHandling.tildaPipeline as TP
from Measurement.XMLImporter import XMLImporter as XmlImp
import Application.Config as Cfg
import MPLPlotter


class DisplayData:
    def __init__(self, file):
        self.pipe = None
        self.file = None
        self.fig = None
        self.cid = None
        self.spec = None
        self.load_spectra(file)
        self.select_pipe()
        self.feed_loaded_spec()
        self.clear_pipe()

    def load_spectra(self, file):
        self.file = file
        self.spec = XmlImp(file, x_as_volt=False)

    def select_pipe(self):
        if self.spec.seq_type in ['trs', 'trsdummy', 'tipa', 'tipadummy']:
            print('loading time resolved spectrum: ', self.file)
            self.pipe = TP.time_resolved_display(self.file)
            self.pipe.start()
            self.fig = MPLPlotter.get_current_figure()
        else:
            print('sorry, only resolved spectra currently supported')

    def feed_loaded_spec(self):
        self.pipe.feed(self.spec)

    def clear_pipe(self):
        self.pipe.clear()

    def con_close_event(self):
        self.cid = self.fig.canvas.mpl_connect('close_event', self.close_spec)

    def close_spec(self, event):
        self.fig.canvas.mpl_disconnect(self.cid)
        Cfg._main_instance.close_spectra_in_main(self.file)  # command flow goes upstream =/
