"""
Created on 10.01.2018

@author: fsommer

Module Description:
"""
import logging
import time
from copy import deepcopy

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

import PyQtGraphPlotter as Pg
import TildaTools as TiTs
from Interface.LiveDataPlottingUi.MakePrePostGrid import PrePostGridWidget

class PreDurPostPlotter(QtWidgets.QVBoxLayout):
    def __init__(self, parent, data_parent):
        QtWidgets.QVBoxLayout.__init__(self, parent)
        self.splitter = QtWidgets.QSplitter(Qt.Vertical)  # splitter for adjusting plot display sizes
        self.addWidget(self.splitter)

        self.data_parent = data_parent

        # plot dict stores all info about the different plots
        # {'plotname':{'widget':wid, 'type':dmm/triton, 'name':dmm/dev, 'chan':ch/-}}
        self.plot_dict = {}


    def create_plot_window(self, args):
        '''
        creates an entry to the plot_dict with all relevant information about this plot.
        calls update_plot_windows to display the new plot along with the existing ones
        :param args: tuple, ((dev_name_str, cb_True/False_bool), pre_post_during_scan_str, track_name_str) for example:
         (('dev0:ch0', False), 'duringScan', 'track0')
        '''
        (dev_name, checkbox_bool), pre_post_during_str, tr_name = args
        plotname = pre_post_during_str + ' - ' + dev_name
        if plotname in self.plot_dict:
            # plotname is already in the plot_dict, the widget has been created already, just show it!
            self.plot_dict[plotname]['plotwid'].show()
        else:
            # if the plotname not exist in the plot_dict yet, create a new plot widget
            logging.info('Creating a new plot window for %s!' %plotname)
            triton_dev_bool = ':' in dev_name  # dmms are not allowed to have ':' in name! (due to xml data struct)
            if triton_dev_bool:
                name, ch = dev_name.split(':')
            else:
                name = dev_name
                ch = ''
            live_plot_wid, live_plot_itm = Pg.create_x_y_widget(
                do_not_show_label=['top', 'right', 'bottom'], y_label= pre_post_during_str + ' - ' +dev_name)
            # add entry to the plot_dict
            self.plot_dict[plotname] = {'plotwid': live_plot_wid,
                                        'plotitem': live_plot_itm,
                                        'istriton': triton_dev_bool,
                                        'devname': dev_name,  # for triton: dev0:ch0 for dmms: dmm0
                                        'name': name,
                                        'chan': ch,
                                        'track': tr_name,
                                        'predurpost': pre_post_during_str}
            # add plot-widget to the plot_layout and update all plots to get them filled with data
            self.splitter.addWidget(self.plot_dict[plotname]['plotwid'])
        self.update_plot_windows()

    def hide_plot_window(self, plotname):
        '''
        hides this plot
        :param plotname: the plotname as stored in the dict.
                For triton devices its dev (dev_name=dev:ch), for dmms dev_name
        '''
        self.plot_dict[plotname]['plotwid'].hide()


    def update_plot_windows(self):
        '''
        updates all active plots with the data stored in the data_label of the corresponding widget
        '''
        for plots in self.plot_dict:  # iterate over all active plots
            # extract basic information of the plot. Needed to find the data
            pre_dur_post_str = self.plot_dict[plots]['predurpost']
            tr_name = self.plot_dict[plots]['track']
            is_triton = self.plot_dict[plots]['istriton']
            name = self.plot_dict[plots]['name']
            chan = self.plot_dict[plots]['chan']
            plot_color = 'r'  # standard color for the plots
            for ind, pdp_str in enumerate(['preScan', 'duringScan', 'postScan']):
                # iterate over the pre/dur/post-widgets to select the plot style color
                if pdp_str == pre_dur_post_str:
                    plot_color = ['k', 'r', 'b'][ind]  # individual colors for pre/dur/postScan data
            if is_triton:
                try:
                    plot_data = self.data_parent.data_dict[tr_name]['triton'][pre_dur_post_str][name][chan]['data']
                except:
                    plot_data = []
            else:
                try:
                    plot_data = self.data_parent.data_dict[tr_name]['measureVoltPars'][pre_dur_post_str]['dmms'][name]['readings']
                except:
                    plot_data = []
            # renew the plot with the received data
            self.plot_dict[plots]['plotitem'].plot(plot_data, pen=plot_color)