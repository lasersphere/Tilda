"""
Created on 

@author: simkaufm

Module Description:Interface for the running simple Counter, which will display the current counters and
the currently selected post acceleration device.
"""

import functools
import os
import logging

import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui

import Tilda.Application.Config as Cfg
from Tilda.PolliFit import PyQtGraphPlotter as Pg
from Tilda.Interface.MQTTSimpleCounter.Ui_simpleCounterRunnning import Ui_SimpleCounterRunning

import random
from paho.mqtt import client as mqtt_client


LCD_COUNTER = False
COUNTER_FMT = ['{:1.2f}', '{:2.1f}', '{:3.0f}', ]
MAGNITUDE = ['  ', ' K', ' M', ' B', ' T', ' Q']
FONT_FAMILY = 'Century Gothic'
I = 3
J = 0
# cts = np.array([999, 9999, 99999, 999999])  #
# cts = np.array([999998, 999999, 1000000, 1000001])  # For testing.
# cts = np.array([666, 6666, 66666, 666666])  #


def gen_counter_sb(parent):
    if LCD_COUNTER:
        widg = QtWidgets.QLCDNumber(parent)
        widg.setDigitCount(6)
        widg.display(0)
        return widg
    widg = QtWidgets.QSpinBox(parent)
    widg.setReadOnly(True)
    font = QtGui.QFont(FONT_FAMILY)
    # font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
    widg.setFont(font)
    widg.setMinimum(0)
    widg.setMaximum(int(1e9))
    widg.setValue(0)
    widg.setSuffix(MAGNITUDE[0])
    widg.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
    widg.setGroupSeparatorShown(False)
    widg.setFrame(False)
    widg.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    return widg

def gen_counter(parent):
    if LCD_COUNTER:
        widg = QtWidgets.QLCDNumber(parent)
        widg.setDigitCount(6)
        widg.display(0)
        return widg
    widg = QtWidgets.QLineEdit(parent)
    widg.setReadOnly(True)
    font = QtGui.QFont(FONT_FAMILY)
    # font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
    widg.setFont(font)
    widg.setText(COUNTER_FMT[0].format(0, ' '))
    widg.setFrame(False)
    widg.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    return widg

broker = 'broker.emqx.io'
port = 1883
topic = "DACvoltage"
client_id = f'subscribe-{random.randint(0, 1000)}'

def connect_mqtt() -> mqtt_client:
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    # client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

client=connect_mqtt()

class SimpleCounterRunningUi(QtWidgets.QMainWindow, Ui_SimpleCounterRunning):
    simple_counter_call_back_signal = QtCore.pyqtSignal(tuple)

    def __init__(self, main_gui, act_pmts, datapoints):
        super(SimpleCounterRunningUi, self).__init__()

        work_dir_before_setup_ui = os.getcwd()
        os.chdir(os.path.dirname(__file__))  # necessary for the icons to appear
        self.setupUi(self)
        os.chdir(work_dir_before_setup_ui)  # change back

        self.main_gui = main_gui

        self.first_call = True
        self.arrayFull = False

        self.sample_interval = 0.2  # seconds fpga samples at 200 ms

        self.act_pmts = act_pmts
        self.datapoints = datapoints
        self.names = []
        self.elements = []  # list of dicts.
        self.add_scalers_to_gridlayout(act_pmts)
        # self.x_data = np.array([np.arange(0, self.datapoints) for i in self.act_pmts])
        self.x_data = np.zeros((len(self.act_pmts), 1))
        self.y_data = np.zeros((len(self.act_pmts), 1))

        self.simple_counter_call_back_signal.connect(self.rcv)

        #self.pushButton_stop.clicked.connect(self.stop)
        #self.pushButton_refresh_post_acc_state.clicked.connect(self.refresh_post_acc_state)
        #self.pushButton_reset_graphs.clicked.connect(self.reset_graphs)
        self.doubleSpinBox.valueChanged.connect(self.set_dac_volt)
        self.checkBox.stateChanged.connect(self.connect_disconnect)
        #self.comboBox_post_acc_control.currentTextChanged.connect(self.set_post_acc_ctrl)

        Cfg._main_instance.start_simple_counter(act_pmts, datapoints,
                                                self.simple_counter_call_back_signal, self.sample_interval)
        self.show()
        #self.MQTTsubscribe()

    def resizeEvent(self, a0: QtGui.QResizeEvent):
        self.resize_counters()
        super().resizeEvent(a0)

    def rcv(self, scaler_liste):
        """ handle new incoming data """
        #if self.first_call:
        #    self.refresh_post_acc_state()
        self.first_call = False
        if not self.arrayFull:
            # As long as the array is not at max length we have to increase its size step by step.
            # Else zero values will appear and mess up the axis scaling.
            new_data = False
            for i, j in enumerate(scaler_liste[0]):
                last_second_sum = np.sum(j)
                number_of_new_data_points = scaler_liste[1][i]
                if number_of_new_data_points:
                    new_data = True
                    self.update_counter(i, last_second_sum)
                    self.y_data[i][-1] = last_second_sum
                    self.x_data[i] += number_of_new_data_points * self.sample_interval
                    self.x_data[i][-1] = 0  # always zero at time 0
                    self.update_plot(i, self.x_data[i], self.y_data[i])
            if self.x_data.shape[1] == self.datapoints:
                # Once full array size is reached we can proceed with normal data handling below
                self.arrayFull = True
            elif new_data:
                # Increase array size by 1 adding zeros to the end.
                # These will be overwritten in the next call and therefore never be displayed in the scaler.
                self.x_data = np.concatenate((self.x_data, np.zeros((len(self.act_pmts), 1))), axis=1)
                self.y_data = np.concatenate((self.y_data, np.zeros((len(self.act_pmts), 1))), axis=1)
        else:
            for i, j in enumerate(scaler_liste[0]):
                last_second_sum = np.sum(j)
                number_of_new_data_points = scaler_liste[1][i]
                if number_of_new_data_points:
                    self.update_counter(i, last_second_sum)
                    self.y_data[i] = np.roll(self.y_data[i], -1)
                    self.x_data[i] = np.roll(self.x_data[i], -1)
                    self.y_data[i][-1] = last_second_sum
                    self.x_data[i] += number_of_new_data_points * self.sample_interval
                    self.x_data[i][-1] = 0  # always zero at time 0
                    self.update_plot(i, self.x_data[i], self.y_data[i])

    def update_counter(self, i, last_second_sum):
        # last_second_sum = cts[i]
        el = self.elements[i]['widg']
        if LCD_COUNTER:
            el.display(last_second_sum)
            return
        if last_second_sum == 0:
            el.setText('   0  ')
            return
        modulo = 2
        digits = int(np.floor(np.log10(np.abs(last_second_sum))))
        if digits > 2:
            modulo = digits % 3
            last_second_sum *= 1e-3 ** (digits // 3)
        n_str = COUNTER_FMT[modulo].format(last_second_sum)[:4].rstrip('.').rjust(4)
        el.setText('{}{}'.format(n_str, MAGNITUDE[digits // 3]))

    def update_plot(self, indic, xdata, ydata):
        plt_data_item = self.elements[indic].get('plotDataItem', None)
        if plt_data_item is None:
            pen = Pg.create_pen('b', width=2)
            self.elements[indic]['plotDataItem'] = Pg.create_plot_data_item(xdata, ydata, pen=pen)
            self.elements[indic]['pltItem'].addItem(self.elements[indic]['plotDataItem'])
            self.elements[indic]['pltItem'].vb.invertX(True)  # as requested by Christian
            self.resize_counters()
        else:
            self.elements[indic]['plotDataItem'].setData(xdata, ydata)

    def stop(self):
        self.close()

    def closeEvent(self, *args, **kwargs):
        Cfg._main_instance.stop_simple_counter()
        self.main_gui.close_simple_counter_win()

    def set_post_acc_ctrl(self, state_name):
        Cfg._main_instance.simple_counter_post_acc(state_name)
        self.refresh_post_acc_state()

    def refresh_post_acc_state(self):
        state_num, state_name = Cfg._main_instance.get_simple_counter_post_acc()
        self.label_post_acc_readback_state.setText(state_name)

    def reset_graphs(self):
        self.x_data = np.zeros((len(self.act_pmts), 1))
        self.y_data = np.zeros((len(self.act_pmts), 1))
        self.arrayFull = False

    def set_dac_volt(self):
        volt_dbl = self.doubleSpinBox.value()
        print(volt_dbl)
        Cfg._main_instance.simple_counter_set_dac_volt(volt_dbl)

    def resize_counters(self):
        if LCD_COUNTER:
            return
        font = None
        scale = 99.
        for pl_dict in self.elements:
            el = pl_dict['widg']
            font = el.font()
            metric = QtGui.QFontMetrics(font)
            scale = min([scale, 0.95 * el.width() / metric.horizontalAdvance(el.text())])
            scale = min([scale, 0.95 * el.height() / metric.height()])
        font.setPointSize(int(scale * font.pointSize()))
        for pl_dict in self.elements:
            pl_dict['widg'].setFont(font)

    def splitter_was_moved(self, caller_ind, pos, ind):
        for pl_ind, pl_dict in enumerate(self.elements):
            if caller_ind != pl_ind:
                pl_dict['splitter'].blockSignals(True)
                pl_dict['splitter'].moveSplitter(pos, ind)
                pl_dict['splitter'].blockSignals(False)
        self.resize_counters()

    def add_scalers_to_gridlayout(self, scalers):
        for i, j in enumerate(scalers):
            try:
                name = 'pmt_' + str(j)
                label_name = 'label_pmt_' + str(j)
                splitter = QtWidgets.QSplitter(self.centralwidget)
                splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
                label = QtWidgets.QLabel(splitter)
                label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                font = QtGui.QFont(FONT_FAMILY)
                # font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
                font.setPointSize(48)
                label.setFont(font)
                label.setObjectName(label_name)
                label.setText(str(j))

                widg = gen_counter(splitter)
                sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding,
                                                   QtWidgets.QSizePolicy.Ignored)
                sizePolicy.setHorizontalStretch(0)
                sizePolicy.setVerticalStretch(0)
                widg.setMinimumWidth(int(self.width() / 5))
                sizePolicy.setHeightForWidth(widg.sizePolicy().hasHeightForWidth())
                widg.setSizePolicy(sizePolicy)
                widg.setObjectName(name)

                plt_widg, plt_item = Pg.create_x_y_widget(x_label='time (s)', y_label='cts')
                plt_widg.setMinimumWidth(int(self.width() / 3))
                splitter.insertWidget(1, plt_widg)
                splitter.setStretchFactor(1, 1)
                splitter.splitterMoved.connect(functools.partial(self.splitter_was_moved, i))
                self.centralwidget.layout().insertWidget(i, splitter)
                self.centralwidget.layout().setStretch(i, 1)
                sc_dict = {'name': name, 'label': label, 'widg': widg,
                           'plotWid': plt_widg, 'pltItem': plt_item, 'splitter': splitter}
                self.elements.append(sc_dict)
            except Exception as e:
                logging.error('error in Interface.SimpleCounter.SimpleCounterRunningUi'
                              '.SimpleCounterRunningUi#add_scalers_to_gridlayout: %s' % e, exc_info=True)

    def MQTTsubscribe(self, client=client):

        def subscribe(client: mqtt_client):
            def on_message(client, userdata, msg):
                print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
                self.doubleSpinBox.setProperty("value", msg.payload.decode())

            client.subscribe(topic)
            client.on_message = on_message

        def run(client=client):
            #client = connect_mqtt()
            subscribe(client)
            client.loop_start()

        run()

    def connect_disconnect(self):
        if self.checkBox.isChecked():
            self.MQTTsubscribe()
        else:
            client.loop_stop()
            client.unsubscribe(topic)

