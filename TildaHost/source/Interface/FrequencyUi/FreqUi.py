"""

Created on '06.08.2021'

@author:'lrenth'

"""

from PyQt5 import QtWidgets, QtCore, QtGui
import Application.Config as Cfg
import configparser

from Interface.FrequencyUi.Ui_Frequency import Ui_Frequency
from Interface.FrequencyUi.AddFreqUi import AddFreqUi


class FreqUi(QtWidgets.QMainWindow, Ui_Frequency):
    """
    A class for setting frequency options
    """
    main_ui_status_call_back_signal = QtCore.pyqtSignal(dict)

    def __init__(self, main_gui):
        super(FreqUi, self).__init__()
        self.options = Cfg._main_instance.local_options # need current options to fill in line edits
        self.freq_list = self.options.freq_dict # dictionary with frequency names and values in MHz
        self.new_freq_name = None   # name of newly added frequency
        self.freq_arith = self.options.freq_arith   # arithmetic to calculate total frequency

        self.setupUi(self)
        self.stored_window_title = 'Frequency'
        self.setWindowTitle(self.stored_window_title)
        self.main_gui = main_gui

        ''' push button functionality '''
        self.pb_add.clicked.connect(self.add_freq)
        self.pb_remove.clicked.connect(self.rem_sel_freq)
        self.pb_edit.clicked.connect(self.edit_freq)

        '''Update List view'''
        for name in self.freq_list: # fill in all frequencies used in current options
            print(str(self.freq_list[name]))
            self.list_frequencies.addItems([name + ': ' + str(self.freq_list[name])])

        '''Update Line Edit'''
        self.le_arit.setText(self.freq_arith)   # fill in current arithmetic from options

        self.show()

    def add_freq(self):
        """
        opens dialog to add a new frequency which is then presented in the listview
        """
        self.open_add_freq_win()
        if self.new_freq_name != None and self.new_freq_name != '': # if new frequency added in the dialog, add to
                                                                    # listview
            self.list_frequencies.addItems([self.new_freq_name + ': ' + str(self.freq_list[self.new_freq_name])])
        else:
            pass

    def rem_sel_freq(self):
        """
        if an item is selected, remove this from the listview and from the frequency dictionary.
        """
        item = self.list_frequencies.currentItem()
        name = ''
        try:
            sep = item.text().index(':')
            i = 0
            while i < sep:  # create key
                name += item.text()[i]
                i += 1

            self.freq_list.pop(name)  # remove from frequency dictionary

            self.list_frequencies.takeItem(self.list_frequencies.row(item))  # update listview
        except ValueError:
            print('ValueError')

    def edit_freq(self):
        """
        opens dialog to edit the selected frequency which is then presented in the listview and updated in dictionary
        """
        item = self.list_frequencies.currentItem()
        name = ''
        value = ''
        try:
            sep = item.text().index(':')
            length = len(item.text())
            i = 0
            while i < sep:  # create key
                name += item.text()[i]
                i += 1
            i += 1
            while i < length:   # get value
                value += item.text()[i]
                i += 1

            self.open_add_freq_win(freq_name=name, freq_val=value)  # open window to edit
            if self.new_freq_name != '':  # if frequency added in the dialog, add to listview
                self.list_frequencies.takeItem(self.list_frequencies.row(item))
                self.list_frequencies.addItems([self.new_freq_name + ': ' + self.freq_list[self.new_freq_name]])
            else:
                pass
        except ValueError:
            print('ValueError')


    def accept(self):
        """
        Accept changes:
        read line edit input for new arithmetic
        save new settings to options object of main instance
        save new settings to ini
        close the dialog
        """
        self.set_arith()
        self.update_options()
        self.update_ini()
        self.closeEvent()

    def reject(self):
        self.closeEvent()

    ''' open windows '''

    def open_add_freq_win(self, freq_name='', freq_val=''):
        AddFreqUi(self, freq_name, freq_val)

    ''' window related '''

    def closeEvent(self, *args, **kwargs):
        """ overwrite the close event """
        if self.main_gui is not None:
            # tell main window that this window is closed.
            self.main_gui.close_freq_win()

    def update_options(self):
        """
        update the options object in main instance to new frequency settings
        """
        Cfg._main_instance.local_options.freq_dict = self.freq_list
        Cfg._main_instance.local_options.freq_arith = self.freq_arith

    def update_ini(self):
        """
        creates a updated ini-file with updated options
        """
        config = configparser.ConfigParser()
        config.sections()
        config.read(Cfg._main_instance.ini_file_path)  # read options.ini
        print(self.freq_arith)
        config['FREQUENCY'] = {'freq_dic': str(self.freq_list), 'arithmetic': self.freq_arith}  # set frequency options
        with open(Cfg._main_instance.ini_file_path, 'w') as optionsfile:    # rewrite ini-File
            config.write(optionsfile)
        print('Updated options.ini')

    def set_arith(self):
        """
        get arithmetic from line input
        """
        self.freq_arith = self.le_arit.text()
