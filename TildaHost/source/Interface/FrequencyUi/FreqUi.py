"""

Created on '06.08.2021'

@author:'lrenth'

"""

from PyQt5 import QtWidgets, QtCore, QtGui
import Application.Config as Cfg
import configparser
import Physics

from Interface.FrequencyUi.Ui_Frequency import Ui_Frequency
from Interface.FrequencyUi.AddFreqUi import AddFreqUi


class FreqUi(QtWidgets.QMainWindow, Ui_Frequency):
    """
    A class for setting frequency options
    """
    main_ui_status_call_back_signal = QtCore.pyqtSignal(dict)

    def __init__(self, main_gui):
        super(FreqUi, self).__init__()
        self.options = Cfg._main_instance.local_options  # need current options to fill in line edits
        # dictionary with frequency names and values in MHz and arithmetic to calculate total frequency:
        self.freq_dict, self.freq_arith = self.options.get_freq_settings()
        # dictionary with frequency names and values in MHz
        self.new_freq_name = None   # name of newly added frequency

        self.setupUi(self)
        self.stored_window_title = 'Frequency'
        self.setWindowTitle(self.stored_window_title)
        self.main_gui = main_gui

        ''' push button functionality '''
        self.pb_add.clicked.connect(self.add_freq)
        self.pb_remove.clicked.connect(self.rem_sel_freq)
        self.pb_edit.clicked.connect(self.edit_freq)

        '''Update List view'''
        for name, value in self.freq_dict.items():  # fill in all frequencies used in current options
            #print(str(self.freq_list[name]))
            self.list_frequencies.addItems(['{}: {}'.format(name, value)])

        '''Update Line Edit'''
        self.le_arit.setText(self.freq_arith)   # fill in current arithmetic from options

        ''' Give feedback on arithmetic '''
        self.check_arith()  # check once on setup
        self.le_arit.editingFinished.connect(self.check_arith)  # and then on each change

        self.show()

    def add_freq(self):
        """
        opens dialog to add a new frequency which is then presented in the listview
        """
        self.open_add_freq_win()
        if self.new_freq_name is not None:  # if new frequency added in the dialog, add to listview
            self.list_frequencies.addItems([self.new_freq_name + ': ' + str(self.freq_dict[self.new_freq_name])])
            self.check_arith()
        else:
            pass

    def rem_sel_freq(self):
        """
        if an item is selected, remove this from the listview and from the frequency dictionary.
        """
        item = self.list_frequencies.currentItem()
        try:
            name, value = item.text().split(': ')
            self.freq_dict.pop(name)  # remove from frequency dictionary
            self.list_frequencies.takeItem(self.list_frequencies.row(item))  # update listview
            self.check_arith()
        except ValueError:
            print('ValueError')

    def edit_freq(self):
        """
        opens dialog to edit the selected frequency which is then presented in the listview and updated in dictionary
        """
        item = self.list_frequencies.currentItem()
        try:
            name, value = item.text().split(': ')

            self.open_add_freq_win(freq_name=name, freq_val=value)  # open window to edit
            if self.new_freq_name is not None:  # if frequency added in the dialog, add to listview

                self.list_frequencies.takeItem(self.list_frequencies.row(item)) # remove old from listview
                self.list_frequencies.addItems(['{}: {}'.format(self.new_freq_name, self.freq_dict[self.new_freq_name])])
                self.check_arith()
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
        Cfg._main_instance.local_options.set_freq(self.freq_dict, self.freq_arith)
        Cfg._main_instance.save_options()  # save to local options file

    def set_arith(self):
        """
        get arithmetic from line input
        """
        self.freq_arith = self.le_arit.text()

    def check_arith(self):
        """
        Will try to calculate the frequency.
        If successful, then the result will be displayed, if not, a wrning is displayed.
        """
        try:
            freq_MHz = eval(self.le_arit.text(),
                            {'__builtins__': None},
                            self.freq_dict)
            freq_wavenum = Physics.wavenumber(freq_MHz)  # Main takes frequency in cm-1
            freq_nm = Physics.wavelenFromFreq(freq_MHz)

            self.label_2.setText("Current: {:.0f} MHz, {:.4f} cm-1, {:.4f} nm".format(freq_MHz, freq_wavenum, freq_nm))
        except Exception as e:
            self.label_2.setText("Given Frequency names must correspond to the name used in the arithmetic.")
