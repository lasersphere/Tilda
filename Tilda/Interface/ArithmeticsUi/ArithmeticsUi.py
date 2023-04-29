"""

Created on '06.08.2021'

@author:'lrenth'

"""

from copy import deepcopy
import numpy as np
from PyQt5 import QtWidgets

import Tilda.Application.Config as Cfg
from Tilda.Interface.ArithmeticsUi.Ui_Arithmetics import Ui_Arithmetics
from Tilda.Interface.ArithmeticsUi.AddObservableUi import AddObservableUi


class ArithmeticsUi(QtWidgets.QDialog, Ui_Arithmetics):
    """
    A class for setting arithmetic options for an observable.
    """

    def __init__(self, main_gui, obs_dict, obs_arith, close_func=None, obs_name='Observable', preview=None):
        super(ArithmeticsUi, self).__init__()
        self.options = Cfg._main_instance.local_options  # need current options to fill in line edits
        # dictionary with observable names, values and arithmetic to calculate total observable.
        self.obs_dict, self.obs_arith = deepcopy(obs_dict), deepcopy(obs_arith)
        self.close_func = close_func
        self.obs_name = obs_name
        self.preview = preview
        # dictionary with observable names and values in MHz
        self.new_obs_name = None   # name of newly added observable
        self.val_accepted = False

        self.setupUi(self)
        self.label.setText('{} Arithmetics'.format(self.obs_name))
        self.stored_window_title = '{} Config'.format(obs_name)
        self.setWindowTitle(self.stored_window_title)
        self.main_gui = main_gui

        ''' Push button functionality '''
        self.pb_add.clicked.connect(self.add_obs)
        self.pb_remove.clicked.connect(self.rem_sel_obs)
        self.pb_edit.clicked.connect(self.edit_obs)

        ''' Update List view '''
        for name, value in self.obs_dict.items():  # fill in all observables used in current options
            # print(str(self.obs_list[name]))
            self.list_observables.addItems(['{}: {}'.format(name, value)])

        ''' Update Line Edit '''
        self.le_arith.setText(self.obs_arith)   # fill in current arithmetic from options

        ''' Give feedback on arithmetic '''
        self.check_arith()  # check once on setup
        self.le_arith.editingFinished.connect(self.check_arith)  # and then on each change

        self.show()

    def add_obs(self):
        """
        opens dialog to add a new observable which is then presented in the listview
        """
        self.open_add_obs_win()
        if self.new_obs_name is not None:  # if new observable added in the dialog, add to listview
            self.list_observables.addItems([self.new_obs_name + ': ' + str(self.obs_dict[self.new_obs_name])])
            self.check_arith()
        else:
            pass

    def rem_sel_obs(self):
        """
        if an item is selected, remove this from the listview and from the observable dictionary.
        """
        item = self.list_observables.currentItem()
        try:
            name, value = item.text().split(': ')
            self.obs_dict.pop(name)  # remove from observable dictionary
            self.list_observables.takeItem(self.list_observables.row(item))  # update listview
            self.check_arith()
        except ValueError:
            print('ValueError')

    def edit_obs(self):
        """
        opens dialog to edit the selected observable which is then presented in the listview and updated in dictionary
        """
        if self.list_observables.currentRow() == -1:
            self.list_observables.setCurrentRow(0)
        item = self.list_observables.currentItem()
        if item is None:
            return
        try:
            name, value = item.text().split(': ')

            self.open_add_obs_win(obs_name=name, obs_val=value)  # open window to edit
            if self.new_obs_name is not None:  # if observable added in the dialog, add to listview

                self.list_observables.takeItem(self.list_observables.row(item))  # remove old from listview
                self.list_observables.addItems(['{}: {}'.format(self.new_obs_name, self.obs_dict[self.new_obs_name])])
                self.check_arith()
            else:
                pass
        except ValueError as e:
            print(repr(e))

    def accept(self):
        """
        Accept changes:
        read line edit input for new arithmetic
        save new settings to options object of main instance
        save new settings to ini
        close the dialog
        """
        self.set_arith()
        self.val_accepted = True
        self.closeEvent()

    def reject(self):
        self.closeEvent()

    def closeEvent(self, *args, **kwargs):
        """ overwrite the close event """
        if self.close_func is not None:
            self.close_func()

    def open_add_obs_win(self, obs_name='', obs_val=''):
        AddObservableUi(self, obs_name, obs_val)

    def set_arith(self):
        """
        get arithmetic from line input
        """
        self.obs_arith = self.le_arith.text()

    def check_arith(self):
        """
        Will try to calculate the observable.
        If successful, then the result will be displayed, if not, a warning is displayed.
        """
        try:
            obs = eval(self.le_arith.text(), {'__builtins__': None}, self.obs_dict)
            if self.preview is None:
                text = 'Preview: {}'.format(obs)
            else:
                vals = [obs if func is None else func(obs) for func in self.preview['functions']]
                text = 'Preview: {}'.format(
                    ', '.join(['{} {}'.format(np.around(val, decimals=decs), unit)
                               for val, unit, decs in zip(vals, self.preview['units'], self.preview['decimals'])]))

            self.label_2.setText(text)
        except Exception as e:
            self.label_2.setText('Given observable names must correspond to the name used in the arithmetic.')
