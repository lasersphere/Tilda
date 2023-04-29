"""

Created on '07.08.2021'

@author:'lrenth'

"""

import logging

from PyQt5 import QtWidgets, QtGui

from Tilda.Interface.ArithmeticsUi.Ui_AddObservable import Ui_AddObservable


class AddObservableUi(QtWidgets.QDialog, Ui_AddObservable):
    """
    Dialog asking for an observable name and observable value in MHz
    """
    def __init__(self, observable_win, name=None, value=None):
        super(AddObservableUi, self).__init__()
        self.setupUi(self)
        self.parent_ui = observable_win
        self.original_name = name

        """Buttons"""
        self.pb_add.clicked.connect(self.add)
        self.pb_cancel.clicked.connect(self.cancel)

        """Line Edit"""
        self.le_name.setText(name)
        self.le_value.setValidator(QtGui.QDoubleValidator())
        self.le_value.setText(value)

        self.exec_()

    def add(self):
        obs_name = self.check_name_rules(self.le_name.text())  # check naming conventions
        obs_val = self.le_value.text()
        try:
            float_val = float(obs_val)
            try:
                self.parent_ui.obs_dict.pop(self.original_name)  # remove old obs name from dictionary
            except AttributeError:
                logging.info('No Attribute %s, skip pop' % self.original_name)
            self.parent_ui.obs_dict[obs_name] = float_val
            self.parent_ui.new_obs_name = obs_name
            self.close()
        except ValueError:
            logging.warning('Observable value needs to be a scalar.')

    def check_name_rules(self, name):
        """
        Implement a few checks to make sure the names are nice (and not empty)
        :param name: str: user-input name
        :return: str: changed name if necessary, else user-input
        """
        name0 = name  # remember original name
        # check for bad string parts.
        if name:
            name = name.replace(':', '')  # no colon allowed because it is the separator!
            name = name.replace(' ', '_')  # no whitespace allowed (best practice)
            # more rules could be added here...
        else:
            # user forgot to give it a name. Use default name
            name = 'new_obs'
        # If the name is new or was changed, we should check that it does not conflict with existing entries
        if name != self.original_name:  # name was changed or is added new
            running_num = 1  # create a running number to attach to duplicates
            run_name = name
            while run_name in self.parent_ui.obs_dict.keys():
                run_name = name+'_{}'.format(running_num)
                running_num += 1
            name = run_name
        # if any changes were made: inform the user with a WARNING
        if name != name0:
            logging.warning('observable name bad or already exists: changed from {} to {}'.format(name0, name))

        return name  # return the new name (or old if it was fine)

    def cancel(self):
        self.close()
