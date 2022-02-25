"""
Created on 18.02.2022

@author: Patrick Mueller
"""

import ast

from PyQt5 import QtWidgets

from Gui.Ui_SpectraFit import Ui_SpectraFit
from SpectraFit import SpectraFit
import TildaTools as TiTs


class SpectraFitUi(QtWidgets.QWidget, Ui_SpectraFit):

    def __init__(self):
        super(SpectraFitUi, self).__init__()
        self.setupUi(self)
        self.main_tilda_gui = None
        self.dbpath = None
        self.items_col = []
        self.items_acol = []

        self.spectra_fit = None

        self.connect()
        self.show()

    def con_main_tilda_gui(self, main_tilda_gui):
        self.main_tilda_gui = main_tilda_gui

    def connect(self):
        """ Connect all the GUI elements. """
        # Files.
        self.c_iso.currentIndexChanged.connect(self.load_files)
        self.b_select_all.clicked.connect(
            lambda checked: self.select_from_list(self.get_items(subset='all'), selected=None))
        self.b_select_col.clicked.connect(
            lambda checked: self.select_from_list(self.get_items(subset='col'), selected=None))
        self.b_select_acol.clicked.connect(
            lambda checked: self.select_from_list(self.get_items(subset='acol'), selected=None))
        self.b_select_favorites.clicked.connect(
            lambda checked: self.select_from_list(self.get_items(subset='fav'), selected=None))
        self.check_multi.stateChanged.connect(self.multi)

        # Parameters.
        self.tab_pars.cellChanged.connect(self.set_par)

        self.b_load_pars.clicked.connect(self.load_pars)
        self.b_reset_pars.clicked.connect(self.reset_pars)
        self.b_save_pars.clicked.connect(self.save_pars)

        # Fit.

    def conSig(self, dbSig):
        dbSig.connect(self.dbChange)

    def dbChange(self, dbpath):
        self.dbpath = dbpath
        self.load_runs()
        self.load_isotopes()

    """ Files """

    def load_runs(self):
        self.c_run.clear()
        it = TiTs.select_from_db(self.dbpath, 'run', 'Runs', caller_name=__name__)
        if it is not None:
            for i, r in enumerate(it):
                self.c_run.insertItem(i, r[0])

    def load_isotopes(self):
        self.c_iso.clear()
        it = TiTs.select_from_db(self.dbpath, 'DISTINCT type', 'Files', addCond='ORDER BY type', caller_name=__name__)
        if it is not None:
            for i, r in enumerate(it):
                self.c_iso.insertItem(i, r[0])

    def load_lines(self):
        pass

    def load_files(self):
        self.list_files.clear()
        it = TiTs.select_from_db(self.dbpath, 'file', 'Files',
                                 [['type'], [self.c_iso.currentText()]], 'ORDER BY date', caller_name=__name__)
        if it is not None:
            for r in it:
                self.list_files.addItem(r[0])
        self.gen_item_lists()

    def gen_item_lists(self):
        self.items_col = []
        self.items_acol = []
        for i in range(self.list_files.count()):
            it = TiTs.select_from_db(self.dbpath, 'colDirTrue', 'Files',
                                     [['file'], [self.list_files.item(i).text()]], caller_name=__name__)
            if it is None:
                continue
            if it[0][0]:
                self.items_col.append(self.list_files.item(i))
            else:
                self.items_acol.append(self.list_files.item(i))

    def get_items(self, subset='all'):
        """
        :param subset: The label of the subset.
        :returns: The specified subset of items from the files list.
        """
        if subset == 'col':
            return self.items_col
        elif subset == 'acol':
            return self.items_acol
        else:
            return [self.list_files.item(i) for i in range(self.list_files.count())]

    def select_from_list(self, items, selected=None):
        """

        :param items: The set of items to select.
        :param selected: Whether to select or deselect the 'items'.
         If None, the 'items' are (de-)selected based on there current selection status.
        :returns: None.
        """
        if self.check_multi.isChecked():  # If multi selection is enabled, (de-)select all items of the set.
            if selected is None:
                selected = True
                selected_items = self.list_files.selectedItems()
                if len(selected_items) > 0 and all(item is selected_items[i] for i, item in enumerate(items)):
                    selected = False
            self.list_files.clearSelection()
            for item in items:
                item.setSelected(selected)
        else:  # If multi selection is disabled, select the next item of the set or deselect if there are none.
            if selected is None or selected:
                selected = True
                i0 = self.list_files.currentRow()
                i = (i0 + 1) % self.list_files.count()
                item = self.list_files.item(i)
                while item not in items:
                    if i == i0:
                        selected = False
                        item = self.list_files.item(i0)
                        break
                    i = (i + 1) % self.list_files.count()
                    item = self.list_files.item(i)
                self.list_files.setCurrentItem(item)
                item.setSelected(selected)
            else:
                self.list_files.currentItem().setSelected(False)
        self.list_files.setFocus()

    def multi(self):
        if self.check_multi.isChecked():
            self.list_files.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        else:
            self.list_files.clearSelection()
            self.list_files.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
            self.list_files.currentItem().setSelected(True)
        self.list_files.setFocus()

    """ Parameters """

    def gen_spectra_fit(self):
        files = [f.text() for f in self.list_files.selectedItems()]
        kwargs = dict(guess_offset=self.check_guess_offset.isChecked(),
                      x_as_freq=self.check_x_as_freq.isChecked(),
                      save_ascii=self.check_save_ascii.isChecked(),
                      show=self.check_auto.isChecked(),
                      fmt=self.edit_fmt.text(),
                      font_size=self.s_fontsize.value())
        return SpectraFit(files, self.dbpath, self.c_run.currentText(), **kwargs)

    def load_pars(self):
        self.spectra_fit = self.gen_spectra_fit()

        self.tab_pars.blockSignals(True)
        self.tab_pars.setRowCount(len(self.spectra_fit.fitter.model.names))
        if self.check_x_as_freq.isChecked():
            pars = self.spectra_fit.get_pars()
        else:
            pars = self.spectra_fit.get_pars()  # TODO get_pars_e()

        for i, (name, val, fix, link) in enumerate(pars):
            w = QtWidgets.QTableWidgetItem(name)
            # w.setFlags(QtCore.Qt.ItemIsEnabled)
            self.tab_pars.setItem(i, 0, w)

            w = QtWidgets.QTableWidgetItem(str(val))
            self.tab_pars.setItem(i, 1, w)

            w = QtWidgets.QTableWidgetItem(str(fix))
            self.tab_pars.setItem(i, 2, w)

            w = QtWidgets.QTableWidgetItem(str(link))
            self.tab_pars.setItem(i, 3, w)
        self.tab_pars.blockSignals(False)

        if self.check_auto.isChecked():
            self.plot()

    def reset_pars(self):
        self.spectra_fit.reset()
        self.load_pars()

    def save_pars(self):
        self.spectra_fit.save_pars()

    def set_par(self, i, j):
        val = None
        try:
            val = ast.literal_eval(self.tab_pars.item(i, j).text())
        except ValueError as e:
            print('error: %s while converting your typed value: %s using 0. / False instead' % (e, val))
            val = 0. if j == 1 else False
        if isinstance(val, float) or isinstance(val, int) \
                or isinstance(val, bool) or isinstance(val, list):
            val = val
        else:
            e = 'val is not a float / int / bool / list'
            print('error: %s while converting your typed value: %s of type: %s using 0. / False instead' % (
                e, type(val), val))
            val = 0. if j == 1 else False
        if j == 1:
            self.spectra_fit.set_val(i, val)
            # if self.spectra_fit.fitter.npar[i] in ['softwGatesWidth', 'softwGatesDelayList', 'midTof']:
            #     if self.main_tilda_gui:
            #         cur_itm = self.fileList.currentItem()
            #         file_path = TiTs.select_from_db(self.dbpath, 'filePath', 'Files',
            #                                         [['file'], [cur_itm.text()]], 'ORDER BY date',
            #                                         caller_name=__name__)
            #         if file_path is None:
            #             return None
            #         file_n = os.path.join(os.path.dirname(self.dbpath), file_path[0][0])
            #         f_win = self.main_tilda_gui.file_plot_wins.get(file_n, False)
            #         if f_win:
            #             f_win.gate_data(self.spectra_fit.fitter.meas) TODO
        elif j == 2:
            self.spectra_fit.set_fix(i, val)
        else:
            self.spectra_fit.set_link(i, val)
        self.tab_pars.blockSignals(True)
        self.tab_pars.item(i, j).setText(str(val))
        self.tab_pars.blockSignals(False)
        if self.check_auto.isChecked():
            self.plot()

    def plot(self):
        self.spectra_fit.plot(clear=True, show=True)
