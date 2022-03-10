"""
Created on 18.02.2022

@author: Patrick Mueller

TODO: Changing the model config, immediate action?
TODO: Arithmetics.
"""

import os
import ast
from copy import deepcopy
from PyQt5 import QtWidgets, QtCore
# noinspection PyProtectedMember
from matplotlib.axes._base import _process_plot_format

from Gui.Ui_SpectraFit import Ui_SpectraFit
from Gui.TRSConfigUi import TRSConfigUi
from SpectraFit import SpectraFit
from Models.Spectra import SPECTRA
import TildaTools as TiTs


colors = ['b', 'g', 'r', 'x', 'm', 'y', 'k']


class SpectraFitUi(QtWidgets.QWidget, Ui_SpectraFit):

    def __init__(self):
        super(SpectraFitUi, self).__init__()
        self.setupUi(self)
        self.trs_config_ui = None
        self.load_lineshapes()
        self.main_tilda_gui = None
        self.dbpath = None
        self.items_col = []
        self.items_acol = []

        self.index_config = 0
        self.index_load = 0
        self.index_marked = 0
        self.fig = None
        self.spectra_fit = None

        self.connect()
        self.show()

    def con_main_tilda_gui(self, main_tilda_gui):
        self.main_tilda_gui = main_tilda_gui

    def connect(self):
        """ Connect all the GUI elements. """
        # Files.
        self.c_run.currentIndexChanged.connect(self.set_run)
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
        sel_model = self.list_files.selectionModel()
        sel_model.selectionChanged.connect(self.set_index)

        # Parameters.
        self.tab_pars.cellChanged.connect(self.set_par_multi)

        self.b_load_pars.clicked.connect(self.load_pars)
        self.b_up.clicked.connect(self.up)
        self.b_down.clicked.connect(self.down)
        self.b_copy.clicked.connect(self.copy_pars)
        self.b_reset_pars.clicked.connect(self.reset_pars)
        self.b_save_pars.clicked.connect(self.save_pars)

        # Model
        self.c_lineshape.currentIndexChanged.connect(self.set_lineshape)
        self.s_npeaks.valueChanged.connect(self.set_npeaks)
        self.check_offset_per_track.stateChanged.connect(self.toggle_offset_per_track)
        self.edit_offset_order.editingFinished.connect(self.set_offset_order)
        self.check_qi.stateChanged.connect(self.toogle_qi)
        self.check_hf_mixing.stateChanged.connect(self.toogle_hf_mixing)
        self.b_racah.clicked.connect(self.set_racah)

        # Fit.
        self.check_chi2.stateChanged.connect(self.toggle_chi2)
        self.c_routine.currentIndexChanged.connect(self.set_routine)
        self.check_guess_offset.stateChanged.connect(self.toggle_guess_offset)
        self.edit_arithmetics.editingFinished.connect(self.set_arithmetics)
        self.b_trsplot.clicked.connect(self.open_trsplot)
        self.check_summed.stateChanged.connect(self.toogle_summed)
        self.check_linked.stateChanged.connect(self.toogle_linked)
        self.check_save_to_db.stateChanged.connect(self.toggle_save_to_db)
        self.b_save_ascii.clicked.connect(self.save_ascii)

        # Plot.
        self.check_x_as_freq.stateChanged.connect(self.toggle_xlabel)
        self.edit_fmt.editingFinished.connect(self.set_fmt)
        self.s_fontsize.editingFinished.connect(self.set_fontsize)
        self.b_plot.clicked.connect(self.plot)
        self.b_trs.clicked.connect(self.open_trs)

        # Action (Fit).
        self.b_fit.clicked.connect(self.fit)

    def conSig(self, dbSig):
        dbSig.connect(self.dbChange)

    def dbChange(self, dbpath):
        self.dbpath = dbpath
        self.load_runs()
        self.load_isotopes()

        self.index_config = 0
        self.index_load = 0
        self.index_marked = 0
        self.fig = None
        self.spectra_fit = self.gen_spectra_fit()

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

    def load_lineshapes(self):
        for i, spec in enumerate(SPECTRA):
            self.c_lineshape.insertItem(i, spec)
        self.c_lineshape.setCurrentText('Voigt')

    def set_run(self):
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

    def set_index(self):
        self.index_load = 0
        # items = self.list_files.selectedItems()
        # if not items:
        #     return
        # try:
        #     self.index_load = items.index(self.list_files.currentItem())
        # except ValueError:
        #     self.index_load = 0

    def mark_loaded(self, items):
        self.list_files.item(self.index_marked).setForeground(QtCore.Qt.GlobalColor.black)
        items[self.index_load].setForeground(QtCore.Qt.GlobalColor.blue)
        self.index_marked = self.list_files.row(items[self.index_load])
        model_file = items[self.index_load].text()
        self.l_model_file.setText(model_file)
        self.index_config = self.index_load
        self.spectra_fit.index_config = self.index_load

    def mark_warn(self, warn):
        for i in warn:
            items = self.list_files.findItems(self.spectra_fit.files[i], QtCore.Qt.MatchFlag.MatchExactly)
            items[0].setForeground(QtCore.Qt.GlobalColor.yellow)

    def mark_errs(self, errs):
        for i in errs:
            items = self.list_files.findItems(self.spectra_fit.files[i], QtCore.Qt.MatchFlag.MatchExactly)
            items[0].setForeground(QtCore.Qt.GlobalColor.red)

    def select_from_list(self, items, selected=None):
        """
        :param items: The set of items to select.
        :param selected: Whether to select or deselect the 'items'.
         If None, the 'items' are (de-)selected based on there current selection status.
        :returns: None.
        """
        # if not self.list_files.selectedItems():
        #     return
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
            self.list_files.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        else:
            self.list_files.clearSelection()
            self.list_files.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
            item = self.list_files.currentItem()
            if item is None:
                item = self.list_files.item(0)
            item.setSelected(True)
        self.list_files.setFocus()

    """ Parameters """

    def _gen_configs(self, files, runs):
        configs = []
        current_config = dict(lineshape=self.c_lineshape.currentText(),
                              npeaks=self.s_npeaks.value(),
                              offset_per_track=self.check_offset_per_track.isChecked(),
                              offset_order=ast.literal_eval(self.edit_offset_order.text()),
                              qi=self.check_qi.isChecked(),
                              hf_mixing=self.check_hf_mixing.isChecked())
        for file, run in zip(files, runs):
            config = TiTs.select_from_db(self.dbpath, 'config', 'FitPars', [['file', 'run'], [file, run]],
                                         caller_name=__name__)
            if config is None:
                config = TiTs.select_from_db(self.dbpath, 'config', 'FitPars', [['run'], [run]], caller_name=__name__)
            if config is None:
                config = current_config
            else:
                config = {**current_config, **ast.literal_eval(config[0][0])}
                if config['lineshape'] not in SPECTRA:
                    config['lineshape'] = 'Voigt'
            configs.append(config)
        if configs:
            self.set_model_gui(configs[self.index_load])
            self.mark_loaded(self.list_files.selectedItems())
        return configs

    def gen_spectra_fit(self):
        files = [f.text() for f in self.list_files.selectedItems()]
        runs = [self.c_run.currentText() for _ in self.list_files.selectedItems()]
        configs = self._gen_configs(files, runs)
        kwargs = dict(routine=self.c_routine.currentText(),
                      absolute_sigma=not self.check_chi2.isChecked(),
                      guess_offset=self.check_guess_offset.isChecked(),
                      arithmetics=self.edit_arithmetics.text(),
                      summed=self.check_summed.isChecked(),
                      linked=self.check_linked.isChecked(),
                      save_to_db=self.check_save_to_db.isChecked(),
                      x_as_freq=self.check_x_as_freq.isChecked(),
                      fmt=self.edit_fmt.text(),
                      fontsize=self.s_fontsize.value())
        return SpectraFit(self.dbpath, files, runs, configs, self.index_load, **kwargs)

    def load_pars(self):
        self.spectra_fit = self.gen_spectra_fit()
        self.set_offset_order()  # Call once to make the length of the list equal to the number of tracks.
        self.update_pars()

    def update_pars(self):
        if not self.list_files.selectedItems():
            return
        self.tab_pars.blockSignals(True)
        self.tab_pars.setRowCount(self.spectra_fit.fitter.models[self.index_config].size)
        if self.check_x_as_freq.isChecked():
            pars = self.spectra_fit.get_pars(self.index_config)
        else:
            pars = self.spectra_fit.get_pars(self.index_config)  # TODO get_pars_e()?

        for i, (name, val, fix, link) in enumerate(pars):
            w = QtWidgets.QTableWidgetItem(name)
            # noinspection PyUnresolvedReferences
            w.setFlags(w.flags() & ~QtCore.Qt.ItemIsEditable)
            self.tab_pars.setItem(i, 0, w)

            w = QtWidgets.QTableWidgetItem(str(val))
            self.tab_pars.setItem(i, 1, w)

            w = QtWidgets.QTableWidgetItem(str(fix))
            self.tab_pars.setItem(i, 2, w)

            w = QtWidgets.QTableWidgetItem(str(link))
            self.tab_pars.setItem(i, 3, w)
        self.tab_pars.blockSignals(False)
        self.plot_auto()

    def update_vals(self):  # Call only if table signals are blocked.
        for i, val in enumerate(self.spectra_fit.fitter.models[self.index_config].vals):
            self.tab_pars.item(i, 1).setText(str(val))
        self.plot_auto()

    def update_fixes(self):  # Call only if table signals are blocked.
        for i, fix in enumerate(self.spectra_fit.fitter.models[self.index_config].fixes):
            self.tab_pars.item(i, 2).setText(str(fix))
        self.plot_auto()

    def update_links(self):  # Call only if table signals are blocked.
        for i, link in enumerate(self.spectra_fit.fitter.models[self.index_config].links):
            self.tab_pars.item(i, 3).setText(str(link))
        self.plot_auto()

    def display_index_load(self):
        items = [self.list_files.findItems(file, QtCore.Qt.MatchFlag.MatchExactly)[0]
                 for file in self.spectra_fit.files]
        sel_model = self.list_files.selectionModel()
        sel_model.blockSignals(True)
        self.list_files.clearSelection()
        for item in items:
            item.setSelected(True)
        sel_model.blockSignals(False)
        self.set_model_gui(self.spectra_fit.configs[self.index_load])
        self.mark_loaded(items)
        self.update_pars()

    def up(self):
        self.index_load = (self.index_config - 1) % len(self.spectra_fit.configs)
        self.display_index_load()

    def down(self):
        self.index_load = (self.index_config + 1) % len(self.spectra_fit.configs)
        self.display_index_load()

    def copy_pars(self):
        if self.spectra_fit.fitter is None:
            return
        for i, model in enumerate(self.spectra_fit.fitter.models):
            tab_dict = {self.tab_pars.item(_i, 0).text():
                        [ast.literal_eval(self.tab_pars.item(_i, _j).text()) for _j in range(1, 4)]
                        for _i in range(self.tab_pars.rowCount())}
            pars = [tab_dict.get(name, [val, fix, link]) for name, val, fix, link in model.get_pars()]
            model.set_pars(pars)

    def reset_pars(self):
        self.spectra_fit.reset()
        self.update_pars()

    def save_pars(self):
        self.spectra_fit.save_pars()

    def set_par_multi(self, i, j):
        self.tab_pars.blockSignals(True)
        for item in self.tab_pars.selectedItems():
            item.setText(self.tab_pars.item(i, j).text())
            self._set_par(item.row(), item.column())
        self.tab_pars.blockSignals(False)
        self.plot_auto()

    def _set_par(self, i, j):  # Call only if table signals are blocked.
        set_x = [self.spectra_fit.set_val, self.spectra_fit.set_fix, self.spectra_fit.set_link][j - 1]
        update_x = [self.update_vals, self.update_fixes, self.update_links][j - 1]

        try:
            val = ast.literal_eval(self.tab_pars.item(i, j).text())
        except (ValueError, TypeError, SyntaxError):
            val = self.tab_pars.item(i, j).text()
        for index in range(len(self.spectra_fit.files)):
            try:
                _i = self.spectra_fit.fitter.models[index].names.index(self.tab_pars.item(i, 0).text())
                set_x(index, _i, val)
            except ValueError:
                continue
        update_x()

    """ Model """

    def set_model_gui(self, config):
        self.c_lineshape.setCurrentText(config['lineshape'])
        self.s_npeaks.setValue(config['npeaks'])
        self.check_offset_per_track.setChecked(config['offset_per_track'])
        self.edit_offset_order.setText(str(config['offset_order']))
        self.check_qi.setChecked(config['qi'])
        self.check_hf_mixing.setChecked(config['hf_mixing'])

    def set_lineshape(self):
        for config in self.spectra_fit.configs:
            config['lineshape'] = self.c_lineshape.currentText()

    def set_npeaks(self):
        for config in self.spectra_fit.configs:
            config['npeaks'] = self.s_npeaks.value()

    def toggle_offset_per_track(self):
        for config in self.spectra_fit.configs:
            config['offset_per_track'] = self.check_offset_per_track.isChecked()

    def set_offset_order(self):
        try:
            offset_order = list(ast.literal_eval(self.edit_offset_order.text()))
            size = len(offset_order)
            for i, config in enumerate(self.spectra_fit.configs):
                n_tracks = len(self.spectra_fit.fitter.meas[i].x)
                _offset_order = [order for order in offset_order]
                if size == 0:
                    _offset_order = [0, ] * n_tracks
                elif size < n_tracks:
                    _offset_order = offset_order + [max(offset_order), ] * (n_tracks - size)
                elif size > n_tracks:
                    _offset_order = offset_order[:(n_tracks - size)]
                config['offset_order'] = [order for order in _offset_order]
            self.edit_offset_order.setText(str(self.spectra_fit.configs[self.index_config]['offset_order']))
        except (ValueError, TypeError, SyntaxError):
            try:
                self.edit_offset_order.setText(str(self.spectra_fit.configs[self.index_config]['offset_order']))
            except IndexError:
                self.edit_offset_order.setText('[0]')

    def toogle_qi(self):
        for config in self.spectra_fit.configs:
            config['qi'] = self.check_qi.isChecked()

    def toogle_hf_mixing(self):
        for config in self.spectra_fit.configs:
            config['hf_mixing'] = self.check_hf_mixing.isChecked()

    def set_racah(self):
        for splitter_model in self.spectra_fit.splitter_models:
            splitter_model.racah()
        self.update_pars()

    """ Fit """

    def toggle_chi2(self):
        self.spectra_fit.absolute_sigma = not self.check_chi2.isChecked()

    def set_routine(self):
        self.spectra_fit.routine = self.c_routine.currentText()

    def toggle_guess_offset(self):
        self.spectra_fit.guess_offset = self.check_guess_offset.isChecked()

    def set_arithmetics(self):
        pass

    def open_trs(self):
        if self.spectra_fit.fitter is None:
            return
        if self.trs_config_ui is not None:
            self.trs_config_ui.deleteLater()
        self.trs_config_ui = TRSConfigUi(self.spectra_fit.fitter.meas[self.index_config].softw_gates)
        self.trs_config_ui.gate_signal.connect(
            lambda gates=self.trs_config_ui.softw_gates: self.set_softw_gates(gates, -1))
        self.trs_config_ui.show()

    def set_softw_gates(self, softw_gates, tr_ind):
        self.spectra_fit.set_softw_gates(self.index_config, softw_gates, tr_ind)
        self.plot_auto()

    # noinspection PyUnusedLocal
    def softw_gates_from_time_res_plot(self, softw_g, tr_ind, soft_b_width, plot_bool):
        """
        this should be connected to the pyqtsignal in the time resolved plotting window.
        -> This will emit new software gates once they are changed there.
        -> update teh parameter table accordingly.
        :param softw_g: list: software gates [[[tr0_sc0_vMin, tr0_sc0_vMax, tr0_sc0_tMin, tr0_sc0_tMax], [tr0_sc1_...
        :param tr_ind: int: track_index to rebin -1 for all
        :param soft_b_width: list: software bin width in ns for each track
        :param plot_bool: bool: plot bool to force a plotting even if nothing has changed.
        """
        self.set_softw_gates(softw_g, tr_ind)

    def open_trsplot(self):
        if self.spectra_fit.fitter is None:
            return
        cur_itm = self.list_files.item(self.index_marked)
        if cur_itm is None:
            return
        if '.xml' not in cur_itm.text():
            print('Only xml files can be opened as TRS plots.')
            return
        if not self.main_tilda_gui:
            print('Get TILDA to open TRS plots.')
            return
        file_path = TiTs.select_from_db(self.dbpath, 'filePath', 'Files',
                                        [['file'], [cur_itm.text()]], 'ORDER BY date',
                                        caller_name=__name__)
        if file_path is None:
            print(str(cur_itm.text()) + ' not found in DB.')
            return
        file_n = os.path.join(os.path.dirname(self.dbpath), file_path[0][0])

        spec_to_plot = deepcopy(self.spectra_fit.fitter.meas[self.index_config])
        spec_to_plot.x = self.spectra_fit.fitter.get_meas_x_in_freq(self.index_config)
        spec_to_plot.x_units = spec_to_plot.x_units_enums.frequency_mhz

        self.main_tilda_gui.load_spectra(
            file_n, spec_to_plot, sum_sc_tr=self.spectra_fit.fitter.st[self.index_config])

        # -> will be loaded to the main of Tilda
        f_win = self.main_tilda_gui.file_plot_wins.get(file_n, False)
        if f_win:
            # connect to the pyqtsignal which is emitted when the liveplot window receives new softw_gates
            # -> use this to update the pars table in gui here.
            # connect to the callback signal when new gates are set in the live plot.
            # list: software gates [[[tr0_sc0_vMin, tr0_sc0_vMax, tr0_sc0_tMin, tr0_sc0_tMax], [tr0_sc1_...
            # int: track_index to rebin -1 for all
            # list: software bin width in ns for each track
            # bool: plot bool to force a plotting even if nothing has changed.
            # new_gate_or_soft_bin_width = QtCore.pyqtSignal(list, int, list, bool)
            f_win.new_gate_or_soft_bin_width.connect(self.softw_gates_from_time_res_plot)

    def toogle_summed(self):
        pass

    def toogle_linked(self):
        pass

    def toggle_save_to_db(self):
        state = self.check_save_to_db.isChecked()
        self.spectra_fit.save_to_db = state

    """ Plot """

    def toggle_xlabel(self):
        self.spectra_fit.x_as_freq = self.check_x_as_freq.isChecked()
        self.plot_auto()

    def set_fmt(self):
        try:
            fmt = self.edit_fmt.text()
            _process_plot_format(fmt)
            self.spectra_fit.fmt = fmt
            self.plot_auto()
        except ValueError:
            self.edit_fmt.setText(self.spectra_fit.fmt)

    def set_fontsize(self):
        fontsize = self.s_fontsize.value()
        if fontsize != self.spectra_fit.fontsize:
            self.spectra_fit.fontsize = fontsize
        self.plot_auto()

    def plot_auto(self):
        if self.check_auto.isChecked():
            self.plot()

    def plot(self):
        self.spectra_fit.plot(clear=True, show=True)

    def save_ascii(self):
        self.spectra_fit.plot(save_path=self.spectra_fit.save_path)

    """ Action (Fit)"""

    def fit(self):
        _, _, info = self.spectra_fit.fit()
        self.tab_pars.blockSignals(True)
        self.update_vals()
        self.tab_pars.blockSignals(False)
        self.mark_warn(info['warn'])
        self.mark_errs(info['errs'])
