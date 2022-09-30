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
from Gui.HFMixingConfigUi import HFMixingConfigUi
from SpectraFit import SpectraFit
from Models.Spectrum import SPECTRA
from Models.Convolved import CONVOLVE
import TildaTools as TiTs


colors = ['b', 'g', 'r', 'x', 'm', 'y', 'k']


class SpectraFitUi(QtWidgets.QWidget, Ui_SpectraFit):

    def __init__(self):
        super(SpectraFitUi, self).__init__()
        self.setupUi(self)

        self.trs_config_ui = None
        self.hf_mixing_config_ui = None
        self.load_lineshapes()
        self.load_convolves()
        self.main_tilda_gui = None
        self.dbpath = None
        self.items_col = []
        self.items_acol = []

        self.index_config = 0
        self.index_load = 0
        self.index_marked = 0
        self.fig = None
        self.spectra_fit = None
        self.thread = QtCore.QThread()

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
        self.c_convolve.currentIndexChanged.connect(self.set_convolve)
        self.s_npeaks.valueChanged.connect(self.set_npeaks)
        self.check_offset_per_track.stateChanged.connect(self.toggle_offset_per_track)
        self.edit_offset_order.editingFinished.connect(self.set_offset_order)
        self.check_qi.stateChanged.connect(self.toogle_qi)
        # self.check_hf_mixing.stateChanged.connect(self.toogle_hf_mixing)
        self.b_hf_mixing.clicked.connect(self.open_hf_mixing)
        self.b_racah.clicked.connect(self.set_racah)

        # Fit.
        self.c_xaxis.currentIndexChanged.connect(
            lambda index, _suppress=False: self.set_x_axis(suppress_plot=_suppress))
        self.c_routine.currentIndexChanged.connect(self.set_routine)
        self.check_chi2.stateChanged.connect(self.toggle_chi2)
        self.check_guess_offset.stateChanged.connect(self.toggle_guess_offset)
        self.check_cov_mc.stateChanged.connect(self.toggle_cov_mc)
        self.s_samples_mc.valueChanged.connect(self.set_samples_mc)
        self.edit_arithmetics.editingFinished.connect(self.set_arithmetics)
        self.check_arithmetics.stateChanged.connect(self.toggle_arithmetics)
        self.b_trsplot.clicked.connect(self.open_trsplot)
        self.b_trs.clicked.connect(self.open_trs)
        self.b_adjust.clicked.connect(self.adjust_uf0)
        self.check_summed.stateChanged.connect(self.toogle_summed)
        self.check_linked.stateChanged.connect(self.toogle_linked)
        self.check_save_to_db.stateChanged.connect(self.toggle_save_to_db)
        # self.check_save_figure.stateChanged.connect(self.toggle_save_figure)

        # Plot.
        self.check_x_as_freq.stateChanged.connect(
            lambda state, _suppress=False: self.toggle_xlabel(suppress_plot=_suppress))
        self.b_plot.clicked.connect(self.plot)
        self.b_save_ascii.clicked.connect(self.save_ascii)
        self.b_save_figure.clicked.connect(self.save_plot)
        self.c_fig.currentIndexChanged.connect(self.set_fig_save_format)
        self.check_zoom_data.stateChanged.connect(
            lambda state, _suppress=False: self.set_zoom_data(suppress_plot=_suppress))
        self.edit_fmt.editingFinished.connect(self.set_fmt)
        self.s_fontsize.editingFinished.connect(self.set_fontsize)

        # Action (Fit).
        # self.b_fit.clicked.connect(self.fit)
        self.b_fit.clicked.connect(self.fit_threaded)

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

    def load_convolves(self):
        for i, spec in enumerate(CONVOLVE):
            self.c_convolve.insertItem(i, spec)
        self.c_convolve.setCurrentText('None')

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
        item = self.list_files.item(self.index_marked)
        if item is not None:
            item.setForeground(QtCore.Qt.GlobalColor.black)
        if items:
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
                if len(selected_items) > 0 and len(selected_items) == len(items) \
                        and all(item is selected_items[i] for i, item in enumerate(items)):
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
        hf_config = dict(enabled_l=False, enabled_u=False, Jl=[0.5, ], Ju=[0.5, ],
                         Tl=[[1.]], Tu=[[1.]], fl=[[0.]], fu=[[0.]], mu=0.)
        current_config = dict(lineshape=self.c_lineshape.currentText(),
                              convolve=self.c_convolve.currentText(),
                              npeaks=self.s_npeaks.value(),
                              offset_per_track=self.check_offset_per_track.isChecked(),
                              offset_order=ast.literal_eval(self.edit_offset_order.text()),
                              qi=self.check_qi.isChecked(),
                              hf_config=hf_config)
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
                if config['convolve'] not in CONVOLVE:
                    config['convolve'] = 'None'
                config['hf_config'] = {**hf_config, **config['hf_config']}
            configs.append(config)
        if configs:
            self.set_model_gui(configs[self.index_load])
            self.mark_loaded(self.list_files.selectedItems())
        return configs

    def gen_spectra_fit(self):
        if self.spectra_fit is not None:
            if self.spectra_fit.fitter is not None:
                try:
                    self.thread.disconnect()  # Disconnect thread from the old fitter.
                except TypeError:  # A TypeError is thrown if there are no connections.
                    pass
                self.spectra_fit.fitter.deleteLater()  # Make sure the QObject, which lives in another thread,
                # is deleted before creating a new one.

        files = [f.text() for f in self.list_files.selectedItems()]
        runs = [self.c_run.currentText() for _ in self.list_files.selectedItems()]
        configs = self._gen_configs(files, runs)
        arithmetics = self.edit_arithmetics.text().strip().lower()
        kwargs = dict(x_axis=self.c_xaxis.currentText(),
                      routine=self.c_routine.currentText(),
                      absolute_sigma=not self.check_chi2.isChecked(),
                      guess_offset=self.check_guess_offset.isChecked(),
                      cov_mc=self.check_cov_mc.isChecked(),
                      samples_mc=self.s_samples_mc.value(),
                      arithmetics=arithmetics if arithmetics else None,
                      summed=self.check_summed.isChecked(),
                      linked=self.check_linked.isChecked(),
                      save_to_db=self.check_save_to_db.isChecked(),
                      x_as_freq=self.check_x_as_freq.isChecked(),
                      fig_save_format=self.c_fig.currentText(),
                      fmt=self.edit_fmt.text(),
                      fontsize=self.s_fontsize.value())
        return SpectraFit(self.dbpath, files, runs, configs, self.index_load, **kwargs)

    def load_pars(self, suppress_plot=False):
        if not self.list_files.selectedItems():
            return
        self.spectra_fit = self.gen_spectra_fit()
        if self.check_arithmetics.isChecked():
            self.update_arithmetics()
        else:
            self.set_arithmetics(suppress_plot=True)
        self.update_pars(suppress_plot=suppress_plot)
        self.edit_offset_order.setText(str(self.spectra_fit.configs[self.index_config]['offset_order']))

    def update_pars(self, suppress_plot=False):
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
        self.plot_auto(suppress_plot)

    def update_vals(self, suppress_plot=False):  # Call only if table signals are blocked.
        for i, val in enumerate(self.spectra_fit.fitter.models[self.index_config].vals):
            self.tab_pars.item(i, 1).setText(str(val))
        self.plot_auto(suppress_plot)

    def update_fixes(self, suppress_plot=False):  # Call only if table signals are blocked.
        for i, fix in enumerate(self.spectra_fit.fitter.models[self.index_config].fixes):
            self.tab_pars.item(i, 2).setText(str(fix))
        self.plot_auto(suppress_plot)

    def update_links(self, suppress_plot=False):  # Call only if table signals are blocked.
        for i, link in enumerate(self.spectra_fit.fitter.models[self.index_config].links):
            self.tab_pars.item(i, 3).setText(str(link))
        self.plot_auto(suppress_plot)

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

    def _parse_fix(self, i, j):
        try:
            return ast.literal_eval(self.tab_pars.item(i, j).text())
        except SyntaxError:
            return self.tab_pars.item(i, j).text()

    def copy_pars(self):
        if self.spectra_fit.fitter is None:
            return
        for i, model in enumerate(self.spectra_fit.fitter.models):
            tab_dict = {self.tab_pars.item(_i, 0).text(): [self._parse_fix(_i, _j) for _j in range(1, 4)]
                        for _i in range(self.tab_pars.rowCount())}
            pars = [tab_dict.get(name, [val, fix, link]) for name, val, fix, link in model.get_pars()]
            model.set_pars(pars, force=True)

    def reset_pars(self):
        self.spectra_fit.reset()
        self.update_pars()

    def save_pars(self):
        self.spectra_fit.save_pars()

    def set_par_multi(self, i, j, suppress_plot=False):
        self.tab_pars.blockSignals(True)
        text = self.tab_pars.item(i, j).text()
        for item in self.tab_pars.selectedItems():
            item.setText(text)
            self._set_par(item.row(), item.column())
        for model in self.spectra_fit.fitter.models:
            model.update()
        self.update_vals()
        self.tab_pars.blockSignals(False)
        self.plot_auto(suppress_plot)

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
        self.c_convolve.setCurrentText(config['convolve'])
        self.s_npeaks.setValue(config['npeaks'])
        self.check_offset_per_track.setChecked(config['offset_per_track'])
        self.edit_offset_order.setText(str(config['offset_order']))
        self.check_qi.setChecked(config['qi'])
        self.check_hf_mixing.setChecked(config['hf_config']['enabled_l'] or config['hf_config']['enabled_u'])

    def set_lineshape(self):
        for config in self.spectra_fit.configs:
            config['lineshape'] = self.c_lineshape.currentText()

    def set_convolve(self):
        for config in self.spectra_fit.configs:
            config['convolve'] = self.c_convolve.currentText()

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
                    if size < len(config['offset_order']):
                        _offset_order = offset_order + config['offset_order'][size:]
                    else:
                        _offset_order = offset_order + [max(offset_order), ] * (n_tracks - size)
                elif size > n_tracks:
                    _offset_order = offset_order[:(n_tracks - size)]
                config['offset_order'] = [order for order in _offset_order]
            self.edit_offset_order.setText(str(self.spectra_fit.configs[self.index_config]['offset_order']))
        except (ValueError, TypeError, SyntaxError, IndexError):
            try:
                self.edit_offset_order.setText(str(self.spectra_fit.configs[self.index_config]['offset_order']))
            except IndexError:
                self.edit_offset_order.setText('[0]')

    def toogle_qi(self):
        for config in self.spectra_fit.configs:
            config['qi'] = self.check_qi.isChecked()

    def toogle_hf_mixing(self):
        # for config in self.spectra_fit.configs:
        #     config['hf_mixing'] = self.check_hf_mixing.isChecked()
        pass

    def open_hf_mixing(self):
        if self.spectra_fit.fitter is None:
            return
        if self.hf_mixing_config_ui is not None:
            self.hf_mixing_config_ui.deleteLater()
        self.hf_mixing_config_ui = HFMixingConfigUi(self.spectra_fit.fitter.iso[self.index_config],
                                                    self.spectra_fit.configs[self.index_config]['hf_config'])
        self.hf_mixing_config_ui.close_signal.connect(self.set_hf_config)
        self.hf_mixing_config_ui.show()

    def set_hf_config(self):
        hf_config = deepcopy(self.hf_mixing_config_ui.config)
        self.check_hf_mixing.setChecked(hf_config['enabled_l'] or hf_config['enabled_u'])
        for config in self.spectra_fit.configs:
            config['hf_config'] = hf_config

    def set_racah(self):
        for splitter_model in self.spectra_fit.splitter_models:
            splitter_model.racah()
        self.update_pars()

    """ Fit """

    def toggle_chi2(self):
        self.spectra_fit.absolute_sigma = not self.check_chi2.isChecked()

    def set_x_axis(self, suppress_plot=False):
        # TODO: Implement DAC voltages as x-axis, replace plot option.
        if self.c_xaxis.currentText() == 'DAC volt (TODO)':
            self.c_xaxis.blockSignals(True)
            self.c_xaxis.setCurrentText(self.spectra_fit.x_axis)
            self.c_xaxis.blockSignals(False)
            return
        self.spectra_fit = self.gen_spectra_fit()
        self.plot_auto(suppress_plot)

    def set_routine(self):
        self.spectra_fit.routine = self.c_routine.currentText()

    def toggle_guess_offset(self):
        self.spectra_fit.guess_offset = self.check_guess_offset.isChecked()

    def toggle_cov_mc(self):
        self.spectra_fit.cov_mc = self.check_cov_mc.isChecked()
        if self.check_cov_mc.isChecked():
            self.s_samples_mc.setEnabled(True)
            self.l_samples_mc.setEnabled(True)
        else:
            self.s_samples_mc.setEnabled(False)
            self.l_samples_mc.setEnabled(False)

    def set_samples_mc(self):
        self.spectra_fit.samples_mc = self.s_samples_mc.value()
    
    def _set_scaler(self, scaler):
        if scaler is None:
            self.spectra_fit.reset_st()
            return
        for i in range(len(self.spectra_fit.fitter.st)):
            self.spectra_fit.fitter.st[i][0] = scaler

    def _set_arithmetics(self, arithmetics):
        self.edit_arithmetics.blockSignals(True)
        self.edit_arithmetics.setText(arithmetics)
        if arithmetics == '':
            arithmetics = self.spectra_fit.arithmetics
            self.edit_arithmetics.setText(arithmetics)
        self.spectra_fit.arithmetics = arithmetics
        self.spectra_fit.fitter.config['arithmetics'] = arithmetics
        self.spectra_fit.fitter.gen_data()
        self.edit_arithmetics.blockSignals(False)

    def set_arithmetics(self, suppress_plot=False):
        if self.spectra_fit.fitter is None:
            self.edit_arithmetics.blockSignals(True)
            self.edit_arithmetics.setText('')
            self.edit_arithmetics.blockSignals(False)
            return
        arithmetics = self.edit_arithmetics.text().strip().lower()
        if arithmetics == self.spectra_fit.fitter.config['arithmetics']:
            return
        if arithmetics == '':
            self._set_scaler(None)
            self._set_arithmetics('')
            self.plot_auto(suppress_plot)
            return
        n = self.spectra_fit.fitter.n_scaler  # Only allow arithmetics with scalers that exist for all files and tracks.
        
        if arithmetics[0] == '[':  # Bracket mode (sum of specified scalers).
            if arithmetics[-1] != ']':
                arithmetics += ']'
            try:
                scaler = sorted(set(eval(arithmetics)))
                while scaler and scaler[-1] >= n:
                    scaler.pop(-1)
                if scaler:
                    self._set_scaler(scaler)
                    self._set_arithmetics(str(scaler))
                    self.plot_auto(suppress_plot)
                else:
                    self._set_arithmetics(self.spectra_fit.arithmetics)
            except (ValueError, SyntaxError):
                self._set_arithmetics(self.spectra_fit.arithmetics)
            return

        variables = {'s{}'.format(i): 4.2 for i in range(n)}
        try:  # Function mode (Specify scaler as variable s#).
            eval(arithmetics, variables)
            self._set_scaler(list(i for i in range(n)))
            self._set_arithmetics(arithmetics)
            self.plot_auto(suppress_plot)
        except (ValueError, TypeError, SyntaxError, NameError):
            self._set_arithmetics(self.spectra_fit.arithmetics)

    def update_arithmetics(self):
        self.edit_arithmetics.blockSignals(True)
        self.edit_arithmetics.setText(self.spectra_fit.arithmetics)
        self.edit_arithmetics.blockSignals(False)

    def toggle_arithmetics(self):
        if self.check_arithmetics.isChecked():
            arithmetics = self.edit_arithmetics.text().strip().lower()
            self._set_scaler(None)
            self._set_arithmetics('')
            self.edit_arithmetics.setReadOnly(True)
            if arithmetics != self.spectra_fit.arithmetics:
                self.plot_auto()
        else:
            self.edit_arithmetics.setReadOnly(False)

    def open_trs(self):
        if self.spectra_fit.fitter is None:
            return
        if self.trs_config_ui is not None:
            self.trs_config_ui.deleteLater()
        self.trs_config_ui = TRSConfigUi(self.spectra_fit.fitter.meas[self.index_config].softw_gates)
        self.trs_config_ui.gate_signal.connect(
            lambda gates=self.trs_config_ui.softw_gates: self.set_softw_gates(gates, -1))
        self.trs_config_ui.show()

    def set_softw_gates(self, softw_gates, tr_ind, suppress_plot=False):
        self.spectra_fit.set_softw_gates(self.index_config, softw_gates, tr_ind)
        self.plot_auto(suppress_plot)

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

    def adjust_uf0(self):
        self.spectra_fit.adjust_uf0(self.check_iterate.isChecked(), self.d_volt.value(), self.d_mhz.value())

    def toogle_summed(self):
        pass

    def toogle_linked(self):
        self.spectra_fit.linked = self.check_linked.isChecked()
        if self.check_linked.isChecked():
            if self.check_cov_mc.isChecked():
                self.check_cov_mc.setChecked(False)
            self.check_cov_mc.setEnabled(False)
        else:
            self.check_cov_mc.setEnabled(True)

    def toggle_save_to_db(self):
        self.spectra_fit.save_to_db = self.check_save_to_db.isChecked()

    """ Plot """

    def toggle_xlabel(self, suppress_plot=False):
        self.spectra_fit.x_as_freq = self.check_x_as_freq.isChecked()
        self.plot_auto(suppress_plot)

    def plot_auto(self, suppress=False):
        if not suppress and self.check_auto.isChecked():
            self.plot()

    def plot(self):
        self.spectra_fit.plot(clear=True, show=True)

    def save_ascii(self):
        self.spectra_fit.plot(ascii_path=self.spectra_fit.ascii_path)

    def save_plot(self):
        self.spectra_fit.plot(plot_path=self.spectra_fit.plot_path)

    def set_fig_save_format(self):
        self.spectra_fit.fig_save_format = self.c_fig.currentText()

    def set_zoom_data(self, suppress_plot=False):
        self.spectra_fit.zoom_data = self.check_zoom_data.isChecked()
        self.plot_auto(suppress_plot)

    def set_fmt(self, suppress_plot=False):
        try:
            fmt = self.edit_fmt.text()
            _process_plot_format(fmt)
            self.spectra_fit.fmt = fmt
            self.plot_auto(suppress_plot)
        except ValueError:
            self.edit_fmt.setText(self.spectra_fit.fmt)

    def set_fontsize(self, suppress_plot=False):
        fontsize = self.s_fontsize.value()
        if fontsize != self.spectra_fit.fontsize:
            self.spectra_fit.fontsize = fontsize
        self.plot_auto(suppress_plot)

    """ Action (Fit)"""

    def fit(self):
        self.mark_loaded(self.list_files.selectedItems())
        _, _, info = self.spectra_fit.fit()
        self.tab_pars.blockSignals(True)
        self.update_vals()
        self.tab_pars.blockSignals(False)
        self.mark_warn(info['warn'])
        self.mark_errs(info['errs'])

    def fit_threaded(self):
        self.mark_loaded(self.list_files.selectedItems())
        if self.spectra_fit.fitter is None:
            return
        self.spectra_fit.fitter.config = self.spectra_fit.gen_config()

        if self.thread is not self.spectra_fit.fitter.thread():
            self.spectra_fit.fitter.moveToThread(self.thread)
            self.spectra_fit.fitter.finished.connect(self.thread.quit)
            self.thread.started.connect(self.spectra_fit.fitter.fit)
            self.thread.finished.connect(self.finish_fit)

        self.enable_gui(False)
        self.thread.start()

    def finish_fit(self):
        self.thread.wait()

        _, _, info = self.spectra_fit.finish_fit()

        self.tab_pars.blockSignals(True)
        self.update_vals(suppress_plot=True)
        self.tab_pars.blockSignals(False)
        self.mark_warn(info['warn'])
        self.mark_errs(info['errs'])
        if self.check_save_figure.isChecked():
            for i, path in enumerate(self.spectra_fit.file_paths):
                self.spectra_fit.plot(index=i, clear=True, show=False, plot_path=os.path.split(path)[0])
        self.plot_auto(suppress=False)
        self.enable_gui(True)

    def enable_gui(self, a0):
        self.vert_files.setEnabled(a0)
        self.vert_parameters.setEnabled(a0)
        self.grid_model.setEnabled(a0)
        self.grid_fit.setEnabled(a0)
        self.grid_plot.setEnabled(a0)
        self.b_fit.setEnabled(a0)
        # self.b_abort.setEnabled(not a0)
        #  TODO: abort during fit. This may be difficult to implement cleanly
        #   since curve_fit actually does not allow intervention.
