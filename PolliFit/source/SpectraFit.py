"""
Created on 18.02.2022

@author: Patrick Mueller
"""

import os
import ast
import sqlite3
import numpy as np

import TildaTools as TiTs
import MPLPlotter as Plot
from DBIsotope import DBIsotope
import Measurement.MeasLoad as MeasLoad
from Fitter import Fitter
import Models.Collection as Mod


class SpectraFit:
    def __init__(self, db, files, runs, configs,
                 routine='curve_fit', absolute_sigma=False, guess_offset=True, arithmetics='',
                 summed=False, linked=False, save_to_db=False, save_ascii=False, x_as_freq=True, fmt='.k', fontsize=10):
        self.db = db
        self.files = files
        self.runs = runs
        self.configs = configs
        self.index_config = 0

        self.routine = routine
        self.absolute_sigma = absolute_sigma
        self.guess_offset = guess_offset
        self.arithmetics = arithmetics
        self.summed = summed
        self.linked = linked
        self.save_to_db = save_to_db

        self.x_as_freq = x_as_freq
        self.save_ascii = save_ascii
        self.fmt = fmt
        self.fontsize = fontsize

        self.file_types = ['.xml']

        self.reset_model = None
        self.fitter = None
        self.gen_fitter()

    def _execute(self, command, *args):
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute(command, *args)
        con.commit()
        con.close()
        
    def load_filepaths(self):
        file_paths = []
        for file in self.files:
            var = TiTs.select_from_db(self.db, 'filePath', 'Files', [['file'], [file]], caller_name=__name__)
            if var:
                file_paths.append(os.path.join(os.path.dirname(self.db), var[0][0]))
            else:
                print(str(file) + ' not found in DB.')
                
        print('\nFile paths:')
        for i, path in enumerate(file_paths):
            print('{}: {}'.format(str(i).zfill(int(np.log10(len(file_paths)))), path))
        return file_paths

    def gen_model(self, config):
        spec_model = eval('Mod.{}()'.format(config['lineshape']))
        split_model = Mod.gen_splitter_model(config)(spec_model)
        npeaks_model = Mod.NPeak(model=split_model, n_peaks=config['npeaks'])
        offset = config['offset_order']
        return Mod.Offset(model=npeaks_model, offsets=offset)

    def gen_config(self):
        return dict(routine=self.routine, absolute_sigma=self.absolute_sigma, guess_offset=self.guess_offset,
                    arithmetics=self.arithmetics, summed=self.summed, linked=self.linked)

    def gen_fitter(self):
        if not self.files:
            self.fitter = None
            return

        models, meas, st, iso = [], [], [], []
        for path, run, config in zip(self.load_filepaths(), self.runs, self.configs):
            var = TiTs.select_from_db(self.db, 'isoVar, lineVar, scaler, track', 'Runs', [['run'], [run]],
                                      caller_name=__name__)
            if var:
                # st: tuple of PMTs and tracks from selected run
                st.append((ast.literal_eval(var[0][2]), ast.literal_eval(var[0][3])))
                linevar = var[0][1]
            else:
                raise ValueError('Run \'{}\' cannot be selected.'.format(run))
            softw_gates_trs = (self.db, run)  # TODO: Get trs gates from parameter list instead of DB.
            # softw_gates_trs = None  # TODO: Temporary force load from file

            meas.append(MeasLoad.load(path, self.db, softw_gates=softw_gates_trs))
            if isinstance(meas[-1], MeasLoad.XMLImporter):
                if meas[-1].seq_type == 'kepco':
                    models.append(Mod.Offset(offsets=[1]))
                else:
                    iso.append(DBIsotope(self.db, meas[-1].type, lineVar=linevar))
                    models.append(self.gen_model(config))
            else:
                raise ValueError('File type not supported. The supported types are {}.'.format(self.file_types))

        self.fitter = Fitter(models, meas, st, iso, self.gen_config())
        self.load_pars()
        self.reset_model = [p for p in models[self.index_config].get_pars()]

    def fit(self):
        if self.fitter is None:
            return
        popt, pcov, info = self.fitter.fit()
        if self.save_to_db:
            self.save_fits(popt, pcov, info)
        return popt, pcov, info

    def get_pars(self, i):
        return self.fitter.get_pars(i)

    def set_val(self, i, j, val):
        self.fitter.set_val(i, j, val)

    def set_fix(self, i, j, fix):
        self.fitter.set_fix(i, j, fix)

    def set_link(self, i, j, link):
        self.fitter.set_link(i, j, link)

    def reset(self):
        if self.reset_model is None:
            return
        for j, (_, val, fix, link) in enumerate(self.reset_model):
            self.set_val(self.index_config, j, val)
            self.set_fix(self.index_config, j, fix)
            self.set_link(self.index_config, j, link)

    def _pars_from_db(self, file, run):
        pars = TiTs.select_from_db(self.db, 'pars', 'FitPars', [['file', 'run'], [file, run]], caller_name=__name__)
        if pars is None:
            pars = TiTs.select_from_db(self.db, 'pars', 'FitPars', [['run'], [run]], caller_name=__name__)
            if pars is None:

                return {}
        pars = ast.literal_eval(pars[0][0])
        return pars

    def load_pars(self):
        self._execute('CREATE TABLE IF NOT EXISTS '
                      '"FitPars"("file" TEXT, "run" TEXT, "config" TEXT, "pars" TEXT, PRIMARY KEY("file", "run"))')
        for i, (file, run) in enumerate(zip(self.files, self.runs)):
            pars = self._pars_from_db(file, run)
            for j, (name, val, fix, link) in enumerate(self.get_pars(i)):
                par = pars.get(name, (val, fix, link))
                self.set_val(i, j, par[0])
                self.set_fix(i, j, par[1])
                self.set_link(i, j, par[2])

    def save_pars(self):
        self._execute('CREATE TABLE IF NOT EXISTS '
                      '"FitPars"("file" TEXT, "run" TEXT, "config" TEXT, "pars" TEXT, PRIMARY KEY("file", "run"))')
        for i, (file, run, config) in enumerate(zip(self.files, self.runs, self.configs)):
            new_pars = {name: (val, fix, link) for (name, val, fix, link) in self.get_pars(i)}
            pars = TiTs.select_from_db(self.db, 'pars', 'FitPars', [['file', 'run'], [file, run]], caller_name=__name__)
            pars = {} if pars is None else ast.literal_eval(pars[0][0])
            new_pars = {**pars, **new_pars}
            self._execute('INSERT OR REPLACE INTO FitPars (file, run, config, pars) VALUES (?, ?, ?, ?)',
                          (file, run, str(config), str(new_pars)))

    def save_fits(self, popt, pcov, info):
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        for i, (file, run) in enumerate(zip(self.files, self.runs)):
            if i in info['errs']:
                continue
            pars = {self.fitter.models[i].names[j]: (pt, np.sqrt(pc[j]), self.fitter.models[i].fixes[j])
                    for j, (pt, pc) in enumerate(zip(popt[i], pcov[i]))}
            cur.execute('INSERT OR REPLACE INTO FitRes (file, iso, run, rChi, pars) '
                        'VALUES (?, ?, ?, ?, ?)', (file, self.fitter.iso[i].name, run, info['chi2'][i], str(pars)))
        con.commit()
        con.close()

    def plot(self, clear=True, show=False):
        if self.fitter is None:
            return
        if clear:
            Plot.clear()

        fig = Plot.plot_model_fit(self.fitter, self.index_config,
                                  x_as_freq=self.x_as_freq, fmt=self.fmt, fontsize=self.fontsize)

        if show:
            Plot.show(True)
            fig.canvas.draw()

    """ Prints """

    def print_pars(self):
        print('Current parameters:')
        for i, file in enumerate(self.files):
            print('File: {}'.format(file))
            for pars in self.get_pars(i):
                print('\t'.join([str(p) for p in pars]))

    def print_files(self):
        print('\nFile paths:')
        for i, file in enumerate(self.files):
            print('{}: {}'.format(str(i).zfill(int(np.log10(len(self.files)))), file))
