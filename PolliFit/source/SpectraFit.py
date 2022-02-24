"""
Created on 18.02.2022

@author: Patrick Mueller
"""

import os
import ast
import numpy as np

import TildaTools as TiTs
from DBIsotope import DBIsotope
import Measurement.MeasLoad as MeasLoad
from Fitter import ModelFitter
import Models.Model as Mod


class SpectraFit:
    def __init__(self, files, db, run, guess_offset=True, x_as_freq=True, save_ascii=False,
                 show=True, fmt='k.', font_size=10):
        self.files = files
        self.db = db
        self.run = run
        self.guess_offset = guess_offset
        self.x_as_freq = x_as_freq
        self.save_ascii = save_ascii
        self.show = show
        self.fmt = fmt
        self.font_size = font_size

        self.file_types = ['.xml']

        self.fitter = None
        self.gen_fitter()
        
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

    def gen_definition(self):
        definition = None
        return Mod.Definition([definition, ] * len(self.files))

    def gen_fitter(self):
        var = TiTs.select_from_db(self.db, 'isoVar, lineVar, scaler, track', 'Runs', [['run'], [self.run]],
                                  caller_name=__name__)
        if var:
            # st: tuple of PMTs and tracks from selected run
            st = (ast.literal_eval(var[0][2]), ast.literal_eval(var[0][3]))
            linevar = var[0][1]
        else:
            raise ValueError('Run \'{}\' cannot be selected.'.format(self.run))
        softw_gates_trs = (self.db, self.run)  # TODO: Get trs gates from parameter list instead of DB.
        # softw_gates_trs = None  # TODO: Temporary force load from file

        models = []
        meas = []
        for path in self.load_filepaths():
            meas.append(MeasLoad.load(path, self.db, softw_gates=softw_gates_trs))
            if isinstance(meas[-1], MeasLoad.XMLImporter):
                if meas[-1].seq_type == 'kepco':
                    models.append(Mod.Offset(offsets=[1]))
                else:
                    # iso = DBIsotope(self.db, meas.type, lineVar=linevar)
                    models.append(Mod.Offset(model=Mod.NPeak(model=Mod.Lorentz(), n_peaks=1), offsets=[1]))
                    # TODO Replace working minimal example.
            else:
                raise ValueError('File type not supported. The supported types are {}.'.format(self.file_types))
        model = models[0]  # TODO Replace with Linked- or Summed-Model.
        self.fitter = ModelFitter(model, meas, st)

    def print_pars(self):
        print('Current parameters:')
        for pars in self.get_pars():
            print('\t'.join([str(p) for p in pars]))

    def get_pars(self):
        return self.fitter.get_pars()

    def save_pars(self):
        pass
        # # Currently, only data for main fit is saved. No isomeres etc.
        # names = self.fitter.npar
        # vals = self.fitter.par
        # fixes = self.fitter.fix
        # links = self.fitter.link
        # i_center = names.index('center')
        # i_int0 = names.index('Int0')
        # # Split at 'center' since this marks the border between "Lines" pars & "Isotopes" pars
        #
        # # Save Lines pars (Pars 0 until center)
        # shape_vals = dict(zip(names[:i_center], vals[:i_center]))
        # shape_fixes = dict(zip(names[:i_center], fixes[:i_center]))
        # line_var = self.fitter.spec.iso.lineVar
        # line_name = self.fitter.spec.iso.shape['name']
        # shape_vals.update({'name': line_name})
        #
        # # Save Isotope data without Int (due to HFS)
        # iso = self.fitter.meas.type
        # isoData = vals[i_center:i_int0]
        # isoDataFix = fixes[(i_center + 1):i_int0]
        #
        # # Save Int
        # relInt = self.fitter.spec.hyper[0].hfInt
        # nrTrans = len(relInt)
        # intData = vals[i_int0:i_int0 + nrTrans]
        # int0 = sum(intData)/sum(relInt)
        #
        # # Save softGates
        # gatesName = names[-3:]
        # gatesData = vals[-3:]
        #
        # try:
        #     con = sqlite3.connect(self.db)
        #     cur = con.cursor()
        #     # Lines pars:
        #     try:
        #         cur.execute('''UPDATE Lines SET shape = ?, fixShape = ? WHERE lineVar = ?''',
        #                 (str(shape), str(shapeFix), str(lineVar)))
        #         con.commit()
        #         print("Saved line pars in Lines!")
        #     except Exception as e:
        #         print("error: Couldn't save line pars. All values correct?")
        #
        #     # Isotopes pars:
        #     try:
        #         cur.execute('''UPDATE Isotopes SET center = ?, Al = ?, Bl = ?, Au = ?, Bu = ?, intScale = ?, fixedAl = ?, fixedBl = ?, fixedAu = ?, fixedBu = ? WHERE iso = ?''',
        #                     (isoData[0], isoData[1], isoData[2], isoData[3], isoData[4], int0, isoDataFix[0], isoDataFix[1], isoDataFix[2], isoDataFix[3], iso))
        #         con.commit()
        #         print("Saved isotope pars in Isotopes!")
        #     except Exception as e:
        #         print("error: Couldn't save Isotopes pars. All values correct?")
        #
        #     # Timegate pars (only when available):
        #     if gatesName[0] == 'softwGatesWidth':
        #         try:
        #             # Save in softwGates
        #
        #             # gates_tr0 = TiTs.calc_soft_gates_from_db_pars(self.fitter.par[-3], self.fitter.par[-2],
        #             #                                               self.fitter.par[-1], voltage_gates=[-1000, 1000])
        #             # softw_gate_all_tr = [gates_tr0 for each in self.fitter.meas.cts]
        #             # cur.execute('''UPDATE Runs SET softwGates = ? WHERE run = ?''',
        #             #             (str(softw_gate_all_tr), self.run))
        #             # con.commit()
        #
        #             # Save in midTof, softwGateWidth and softwGateDelayList
        #             cur.execute('''UPDATE Runs SET softwGateWidth = ?, softwGateDelayList = ? WHERE run = ?''',
        #                         (float(gatesData[0]), str(gatesData[1]), self.run))
        #             con.commit()
        #             cur.execute('''UPDATE Isotopes SET midTof = ? WHERE iso = ?''', (gatesData[2], iso))
        #             con.commit()
        #             print("Saved gate pars in Runs & Isotopes!")
        #         except Exception as e:
        #             print("error: Coudln't save softwGates. All values correct?")
        #
        #     con.close()
        #
        # except Exception as e:
        #     print("error: No database connection possible. No line pars have been saved!")

    def plot(self):
        if not self.show:
            return

    """ Prints """
    def print_files(self):
        print('\nFile paths:')
        for i, file in enumerate(self.files):
            print('{}: {}'.format(str(i).zfill(int(np.log10(len(self.files)))), file))
