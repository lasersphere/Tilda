"""
Created on 18.02.2022

@author: Patrick Mueller
"""

import sqlite3

import TildaTools as TiTs
from Fitter import Fitter
import Model as Mod


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


        self.model = Mod.Model()
        self.fitter = Fitter()

    def print_pars(self):
        print('Current parameters:')
        for pars in zip(self.fitter.names, self.fitter.vals, self.fitter.fixes, self.fitter.links):
            print('\t'.join([str(p) for p in pars]))

    def get_pars(self):
        return zip(self.fitter.names, self.fitter.vals, self.fitter.fixes, self.fitter.links)

    def get_pars_e(self):
        return self.fitter.pars_to_e()

    def set_par(self, i, val):
        if self.x_as_freq:
            self.fitter.set_par(i, val)
        else:
            self.fitter.set_par_e(i, val)

        # if self.fitter.npar[i] in ['softwGatesWidth', 'softwGatesDelayList', 'midTof']:
        #     # one of the gate parameter was changed -> gate data again
        #     # then data needs also to be gated again.
        #     gates_tr0 = TiTs.calc_soft_gates_from_db_pars(self.fitter.par[-3], self.fitter.par[-2], self.fitter.par[-1])
        #     softw_gate_all_tr = [gates_tr0 for each in self.fitter.meas.cts]
        #     self.fitter.meas.softw_gates = softw_gate_all_tr
        #     self.fitter.meas = TiTs.gate_specdata(self.fitter.meas) TODO

        self.plot()

    def set_fix(self, i, fix):
        self.fitter.set_fix(i, fix)

    def set_link(self, i, link):
        self.fitter.set_link(i, link)

    def reset(self):
        self.fitter.reset()
        self.plot()

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
