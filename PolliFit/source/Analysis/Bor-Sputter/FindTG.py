import sqlite3
import os
from openpyxl import Workbook, load_workbook


#### Find timeGates with best signal to noise ratio ####

class FindTG:

    def __init__(self):
        #workingdir = 'C:\\Users\\Laura Renth\\Desktop\\Daten\\Promotion\\Bor\\Sputter source\\2021-03-Data' #working dir IKP
        self.workingdir = 'C:\\Users\\Laura Renth\\ownCloud\\User\\Laura\\2021-03-Data'  #working dir IKP Owncloud
        self.db = os.path.join(self.workingdir, 'B-_Auswertung.sqlite')
        # isotope and run to investigate
        self.isotope = '11B'
        self.run = 'sym2'
        self.startTGwidth = 2
        self.startTGcenter = 1
        self.prepDB(self.startTGcenter, self.startTGwidth)

    def writeToFile(self):
        ### write timeGates and rChi to excelfile

        # open Workbook to store result in
        try:
            wb = load_workbook(os.path.join(self.workingdir, 'FindTG.xlsx'))
            ws = wb.active
        except:
            wb = Workbook()
            ws = wb.active
            ws.title = 'rChi'
            ws['A1'] = 'TGcenter'
            ws['B1'] = 'TGwdith'
            ws['C1'] = 'rChi'
            wb.save(os.path.join(self.workingdir, 'FindTG.xlsx'))

        # get current TG from db
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''SELECT softwGateWidth FROM Runs WHERE run = ?''', (self.run,))
        width = cur.fetchall()[0][0]
        cur.execute('''SELECT midTof FROM Isotopes WHERE iso = ?''', (self.isotope,))
        midTOF = cur.fetchall()[0][0]

        # find results in database
        cur.execute('''SELECT rChi FROM FitRes WHERE iso = ? AND run = ?''', (self.isotope, self.run,))
        paras = cur.fetchall()
        print(paras)
        con.close()

        # calculate mean rChi
        sum = 0
        for x in paras:
            sum+=x[0]

        mean = sum/len(paras)
        print(mean)

        # write to Workbook
        max=row=ws.max_row
        ws.cell(row=max+1, column=1, value=midTOF)
        ws.cell(row=max+1, column=2, value=width)
        ws.cell(row=max+1, column=3, value=mean)

        wb.save(os.path.join(self.workingdir, 'FindTG.xlsx'))

    def prepDB(self, TGcenter, TGwidth):
        # write TGcenter and TGwidth to Isotope
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''UPDATE Isotopes SET midTof = ? WHERE iso = ? ''', (TGcenter, self.isotope,))
        cur.execute('''UPDATE Runs SET softwGateWidth = ? WHERE run = ?''', (TGwidth, self.run,))
        con.commit()
        con.close()


search = FindTG()
search.writeToFile()