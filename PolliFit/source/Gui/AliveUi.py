'''
Created on 29.07.2016

@author: K. Koenig
'''

import ast
import copy
import sqlite3
from datetime import datetime

import numpy as np
from PyQt5 import QtWidgets, QtCore

import AliveTools
import Physics
import functools
import TildaTools as TiTs
import Analyzer
import MPLPlotter as plot
from Gui.Ui_Alive import Ui_Alive


class AliveUi(QtWidgets.QWidget, Ui_Alive):
    def __init__(self):
        super(AliveUi, self).__init__()
        self.setupUi(self)

        self.runSelect.currentTextChanged.connect(self.loadIsos)
        self.isoSelect.currentIndexChanged.connect(self.loadFiles)
        self.isoSelect_2.currentIndexChanged.connect(self.loadFiles)

        self.fileList.itemChanged.connect(self.recalc)
        self.fileList_2.itemChanged.connect(self.recalc)

        self.pB_compareAuto.clicked.connect(self.compareAuto)
        self.pB_compareIndividual.clicked.connect(self.compareIndividual)

        self.dbpath = None

        self.show()

    def compareAuto(self):
        self.recalc()

        plot_data=[]
        Error_data=[]
        ref_xData=[]
        ref_timeData=[]
        xData=[]
        timeData=[]
        offsetVolt=[]


        all_Files=AliveTools.get_listAll_from_db(self.dbpath)[0]
        HV_files=AliveTools.get_listAll_from_db(self.dbpath)[1]

        for file in HV_files:
            hv = AliveTools.calculateVoltage(self.dbpath, file['filename'], self.run)



            ref_Files=AliveTools.find_ref_files(file,all_Files)
            if ref_Files == []:
                print('no reference files in database')

            errorData=[]
            Delta=[]
            ref_x_data=[]
            ref_time_data=[]
            x_data=[]
            time_data=[]
            offset_volt=[]

            for element in ref_Files:
                ref = AliveTools.calculateVoltage(self.dbpath, element['filename'], self.run)
                ref_x_data = ref_x_data +[AliveTools.get_nameNumber_and_time(element['filename'])[0]]
                ref_time_data = ref_time_data +[AliveTools.get_nameNumber_and_time(element['filename'])[1]]
                x_data = x_data +[AliveTools.get_nameNumber_and_time(file['filename'])[0]]
                time_data = time_data +[AliveTools.get_nameNumber_and_time(file['filename'])[1]]
                offset_volt=offset_volt+[AliveTools.get_offsetVolt_from_db(self.dbpath, file['filename'])]


                Delta = Delta +[(hv[0] - ref[0])]
                Delta_max = (hv[1] - ref[2])-(hv[0] - ref[0])
                Delta_min =(hv[0] - ref[0])-(hv[2] - ref[1])
                errorData=errorData+[[Delta_min,Delta_max]]

            plot_data=plot_data+[Delta]
            Error_data=Error_data + [errorData]
            ref_xData=ref_xData+[ref_x_data]
            ref_timeData=ref_timeData+[ref_time_data]
            xData=xData+[x_data]
            timeData=timeData+[time_data]
            offsetVolt=offsetVolt+[offset_volt]

        self.plotdata=AliveTools.changeListFormat(plot_data)
        self.Error=AliveTools.changeListFormat(Error_data)
        self.ref_x_data=AliveTools.changeListFormat(ref_xData)
        self.x_data=AliveTools.changeListFormat(xData)
        self.ref_time_data=AliveTools.changeListFormat(ref_timeData)
        self.time_data=AliveTools.changeListFormat(timeData)
        self.offset_volt=AliveTools.changeListFormat(offsetVolt)

        self.saving()





    def compareIndividual(self):
        self.recalc()
        self.plotdata = []

        if len(self.chosenFiles) > 0:
            if len(self.chosenFiles_2) > 0:

                self.numberOfRef = len(self.chosenFiles)
                self.numberOfHV = len(self.chosenFiles_2)

                refVolt = []
                hvVolt = []
                x_data=[]
                time_data=[]
                ref_x_data=[]
                ref_time_data=[]
                offsetVolt=[]
                self.x_data =[]
                self.time_data=[]
                self.ref_x_data=[]
                self.ref_time_data=[]
                self.Error=[]
                self.offset_volt=[]


                for file in self.chosenFiles:
                    ref = AliveTools.calculateVoltage(self.dbpath, file, self.run)
                    refVolt = refVolt + [ref]
                    ref_x_data = ref_x_data +[AliveTools.get_nameNumber_and_time(file)[0]]
                    ref_time_data = ref_time_data +[AliveTools.get_nameNumber_and_time(file)[1]]

                for file in self.chosenFiles_2:
                    hvVolt = hvVolt + [AliveTools.calculateVoltage(self.dbpath, file, self.run)]
                    x_data = x_data +[AliveTools.get_nameNumber_and_time(file)[0]]
                    time_data = time_data +[AliveTools.get_nameNumber_and_time(file)[1]]
                    offsetVolt=offsetVolt+[AliveTools.get_offsetVolt_from_db(self.dbpath, file)]


                print(hvVolt)
                for a in range(len(refVolt)):
                    data = []
                    Error=[]
                    ref_xData=[]
                    ref_timeData=[]

                    for b in range(len(hvVolt)):
                        Delta = (hvVolt[b][0] - refVolt[a][0])
                        Delta_max = (hvVolt[b][1] - refVolt[a][2])
                        Delta_min = (hvVolt[b][2] - refVolt[a][1])

                        #Delta = (hv_measurement[0] - ref_measurement + hv_measurement[1])/hv_measurement[1]*1000000
                        data = data + [Delta]
                        Error=Error+ [[Delta-Delta_min,Delta_max-Delta]]
                        ref_xData=ref_xData+[ref_x_data[a]]
                        ref_timeData=ref_timeData+[ref_time_data[a]]

                    self.plotdata = self.plotdata + [data]
                    self.Error=self.Error+[Error]
                    self.offset_volt=self.offset_volt+[offsetVolt]
                    self.x_data=self.x_data+[x_data]
                    self.time_data=self.time_data+[time_data]
                    self.ref_x_data=self.ref_x_data+[ref_xData]
                    self.ref_time_data=self.ref_time_data+[ref_timeData]


                self.saving()
            else:
                print('select HV measurement')
        else:
            print('select reference measurement')



    def conSig(self, dbSig):
        dbSig.connect(self.dbChange)

    def loadIsos(self, run):
        self.isoSelect.clear()
        self.isoSelect_2.clear()

        it = TiTs.select_from_db(self.dbpath, 'DISTINCT iso', 'FitRes', [['run'], [run]], 'ORDER BY iso',
                                 caller_name=__name__)
        if it:
            for i, e in enumerate(it):
                self.isoSelect.insertItem(i, e[0])
                self.isoSelect_2.insertItem(i, e[0])


    def loadRuns(self):
        self.runSelect.clear()
        runit = TiTs.select_from_db(self.dbpath, 'run', 'Runs', caller_name=__name__)
        if runit:
            for i, r in enumerate(runit):
                self.runSelect.insertItem(i, r[0])

    def loadFiles(self):
        num = ['', '_2']
        for i in num:
            getattr(self, 'fileList'+str(i)).clear()
        try:

            self.iso = getattr(self, 'isoSelect'+str(num[0])).currentText()
            self.iso_2 = getattr(self, 'isoSelect'+str(num[1])).currentText()
            self.run = self.runSelect.currentText()
            self.par = 'center'

            #self.files = Analyzer.getFiles(self.iso, self.run, self.dbpath, files)

            self.vals, self.errs, self.dates, self.files = Analyzer.extract(getattr(self,'iso'+str(num[0])),
                                                                            self.par, self.run, self.dbpath, prin=False)
            self.vals_2, self.errs_2, self.dates_2, self.files_2 = Analyzer.extract(getattr(self,'iso'+str(num[1])),
                                                                            self.par, self.run, self.dbpath, prin=False)
            for j in num:
                r = TiTs.select_from_db(self.dbpath, 'config, statErrForm, systErrForm', 'Combined',
                                        [['iso', 'parname', 'run'], [getattr(self, 'iso'+str(j)), self.par, self.run]],
                                        caller_name=__name__)
                select = [True] * len(getattr(self, 'files'+str(j)))
                if r:
                    if j == '':
                        self.statErrForm = r[0][1]
                        self.systErrForm = r[0][2]
                    else:
                        self.statErrForm_2 = r[0][1]
                        self.systErrForm_2 = r[0][2]
                    cfg = ast.literal_eval(r[0][0])
                    for i, f in enumerate(self.files):
                        if cfg:
                            select[i] = True
                        elif f not in cfg:
                            select[i] = False
                else:
                    if j == '':
                        self.statErrForm = 0
                        self.systErrForm = 0
                    else:
                        self.statErrForm_2 = 0
                        self.systErrForm_2 = 0
                getattr(self,'fileList'+str(j)).blockSignals(True)

                for f, s in zip(getattr(self, 'files'+str(j)), select):
                    w = QtWidgets.QListWidgetItem(f)
                    w.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                    if s:
                        w.setCheckState(QtCore.Qt.Checked)
                    else:
                        w.setCheckState(QtCore.Qt.Unchecked)
                    getattr(self, 'fileList'+str(j)).addItem(w)

                getattr(self, 'fileList'+str(j)).blockSignals(False)

            self.recalc()

        except Exception as e:
            print('error: while loading files in AliveUi: %s' % e)

    def recalc(self):
        select = []
        self.chosenFiles = []
        self.chosenVals = []
        self.chosenErrs = []
        self.chosenDates = []
        self.val = 0
        self.err = 0
        self.redChi = 0
        self.systeErr = 0

        for index in range(self.fileList.count()):
            if self.fileList.item(index).checkState() != QtCore.Qt.Checked:
                select.append(index)
        if len(self.vals) > 0 and len(self.errs) > 0:
            self.chosenVals = np.delete(copy.deepcopy(self.vals), select)
            self.chosenErrs = np.delete(copy.deepcopy(self.errs), select)
            self.chosenDates = np.delete(copy.deepcopy(self.dates), select)
            self.chosenFiles = np.delete(copy.deepcopy(self.files), select)

        select_2 = []
        self.chosenFiles_2 = []
        self.chosenVals_2 = []
        self.chosenErrs_2 = []
        self.chosenDates_2 = []
        self.val_2 = 0
        self.err_2 = 0
        self.redChi_2 = 0
        self.systeErr_2 = 0

        for index in range(self.fileList_2.count()):
            if self.fileList_2.item(index).checkState() != QtCore.Qt.Checked:
                select_2.append(index)
        if len(self.vals_2) > 0 and len(self.errs_2) > 0:
            self.chosenVals_2 = np.delete(copy.deepcopy(self.vals_2), select_2)
            self.chosenErrs_2 = np.delete(copy.deepcopy(self.errs_2), select_2)
            self.chosenDates_2 = np.delete(copy.deepcopy(self.dates_2), select_2)
            self.chosenFiles_2 = np.delete(copy.deepcopy(self.files_2), select_2)

    def dbChange(self, dbpath):
        self.dbpath = dbpath
        self.loadRuns()  # might still cause some problems

    def saving(self):
        plot.clear()
        plot.AlivePlot(self.x_data[0], self.plotdata, self.Error[0])
        plot.show(True)
        print(self.x_data)
        print(self.plotdata)
        print(self.Error)
        print(self.ref_x_data)
        print(self.ref_time_data)
        print(self.time_data)
        print(self.offset_volt)

        #Zeitstempel erzeugen
        t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        ind = self.dbpath.rfind('/')
        ind_2 = self.dbpath.rfind('.')
        savePath = self.dbpath[:ind]
        name = self.dbpath[ind+1:ind_2]


        with open(savePath + '/Results_'+name+'_'+t+'.txt', 'w+') as out_file:
            out_string ='number ; time ; Voltage ; neg. Error ; pos. Error; HV div. volt ; Ref. number ; Ref. time \n'
            for a in range(len(self.plotdata)):
                for b in range(len(self.plotdata[a])):

                    out_string += str(self.x_data[a][b])
                    out_string += ' ; '
                    out_string += self.time_data[a][b]
                    out_string += ' ; '
                    out_string += str(self.plotdata[a][b])
                    out_string += ' ; '
                    out_string +=str(-self.Error[a][b][0])
                    out_string += ' ; '
                    out_string +=str(self.Error[a][b][1])
                    out_string += ' ; '
                    out_string +=str(self.offset_volt[a][b])
                    out_string += ' ; '
                    out_string +=str(self.ref_x_data[a][b])
                    out_string += ' ; '
                    out_string +=str(self.ref_time_data[a][b])
                    out_string += '\n'

                out_string += '\n'
            out_file.write(out_string)


        #with open(savePath + '/Results_'+name+'_'+t+'.txt', 'w+') as out_file:
        #    out_string =''
        #    for a in range(len(self.ref_x_data)):
        #        out_string += 'reference measurement number = '+str(self.ref_x_data[a])
        #        out_string += ' reference measurement time = '+self.ref_time_data[a] + '\n'
        #        out_string +='number ; time ; Voltage ; neg. Error ; pos. Error \n'
        #        for i in range(len(self.x_data)):
        #            out_string += str(self.x_data[i])
        #            out_string += ' ; '
        #            out_string += self.time_data[i]
        #            out_string += ' ; '
        #            out_string += str(self.plotdata[a][i])
        #            out_string += ' ; '
        #            out_string +=str(-self.Error[a][i][0])
        #            out_string += ' ; '
        #            out_string +=str(self.Error[a][i][1])
        #            out_string += '\n'
        #        out_string += '\n'
        #    out_file.write(out_string)
