"""
Created on 06.06.2014

@author: hammen
"""

import os
from lxml.etree import XMLParser, parse
import sqlite3

from PyQt5 import QtWidgets

from Tilda.PolliFit import Tools
from Tilda.PolliFit.Gui.Ui_Crawler import Ui_Crawler
from Tilda.PolliFit.Measurement.XMLImporter import METADATA_SYSTEMS, METADATA_CHANNELS


MAX_CHANNEL_FILES = 20


class CrawlerUi(QtWidgets.QWidget, Ui_Crawler):

    def __init__(self):
        super(CrawlerUi, self).__init__()
        self.setupUi(self)
        
        self.bcrawl.clicked.connect(self.crawl)
        self.b_load_channels.clicked.connect(lambda: self.load_channels_from_folder(force=True))
        self.pushButton_save_sql.clicked.connect(self.save_sql_cmd)
        self.pushButton_load_sql.clicked.connect(self.load_sql_cmd)

        self.dbpath = None
        
        self.show()
        
    def conSig(self, dbSig):
        dbSig.connect(self.dbChange)

    def get_channels_from_xml(self, file):
        channels = set()
        p = XMLParser(huge_tree=True)
        root = parse(file, parser=p)
        try:
            header = root.find('tracks').find('track0').find('header')
        except AttributeError:
            return channels
        pre_dur_post = ['preScan', 'duringScan', 'postScan']
        for msys in METADATA_SYSTEMS:
            _msys = header.find(msys)
            if _msys is None:
                continue
            for pdp in pre_dur_post:
                _pdp = _msys.find(pdp)
                if _pdp is None:
                    continue
                if msys == 'sql':
                    channels = channels.union(set(ch.tag for ch in _pdp))
                elif msys == 'measureVoltPars':
                    dev = _pdp.find('dmms')
                    if dev is None:
                        continue
                    channels = channels.union(set(f'{dev.tag}.{ch.tag}' for ch in dev))
                else:
                    for dev in _pdp:
                        channels = channels.union(set(f'{dev.tag}.{ch.tag}' for ch in dev))
        return channels

    def load_channels_from_folder(self, force=False):
        files = list(os.path.join(_files[0], f) for _files in os.walk(os.path.dirname(self.dbpath))
                     for f in _files[2] if os.path.splitext(f)[1] == '.xml')
        if len(files) > MAX_CHANNEL_FILES and not force:
            self.l_load_channels.setText(f'Number of xml files exceeded {MAX_CHANNEL_FILES}.'
                                         f' Please load channels manually.')
            files = []
        if files:
            self.l_load_channels.setText('')
        channels = set()
        for f in files:
            channels = channels.union(self.get_channels_from_xml(f))
        channels = sorted(channels)
        for c_box in [self.c_accVolt, self.c_offset, self.c_laserFreq]:
            c_box.clear()
            c_box.addItems(['Automatic', 'None'])
            c_box.addItems(channels)
            c_box.setCurrentIndex(0)

    def get_meta_data_channels(self):
        meta_data_channels = {mtype: eval(f'self.c_{mtype}', {'self': self}).currentText().replace('Automatic', '')
                              for mtype in METADATA_CHANNELS.keys()}
        meta_data_channels = {mtype: eval(dev_ch) if dev_ch == 'None' else dev_ch
                              for mtype, dev_ch in meta_data_channels.items()}
        return meta_data_channels
    
    def crawl(self):
        t = self.path.text()
        path = t if t != '' else '.'
        Tools.crawl(self.dbpath, path=path, rec=self.recursive.isChecked(),
                    add_miss_cols=True, meta_data_channels=self.get_meta_data_channels())
        if self.lineEdit_sql_cmd.text():
            con = sqlite3.connect(self.dbpath)
            cur = con.cursor()
            cur.execute(''' %s ''' % self.lineEdit_sql_cmd.text())
            con.commit()
            con.close()

    def dbChange(self, dbpath):
        self.dbpath = dbpath
        self.load_channels_from_folder(force=False)
        sql_file = os.path.join(os.path.split(self.dbpath)[0], 'sql_cmd.txt')
        if os.path.isfile(sql_file):
            self.load_sql_cmd(False, path=sql_file)

    def save_sql_cmd(self):
        start_path = os.path.join(os.path.split(self.dbpath)[0], 'sql_cmd.txt')
        path, ending = QtWidgets.QFileDialog.getSaveFileName(
            QtWidgets.QFileDialog(), 'save sql cmd as txt', start_path,
            '*.txt')
        if path:
            if os.path.isfile(path):
                full_path = path
            else:
                full_path = path + ending[1:]
            text_file = open(full_path, 'w')
            text_file.write(self.lineEdit_sql_cmd.text())
            text_file.close()
            print('saved sql cmd to: ', full_path)

    def load_sql_cmd(self, buttonres, path=''):
        if path == '':
            path, ending = QtWidgets.QFileDialog.getOpenFileName(
                QtWidgets.QFileDialog(), 'load sql cmd from txt', os.path.split(self.dbpath)[0],
                '*.txt')
        if path:
            text_file = open(path, 'r')
            self.lineEdit_sql_cmd.setText(text_file.readline())
            text_file.close()
