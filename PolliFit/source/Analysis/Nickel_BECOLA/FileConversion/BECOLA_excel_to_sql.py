'''
Created 30.09.2018

@author: fsommer
'''
from openpyxl import Workbook, load_workbook  # Needed to read/write data to xlsx
from string import ascii_uppercase
import os

import csv, os, sqlite3, re

import numpy as np

from pylab import plot, meshgrid, cm, imshow, contour, clabel, colorbar, axis, title, show


class BECOLAExelcuter():
    '''
    This object reads a BECOLA excel file that contains run information and extracts the crucial information

    Data structure for the excel sheet is (A-M) (Sheet: '1361'):
    DATE (or Time) >> Run # >> Isotope >> Laser Freq. >> Laser Background (cps) >> Notes >> Scans >> Fill Time (ms) >> Release Time (ms) >>
     Starting Voltage (V) >> V Steps >> V Step Size (V) >> Seqs >> ...

    Identifier for an interesting row should be the run #.
    '''

    def __init__(self, path, run_no):
        '''Read the file'''

        # print("BECOLAExelcuter is reading excel files in folder")
        super(BECOLAExelcuter, self).__init__()

        self.working_dir = path

        self.run_no = run_no

        # print('Working directory is %s' %self.working_dir)
        # dict containing all workbooks loaded from the xlsx files. Format is:
        # {'filename': Workbook}
        self.wb_dict = {}

        # dict containing all names/identifiers and the respective points scored in each exercise
        # {'nameA':{'ex1':int, 'ex2':int,...},'nameB':{},...}
        self.result_dict = {}
        # also include a list of the identifiers for sorted extraction
        self.ident_list = []

        # load all xml files from the folder TODO: Ask user to confirm the folder?
        # print('Loading files for search...')
        self.load_xlsx_in_dir(self.working_dir)

    def run(self):
        # search all workbooks for a specific run number
        retlist = []
        for workbooks in self.wb_dict:
            colrow_tup = self.search_wb_for_keyword(self.run_no, self.wb_dict[workbooks])
            if colrow_tup:  # run was found in excel sheet
                retlist = self.extract_row_as_list(colrow_tup[1], self.wb_dict[workbooks].worksheets[0],
                                                   end_by_empty=False)
                # print(retlist)
        return retlist

    def load_single_xlsx_file(self, filepath):
        # print('Loading .xlsx file from %s' % str(filepath))
        new_wb = load_workbook(filepath, data_only=True)  # data only removes formulas
        return new_wb

    def load_xlsx_in_dir(self, directory):
        # There is supposed to be only ONE .xlsx file in this folder!!
        for file in os.listdir(directory):  # crawl through the folder
            if os.path.isfile(file) and '.xlsx' in file:  # Pick only .xlsx files
                if file not in self.wb_dict:
                    # import xlsx-file to workbook
                    new_wb = self.load_single_xlsx_file(os.path.join(directory, file))
                # add imported workbook to the workbook dict
                if new_wb is not None:  # skip previously created files
                    self.wb_dict[file] = new_wb
                return ()

    def convert_to_xlsx(self, filepath):
        try:
            import pyexcel as pe  # used to convert other formats to xlsx
            if '.ods' in filepath or '.xls' in filepath or '.csv' in filepath:
                # convertable
                print('Converting file %s to .xlsx-file format.' % filepath)
                new_filename = filepath.split('.')[-2] + '.xlsx'
                data = pe.get_array(file_name=filepath)
                pe.save_as(array=data, dest_file_name=new_filename)
                return new_filename
            else:
                return None
        except Exception as e:
            print('Could not convert file. Error: ', e)
            return None

    def search_wb_for_keyword(self, keyword, workbook):
        """
        Will search for a keyword in the workbook and return the column and row this keyword appears
        :param keyword: Should be a special case word that is known to be the header of a column (or row)
        :param workbook: The workbook to be seached
        :return: tuple containing column and row (str, int) of the keyword
        """
        ret = ''
        for sheet in workbook:
            if sheet.title == '1361':
                for rows in sheet.rows:
                    for cell in rows:
                        if cell.value == keyword or cell.value == int(keyword):
                            ret = (cell.column, cell.row)
        if ret is '':
            # print('Keyword %s not found. Please provide a valid keyword!' % keyword)
            return False
        else:
            return ret

    def extract_row_as_list(self, row, worksheet, start_par='A', end_par='M', end_by_empty=True):
        """
        Example: extract_column_as_dict('A', new_wb.worksheets[0], end_par=3)
        :param row: int, Row as int (e.g. '2')
        :param worksheet: ws, Needs a worksheet, since there's a column in each worksheet
        :param start_par: str, optional, start to extract values in this column
        :param end_par: str, optional, stop extracting values in this column.
        :param end_by_empty: bool, if True will extract values from the list until it finds an empty cell
        :return: dict of values of that column and their position {val: (),...}
        """
        start_cell_str = start_par + str(row)
        end_cell_str = end_par + str(row)
        selection = worksheet[start_cell_str:end_cell_str]  # gives a tuple of tuples ((cell1,),(cell2,),...)
        ret = []  # prepares a dict for the return values
        end_by_empty_condition = False
        for cells in selection[0]:
            val = cells.value
            if val is None and end_by_empty:
                end_by_empty_condition = True
                break
            else:
                ret.append(val)
        # if end_by_empty_condition:
        #     print('extraction finished due to empty cell')
        return ret


if __name__ == '__main__':
    path = os.getcwd()  # path to excel files that are to be searched
    keyword = '6301'  # keyword that is to be searched in excel files. Row that contains it will be extracted.

    imper = BECOLAExelcuter(path, keyword)

    print(imper.run())
