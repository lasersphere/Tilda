"""

Created on '11.05.2015'

@author:'simkaufm'

"""
"""
Module to convert the C Api Output to Python input
Just Run this on your header File created by the NI C Api generator and copy the Console Output to your Python File
"""
import re

class CApiAnalyser():
    def __init__(self):
        pass

    def findLastUnderScore(self, string):
        for i in re.finditer('_',string):
            index = i.end()
        return index

    def analyser(self, liste, val):
        for i,j in enumerate(liste):
                if 'Bool_' in j:
                    print(j[self.findLastUnderScore(j):j.index('=')-1] + ' = {\'ref\':' + str(j[j.index('=')+2:-2]) + ', \'val\':ctypes.c_bool(), \'ctr\':' + str(val) + '}')
                elif 'U8_' in j:
                    print(j[self.findLastUnderScore(j):j.index('=')-1] + ' = {\'ref\':' + str(j[j.index('=')+2:-2]) + ', \'val\':ctypes.c_ubyte(), \'ctr\':' + str(val) + '}')
                elif 'I8_' in j:
                    print(j[self.findLastUnderScore(j):j.index('=')-1] + ' = {\'ref\':' + str(j[j.index('=')+2:-2]) + ', \'val\':ctypes.c_byte(), \'ctr\':' + str(val) + '}')
                elif 'U16_' in j:
                    print(j[self.findLastUnderScore(j):j.index('=')-1] + ' = {\'ref\':' + str(j[j.index('=')+2:-2]) + ', \'val\':ctypes.c_uint(), \'ctr\':' + str(val) + '}')
                elif 'I16_' in j:
                    print(j[self.findLastUnderScore(j):j.index('=')-1] + ' = {\'ref\':' + str(j[j.index('=')+2:-2]) + ', \'val\':ctypes.c_int(), \'ctr\':' + str(val) + '}')
                elif 'U32_' in j:
                    print(j[self.findLastUnderScore(j):j.index('=')-1] + ' = {\'ref\':' + str(j[j.index('=')+2:-2]) + ', \'val\':ctypes.c_ulong(), \'ctr\':' + str(val) + '}')
                elif 'I32_' in j:
                    print(j[self.findLastUnderScore(j):j.index('=')-1] + ' = {\'ref\':' + str(j[j.index('=')+2:-2]) + ', \'val\':ctypes.c_long(), \'ctr\':' + str(val) + '}')
    

    def CApiFileHandler(self, path):
        with open(path,'r') as myfile:
            inhalt = myfile.readlines()
            signature = [s for s in inhalt if "Signature" in s]
            signature = signature[0][signature[0].index('=')+3:-3]
            print('\'\'\'Bitfile Signature:\'\'\'')
            print('bitfileSignature = ' + '\'' + signature + '\'')
            indicators = [s for s in inhalt if "Indicator" in s]
            controls = [s for s in inhalt if "Control" in s]
            print('\'\'\'Indicators:\'\'\'')
            self.analyser(indicators, False)
            print('\'\'\'Controls:\'\'\'')
            self.analyser(controls, True)




outputplease = CApiAnalyser()
outputplease.CApiFileHandler('D:\\Workspace\\PyCharm\\Tilda\\TildaTarget\\bin\\TimeResolvedSequencer\\NiFpga_TRS.h')
