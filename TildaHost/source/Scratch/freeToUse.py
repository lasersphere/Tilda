"""

Created on '22.06.2015'

@author:'simkaufm'

"""
# import pickle
# import Service.FolderAndFileHandling as fileh
# import os
# import Service.draftScanParameters as draftScan
# import Service.Formating as form

# path = 'D:\\Workspace\\Testdata'
# # print(path)
# file = 'D:\Workspace\Testdata\\raw\\20150622_185322_trs_simontium_27_track0_0.raw'
# # print(fileh.nameFile(path,'raw', '20150622_183238_trs_', 'simontium_27'))
# # data = pickle.load(open(file, 'rb'))
# # print(data, type(data), data[0])
#
# # for nfile in os.listdir(os.path.split(file)[0]):
# #     print(nfile)
# #     if nfile.endswith('.raw'):
# #         print(nfile)
# #         print(fileh.loadPickle(os.path.join(path, 'raw', nfile)))
#
# # print(fileh.createXmlFileOneIsotope(draftScan.draftScanDict))
#
# ele = fileh.loadXml('D:\Workspace\Testdata\sums\\20150623_114059_trs_sum_simontium_27.xml')
#
#
# dicti = draftScan.draftTrackPars
# form.xmlAddCompleteTrack(ele, dicti, 1)

# print(512 >> 2)
# print(8192 >> 2)

class printer():
    def __init__(self):
        print('hi There')

    def druck(self):
        return 'Hello World!'