"""

Created on '22.06.2015'

@author:'simkaufm'

"""
import pickle
import Service.FolderAndFileHandling as fileh

path = 'D:\\Workspace\\Testdata'
# print(path)
file = 'D:\Workspace\Testdata\\raw\\20150622_185322_trs_simontium_27_track0_0.raw'
print(fileh.nameFile(path,'raw', '20150622_183238_trs_', 'simontium_27'))
data = pickle.load(open(file, 'rb'))
print(data, type(data), data[0])

print(fileh.loadPickle(file))