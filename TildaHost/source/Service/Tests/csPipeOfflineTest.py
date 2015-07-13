__author__ = 'Simon-K'


import Service.FolderAndFileHandling as FileHandle
import Service.AnalysisAndDataHandling.tildaPipeline as TildaPipe
import Service.Formating as Form


import os

# TildaPipe.CsPipe()

path = 'C:\\Workspace\\TildaTestData\\raw'
prun0 = os.path.join(path, 'run0')
filesrun0 = [file for file in os.listdir(prun0) if file.endswith('.raw')]
xmlFile = os.path.join(prun0, [file for file in os.listdir(prun0) if file.endswith('.xml')][0])
scandict = FileHandle.scanDictionaryFromXmlFile(xmlFile, 0, {})

for key, val in scandict.items():
    scandict[str(key)] = Form.convertStrValuesInDictToFloat(scandict[str(key)])
print(scandict)

# for file in filesrun0:
#     print(FileHandle.loadPickle(os.path.join(prun0, file)))