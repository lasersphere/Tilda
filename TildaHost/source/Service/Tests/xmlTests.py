"""

Created on '28.05.2015'

@author:'simkaufm'

"""

import numpy as np
import os.path

import pickle
from Driver.DataAcquisitionFpga.TimeResolvedSequencerConfig import TRSConfig
import Service.FolderAndFileHandling as filehand
import random
import Service.Formating as form
import Service.FolderAndFileHandling as handl
import lxml.etree as ET
import Service.draftScanParameters as draftDicts

isotopeData = draftDicts.draftIsotopePars

np.set_printoptions(threshold=np.nan)

# path = 'TildaHost\\source\\Scratch\\exampleTRSRawData.py'
# file = os.path.join(filehand.findTildaFolder(), path)
# trsExampleData = pickle.load(open(file, 'rb'))[1:]
# print(trsExampleData)

bodyRoot = form.xmlCreateIsotope(isotopeData)
exampleVoltArray = np.random.randint(10, size=20)
exampleScalerArray = np.random.randint(100, size=(20, 2000, 8))
exampleTimeArray = np.random.randint(0, 10, size=2000)
dicti = draftDicts.draftScanDict
data = (np.random.randint(10, size=20), np.random.randint(0, 10, size=20), np.random.randint(100, size=(20, 20, 3)))
form.xmlAddCompleteTrack(bodyRoot, dicti, data)
#
# dict['pipeInternals'].update(activeTrackNumber=1)
# form.xmlAddCompleteTrack(bodyRoot, dict, data)
# # print(ET.tostring(bodyRoot, pretty_print=True))
# handl.saveXml(bodyRoot, 'DummyData2.xml')

newdict = filehand.scanDictionaryFromXmlFile('DummyData2.xml', 0, dicti)
print(newdict)

#
#loadedText = handl.loadXml('DummyData.xml').find('tracks').find('track0').find('voltArray').text
#loadedText = loadedText.replace('\\n','').replace('[', '').replace(']', '')
#print(loadedText[2:-2])
#nump = np.fromstring(loadedText[2:-2], dtype=exampleVoltArray.dtype, sep=' ')
#print(nump, type(nump))
#loadedTextnp = np.fromstring(loadedText)
#print(loadedTextnp)

#track0 =  handl.loadXml('DummyData.xml').find('tracks').find('track0')
#voltText = track0.find('voltArray').text
#scalerText = track0.find('scalerArray').text

#scalerText = scalerText.replace('\\n', '').replace('[', '').replace(']', '').replace('  ', ' ')

#scalerText = np.fromstring(scalerText[1:-1], dtype=exampleScalerArray.dtype, sep=' ').reshape(exampleScalerArray.shape)
#print('done', type(exampleScalerArray.shape))


#print(type(form.numpyArrayFromString(scalerText, exampleScalerArray.shape)))


# rootele = handl.loadXml('DummyData.xml')
# text = form.xmlGetDataFromTrack(rootele, 0, 'scalerArray')
# nump = form.numpyArrayFromString(text, exampleScalerArray.shape)
#
# print(nump)