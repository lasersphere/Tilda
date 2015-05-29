"""

Created on '28.05.2015'

@author:'simkaufm'

"""

import numpy as np
import random
import Service.Formating as form
import Service.FolderAndFileHandling as handl
import lxml.etree as ET

isotopeData = {'version': '0.1', 'type': 'trs', 'isotope': 'simontium_27',
               'nOfTracks': '1', 'colDirTrue': 'False', 'accVolt': '999.8',
               'laserFreq': '12568.73'}

np.set_printoptions(threshold=np.nan)

bodyRoot = form.xmlCreateIsotope(isotopeData)
exampleVoltArray = np.random.randint(10, size=20)
exampleScalerArray = np.random.randint(100, size=(20, 2000, 8))
exampleTimeArray = np.random.randint(0, 10, size=2000)

# form.xmlAddDataToTrack(bodyRoot, 0, 'scalerArray', np.array_str(exampleScalerArray))
# form.xmlAddDataToTrack(bodyRoot, 0, 'voltArray', np.array_str(exampleVoltArray))
# form.xmlAddDataToTrack(bodyRoot, 0, 'timeArray', np.array_str(exampleTimeArray))
# handl.saveXml(bodyRoot, 'DummyData.xml')

loadedText = handl.loadXml('DummyData.xml').find('tracks').find('track0').find('scalerArray').text
loadedText = loadedText.replace('\\n','')
# loadedTextnp = np.fromstring(loadedText, dtype=)
# print(loadedTextnp)
