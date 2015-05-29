"""

Created on '28.05.2015'

@author:'simkaufm'

"""

import numpy as np
import Service.Formating as form
import Service.FolderAndFileHandling as handl
import lxml.etree as ET

isotopeData = {'version': '0.1', 'type': 'trs', 'isotope': 'simontium_27', 'nOfTracks': '1',
               'colDirTrue': 'False', 'accVolt': '999.8', 'laserFreq': '12568.73',
               'saveToFile': 'DummyData.xml'}
# print(ET.tostring(form.xmlFormat(isotopeData), pretty_print=True))
# print(ET.SubElement(form.xmlFormat(isotopeData), 'track0'))

# tree = ET.parse(isotopeData['saveToFile'])
# print(tree)

bodyRoot = form.xmlFormatBody(isotopeData)
exampleVoltArray = np.random.randint(10, size=20)
exampleScalerArray = np.random.randint(100, size=(20, 2000, 8))
exampleTimeArray = np.random.randint(0, 10, size=2000)
form.xmlAddDataToTrack(bodyRoot, 0, 'scalerArray', exampleScalerArray)
form.xmlAddDataToTrack(bodyRoot, 2, 'voltArray', exampleVoltArray)
form.xmlAddDataToTrack(bodyRoot, 3, 'timeArray', exampleTimeArray)
handl.saveXml(bodyRoot, isotopeData['saveToFile'])