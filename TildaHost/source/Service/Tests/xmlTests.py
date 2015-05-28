"""

Created on '28.05.2015'

@author:'simkaufm'

"""

import Service.Formating as form
import lxml.etree as ET

isotopeData = {'version': '0.1', 'type': 'trs', 'isotope': 'simontium_27', 'nOfTracks': '1',
               'colDirTrue': 'False', 'accVolt': '999.8', 'laserFreq': '12568.73',
               'saveToFile': 'DummyData.xml'}
# print(ET.tostring(form.xmlFormat(isotopeData), pretty_print=True))
# print(ET.SubElement(form.xmlFormat(isotopeData), 'track0'))

tree = ET.parse(isotopeData['saveToFile'])
print(tree)
