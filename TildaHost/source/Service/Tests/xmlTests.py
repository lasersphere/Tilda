"""

Created on '28.05.2015'

@author:'simkaufm'

"""

import numpy as np

import Service.Formating as form
import Service.FolderAndFileHandling as handl
import Service.draftScanParameters as draftDicts
import Measurement.XMLImporter as pollixmlimp


# """creation of an xml file:"""
#
# scandict = draftDicts.draftScanDict
# scandict['isotopeData']['nOfTracks'] = 3
#
# def data():
#     return np.random.randint(100, size=(20, 2))
#
# np.set_printoptions(threshold=np.nan)
#
# bodyRoot = form.xmlCreateIsotope(scandict['isotopeData'])
# # exampleVoltArray = np.random.randint(10, size=20)
# exampleScalerArray = np.random.randint(100, size=(20, 2000, 8))
# # exampleTimeArray = np.random.randint(0, 10, size=2000)
#
#
# form.xmlAddCompleteTrack(bodyRoot, scandict, data())
#
# scandict['pipeInternals'].update(activeTrackNumber=1)
# form.xmlAddCompleteTrack(bodyRoot, scandict, data())
# scandict['pipeInternals'].update(activeTrackNumber=2)
# form.xmlAddCompleteTrack(bodyRoot, scandict, data())
# # print(ET.tostring(bodyRoot, pretty_print=True))
# handl.saveXml(bodyRoot, 'DummyData2.xml', False)

"""loading of an xml File"""
# loadedD = handl.getAllTracksOfXmlFileInOneDict('DummyData2.xml')
# print(loadedD)


# import
xml = pollixmlimp.XMLImporter('D:/Workspace/PyCharm/Tilda/TildaHost/source/Service/Tests/DummyData2.xml')
print(xml.x, xml.cts, xml.err)