"""

Created on '07.08.2015'

@author:'simkaufm'

"""

import matplotlib.pyplot as plt

from Measurement.XMLImporter import XMLImporter
import MPLPlotter

path = 'D:/Workspace/PyCharm/Tilda/PolliFit/test/Project/Data/testTilda.xml'
file = XMLImporter(path)
# MPLPlotter.plot(file.x[0], file.cts[0])
# MPLPlotter.show()
plt.plot(file.x[0], file.cts[0])
plt.show()