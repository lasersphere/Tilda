import os

from Tilda.PolliFit import Tools
from Tilda.PolliFit.Measurement import XMLImporter

path = 'C:\Sn_data\Sn_beamtime_Tilda_active_data\sums'
db = 'C:\Sn_data\Sn_beamtime_Tilda_active_data\Sn_beamtime_Tilda_active_data.sqlite'

sn124 = Tools.fileList(db, '124_Sn')
print(sn124)
sn124_full_path = [os.path.join(path, sn124[i]) for i, j in enumerate(sn124)]
XMLImporter(sn124_full_path[0])
