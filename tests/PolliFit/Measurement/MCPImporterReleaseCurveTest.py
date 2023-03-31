"""
Created on 

@author: simkaufm

Module Description:
"""

import os


from Tilda.PolliFit.Measurement.MCPImporter import MCPImporter

interesting_files = [
    # '66Ni_release_curve_SEM_Scaler0_rilis_OFF.mcp', '66Ni_release_curve_SEM_Scaler0_rilis_ON.mcp',
    # '70Ni_release_curve_SEM_Scaler0_rilis_ON_04.mcp', '70Ni_release_curve_SEM_Scaler0_rilis_OFF_05.mcp',
    '70Ni_release_curve_SEM_Scaler0_rilis_OFF_06.mcp', '70Ni_release_curve_SEM_Scaler0_rilis_ON_07.mcp',
    '70Ni_release_curve_SEM_Scaler0_rilis_ON_08.mcp'
]

file_folder = 'E:\\Workspace\\OwnCloud\\Projekte\\COLLAPS\\Nickel\\Measurement_and_Analysis_Simon\\Ni_workspace\\ReleaseCurves'

files = [os.path.normpath(os.path.join(file_folder, f)) for f in interesting_files]

for i,f in enumerate(files):
    meas = MCPImporter(f)
    print(f)
    print('scans: ', meas.nrScans)
    for x, y in zip(meas.x[0], meas.cts[0][0]):
        print('%d\t%d' % (x, y))