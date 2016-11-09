"""
Created on 

@author: simkaufm

Module Description:
"""

import Measurement.MeasLoad as Meas

# mcp_file = os.path.normpath('E:\\Workspace\\AddedTestFiles\\60Ni_no_protonTrigger_Run019.mcp')
#
# desired_xml_saving_path = 'E:\\Workspace\\AddedTestFiles\\test2.xml'
# example_xml_file = 'E:/Workspace/AddedTestFiles/neu_trsdummy_141.xml'
# # Tools.createDB('E:\\Workspace\\AddedTestFiles\\database.sqlite')
# db = 'E:\\Workspace\\AddedTestFiles\\database.sqlite'
# # Tools.crawl(db)
# files = ['60Ni_no_protonTrigger_Run019.mcp', '60Ni_no_protonTrigger_Run035.mcp', '60Ni_no_protonTrigger_Run036.mcp',
#          '60Ni_no_protonTrigger_Run037.mcp', '60Ni_no_protonTrigger_Run044.mcp', '60Ni_no_protonTrigger_Run045.mcp',
#          '60Ni_no_protonTrigger_Run053.mcp', '60Ni_no_protonTrigger_Run054.mcp', '60Ni_no_protonTrigger_Run055.mcp',
#          '60Ni_no_protonTrigger_Run058.mcp', '60Ni_no_protonTrigger_Run059.mcp', '60Ni_no_protonTrigger_Run060.mcp',
#          '60Ni_no_protonTrigger_Run067.mcp', '60Ni_no_protonTrigger_Run068.mcp', '60Ni_no_protonTrigger_Run72.mcp',
#          '60Ni_no_protonTrigger_Run076.mcp', '60Ni_no_protonTrigger_Run077.mcp', '60Ni_no_protonTrigger_Run078.mcp',
#          '60Ni_no_protonTrigger_Run082.mcp', 'neu_trsdummy_141.xml']
# print(mcp_file)
# # meas = MCPImporter(mcp_file)
# meas = XMLImporter(example_xml_file)
#
# to_add = [(1, Meas.load(os.path.join(os.path.dirname(desired_xml_saving_path), file), db, raw=True)) for file in
#           files[1:]]
#
# dacs = []
# for tr_ind in range(meas.nrTracks):
#     dac_start = meas.x[tr_ind][0]
#     dac_stopp = meas.x[tr_ind][0]
#     dac_stepsize = meas.x[tr_ind][1] - meas.x[tr_ind][1]
#     dacs.append((dac_start, dac_stopp, dac_stepsize))
# # just extract needed values, if not available anyhow.
# # * needed & available in file, ! needed, but not be available in file, - only needed in db not relevant for adding
# # accVolt*, laserFreq(*)!, colDirTrue*, line-, type*, voltDivRatio-, lineMult-, lineOffset-, offset*
# meas2 = MCPImporter(mcp_file)
# meas_xml = XMLImporter(example_xml_file)
#
# cts_mcp = meas.cts
# cts_tilda = meas_xml.cts
#
#
# # measures = [(-1, meas2), (-1, meas2)]
#
#
# # add files
# # par_spec, added_files = add_specdata(meas, to_add)
# #
# # # create empty xml file
# # scan_dict = create_scan_dict_from_spec_data(meas, desired_xml_saving_path, db)
# # scan_dict['isotopeData']['addedFiles'] = added_files
# # TildaTools.createXmlFileOneIsotope(scan_dict, filename=desired_xml_saving_path)
# # # call savespecdata in filehandl (will expect to have a .time_res)
# # TildaTools.save_spec_data(meas, scan_dict)
# # MPLPlotter.plot(meas.getArithSpec([4], -1))
# # # MPLPlotter.show(True)
# spec, files, save_name = TildaTools.add_specdata(meas, to_add, os.path.dirname(desired_xml_saving_path))
#
# meas_import = XMLImporter(save_name)
# # print('after import: offset: %s, accVolt: %s' % (meas_import.offset, meas_import.accVolt))
# #
# # # print(create_scan_dict_from_spec_data(meas))
# # # print(create_scan_dict_from_spec_data(meas_xml))
# # MPLPlotter.plot(meas_import.getArithSpec([0], -1))
# # MPLPlotter.show(True)
#
# from Interface.LiveDataPlottingUi.LiveDataPlottingUi import TRSLivePlotWindowUi
# from PyQt5 import QtWidgets
# import sys
#
# app = QtWidgets.QApplication(sys.argv)
# mwin = TRSLivePlotWindowUi(save_name, subscribe_as_live_plot=False)
# mwin.new_data(meas_import)
# app.exec()
imp = Meas.load(
    'E:/Workspace/AddedTestFiles/neu_trsdummy_141_sum.xml', 'E:/Workspace/AddedTestFiles/AddedTestFiles.sqlite', True, False)
