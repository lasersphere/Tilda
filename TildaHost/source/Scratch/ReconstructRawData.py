"""
Module for reconstructing raw data to an xml file.

"""
import os
from datetime import datetime
import json
from PyQt5 import QtCore
import XmlOperations as XmlOps
import TildaTools as TiTs
from Measurement.XMLImporter import XMLImporter
import numpy as np

import Service.FileOperations.FolderAndFileHandling as FilesHandl
from Service.AnalysisAndDataHandling.tildaPipeline import find_pipe_by_seq_type

# filenames = ['26Al_trs_run142.xml']
work_dir = 'E:/temp4'
raw_files = os.path.join(work_dir, 'raw')
sums_dir = os.path.join(work_dir, 'sums')
filenames = os.listdir(sums_dir)


# filenames = ['one_trsdummy_run000.xml']

# work_dir = 'E:/TildaDebugging2'
# raw_files = os.path.join(work_dir, 'raw')
# sums_dir = os.path.join(work_dir, 'sums')
# filenames = os.listdir(sums_dir)


def find_go(file_name, workdir):
    """ find all files that this file was a go on """
    root = TiTs.load_xml(os.path.join(workdir, 'sums', file_name))
    go_on = ''
    go = XmlOps.xmlFindOrCreateSubElement(root, 'header').find('continuedAcquisitonOnFile')
    if go is not None:
        go_on = go.text
    return go_on


def reconstruct_file_from_raw(file_name, raw_files, workdir):
    """ reconstruct a file from the raw data files. Will find all raw data, even its a go on something. """
    # workdir = 'G:/Experiments/Collaps/Collaps_items/data_online/Al/Al_2017_Tilda'
    # raw_files = os.path.join(workdir, 'raw')
    max_element_fed = 1
    xml_files = [file_name]
    go_on = find_go(file_name, workdir)
    while go_on:
        xml_files.insert(0, go_on)
        go_on = find_go(go_on, workdir)
    print('xml_files: ', xml_files)
    raw_files_this_xml = []
    for each in xml_files:
        raw_files_this_xml += [
            os.path.normpath(os.path.join(raw_files, file))
            for file in os.listdir(raw_files) if each.split('.')[0] in file]
    file_name = xml_files[-1]
    # print(raw_files_this_xml)
    pipedata_files = [file for file in raw_files_this_xml if file.endswith('jpipedat')]
    print('pipedata files: ')
    for pipedata_file in pipedata_files:
        print('\t', pipedata_file)
    hdf5_files = [file for file in raw_files_this_xml if file.endswith('hdf5')]
    print('hdf5 files: ')
    for hdf5_file in hdf5_files:
        print('\t', hdf5_file)
    if len(pipedata_files) >= 2:
        scan_dict = FilesHandl.load_json(pipedata_files[-2])
        scan_dict_done = FilesHandl.load_json(pipedata_files[-1])
    else:
        # acquisition was not completed, give it a shot anyhow.
        scan_dict = FilesHandl.load_json(pipedata_files[0])
        scan_dict_done = FilesHandl.load_json(pipedata_files[0])
    # print(scan_dict)#
    # TiTs.print_dict_pretty(scan_dict)
    reconstructed_folder = os.path.normpath(os.path.join(workdir, 'reconstructed', 'sums')).replace('\\', '\\\\')
    new_file_name = os.path.join(reconstructed_folder, file_name)
    if not os.path.isdir(reconstructed_folder):
        print('creating folder: %s' % reconstructed_folder)
        os.mkdir(os.path.split(reconstructed_folder)[0])
        os.mkdir(reconstructed_folder)
        print('%s was created' % reconstructed_folder)
    scan_dict['pipeInternals']['workingDirectory'] = os.path.join(workdir, 'reconstructed')

    scan_dict['pipeInternals']['curVoltInd'] = 0
    scan_dict['isotopeData']['isotopeStartTime'] = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    new_file_name = TiTs.createXmlFileOneIsotope(scan_dict, filename=new_file_name)
    scan_dict['pipeInternals']['activeXmlFilePath'] = new_file_name

    pipe = find_pipe_by_seq_type(scan_dict,
                                 callback_sig=None,
                                 live_plot_callback_tuples=(None, None, None, None, None),
                                 fit_res_callback_dict=None,
                                 scan_complete_callback=None,
                                 dac_new_volt_set_callback=None)
    # print(pipe, type(pipe))
    print('new file name: ', new_file_name)
    # print('file renamed to: ', new_file_name)
    pipe.start()
    if len(hdf5_files):
        for file in hdf5_files:
            data = FilesHandl.load_hdf5(file)
            # feed only step by step!
            # TiTs.translate_raw_data(data)

            for i in range(0, data.size, max_element_fed):
                pipe.feed(data[i:i + max_element_fed])
            if i + max_element_fed < data.size:
                pipe.feed(data[i:data.size])

        save_start = datetime.now()
        # print('done loading raw data saving now. ', save_start)
        pipe.save()
        save_done = datetime.now()
        # print('saving took: ', save_done - save_start)
        # new_file_name = 'G:/Experiments/Collaps/Collaps_items/data_online/Al/Al_2017_Tilda/reconstructed/sums/27Al_trs_run063.xml'
        root = TiTs.load_xml(new_file_name)
        header = XmlOps.xmlFindOrCreateSubElement(root, 'header')
        XmlOps.xmlFindOrCreateSubElement(header, 'isotopeStartTime',
                                         value=scan_dict_done['isotopeData']['isotopeStartTime'])
        tracks = XmlOps.xmlFindOrCreateSubElement(root, 'tracks')
        for track in tracks:
            # print(track, track.tag)
            tr_header = XmlOps.xmlFindOrCreateSubElement(track, 'header')
            XmlOps.xmlFindOrCreateSubElement(tr_header, 'workingTime',
                                             str(scan_dict_done.get(track.tag, {}).get('workingTime', str([]))))
        TiTs.save_xml(root, path=new_file_name)
        return new_file_name
    else:
        raise Exception('no raw data was saved for this one!')


def compare_xml(sample_file, compare_files):
    """ compare a list of xml files to a given sample file """
    res_lis = []
    not_equal = []
    sample_meas = XMLImporter(sample_file)
    for compare_file in compare_files:
        if compare_file != sample_file:
            compare_meas = XMLImporter(compare_file)
            if compare_meas.seq_type in ['trs', 'trsdummy']:
                same = np.array_equal(compare_meas.time_res_zf, sample_meas.time_res_zf)
                same_t = np.array_equal(compare_meas.t, sample_meas.t)
            elif compare_meas.seq_type in ['cs', 'csdummy']:
                same = np.array_equal(compare_meas.cts, sample_meas.cts)
                same_t = None
            res_lis.append((compare_file, same, same_t))
            if not same:
                not_equal.append((compare_file, same, same_t))
    return not_equal



not_equal_files = []
not_eq_ct = 0
eq_ct = 0
tot_ct = 0
not_saved_files = []
excpetion_on_files = []
no_xml = []
for file in filenames:
    print('------------- working on: %s ----------' % file)
    try:
        if file.endswith('.xml'):  # no files that were not saved!
            tot_ct += 1
            print('file')
            new_file = reconstruct_file_from_raw(file, raw_files, work_dir)
            print('reconstruction done')
        #     if os.path.getsize(os.path.join(work_dir, 'sums', file)) > 6000:
        #         comp_res = compare_xml(os.path.join(sums_dir, file), [new_file])
        #         if len(comp_res):
        #             not_equal_files.append((file, comp_res))
        #             not_eq_ct += 1
        #             print(os.path.join(sums_dir, file), ' and ', new_file, '  are NOT equal')
        #         else:
        #             eq_ct += 1
        #             print(os.path.join(sums_dir, file), ' and ', new_file, '  are equal')
        #     else:
        #         not_saved_files.append(file)
        # else:
        #     no_xml.append(file)
    except Exception as e:
        excpetion_on_files.append((file, e))
    print('------------- done with: %s ----------' % file)

filenames = [os.path.join('E:/Workspace/Al_Collaps_analysis/reconstructed/sums', fi) for fi in os.listdir('E:/Workspace/Al_Collaps_analysis/reconstructed/sums')]
print('not equal files: ',
      compare_xml(os.path.join('E:/Workspace/Al_Collaps_analysis/reconstructed/sums', '10_feed_29Al_trs_run228.xml'),
                  filenames))

print('total files: %s, equal files: %s,'
      ' not equal files: %s, percentage_not_equal: %.2f %%' % (
          tot_ct, eq_ct, not_eq_ct, (not_eq_ct / max(1, (eq_ct + not_eq_ct))) * 100))

print('files that are not equal after reconstruction:')
for not_equal_file in not_equal_files:
    print('\t', not_equal_file)

print('files not saved:')
for f in not_saved_files:
    print('\t', f)

print('files with exception:')
for f in excpetion_on_files:
    print('\t', f)

print('%d files that are no xml: ' % len(no_xml))
for f in no_xml:
    print('\t', f)
