import os

from Measurement.XMLImporter import XMLImporter

run68= 'C:\\Collaps_Ni_selected_files\\Ni_tipa_068.xml'
run69 = 'C:\\Collaps_Ni_selected_files\\Ni_tipa_069.xml'


# print(spec.t_proj[0].shape, len(spec.t))
txt_path_68 = os.path.join(os.path.split(run68)[0], 'Ni_tipa_068.txt')
txt_path_69 = os.path.join(os.path.split(run69)[0], 'Ni_tipa_069.txt')

for xmlfile in [run68, run69]:
    spec = XMLImporter(xmlfile, False)
    time_proj = spec.t_proj[0]
    path = os.path.join(os.path.split(xmlfile)[0], os.path.split(xmlfile)[1].replace('.xml', '.txt'))
    # print(path)
    file = open(path, 'w')
    file.write('%s \t %s \t %s \t %s \t %s\n' % ('t', 'pmt0', 'pmt1', 'pmt2', 'pmt3'))
    for i, j in enumerate(spec.t[0]):
        file.write('%d \t %d \t %d \t %d \t %d\n' % (spec.t[0][i], time_proj[0][i], time_proj[1][i], time_proj[2][i], time_proj[3][i]))
    file.close()