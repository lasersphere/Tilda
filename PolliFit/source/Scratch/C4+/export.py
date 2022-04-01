import ast
from datetime import datetime
import TildaTools as TiTs

dbpath = "C:\\Users\\pimgram\\IKP ownCloud\\Projekte\\KOALA\\C4+\\Online-Auswertung\\2022-02-16\\C4+_P2_cw.sqlite"
savepath = "C:\\Users\\pimgram\\IKP ownCloud\\Projekte\\KOALA\\C4+\\Online-Auswertung\\2022-02-16\\export.txt"

data_fitRes = TiTs.select_from_db(dbpath, 'file, rChi, pars', 'FitRes')
data_files = TiTs.select_from_db(dbpath, 'file, date', 'Files')

list = []
for file in data_fitRes:
    file_name = file[0]
    file_rChi = file[1]
    file_pars = ast.literal_eval(file[2])
    file_center = file_pars['center']

    for f in data_files:
        if file_name == f[0]:
            file_date = f[1]
            dt_obj = datetime.strptime(file_date, '%Y-%m-%d %H:%M:%S')
            file_time = dt_obj.timestamp()
            break

    list.append((file_time, file_name, file_center, file_rChi))

file_write = open(savepath, 'x')
file_write.write('DateTime / s\tFile Name\tCenter / MHz\tCenter_d / MHz\trChi\n')
for fi in list:
    st = str(fi[0]) + '\t'
    st += fi[1] + '\t'
    st += str(fi[2][0]) + '\t' + str(fi[2][1]) + '\t'
    st += str(fi[3]) + '\n'

    file_write.write(st)

file_write.close()