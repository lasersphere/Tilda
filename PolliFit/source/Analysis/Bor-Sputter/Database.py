import os
import sqlite3
from openpyxl import Workbook, load_workbook
import Physics


###### Filling database with files, adding correct voltage, laser freqeuncy and isotope. Needs an excel file with information ######

#workingdir = 'C:\\Users\\Laura Renth\\Desktop\\Daten\\Promotion\\Bor\\Sputter source\\2021-03-Data' #working dir IKP
workingdir = 'C:\\Users\\Laura Renth\\ownCloud\\User\\Laura\\KOALA\\2021-03-Data'  #working dir IKP Owncloud
#workingdir = 'D:\\Owncloud\\User\\Laura\\KOALA\\2021-03-Data'  #working dir hp Owncloud
db = os.path.join(workingdir, 'B-_Auswertung.sqlite')
files = []  #list of all files

### open protocol workbook
wb = load_workbook(os.path.join(workingdir, 'Protokoll.xlsx'))
ws = wb.active

### fill files list
for row in ws.values:   #iterate through protocol
    files.append(row[0] + '.xml')

del files[0]
print(files)

### remove Files from database
con = sqlite3.connect(db)
cur = con.cursor()
cur.execute('''SELECT file FROM FILES''')
dbFiles = cur.fetchall()    #list of all files in data base
print(dbFiles)
con.commit()
con.close()

for file in dbFiles:    #iterate through all files in data base
    delete = True
    for f in files:
        if file[0]==f:
            delete = False  #if file in xlsx list, keep it
    if delete:
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute('''DELETE FROM Files WHERE file = ?''', (file[0],)) #else delete it
        con.commit()
        con.close()

### insert correct voltage and laser frequency
con = sqlite3.connect(db)
cur = con.cursor()
cur.execute('''SELECT file FROM FILES''',)
dbFiles = cur.fetchall()    #list of all files in data base

for file in dbFiles:
    for row in ws.values:
        if row[0] == file[0][:-4]:
            cur.execute('''UPDATE Files SET accVolt = ? WHERE file = ?''', (row[7], file[0],))
            cur.execute('''UPDATE Files SET laserFreq = ? WHERE file = ?''', (row[3]*4, file[0],))
con.commit()
con.close()

### correct the isotope name
con = sqlite3.connect(db)
cur = con. cursor()
for file in dbFiles:
    cur.execute('''SELECT type FROM FILES WHERE file = ?''', (file[0],))
    type = cur.fetchall()[0][0]
    if type =='11B_D2':
        line = 'D2'
    elif type == '10B_D2':
        line = 'D2'
    else:
        line = 'D1'
    cur.execute('''UPDATE Files SET line = ? WHERE file = ?''', (line, file[0],))
con.commit()
con.close()

print(Physics.relDoppler(4*299412500,-562017))