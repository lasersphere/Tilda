"""

Created on '22.06.2015'

@author:'simkaufm'

"""
import Tools as Tls

# db = 'R:\Projekte\TRIGA\Measurements and Analysis_Simon\Bunchermessungen2016\Bunchermessungen2016.sqlite'
# isoL = ['40_Ca', '43_Ca', '44_Ca', '48_Ca']
# Tls.centerPlot(db, isoL)
file = 'E:\\Quote.txt'
stream = open(file)

compl = ''
go = True
while go:
    i = 0
    line = ''
    while i < 9:
        read = stream.readline()
        if read == '':
            go = False
            break
        line += read
        i += 1
    line = line.replace('\n', '\t')
    compl += line + '\n'
print(compl)
stream.close()