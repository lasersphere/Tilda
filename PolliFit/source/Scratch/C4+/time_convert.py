from datetime import datetime

path = "C:\\Users\\pimgram\\IKP ownCloud\\Projekte\\KOALA\\C4+\\Online-Auswertung\\2022-02-16\\cathode_teil1.txt"

file = open(path)

l = []
for i, line in enumerate(file):
    if i == 0:
        header = line
    else:
        test = line.split('\t')
        date = '2022-02-16 ' + test[0]
        dt_obj = datetime.strptime(date, '%Y-%m-%d %H:%M:%S,%f')
        time = dt_obj.timestamp()
        current = float(test[1].replace(',', '.'))
        l.append((time, current))

file.close()

path2 = "C:\\Users\\pimgram\\IKP ownCloud\\Projekte\\KOALA\\C4+\\Online-Auswertung\\2022-02-16\\cathode_teil1_conv.txt"

new_file = open(path2, 'x')

new_file.write('Time / s\tCurrent / mA\n')

for li in l:
    s = str(li[0]) + '\t'
    s += str(li[1]) + '\n'
    new_file.write(s)

new_file.close()