import numpy as np

fh=open('Sall_irt.csv','r')
line=fh.readline() # header row

pairlist=[]

fo=open('Sall_fixed.csv','w')

for line in fh:
    line=line.split('\n')[0].split(',')
    pair=np.sort([line[1],line[2]])
    pairstring=''.join(pair)
    if pairstring not in pairlist:
        pairlist.append(pairstring)
        newline=','.join([line[0], pair[0], pair[1], line[3], line[4], line[5]])
        fo.write(newline+"\n")

fh.close()
fo.close()
