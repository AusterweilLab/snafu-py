import rw
import numpy as np

steps=[]
ehs=[]
meaneh=[]
meanstep=[]

fh1 = open('ehmean.csv','w')
fh2 = open('eh.csv','w')

for i in range(10000):
    print i
    g,a=rw.genG(20,4,.3)
    X,step=rw.genX(g,use_irts=1)
    eh=rw.expectedHidden([X],a,len(a))[0]
    steps.append(step)
    ehs.append(eh)
    meaneh.append(np.mean(eh))
    meanstep.append(np.mean(step))

for num, i in enumerate(meaneh):
    fh1.write(str(i) + "," + str(meanstep[num-1]) + "\n")

ehs=rw.flatten_list(ehs)
steps=rw.flatten_list(steps)

for num, i in enumerate(ehs):
    fh2.write(str(i) + "," + str(steps[num-1]) + "\n")

fh1.close()
fh2.close()
