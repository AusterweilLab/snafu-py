import pickle
import numpy as np

outname="individual_uinvite.csv"
fo=open(outname,'w')

for subnum in range(50):
    picklename="h_" + str(subnum) + ".pickle"
    fh=open(picklename,'r')
    alldata = pickle.load(fh)
    graph=alldata['graph']
    items=alldata['items']
    fh.close()

    for i in range(len(graph)):
        for j in range(len(graph)):
            if i>j:
                item1=items[i]
                item2=items[j]
                itempair=np.sort([item1,item2])
                subj="S"+str(101+subnum)
                fo.write(subj + ",uinvite_hierarchical," + itempair[0] + "," + itempair[1] +  "," + str(graph[i,j]) + "\n")

fo.close()
