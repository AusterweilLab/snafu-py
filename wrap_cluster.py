import numpy as np
import rw

def no_persev(x):
    seen = set()
    seen_add = seen.add
    return [i for i in x if not (i in seen or seen_add(i))]

visits={}
with open("JA_AnimalNaming.csv","r") as fh:
    fh.readline()
    lines=[line.strip("\n").split(",") for line in fh]
    ids=list(set([line[0] for line in lines]))
    for line in lines:
        if line[0] not in visits.keys():
            visits[line[0]] = []
        if line[1] not in visits[line[0]]:
            visits[line[0]].append(line[1])

with open('wrap_clustering.csv','w') as fh:
    for subj in ids:
        Xs, items, numnodes, irts = rw.readX(subj,"animals","JA_AnimalNaming.csv",ignorePerseverations=False,spellfile="categories/zemla_spellfile.csv")
        Xs = rw.numToAnimal(Xs, items)
        p=[rw.perseverations(x) for x in Xs]
        Xs = [no_persev(x) for x in Xs]
        
        cf=[rw.clusterSize(x, "categories/troyer_hills_zemla_animals.csv") for x in Xs]
        cs=[rw.clusterSize(x, "categories/troyer_hills_zemla_animals.csv", clustertype="static") for x in Xs]
        i=[rw.intrusions(x, "categories/troyer_hills_zemla_animals.csv") for x in Xs]
        for visnum, visid in enumerate(visits[subj]):
            vec = [subj, visid, np.mean(cf[visnum]), len(cf[visnum]), np.mean(cs[visnum]), len(cs[visnum]), len(i[visnum]), len(p[visnum])]
            vec = [str(j) for j in vec]
            fh.write(",".join(vec))
            fh.write('\n')
