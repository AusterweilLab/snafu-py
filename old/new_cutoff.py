import pickle
import rw

usf_graph, usf_items = rw.read_csv("./snet/USF_animal_subset.snet")

for i in range(1,14):
    filename="uinvite_noirt_0_"+str(i)
    fh=open(filename,'r')
    priordict=pickle.load(fh)
    fh.close()
    uinvite_group_graph = rw.priorToGraph(priordict, usf_items,cutoff=0.7)
    print rw.cost(uinvite_group_graph, usf_graph)



