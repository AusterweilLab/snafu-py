import rw.rw as rw
import networkx

f=open('sim_results.csv')
dat=f.read().split('\n')
f.close()

outfile='sdt.csv'
fo=open(outfile,'a', 0) # write/append to file with no buffering

for line in range(1,len(dat)-1):
    linearr=dat[line].split(',')
    numnodes=int(linearr[10])
    numlinks=int(linearr[11])
    probRewire=float(linearr[12])
    graph_seed=int(linearr[14])
    g,a=rw.genG(numnodes,numlinks,probRewire,seed=graph_seed)

    x_seed=int(linearr[17])
    numx=int(linearr[15])
    [Xs,irts]=zip(*[rw.genX(g, seed=x_seed+i,use_irts=1) for i in range(numx)])

    g1=rw.noHidden(Xs, numnodes)          # directly connect Xs into a graph
    g2=rw.hashToGraph(linearr[23], numnodes) # irts
    g3=rw.hashToGraph(linearr[24], numnodes) # no irts
    sdt1=rw.costSDT(g1, a)
    sdt2=rw.costSDT(g2, a)
    sdt3=rw.costSDT(g3, a)

    fo.write(str(sdt1[0]) + ',' +
             str(sdt1[1]) + ',' +
             str(sdt1[2]) + ',' +
             str(sdt1[3]) + ',' +
             str(sdt2[0]) + ',' +
             str(sdt2[1]) + ',' +
             str(sdt2[2]) + ',' +
             str(sdt2[3]) + ',' +
             str(sdt3[0]) + ',' +
             str(sdt3[1]) + ',' +
             str(sdt3[2]) + ',' +
             str(sdt3[3]) + '\n')

