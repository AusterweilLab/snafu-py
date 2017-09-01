import rw
import math

# double check number of nodes is correct
graph=rw.Toygraphs({'graphtype': "smallworld",
                   'numnodes':50,
                   'numlinks': 4,
                   'probRewire': .3})

prior=rw.genSWPrior(graph)      # same vals used in generating test_case_prior.csv

outfile=open('swvals3.csv','w')

with open('test_case_prior2.csv','r') as fh:
    next(fh) # skip header row
    for line in fh:
        line=line.split(',')
        
        tg=rw.hashToGraph(line[10])     # true graph
        nrw=rw.hashToGraph(line[14])    # naive random walk
        uinp=rw.hashToGraph(line[22])   # u-invite no prior
        uip=rw.hashToGraph(line[30])    # u-invite w/ prior
        numx=str(line[7])
        
        outfile.write(numx + ',')
        outfile.write(str(math.log(rw.evalSWPrior(rw.smallworld(tg),prior))) + ',')
        outfile.write(str(math.log(rw.evalSWPrior(rw.smallworld(nrw),prior))) + ',')
        outfile.write(str(math.log(rw.evalSWPrior(rw.smallworld(uinp),prior))) + ',')
        outfile.write(str(math.log(rw.evalSWPrior(rw.smallworld(uip),prior))))
        print rw.evalSWPrior(rw.smallworld(tg),prior)
        outfile.write('\n')

outfile.close()
