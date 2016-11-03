import rw
import math

# double check number of nodes is correct
prior=rw.genPrior(50,4,.3)      # same vals used in generating test_case_prior.csv

outfile=open('swvals2.csv','w')

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
        outfile.write(str(math.log(prior(rw.smallworld(tg))[0])) + ',')
        outfile.write(str(math.log(prior(rw.smallworld(nrw))[0])) + ',')
        outfile.write(str(math.log(prior(rw.smallworld(uinp))[0])) + ',')
        outfile.write(str(math.log(prior(rw.smallworld(uip))[0])))
        print prior(rw.smallworld(tg))[0]
        outfile.write('\n')

outfile.close()
