#!/bin/sh

NUMGRAPHS=4
NUMSUBS=20

# change graph layout parameter for all at once using, e.g.
# dot -Tpdf S105_4.dot -o S105_4.pdf -Glayout=neato

# see http://www.graphviz.org/doc/info/attrs.html#d:layout for list of attributes
# see http://www.graphviz.org/content/command-line-invocation for command-line
# instructions, or put variables in dot files themselves

for i in $(seq 0 $NUMGRAPHS); do 
    for j in $(seq -s " " -w 1 $NUMSUBS); do
        #dot -Tpdf S1${j}_${i}.dot -o S1${j}_${i}.pdf
        dot -Kneato -Gmode=ipsep -Goverlap=ipsep -Gsplice=true -Gepsilon=.00000000001 -Gesep=+1 -Gsep=+10 -Gsplines=true -Gstart=${i} -Gdamping=.99 -Tpdf S1${j}_0.dot -o S${j}_0_neato${i}.pdf
    done
done

echo "Finished generating pdfs"
