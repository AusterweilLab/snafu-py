#!/bin/sh

for i in $(seq 1 10); do 
    dot -Kneato -Gmode=ipsep -Goverlap=ipsep -Gsplice=true -Gepsilon=.00000000001 -Gesep=+1 -Gsep=+10 -Gsplines=true -Gstart=${i} -Gdamping=.99 -Tpdf S111_0.dot -o S111_0_neato${i}.pdf
done
