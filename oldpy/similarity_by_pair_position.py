import numpy as np
from __future__ import division
import rw

# read in data from similarity experiment
sims={}

def loadRatings(f, sims):
    with open(f,'r') as ratings:
        for line in ratings:
            line=line.split(',')
            knowboth = line[5]
            if knowboth=="false":
                items = np.sort([line[2],line[3]])
                rating = line[4]
                if items[0] not in sims.keys():
                    sims[items[0]] = {}
                if items[1] not in sims[items[0]].keys():
                    sims[items[0]][items[1]] = (int(rating), 1)
                else:
                    currentrating = sims[items[0]][items[1]][0]
                    numratings = sims[items[0]][items[1]][1] + 1
                    sims[items[0]][items[1]] = (((currentrating + int(rating)) / numratings), numratings)
    return sims

# human ratings
sims = loadRatings('joe1.csv', sims)
sims = loadRatings('jeff1.csv', sims)
sims = loadRatings('joe2.csv', sims)

subj="S101"
category="animals"

Xs, items, irts, numnodes = rw.readX(subj,category,'/Users/jcz/Dropbox/projects/semnet/rw/Spring2015/results_cleaned.csv')
