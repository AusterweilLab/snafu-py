#!/usr/bin/python

# * Excludes lists that had to be re-done (<5 items)
# * S101 modified to separate items entered on the same line (RT lost)
# * Lower case 
# * Remove plurals and spaces
# * still needs a significant amount of manual processing to clean up!! mostly spelling mistakes

import json
import os
import csv

logdir=('./test/')
outfile='pilot.csv'

datafiles=os.listdir(logdir)
datafiles=[df for df in datafiles if "data" in df]

header=['id','cond','game','category','item','firstkey','RT','RTstart','shortlist']
fulldata=[]

for df in datafiles:
    ok=df.split('_')[0]
    subj=ok[:-1]
    cond=ok[-1]
    with open(logdir+df) as json_data:
        data=json.load(json_data)
    for gamenum, game in enumerate(data):
        shortlist = 1 if len(game["items"]) <= 5 else 0
        category=game["category"]
        for i, item in enumerate(game["items"]):
            if len(game["firsttime"]) < len(game["times"]):
                abc=1
                if i==0:
                    firsttime=""
                else:
                    firsttime=game["firsttime"][i-1]-game["times"][i-1]
            else:
                abc=2
                if i==0:
                    firsttime=game["firsttime"][i]-game["starttime"]
                else:
                    firsttime=game["firsttime"][i]-game["times"][i-1]
            rtstart=game["times"][i]-game["starttime"]
            if i==0:
                rt=rtstart
            else:
                rt=game["times"][i]-game["times"][i-1]
            cleanitem=item.lower().replace(" ","").replace("?","").replace("'","")
            line=[subj, cond, gamenum, category, cleanitem, firsttime, rt, rtstart, shortlist]
            fulldata.append(line)

with open(outfile,'wb') as csvfile:
    w = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    w.writerow(header)
    for i in fulldata:
        w.writerow(i)

