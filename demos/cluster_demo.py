# This file will code each item in your fluency data as either a cluster switch
# or not -- useful for item level analysis

import snafu
import os

os.makedirs("demos_data", exist_ok=True)



fluencydata = snafu.load_fluency_data("../fluency_data/snafu_sample.csv", 
                                        category="animals",
                                        removeNonAlphaChars=True,
                                        spell="../spellfiles/animals_snafu_spellfile.csv", 
                                        group=["Experiment1"])

schemefile = "../schemes/animals_snafu_scheme.csv"

# Replaces each item in each list with its clusters (if multiple, separated by semicolon)
clusterlabels = snafu.labelClusters(fluencydata.labeledXs, scheme=schemefile, labelIntrusions=True)

# code each item as a cluster switch, non-cluster switch, or intrusion
switchlists = []
for fluencylist in clusterlabels:
    switchlist = []
    prev_clusters = []
    curr_clusters = []
    for itemnum, item in enumerate(fluencylist):
        if item == "intrusion":
            switchlist.append("intrusion")
            continue
        curr_clusters = item.split(';')
        matching_clusters = list(set(prev_clusters) & set(curr_clusters))
        if (len(matching_clusters) > 0) or (len(prev_clusters) == 0): # if no overlap, or if first item
            switchlist.append(0)
        else:
            switchlist.append(1)
        prev_clusters = curr_clusters 
    switchlists.append(switchlist)

# write data to file
with open('demos_data/switches.csv','w') as fh:
    fh.write('id,listnum,category,item,switch\n')
    for eachlistnum, eachlist in enumerate(fluencydata.listnums):
        subj = eachlist[0]
        listnum = eachlist[1]
        for itemnum, item in enumerate(fluencydata.labeledXs[eachlistnum]):
            to_write = [subj, str(listnum), "animals", item, str(switchlists[eachlistnum][itemnum])]
            fh.write(','.join(to_write))
            fh.write("\n")



