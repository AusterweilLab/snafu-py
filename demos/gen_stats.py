# DEMO 

import snafu

filepath = "../fluency_data/snafu_sample.csv"

# load data from Experiment 1, animal category
# fix spelling
# keep lists organized by subject (hierarchical)
# don't remove intrusions or perseverations

filedata = snafu.load_fluency_data(filepath,
                                   category="animals",
                                   spell="../spellfiles/animals_snafu_spellfile.csv",
                                   group=["Experiment1"],
                                   hierarchical=True)

snafu.perseverations(filedata.labeledXs)
snafu.intrusions(filedata.labeledXs, "../schemes/animals_snafu_scheme.csv")
snafu.clusterSize(filedata.labeledXs, "../schemes/animals_snafu_scheme.csv", clustertype="fluid")
snafu.clusterSwitch(filedata.labeledXs, "../schemes/animals_snafu_scheme.csv", clustertype="fluid")
snafu.wordFrequency(filedata.labeledXs, data="../frequency/subtlex-us.csv", missing=0.5)


snafu.clusterSize(filedata.labeledXs, 2, clustertype="fluid")
snafu.clusterSwitch(filedata.labeledXs, 2, clustertype="fluid")

