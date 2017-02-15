# jump = 0.05, w/ prior weighted 10%
# n,k,p = 20,4,.3

library(data.table)

dat<-fread('test_case_prior.csv') # or tmp.csv

# cost goes down with more lists, prior doesn't seem to do consistently better (or worse)
dat[,.(mean(cost_invite),mean(cost_invitenoprior)),keyby=numx]

# P(true graph) is consistently worse
dat[,.(mean(truegraphval),mean(bestval_invite)),keyby=numx]

# Though per list, gap is closing
dat[,.(mean(truegraphval-bestval_invite)/numx),keyby=numx]
