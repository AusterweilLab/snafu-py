library(data.table)
library(ggplot2)
library(reshape2)
library(tidyr)

dat <- fread('usf_sims.csv')

names(dat) <- c("numlists","listlength","sample","rw_hit","rw_miss","rw_fa","rw_cr","ui_hit","ui_miss","ui_fa","ui_cr")
dat <- dat[,lapply(.SD,mean),keyby=.(numlists,listlength)]
dat[,sample:=NULL]
dat[,ui_cost:=ui_miss+ui_fa]
dat[,rw_cost:=rw_miss+rw_fa]
dat[,numedges:=ui_hit+ui_fa+ui_miss+ui_cr] # same for UI and RW


# impute miss/cr from missing nodes
misses<-fread('usf_sims_misses.csv')
names(misses)<-c("numlists","listlength","missavg","cravg")
setkey(dat,numlists,listlength)
setkey(misses,numlists,listlength)
dat<-merge(dat,misses)

dat[,ui_costwhole:=ui_cost+missavg] # cost including nodes not in X
dat[,rw_costwhole:=rw_cost+missavg]

dat2<-melt(dat,id.vars=c("numlists","listlength","numedges","missavg","cravg"))
dat2<-separate(dat2,variable,sep="_",into=c("model","measure"))

dat3<-spread(dat2, key=measure, value=value)

ggplot(dat2[measure!="costwhole" & measure!="cr" & measure!="miss" & measure!="cost"],aes(y=value,x=numlists,color=model,linetype=measure)) + geom_line() + facet_wrap(~ listlength)
ggplot(dat3,aes(y=costwhole,x=numlists,color=model)) + geom_line() + facet_wrap(~ listlength)
