library(data.table)
library(ggplot2)

linenum<-10

dat <- readLines('records.csv')
l1 <- strsplit(dat[linenum],split=",")
method<-as.numeric(l1[[1]][1])
methodnames<-c("Prune","Triangles","Other")

l2 <- unlist(lapply(l1[[1]][c(2:length(l1[[1]]))], function(i) { if (i=="-inf") { NA } else { as.numeric(i) } }))

# list of changed edges that successfully improved the graph
# those are edges signed + in the data
changed<-unlist(lapply(l2,function(i) { if ((i<0) || is.na(i)) { FALSE } else { TRUE }}))

# make all LL negative again
l2 <- -abs(l2)

plotdat<-data.table(x=seq(length(l2)), y=l2, changed=changed)
plotplot<-ggplot(plotdat,aes(x=x,y=y)) + geom_point(aes(color=changed)) + geom_line() + ylab("LL") + ggtitle(methodnames[method+1]) + theme(legend.position="none")
