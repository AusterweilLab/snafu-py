library(data.table)
library(ggplot2)

dat <- readLines('records.csv')

l1 <- strsplit(dat[1],split=",")
method<-l1[[1]][1]
l2<-unlist(lapply(l1[[1]][c(2:length(l1[[1]]))], function(i) { if (i=="-inf") { NA } else { as.numeric(i) } }))
plotdat<-data.table(x=seq(length(l2)),y=l2)
ggplot(plotdat,aes(x=x,y=y)) + geom_line()
