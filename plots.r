library(lubridate)
library(ggplot2)
library(data.table)

dat<-fread('log.csv')
dat[,time2:=parse_date_time(time,c("hms"))]
dat2<-dat[,.(time2=mean(time)),by=.(method,numx)]

ggplot(dat2,aes(x=numx,y=time2,color=method)) + geom_line()

ggplot(dat[startGraph=="windowgraph" & method=="uinvite",mean(timeq),keyby=.(followtype,numx)],aes(x=numx,y=V1,color=followtype)) + geom_line()
