library(lubridate)
library(ggplot2)
library(data.table)
library(ez)

dat<-fread('logall.csv')
dat[,timetaken:=parse_date_time(time,c("hms"))]

followtypedat<-dat[startGraph=="windowgraph" & method=="uinvite" & triangle_limit==100]
ggplot(followtypedat[,.(cost=mean(cost)),keyby=.(numx,followtype)],aes(x=numx,y=cost,color=followtype)) + geom_line() + ylim(0,165)
ggplot(followtypedat[,.(timetaken=mean(timetaken)),keyby=.(numx,followtype)],aes(x=numx,y=timetaken,color=followtype)) + geom_line()

limitdat<-dat[startGraph=="windowgraph" & method=="uinvite" & followtype=="avg"]
ggplot(limitdat[,.(cost=mean(cost)),keyby=.(numx,triangle_limit)],aes(x=numx,y=cost,color=factor(triangle_limit))) + geom_line() + ylim(0,165)
ggplot(limitdat[,.(timetaken=mean(timetaken)),keyby=.(numx,triangle_limit)],aes(x=numx,y=timetaken,color=factor(triangle_limit))) + geom_line()

startgraphdat<-dat[method=="uinvite" & followtype=="avg" & triangle_limit==100]
ggplot(startgraphdat[,.(cost=mean(cost)),keyby=.(numx,startGraph)],aes(x=numx,y=cost,color=factor(startGraph))) + geom_line() + ylim(0,165)
ggplot(startgraphdat[,.(timetaken=mean(timetaken)),keyby=.(numx,startGraph)],aes(x=numx,y=timetaken,color=factor(startGraph))) + geom_line()

methoddat<-dat[followtype=="avg" & triangle_limit==100 & startGraph=="windowgraph"]
ggplot(methoddat[,.(cost=mean(cost)),keyby=.(numx,method)],aes(x=numx,y=cost,color=factor(method))) + geom_line() + ylim(0,400)
ggplot(methoddat[,.(timetaken=mean(timetaken)),keyby=.(numx,method)],aes(x=numx,y=timetaken,color=factor(method))) + geom_line()

source('basicplot.r')
library(Rmisc)
uinvitedat<-dat[followtype=="avg" & triangle_limit==100 & startGraph=="windowgraph" & (method=="uinvite" | method=="uinvite_prior" | method=="uinvite_irt" | method=="uinvite_irt_prior")]
bbar(uinvitedat[,.(cost),by=method])

uinvitedat<-dat[followtype=="avg" & triangle_limit==100 & startGraph=="windowgraph" & (method=="uinvite" | method=="uinvite_prior")]
ggplot(uinvitedat[,.(cost=mean(cost)),keyby=.(numx,method)],aes(x=numx,y=cost,color=factor(method))) + geom_line() + ylim(0,175)

uinvitedat<-dat[followtype=="avg" & triangle_limit==100 & startGraph=="windowgraph" & (method=="uinvite" | method=="uinvite_irt")]
ggplot(uinvitedat[,.(cost=mean(cost)),keyby=.(numx,method)],aes(x=numx,y=cost,color=factor(method))) + geom_line() + ylim(0,175)

