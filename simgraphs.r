library(data.table)
library(ggplot2)
library(gridExtra)

usf_hit <- 393
usf_cr <- 12327
usf_total <- 12720

#dat <- fread('sim_all_but_uinvite.csv')
dat <- fread('simgood.csv')
dat[,modeltype:="uinvite"]
dat[method=="rw",modeltype:="other"]
dat[method=="fe",modeltype:="other"]
dat[method=="goni",modeltype:="other"]
dat[method=="kenett",modeltype:="other"]
dat[method=="chan",modeltype:="other"]

dat<- dat[,lapply(.SD,mean),by=.(method,modeltype,subjects)]

dat[,hit_p := hit/usf_hit]
dat[,fa_p := fa/usf_cr]
dat[,cost_p := cost/usf_total]

bbrocdat <- fread('bb.csv')
zibrocdat <- fread('zib.csv')
bbrocdat[,ptsize:=1]
zibrocdat[,ptsize:=1]
bbrocdat[cutoff==0.5,ptsize:=3]
zibrocdat[cutoff==0.5,ptsize:=3]
bbrocdat[,type:="bb"]
zibrocdat[,type:="zibb"]
rocdat <- rbind(bbrocdat,zibrocdat)

#ggplot(rocdat,aes(y=hit,x=fa,color=name)) + geom_point(aes(size=factor(ptsize))) + geom_line() + coord_cartesian(xlim=c(0,313),ylim=c(0,313))

#ggplot(dat[p=="p7" & b=="b1"],aes(x=fa,y=hit,color=a)) + geom_line() + geom_point(aes(size=factor(cutoff==0.5))) + xlim(0,393) + ylim(0,393)

#p4 <- ggplot(dat, aes(x=subjects,y=cost_p,color=method)) + geom_line() + ylim(0,0.08)
#p5 <- ggplot(dat, aes(x=subjects,y=hit_p,color=method)) + geom_line()
#p6 <- ggplot(dat, aes(x=subjects,y=fa_p,color=method)) + geom_line()
#grid.arrange(p4,p5,p6,ncol=3)

mainPlot <- function() {
    p1 <- ggplot(dat, aes(x=subjects,y=cost_p,color=method,linetype=modeltype)) + geom_line() + ylab("Cost (miss + false alarm)") + xlab("# of subjects")
    p2 <- ggplot(dat, aes(x=subjects,y=hit_p,color=method,linetype=modeltype)) + geom_line() + ylab("Hits") + xlab("# of subjects")
    p3 <- ggplot(dat, aes(x=subjects,y=fa_p,color=method,linetype=modeltype)) + geom_line() + ylab("False alarms") + xlab("# of subjects")
    grid.arrange(p1,p2,p3,ncol=3)
}

plotROC <- function() {
    bbroc <- ggplot(bbrocdat,aes(y=hit,x=fa,color=name)) + geom_point(aes(size=factor(ptsize))) + geom_line() + coord_cartesian(xlim=c(0,313),ylim=c(0,313)) + geom_abline(linetype="dashed")
    zibroc <- ggplot(zibrocdat,aes(y=hit,x=fa,color=name)) + geom_point(aes(size=factor(ptsize))) + geom_line() + coord_cartesian(xlim=c(0,313),ylim=c(0,313)) + geom_abline(linetype="dashed")
    grid.arrange(bbroc,zibroc,ncol=2)
}
