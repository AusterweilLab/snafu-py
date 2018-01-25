library(data.table)
library(ggplot2)
library(gridExtra)

#Extract Legend 
g_legend<-function(a.gplot){ 
  tmp <- ggplot_gtable(ggplot_build(a.gplot)) 
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box") 
  legend <- tmp$grobs[[leg]] 
  return(legend)} 


#dat<-fread('usf.csv')
dat<-fread('usf.csv')
dat2<-dat[,mean(cost),keyby=.(method,listnum)]
dat3<-dat[,mean(hit),keyby=.(method,listnum)]
dat4<-dat[,mean(fa),keyby=.(method,listnum)]

dat4[,method:=factor(method)]
levels(dat4$method)<-c("PF","FE","CN","CBN","RW","U-INVITE","Hierarchical U-INVITE")
dat4[,Method:=method]

usf1 <- ggplot(dat2,aes(x=listnum,y=V1,color=method)) + geom_line(size=0.6) + theme_classic() + xlab("Number of psuedo-participants") + ylab("Cost") + theme(legend.position="none") + scale_y_continuous(limits=c(0,1000),expand=c(0,0))
usf2 <- ggplot(dat3,aes(x=listnum,y=V1,color=method)) + geom_line(size=0.6) + theme_classic() + xlab("Number of psuedo-participants") + ylab("Hits") + theme(legend.position="none") + geom_hline(yintercept=396,linetype="dashed") + scale_y_continuous(limits=c(0,400),expand=c(0,0))
usf3 <- ggplot(dat4,aes(x=listnum,y=V1,color=method)) + geom_line(size=0.6) + theme_classic() + xlab("Number of psuedo-participants") + ylab("False alarms") + theme(legend.position="none") + scale_y_continuous(limits=c(0,1000),expand=c(0,0))

usf_legend <- ggplot(dat4,aes(x=listnum,y=V1,color=Method)) + geom_line(size=0.6) + theme_classic() + xlab("Number of psuedo-participants") + ylab("False alarms")
usf_legend <- g_legend(usf_legend) 

ggsave("usf_fit.eps",arrangeGrob(usf1,usf2,usf3,usf_legend,ncol=4),units="in",width=11)
