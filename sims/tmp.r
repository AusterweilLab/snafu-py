library(data.table)
library(ggplot2)
library(reshape2)

dat<-fread('tmp.csv')
newdat<-dat[,.(rw=mean(cost_rw),invite=mean(cost_noirts),irt5=mean(cost_irt5),irt9=mean(cost_irt9)),by=numx]
newdat2<-melt(newdat,id.vars=c("numx"))
colnames(newdat2)<-c("numx","Method","cost")
newdat2[,Method:=toupper(Method)]
newdat2[Method=="INVITE",Method:="U-INVITE"]
plotbasic <- theme_bw() + theme_classic(base_size=18) + theme(legend.key=element_blank(), plot.title=element_text(face="bold"))
ggplot(newdat2,aes(x=numx,y=cost,group=Method,color=Method)) + geom_line(size=2) + plotbasic + xlab("Number of lists") + ylab("Cost") + scale_y_continuous(expand=c(0,0), limits=c(0,30)) + theme(legend.position = c(.8,.8)) 
