library(data.table)
library(ggplot2)
library(gridExtra)

dat <- fread('sim_all_but_uinvite.csv')

dat<- dat[,lapply(.SD,mean),by=.(method,subjects)]
p1 <- ggplot(dat, aes(x=subjects,y=cost,color=method)) + geom_line()
p2 <- ggplot(dat, aes(x=subjects,y=hit,color=method)) + geom_line()
p3 <- ggplot(dat, aes(x=subjects,y=fa,color=method)) + geom_line()
grid.arrange(p1,p2,p3,ncol=3)
