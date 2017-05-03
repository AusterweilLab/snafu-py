library(data.table)
library(ggplot2)
library(gridExtra)

usf_hit <- 393
usf_cr <- 12327
usf_total <- 12720

dat <- fread('sim_all_but_uinvite.csv')
dat<- dat[,lapply(.SD,mean),by=.(method,subjects)]

dat[,hit_p := hit/usf_hit]
dat[,fa_p := fa/usf_cr]
dat[,cost_p := cost/usf_total]

p4 <- ggplot(dat, aes(x=subjects,y=cost_p,color=method)) + geom_line() + ylim(0,0.08)
p5 <- ggplot(dat, aes(x=subjects,y=hit_p,color=method)) + geom_line()
p6 <- ggplot(dat, aes(x=subjects,y=fa_p,color=method)) + geom_line()
grid.arrange(p4,p5,p6,ncol=3)

p1 <- ggplot(dat, aes(x=subjects,y=cost,color=method)) + geom_line() + ylim(0,1000)
p2 <- ggplot(dat, aes(x=subjects,y=hit,color=method)) + geom_line()
p3 <- ggplot(dat, aes(x=subjects,y=fa,color=method)) + geom_line()
grid.arrange(p1,p2,p3,ncol=3)

# roc curve
# ggplot(dat,aes(x=fa_p,y=hit_p,color=method)) + geom_line()
