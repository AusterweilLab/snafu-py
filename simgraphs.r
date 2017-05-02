library(data.table)
library(ggplot2)
library(gridExtra)

usf_hit <- 393
usf_cr <- 12720

dat <- fread('sim_all_but_uinvite.csv')
dat<- dat[,lapply(.SD,mean),by=.(method,subjects)]

dat[,hit_p := hit/usf_hit]
dat[,fa_p := fa/usf_cr]

p1 <- ggplot(dat, aes(x=subjects,y=cost,color=method)) + geom_line()
p2 <- ggplot(dat, aes(x=subjects,y=hit,color=method)) + geom_line()
p3 <- ggplot(dat, aes(x=subjects,y=fa,color=method)) + geom_line()
grid.arrange(p1,p2,p3,ncol=3)

# roc curve
# ggplot(dat,aes(x=fa_p,y=hit_p,color=method)) + geom_line()
