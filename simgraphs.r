library(data.table)
library(ggplot2)

dat <- fread('sim_methods.csv')

dat<- dat[,lapply(.SD,mean),by=.(method,listnum)]
ggplot(dat, aes(x=listnum,y=cost,color=method)) + geom_line()
