library(data.table)
library(ggplot2)

dat <- fread('irt_simulation_data.csv',header=TRUE)

irt <- dat[,lapply(.SD,mean),by=type]
irt_plot <- irt[,.SD,.SDcols=2:ncol(irt)]
y1 <- as.numeric(irt_plot[1])
y2 <- as.numeric(irt_plot[2])
len <- length(y1)

type <- c(rep("er",len),rep("ws",len))
plotdat <- data.table(type,x=rep(x=1:len,2),y=c(y1,y2))

ggplot(plotdat,aes(x=x,y=y,color=type)) + geom_line()
