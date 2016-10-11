library(data.table)

dat<-fread('all.csv')
usf<-fread('../../USF_animal_subset.csv',header=F)
usfprime<-fread('../../USF_prime.csv',header=F)
animals<-fread('../../USFanimals.csv',header=F)

# some pairs can't be verified because one of the animals is not in USF
dat[,animalsinusf:=0]

for (row in seq(nrow(dat))) {
    node1<-dat[row,node1]
    node2<-dat[row,node2]

    # look for pair both ways (undirected graph)
    a<-nrow(usf[V1==node1 & V2==node2])
    b<-nrow(usf[V1==node2 & V2==node1])
    dat[row,inusf:=(a+b)>0]     # is edge in USF network?

    # look for pair both ways (undirected graph) in USF'
    a<-nrow(usfprime[V1==node1 & V2==node2])
    b<-nrow(usfprime[V1==node2 & V2==node1])
    dat[row,inusfprime:=(a+b)>0]     # is edge in USF network?

    a<-nrow(animals[V1==node1])
    b<-nrow(animals[V1==node2])
    ab<-a+b
    dat[row,animalsinusf:=(ab==2)]  # are both animals in USF network (is edge verifiable?)
    
}
