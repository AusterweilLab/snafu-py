library(R.matlab)
library(data.table)
library(ggplot2)

#allsubs<-c("S101","S102","S103","S104","S105","S106","S107","S108","S109","S110","S111","S112","S113","S114","S115","S116","S117","S118","S119","S120","S101","S102","S103","S104","S105","S106","S107","S108","S109","S110","S111","S112","S113","S114","S115","S116","S117","S118","S119","S120")
allsubs <-  sapply(seq(100,150),function(i) { paste("S",i,sep="")[1] })

# human data
humans<-fread('../humans2017.csv')
setkey(humans,subj)

# MATLAB files
rawmat<-readMat('../snet/BEAGLEdata.mat')
beagle<-rawmat[5]$BEAGLE.sim
lbls<-rawmat[4]$BEAGLE.labels
lbls<-unlist(lbls)
cats<-rawmat$BEAGLE.animalClassesIdxMatrix

checkBeagle <- function(word1,word2) {
    word1<-toupper(word1)
    word2<-toupper(word2)
    idx1<-match(word1,lbls)
    idx2<-match(word2,lbls)
    beagle[idx1,idx2]
}

# extract BEAGLE values and add to data table
loadBEAGLEvals <- function() {
    beaglevals<-c()
    for (i in seq(nrow(humans))) {
        node1<-toupper(humans[i,node1])
        node2<-toupper(humans[i,node2])
        idx1<-match(node1,lbls)
        idx2<-match(node2,lbls)

        if (is.na(idx1) | is.na(idx2)) {
            beaglevals<-c(beaglevals,NA)
        } else {
            beaglevals<-c(beaglevals,beagle[idx1,idx2])
        }
    }
    # append beagle values to human data
    humans[,beagle:=beaglevals]
}

# use BEAGLE x Troyer category matrix (771x22) to see if links share a category
sameCats <- function() {
    catvals<-c()
    for (i in seq(nrow(humans))) {
        node1<-toupper(humans[i,node1])
        node2<-toupper(humans[i,node2])
        idx1<-match(node1,lbls)
        idx2<-match(node2,lbls)
        sharecat<-any(cats[idx1,]>0 & cats[idx2,]>0)
        catvals<-c(catvals,sharecat)
    }
    humans[,sharecat:=catvals]
}

# Find avg BEAGLE value of all possible link combos (per participant)
beagleRand <- function() {
    for (sb in unique(humans[,subj])) {
        nodes<-unique(c(humans[sb,][,node1],humans[sb,][,node2]))
        nodeidx<-unlist(lapply(nodes, function(node) { match(toupper(node),lbls) }))
        nodeidx<-nodeidx[!is.na(nodeidx)]
        combos<-combn(nodeidx,2)
        beaglevals<-c()
        for (i in seq(ncol(combos))) {
            beaglevals<-c(beaglevals,beagle[combos[1,i],combos[2,i]])
        }
        humans[sb,beaglerand:=mean(beaglevals)]
    }
}

# probability of two random nodes in graph sharing a category
catRand <- function() {
    for (sb in unique(humans[,subj])) {
        nodes<-unique(c(humans[sb,][,node1],humans[sb,][,node2]))
        nodeidx<-unlist(lapply(nodes, function(node) { match(toupper(node),lbls) }))
        nodeidx<-nodeidx[!is.na(nodeidx)]
        combos<-combn(nodeidx,2)
        catvals<-c()
        for (i in seq(ncol(combos))) {
            sharecat<-any(cats[combos[1,i]]>0 & cats[combos[2,i]]>0)
            catvals<-c(catvals,sharecat)
        }
        humans[sb,catrand:=mean(catvals)]
    }
}

loadBEAGLEvals()
beagleRand()
humans[,beaglediff:=beagle-beaglerand]

sameCats()
catRand()

humans2<-melt(humans,id.vars=c("subj","node1","node2","beaglediff","beaglerand","beagle","catrand","sharecat"))

# BEAGLE test: INVITE vs IRT
t.test(humans[uinvite==0 & irt==1,mean(beaglediff,na.rm=T),by=subj][,V1])  # adds quality links!
t.test(humans[uinvite==1 & irt==0,mean(beaglediff,na.rm=T),by=subj][,V1])  # removes quality links :(
t.test(humans[uinvite==1 & irt==0,mean(beaglediff,na.rm=T),by=subj][,V1],humans[uinvite==0 & irt==1,mean(beaglediff,na.rm=T),by=subj][,V1])

# Troyer test: INVITE vs IRT
t.test(humans[uinvite==1,mean(sharecat,na.rm=T),keyby=subj][,V1],humans[irt==1,mean(sharecat,na.rm=T),keyby=subj][,V1],paired=T)



#> humans2[value==1,.(.N,mean(beaglediff,na.rm=T)),keyby=variable]
#   variable    N          V2
#1:       rw 2784 0.078526046
#2:       fe   73 0.139276736
#3:     goni  623 0.105313114
#4:     chan  439 0.086944563
#5:   kenett 1083 0.004109446

#> humans2[value==1,.(.N,mean(sharecat,na.rm=T)),keyby=variable]
#   variable    N        V2
#1:       rw 2784 0.3397004
#2:       fe   73 0.3972603
#3:     goni  623 0.6900000
#4:     chan  439 0.5780822
#5:   kenett 1083 0.1132300

# ggplot(humans2[beaglediff!="NA" & value==1],aes(beaglediff,color=variable)) + geom_density()
# > humans2[,.(.N,mean(beaglediff,na.rm=T)),keyby=.(value,variable)]
