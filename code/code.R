doge<-as.data.frame(read.csv('C:/Users/PATRICE LANSIAUX/Desktop/crypto/papers/doge VS LTC/data/doge.csv'))
doge<-doge[,-1]

ltc<-as.data.frame(read.csv('C:/Users/PATRICE LANSIAUX/Desktop/crypto/papers/doge VS LTC/data/ltc.csv'))
ltc<-ltc[,-1]

muinther::pearsontable(ltc)
muinther::pearsontable(doge)

entropy_outputs_DOGE <- as.data.frame(read.csv("I:/muinther/R/muinther/inst/temp/entropy_outputs.csv"))

loop(ltc,1,7)
loop(doge,1,7)

names<-colnames(ltc)

for (i in 0:6) {
  y = i + 1
  entropy_outputs_DOGE[, 1:2] <- replace(entropy_outputs_DOGE[, 1:2],entropy_outputs_DOGE[, 1:2]==i,names[y])
}

for (i in 0:6) {
  y = i + 1
  entropy_outputs_LTC[, 1:2] <- replace(entropy_outputs_LTC[, 1:2],entropy_outputs_LTC[, 1:2]==i,names[y])
}

heatmap2(entropy_outputs_LTC)
heatmap2(entropy_outputs_DOGE)

write.table(entropy_outputs_LTC, "C:/Users/PATRICE LANSIAUX/Desktop/crypto/papers/doge VS LTC/data/entropy_outputs_LTC.csv", row.names=FALSE, sep=",",dec=".", na=" ")
write.table(entropy_outputs_DOGE, "C:/Users/PATRICE LANSIAUX/Desktop/crypto/papers/doge VS LTC/data/entropy_outputs_DOGE.csv", row.names=FALSE, sep=",",dec=".", na=" ")
system.file("python","loop.py", package = "muinther"))

r<-as.numeric(z[,4])
p<-as.numeric(z[,3])
z<-cbind(z[,1:2],p,r)
