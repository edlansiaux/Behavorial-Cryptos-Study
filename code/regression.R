DOGE <- data.frame(c(1:2499),DOGE_USD[,5],DOGE_tweets[,8] )
i=0
dat <- c()
for (i in 1:nrow(DOGE)){
  dat <- DOGE[i,]
    if (dat[2]=="null"){
      DOGE<-DOGE[-i,]
    }
}
LTC <- data.frame(c(1:2499), LTC_USD[,5],LTC_tweets[,8] )
i=0
dat <- c()
for (i in 1:nrow(LTC)){
  dat <- LTC[i,]
  if (dat[2]=="null"){
    LTC<-LTC[-i,]
  }
}

#Doge regression
time <- DOGE[,1]
price <- DOGE[,2]
tweet <- DOGE[,3]
dogedroite<-lm(price~time+tweet)
print(dogedroite)
##Trend coefficient
a1 <- coef(dogedroite)[1]
##Time coefficient
a2 <- coef(dogedroite)[2]
##Tweet coefficient
a3 <- mean(coef1[-c(1:2),])
##Prediction quality
###MANOVA
dogeanov<- anova(dogedroite)
formattable::formattable(dogeanov)
###statistical residues analysis
layout(matrix(1:4,2,2))
plot(dogedroite, main="Dogecoin residues analysis")
####Studentized residues
res <- rstudent(dogedroite)
plot(res,ylab="Studentized residues",xlab="",main="Dogecoin Student residues",ylim=c(-2.5,2.5))
abline(h=c(-2,0,2),lty=c(2,1,2),col=c("red","blue","red"))

#LTC regression
timel <- LTC[,1]
pricel <- LTC[,2]
tweetl <- LTC[,3]
ltcdroite<-lm(pricel~timel+tweetl)

##Trend coefficient
b1 <- coef(ltcdroite)[1]
##Time coefficient
b2 <- coef(ltcdroite)[2]
##Tweet coefficient
b3 <- mean(coef[-c(1:2),])
##Prediction quality
###MANOVA
ltcanov<- anova(ltcdroite)
formattable::formattable(ltcanov)
###statistical residues analysis
layout(matrix(1:4,2,2))
plot(ltcdroite, main="Litecoin residues analysis")
####Studentized residues
res <- rstudent(ltcdroite)
plot(res,ylab="Studentized residues",xlab="",main="Litecoin Student residues",ylim=c(-2.5,2.5))
abline(h=c(-2,0,2),lty=c(2,1,2),col=c("red","blue","red"))
