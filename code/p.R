dogedata <- data.frame(DOGE[,7], DOGE_USD[,6])
colnames(dogedata)<-c("doge_tweets","doge_price")

ltcdata<- data.frame(as.factor(LTC[,7]), as.factor(LTC_USD[,6]))
colnames(ltcdata)<-c("ltc_tweets","ltc_price")

data <- cbind(ltcdata,dogedata)

#incomplete data suppression
for(i in 1:nrow(data)){
  dat <- data[i,]
  for (y in 1:length(dat)){
    if (dat[y]=="null"){
      data <- data[-i,]
    }
  }
}
data<- data.frame(as.numeric(data[,1]),as.numeric(data[,2]),as.numeric(data[,3]),as.numeric(data[,4]))
colnames(data)<-c("ltc_tweets","ltc_price","doge_tweets","doge_price")
#muinther usage
pearson <- pearsontable(data)
loop(data,1,4)

var_1 <- colnames(data)
for (i in 0:3) {
  y = i + 1
  entropy_outputs[, 1:2] <- replace(entropy_outputs[, 1:2],entropy_outputs[, 1:2]==i,var_1[y])
}

heat <- heatmap2(entropy_outputs)


#outputs

ggpubr::ggarrange(pearson[[1]], heat, ncol = 2, labels = c("A", "B"))
