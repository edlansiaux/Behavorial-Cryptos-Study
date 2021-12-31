LTCUSD <-  LTC_USD[,c(1,6)]
DOGEUSD <-  DOGE_USD[,c(1,6)]

doge_usd <- DOGEUSD[,2]
ltc_usd <- LTCUSD[,2]


#ADF stats for non-stationary data

## DOGE non-stationary data ADF
doge_usd <- as.numeric(doge_usd)
doge_usd <- doge_usd[!is.na(doge_usd)]
nlagdoge = floor(4*(length(doge_usd)/100)^(2/9))
adfdoge <- aTSA::adf.test(doge_usd, nlag=nlagdoge)
view(adfdoge)

##LTC non-stationary data ADF
ltc_usd <- as.numeric(ltc_usd)
ltc_usd <- ltc_usd[!is.na(ltc_usd)]
nlagltc = floor(4*(length(ltc_usd)/100)^(2/9))
adfltc <- aTSA::adf.test(ltc_usd, nlag=nlagltc)

##SUM UP OUTPUTS IN A SAME csv
### DOGE CSV
adfdoge_csv<-cbind(adfdoge[["type1"]], adfdoge[["type2"]], adfdoge[["type3"]])
colnames(adfdoge_csv)<-c("type1_lag","type1_ADF","type1_p.value","type2_lag","type2_ADF","type2_p.value","type3_lag","type3_ADF","type3_p.value")

###LTC CSV
adfltc_csv<-cbind(adfltc[["type1"]], adfltc[["type2"]], adfltc[["type3"]])
colnames(adfltc_csv)<-c("type1_lag","type1_ADF","type1_p.value","type2_lag","type2_ADF","type2_p.value","type3_lag","type3_ADF","type3_p.value")


#ADF stats for stationary data
##DOGE stationary data ADF
doge_usd_log <- log(doge_usd)
nlagdogelog = floor(4*(length(doge_usd_log)/100)^(2/9))
adfdoge_non <- aTSA::adf.test(doge_usd_log, nlag=nlagdogelog)

##LTC stationary data ADF
ltc_usd_log <- log(ltc_usd)
nlagltclog = floor(4*(length(ltc_usd_log)/100)^(2/9))
adfltc_non <- aTSA::adf.test(ltc_usd_log, nlag=nlagltclog)

##CSV sum up
### DOGE CSV
adfdoge_non_csv<-cbind(adfdoge_non[["type1"]], adfdoge_non[["type2"]], adfdoge_non[["type3"]])
colnames(adfdoge_non_csv)<-c("type1_lag","type1_ADF","type1_p.value","type2_lag","type2_ADF","type2_p.value","type3_lag","type3_ADF","type3_p.value")

###LTC CSV
adfltc_non_csv<-cbind(adfltc_non[["type1"]], adfltc_non[["type2"]], adfltc_non[["type3"]])
colnames(adfltc_non_csv)<-c("type1_lag","type1_ADF","type1_p.value","type2_lag","type2_ADF","type2_p.value","type3_lag","type3_ADF","type3_p.value")

#OUTPUTS
##Means for each parameters
###DOGE non-stationary data
i = 0 
mean1 <- c()
while (i < ncol(as.data.frame(adfdoge_csv))+1){
  mean1 <- cbind(mean1,mean(adfdoge_csv[,i]))
  i = i+1
}
mean1 <- mean1[-1]
mean1[c(1,4,7)]  <- gsub(3.5, "mean", mean1[c(1,4,7)])
adfdoge_csv <- rbind(adfdoge_csv,mean1)
write.csv(x = adfdoge_csv, file = "adfdoge.csv")

###DOGE stationary data
i = 0 
mean2 <- c()
while (i < ncol(as.data.frame(adfdoge_non_csv))+1){
  mean2 <- cbind(mean2,mean(adfdoge_non_csv[,i]))
  i = i+1
}
mean2 <- mean2[-1]
mean2[c(1,4,7)]  <- gsub(3.5, "mean", mean2[c(1,4,7)])
adfdoge_non_csv <- rbind(adfdoge_non_csv,mean2)
write.csv(x = adfdoge_non_csv, file = "adfdoge_non.csv")

### LTC non-stationary data
i = 0 
mean3 <- c()
while (i < ncol(as.data.frame(adfltc_csv))+1){
  mean3 <- cbind(mean3,mean(adfltc_csv[,i]))
  i = i+1
}
mean3 <- mean3[-1]
mean3[c(1,4,7)]  <- gsub(3.5, "mean", mean3[c(1,4,7)])
adfltc_csv <- rbind(adfltc_csv,mean3)
write.csv(x = adfltc_csv, file = "adfltc.csv")

###LTC stationary data
i = 0 
mean4 <- c()
while (i < ncol(as.data.frame(adfltc_non_csv))+1){
  mean4 <- cbind(mean4,mean(adfltc_non_csv[,i]))
  i = i+1
}
mean4 <- mean4[-1]
mean4[c(1,4,7)]  <- gsub(3.5, "mean", mean4[c(1,4,7)])
adfltc_non_csv <- rbind(adfltc_non_csv,mean4)
write.csv(x = adfltc_non_csv, file = "adfltc_non.csv")

##Final table
final<-rbind(mean1,mean2,mean3,mean4)
colnames(final)<-c("type1_lag","type1_ADF","type1_p.value","type2_lag","type2_ADF","type2_p.value","type3_lag","type3_ADF","type3_p.value")
final<-final[,c(2,3,5,6,8,9)]
rownames(final)<-c("Doge non-stationary data","Doge stationary data","LTC non-stationary data", "LTC stationary data")
write.csv(x = final, file = "final table.csv")
formattable::formattable(as.data.frame(final))

##Trends chart
p <- ggplot2::theme(
  plot.title = ggplot2::element_text(color="black", size=10, face="bold"),
  axis.title.x = ggplot2::element_text(color="black", size=8, face="italic"),
  axis.title.y = ggplot2::element_text(color="black", size=8, face="italic")
)

###Non-stationary data chart
time <- as.numeric(c(1:length(ltc_usd))) 
doge_usd <- doge_usd[,-c(2,3)]
doge_usd <- cbind(doge_usd, time, "dogecoin")
ltc_usd <- cbind(ltc_usd, time, "litecoin")
data <- rbind(doge_usd,ltc_usd)
colnames(data)<-c("price","time","coin")
data <- data.frame(as.numeric(data[,1]),as.numeric(data[,2]),data[,3])
chart1 <- ggplot2::ggplot(data, ggplot2::aes(x=data[,2], y=data[,1], fill=data[,3], color=data[,3])) + ggplot2::geom_area() + ggplot2::geom_point() + ggplot2::ylim(0,400) + ggplot2::xlim(1,2483) + ggplot2::labs(x="Time (day)") + ggplot2::labs(fill="Coin") + ggplot2::labs(y="Price ($)") + ggplot2::labs(color="Coin") + ggplot2::ggtitle("Trend analysis of non-stationary data")
chart2 <- ggplot2::ggplot(data, ggplot2::aes(x=data[,2], y=data[,1], fill=data[,3], color=data[,3])) + ggplot2::geom_area() + ggplot2::geom_point() + ggplot2::xlim(1,2483) +  ggplot2::scale_y_sqrt()+ ggplot2::labs(x="Time (day)") + ggplot2::labs(fill="Coin") + ggplot2::labs(y="Price ($)") + ggplot2::labs(color="Coin") + ggplot2::ggtitle("Trend analysis of non-stationary data with Y square scale")
chart1 <- chart1 + p
chart2 <- chart2 + p
###stationary data chart
doge_usd_log <- cbind(doge_usd_log, time, "dogecoin")
ltc_usd_log <- cbind(ltc_usd_log, time, "litecoin")
data2 <- rbind(doge_usd_log,ltc_usd_log)
data2 <- data.frame(as.numeric(data2[,1]),as.numeric(data2[,2]),data2[,3])
chart3 <- ggplot2::ggplot(data2, ggplot2::aes(x=data2[,2], y=data2[,1], fill=data2[,3], color=data2[,3])) + ggplot2::geom_area() + ggplot2::geom_point() + ggplot2::labs(x="Time (day)") + ggplot2::labs(fill="Coin") + ggplot2::labs(y="Price ($)") + ggplot2::labs(color="Coin") + ggplot2::ggtitle("Trend analysis of stationary data")
chart3 <- chart3 + p
###Final chart output
figure <- ggpubr::ggarrange(chart1, chart2, chart3, 
          labels = c("A.", "B.", "C."),
          font.label = list(size = 10, color = "black", face = "bold", family = NULL),
          ncol = 1, nrow = 3)
ggpubr::annotate_figure(figure, fig.lab="Figure 3. PRICE TREND ANALYSIS", fig.lab.face = "bold", fig.lab.pos="top", fig.lab.size = 12)
