colnames(DOGE)<-c("date","price","tweets")
colnames(LTC)<-c("date","price","tweets")

DOGE <- cbind(DOGE[,1],DOGEUSD[,2],DOGE[,8])
DOGE <- as.data.frame(DOGE)

LTC <- cbind(LTC[,1],LTCUSD[,2],LTC[,8])
LTC <- as.data.frame(LTC)

Doge11<- DOGE
LTC11 <- LTC
DOGE <- as.matrix(Doge11)

#dogecoin autoarima

##Non stationnary doge forecast
doge_general<-plot(DOGE)
DOG <- DOGE
DOGE<-cbind(as.numeric(DOGE[,1]),as.numeric(DOGE[,2]),as.numeric(DOGE[,3]))

arimadoge <- forecast::auto.arima(DOGE[,2], xreg = as.matrix(DOGE[,-2]))
fcdoge <- forecast::forecast(arimadoge, level = c(99), h = 36, xreg = as.matrix(DOGE[,-2]))
forecastnondoge<-forecast::autoplot(fcdoge)
forecastnondoge[["labels"]][["title"]]<-"A. Dogecoin non-stationary with ARIMA(2,1,2)"
forecastnondoge[["labels"]][["y"]][["yvar"]]<- "Dogecoin price ($)"
forecastnondoge

##Stationnary doge forecast

logDOGE <- log(as.numeric(DOG[,2]))
DOG[,2]<-logDOGE
plot(DOG)
logiDOGE<-cbind(as.numeric(DOGE[,1]),as.numeric(logDOGE),as.numeric(DOGE[,3]))
colnames(logiDOGE)<-c("date","price","tweets")
arimalogdog <- forecast::auto.arima(logiDOGE[,2], xreg = as.matrix(logiDOGE[,-2]), stationary = TRUE)
fclogdog <- forecast::forecast(arimalogdog, level = c(99), h = 36, xreg = as.matrix(logiDOGE[,-2]))
forecastdoge<-forecast::autoplot(fclogdog)
forecastdoge[["labels"]][["title"]]<-"B. Dogecoin stationary with ARIMA(0,0,0)"
forecastdoge[["labels"]][["y"]][["yvar"]]<- "Dogecoin log price ($)"
forecastdoge
#litecoin autoarima

##non-stationary ltc forecast
plot(LTC)
LTC[,2]<-as.numeric(LTC[,2])/10
LTC<-cbind(as.numeric(LTC[,1]),as.numeric(LTC[,2]),as.numeric(LTC[,3]))
arimaltcnon <- forecast::auto.arima(LTC[,2], xreg = as.matrix(LTC[,-2]))
fcltcnon <- forecast::forecast(arimaltcnon, level = c(99), h = 1, xreg = as.matrix(LTC[,-2]))
forecastltcnon <- forecast::autoplot(fcltcnon, ts.connect = FALSE)
forecastltcnon[["labels"]][["title"]]<-"C. Litecoin non-stationary with ARIMA(1,1,2)"
forecastltcnon[["labels"]][["y"]][["yvar"]]<- "Litecoin price ($)"
forecastltcnon
##Stationary ltc forecast
logLTC <- log(as.numeric(LTC[,2]))
LTC1 <- LTC
LTC1[,2]<-logLTC
plot(LTC1)
logiTC<-cbind(as.numeric(LTC[,1]),as.numeric(logLTC),as.numeric(LTC[,3]))

colnames(logiTC)<-c("date","price","tweets")
arimaltc <- forecast::auto.arima(logiTC[,2], xreg = as.matrix(logiTC[,-2]), stationary = TRUE)
fcltc <- forecast::forecast(arimaltc, level = c(99), h = 1, xreg = as.matrix(logiTC[,-2]))
forecastltc <- forecast::autoplot(fcltc, ts.connect = FALSE,   conf.int = FALSE)
forecastltc[["labels"]][["title"]]<-"D. Litecoin stationary with ARIMA(0,0,0)"
forecastltc[["labels"]][["y"]][["yvar"]]<- "Litecoin log price ($)"
forecastltc

#Forecast outputs presentation
gridExtra::grid.arrange(forecastnondoge,forecastdoge,forecastltcnon, forecastltc, ncol=2, nrow = 2) 

#prediction average error 
errordogelog <- (forecastdoge[["layers"]][[2]][["data"]][["ymax"]]+forecastdoge[["layers"]][[2]][["data"]][["ymin"]])/2
errordogelog<- as.data.frame(errordogelog)
errordogelog <- errordogelog[1:2482,]
doge <- log(DOGE[,2])  
for (i in 1:2482){
  errordogelog[i] <- abs(errordogelog[i]-doge[i])/doge[i]
}
errordogelog <- average(errordogelog)


errordoge <- abs(forecastnondoge[["layers"]][[3]][["data"]][["yvar"]]-DOGE[,2])/-DOGE[,2]
errordoge <- (forecastnondoge[["layers"]][[2]][["data"]][["ymax"]]+forecastnondoge[["layers"]][[2]][["data"]][["ymin"]])/2
errordoge<- as.data.frame(errordoge)
errordoge <- errordoge[1:2482,]
doge <- DOGE[,2]  
for (i in 1:2482){
  errordoge[i] <- abs(errordoge[i]-doge[i])/doge[i]
}
errordoge <- mean(errordoge)

errorltc <- (forecastltcnon[["layers"]][[2]][["data"]][["ymax"]]+forecastltcnon[["layers"]][[2]][["data"]][["ymin"]])/2
errorltc<- as.data.frame(errorltc)
errorltc <- errorltc[1:2482,]
ltc <- LTC[,2]  
for (i in 1:2482){
  errorltc[i] <- abs(errorltc[i]-ltc[i])/ltc[i]
}
errorltc <- mean(errorltc)

errorltclog <- (forecastltc[["layers"]][[2]][["data"]][["ymax"]]+forecastltc[["layers"]][[2]][["data"]][["ymin"]])/2
errorltclog<- as.data.frame(errorltclog)
errorltclog <- errorltclog[1:2482,]
ltc <- log(LTC[,2])  
for (i in 1:2482){
  errorltclog[i] <- abs(errorltclog[i]-ltc[i])/ltc[i]
}
errorltclog <- mean(errorltclog)

models <- c("Dogecoin non-stationary with ARIMA(2,1,2)","Dogecoin stationary with ARIMA(0,0,0)","Litecoin non-stationary with ARIMA(1,1,2)","Litecoin stationary with ARIMA(0,0,0)")
errors <- c(errordoge,errordogelog,errorltc,errorltclog)
table<- cbind(models, errors)
colnames(table)<-c("ARIMA model","Average error of prediction (%)")
table <- as.data.frame(table)
formattable::formattable(table)
