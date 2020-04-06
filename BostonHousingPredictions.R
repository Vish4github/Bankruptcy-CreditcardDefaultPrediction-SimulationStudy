
# Libraries ---------------------------------------------------------------
library(MASS)
library(leaps)
library(rpart)
library(rpart.plot)
library(randomForest)
library(gbm)
library(dplyr)
library(ROCR)
library(ipred)
library('neuralnet')
library(mgcv)

# Dataset -----------------------------------------------------------------

data(Boston)

# Sampling for train and test ---------------------------------------------

set.seed(13255870)
index <- sample(nrow(Boston),nrow(Boston)*0.70)
boston.train <- Boston[index,]
boston.test <- Boston[-index,]

View(head(boston.train))
summary(boston.train)

View(cor(boston.train[,!names(boston.train) %in% c("medv")], boston.train$medv))

# Linear Regression -------------------------------------------------------

Boston.lm<-lm(medv~.,data=boston.train)
summary(Boston.lm)



# Best regression selection -----------------------------------------------------

Boston.bck<-step(Boston.lm,direction='backward',k=log(nrow(boston.train)))
#Boston.bck<-step(Boston.lm,direction='backward',k=2)#10
Boston.nm<-lm(medv~1,data = boston.train)
Boston.fwd<-step(Boston.nm,scope=list(lower=Boston.nm,upper=Boston.lm),direction = 'forward') #11
Boston.stepwise<-step(Boston.nm,scope=list(lower=Boston.nm,upper=Boston.lm),direction = 'both') #11
Boston.bestsub<-regsubsets(medv~.,data=boston.train,nbest=1,nvmax=15,method=c('exhaustive'))
summary(Boston.bestsub)
summary(Boston.bck)

BIC(Boston.bck)
#medv ~ crim + zn + nox + rm + dis + rad + tax + ptratio + black + lstat

# In sample MSE -----------------------------------------------------------

Boston.bck.insample.pred<-predict(Boston.bck,newdata = boston.train)
mean((boston.train$medv-Boston.bck.insample.pred)^2)


# Out of sample prediction ------------------------------------------------

Boston.bck.outsample.pred<-predict(Boston.bck,newdata = boston.test)
mean((boston.test$medv-Boston.bck.outsample.pred)^2)

summary(Boston.bck)


#Trees
# Simple Regression Tree ---------------------------------------------------------

Boston.rpart<-rpart(medv~.,data=boston.train)
prp(Boston.rpart,digits=4,extra=1)

Boston.tree.insample.pred<-predict(Boston.rpart,newdata = boston.train)
mean((boston.train$medv-Boston.tree.insample.pred)^2)

Boston.tree.outsample.pred<-predict(Boston.rpart,newdata = boston.test)
mean((boston.test$medv-Boston.tree.outsample.pred)^2)

Boston.largetree<-rpart(medv~.,data=boston.train,cp=0.001)
prp(Boston.largetree,digits=4,extra=1)
plotcp(Boston.largetree)
printcp(Boston.largetree)

Boston.treepruned<-prune(Boston.largetree,cp=0.0077434)
prp(Boston.treepruned,digits=4,extra=1)


# In sample prediction and MSE ----------------------------------------------------

Boston.treepruned.insample.pred<-predict(Boston.treepruned,newdata = boston.train)
mean((boston.train$medv-Boston.treepruned.insample.pred)^2)


# Out of sample prediction and MSE ----------------------------------------

Boston.treepruned.outsample.pred<-predict(Boston.treepruned,newdata = boston.test)
mean((boston.test$medv-Boston.treepruned.outsample.pred)^2)


# Bagging -----------------------------------------------------------------

Boston.bag<- randomForest(medv~., data = boston.train, ntree=100,mtry=ncol(boston.train)-1) 
#mtry= differentiating factor between RF and Bagging
#Bagging - mtry = no of predictors randomly selected in Random forests , in bagging all predcitors are selected
#ntree = bootstrap samples
Boston.bag$mse


# In sample prediction and MSE --------------------------------------------

Boston.bag.insample.pred<-predict(Boston.bag,newdata = boston.train)
mean((boston.train$medv-Boston.bag.insample.pred)^2)


# Out of sample prediction and MSE ----------------------------------------

Boston.bag.outsample.pred<- predict(Boston.bag, newdata = boston.test)
mean((boston.test$medv-Boston.bag.outsample.pred)^2)

# Out of bag prediction ---------------------------------------------------
Boston.bag$mse[100]  #out of bag prediction same as insample


# Variation of MSE with the number of trees -------------------------------

ntree<-c(1,3,5,seq(10,200,10)) #no of trees to try
MSE.test<-rep(0,length(ntree))

for(i in 1:length(ntree))
{
  boston.bag1<- randomForest(medv~., data = boston.train, ntree=ntree[i],mtry=ncol(boston.train)-1) #mtry= differentiating factor between RF and Bagging
  boston.bag.pred1<- predict(boston.bag1, newdata = boston.test)
  MSE.test[i]<- mean((boston.test$medv-boston.bag.pred1)^2)
}

MSE.test
plot(ntree, MSE.test, type = 'l', col=2, lwd=2, xaxt="n")
axis(1, at = ntree, las=1)  #ntree is not treated as atuning parameter


# Random Forest -----------------------------------------------------------

Boston.rf<- randomForest(medv~., data = boston.train, importance=TRUE,ntree=500,mtry=floor((ncol(boston.train)-1)/3)) 
#decorrelate the tree - mtry = choose (ncol-1)/3 predictor variables 
# always p/3 to be used
Boston.rf

Boston.rf$importance
varImpPlot(Boston.rf)

plot(Boston.rf$mse, type='l', col=2, lwd=2, xlab = "ntree", ylab = "OOB Error")


# Insample prediction and MSE ---------------------------------------------

Boston.rf.insample.pred<- predict(Boston.rf)
mean((boston.train$medv-Boston.rf.insample.pred)^2)


# Out of sample prediction and MSE --------------------------------------------
Boston.rf.outsample.pred<- predict(Boston.rf, newdata = boston.test)
mean((boston.test$medv-Boston.rf.outsample.pred)^2)

#specify the mtry values 

oob.err<- rep(0, 13)
test.err<- rep(0, 13)
for(i in 1:13){
  fit<- randomForest(medv~., data = boston.train, mtry=i)
  oob.err[i]<- fit$mse[500]  #default no of trees in RF = 500
  test.err[i]<- mean((boston.test$medv-predict(fit, boston.test))^2)
  cat(i, " ")
}

matplot(cbind(test.err, oob.err), pch=15, col = c("red", "blue"), type = "b", ylab = "MSE", xlab = "mtry")
legend("topright", legend = c("test Error", "OOB Error"), pch = 15, col = c("red", "blue"))


# Boosting ----------------------------------------------------------------

Boston.boost<- gbm(medv~., data = boston.train, distribution = "gaussian", n.trees = 10000, shrinkage = 0.01, 
                   interaction.depth = 8)
#gaussian for continuous response
#interaction depth = how deep each tree is
# shrinkage = learning rate(each step)
# n trees = 10000  
summary(Boston.boost)
mean(Boston.boost$train.error)


par(mfrow=c(1,1))
plot(Boston.boost, i="lstat")
plot(Boston.boost, i="rm")



# In sample prediction and MSE --------------------------------------------

Boston.boost.insample.pred<- predict(Boston.boost,newdata = boston.train, n.trees = 10000)
mean((boston.train$medv-Boston.boost.insample.pred)^2)


# Prediction on testing sample --------------------------------------------

Boston.boost.outsample.pred<- predict(Boston.boost, newdata=boston.test, n.trees = 10000)
mean((boston.test$medv-Boston.boost.outsample.pred)^2)


# No of trees - can be used to see if we get stuck at a local minima-------------------------------------------------------------
ntree<- seq(100, 10000, 100)
predmat<- predict(Boston.boost, newdata = boston.test, n.trees = ntree)
err<- apply((predmat-boston.test$medv)^2, 2, mean)
plot(ntree, err, type = 'l', col=2, lwd=2, xlab = "n.trees", ylab = "Test MSE")
abline(h=min(test.err), lty=2)





# Nueral Nets -------------------------------------------------------------

#install.packages('neuralnet',repos='http://cran.us.r-project.org')

data(Boston)

maxs <- apply(Boston, 2, max) 
mins <- apply(Boston, 2, min)

#response is also scaled? - needs to be in 0 to  1 range
set.seed(13255870)
scaled <- as.data.frame(scale(Boston, center = mins, scale = maxs - mins))

indexnn<-sample(nrow(Boston),nrow(Boston)*0.7)
Boston.train<-scaled[indexnn,]
Boston.test<-scaled[-indexnn,]
boston.testn<-Boston[-indexnn,]
boston.testn$medv


names<-colnames(Boston.train)
predictors<-names[!names%in%"medv"]
predictor_vars<-paste(predictors,collapse = '+')
form<-as.formula(paste("medv~",predictor_vars,collapse = '+'))
#neuralnet doesnt allow medv~. like others, say lm()

nn<-neuralnet(form,data=Boston.train,hidden=c(5,3),linear.output=T)
#hidden layer = 2 , first with 5 nuerons and second with 3

plot(nn)


# In sample ---------------------------------------------------------------
pr.nn_train <- compute(nn,Boston.train[,1:13])
predictions_train<-pr.nn_train$net.result*(max(Boston$medv)-min(Boston$medv))+min(Boston$medv)
actualValues_train<-(Boston.train$medv)*((max(Boston$medv)-min(Boston$medv))+min(Boston$medv))
MSE.nn_train <- sum((actualValues_train - predictions_train)^2)/nrow(Boston.train)
MSE.nn_train







pr.nn <- compute(nn,Boston.test[,1:13])
pr.nn$net.result  #probabilities

predictions<-pr.nn$net.result*(max(Boston$medv)-min(Boston$medv))+min(Boston$medv)
#unscaling

actualValues<-(Boston.test$medv)*((max(Boston$medv)-min(Boston$medv))+min(Boston$medv))

MSE.nn <- sum((actualValues - predictions)^2)/nrow(Boston.test)
MSE.nn


68# Randomly selecting the starting points to make sure the convergence is global --------------------------------------------------------


set.seed(13255870)
cv.error <- NULL
cv.error_train<-NULL
k <- 10

library(plyr) 
pbar <- create_progress_bar('text')
pbar$init(k)

for(i in 1:k){
  index <- sample(1:nrow(Boston),round(0.9*nrow(Boston)))
  train.cv <- scaled[index,]
  test.cv <- scaled[-index,]
  
  nn <- neuralnet(form,data=train.cv,hidden=c(5,3),linear.output=T)
  
  
  pr.nn_train <- compute(nn,train.cv[,1:13])
  predictions_train<-pr.nn_train$net.result*(max(Boston$medv)-min(Boston$medv))+min(Boston$medv)
  actualValues_train<-(train.cv$medv)*((max(Boston$medv)-min(Boston$medv))+min(Boston$medv))
  cv.error_train[i] <- sum((actualValues_train - predictions_train)^2)/nrow(train.cv)
  
  pr.nn <- compute(nn,test.cv[,1:13])
  pr.nn <- pr.nn$net.result*(max(Boston$medv)-min(Boston$medv))+min(Boston$medv)
  test.cv.r <- (test.cv$medv)*(max(Boston$medv)-min(Boston$medv))+min(Boston$medv)
  cv.error[i] <- sum((test.cv.r - pr.nn)^2)/nrow(test.cv)
  pbar$step()
}

mean(cv.error)
mean(cv.error_train)
cv.error
cv.error_train



# CV for lm() -------------------------------------------------------------

library(boot)
set.seed(200)
lm.fit <- glm(medv~.,data=Boston)
cv.glm(Boston,lm.fit,K=10)$delta[1]



# GAM ---------------------------------------------------------------------


gam_boston<-gam(medv ~ s(crim)+s(zn)+s(indus)+chas+s(nox)
                +s(rm)+s(age)+s(dis)+rad+s(tax)+s(ptratio)
                +s(black)+s(lstat),data=boston.train)

summary(gam_boston)

gam_boston_correct<-gam(medv ~ s(crim)+s(zn)+s(indus)+chas+s(nox)
                        +s(rm)+age+s(dis)+rad+s(tax)+s(ptratio)
                        +black+s(lstat),data=boston.train)

summary(gam_boston_correct)


# Insample Performance ----------------------------------------------------

train_pred<- predict(gam_boston_correct)
test_pred<- predict(gam_boston_correct,newdata = boston.test)

MSE_out<-sum((test_pred - boston.test$medv)^2)/nrow(boston.test)
MSE_in<-sum((train_pred - boston.train$medv)^2)/nrow(boston.train)

#MSE_out_2<-mean((test_pred -boston.test$medv)^2)

AIC(gam_boston_correct)
BIC(gam_boston_correct)
gam_boston_correct$deviance

