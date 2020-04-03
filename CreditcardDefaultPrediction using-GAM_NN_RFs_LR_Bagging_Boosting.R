# Classification ----------------------------------------------------------

# Libraries ---------------------------------------------------------------
library(MASS)
library('boot')
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
library(nnet)


# load credit card data
credit.data <- read.csv("C:/Users/Vishnu/Desktop/Spring 2020/Data Mining/Data Mining 1/credit_default.csv", header=T)
# convert categorical variables
credit.data$SEX<- as.factor(credit.data$SEX)
credit.data$EDUCATION<- as.factor(credit.data$EDUCATION)
credit.data$MARRIAGE<- as.factor(credit.data$MARRIAGE)
credit.data<-rename(credit.data,default=default.payment.next.month)
colnames(credit.data)
credit.data$default<- as.factor(credit.data$default)



# random splitting
index <- sample(nrow(credit.data),nrow(credit.data)*0.70)
credit.train = credit.data[index,]
credit.test = credit.data[-index,]


# Logistic model with all X variables  ------------------------------------

credit.glm<-glm(default~.,family = binomial,data = credit.train)
summary(credit.glm)
credit.glm$coefficients

#credit.glm_probit<-glm(default~.,family = binomial(link = 'probit'),data = credit.train)
#summary(credit.glm_probit)
#credit.glm_probit$coefficients

#credit.glm_cloglog<-glm(DLRSN~.,family = binomial(link = 'cloglog'),data = bankruptcy.train)
#summary(credit.glm_cloglog)
#credit.glm_cloglog$coefficients

# Best models with AIC, BIC and LASSO --------------------------------------------

credit.glm.back <- step(credit.glm)  #k=2
summary(credit.glm.back)

credit.glm.back.BIC <- step(credit.glm, k=log(nrow(credit.train))) 
Finalmodel<-summary(credit.glm.back.BIC)
Finalmodel$deviance
BIC(credit.glm.back.BIC)

library(car)
vif(credit.glm)

credit.glm.back$deviance
credit.glm.back.BIC$deviance


nrow(credit.train)

costfunc = function(obs, pred.p, pcut)
{
  weight1 = 5   # define the weight for "true=1 but pred=0" (FN) placing weight on FN such that we need to avoid cases wherewe predict it wont default but it has in the actual data
  weight0 = 1    # define the weight for "true=0 but pred=1" (FP)
  c1 = (obs==1)&(pred.p<pcut) # count for "true=1 but pred=0"   (FN)
  #print(sum(c1))
  c0 = (obs==0)&(pred.p>=pcut)   # count for "true=0 but pred=1"   (FP)
  #print (sum(c0))
  cost = mean(weight1*c1 + weight0*c0)  # misclassification with weight
  return(cost) # you have to return to a value when you write R functions
} 



# Define a seq, need to search for opt cut off from this ------------------

pred.glm.credittrain<-predict(credit.glm.back.BIC,credit.train,type='response')

p.seq = seq(0.01, 1, 0.01) 
cost = rep(0, length(p.seq))  
for(i in 1:length(p.seq)){ 
  cost[i] = costfunc(obs = credit.train$default, pred.p = pred.glm.credittrain, pcut = p.seq[i])  
}

min(cost)
plot(p.seq, cost)

optimal.pcut.glm0 = p.seq[which(cost==min(cost))]

class.glm0.train.opt<- (pred.glm.credittrain>optimal.pcut.glm0)*1
table(credit.train$default, class.glm0.train.opt, dnn = c("True", "Predicted"))

MR<- mean(credit.train$default!= class.glm0.train.opt)
sum(credit.train$default!= class.glm0.train.opt)
FPR<- sum(credit.train$default==0 & class.glm0.train.opt==1)/sum(credit.train$default==0)
FNR<- sum(credit.train$default==1 & class.glm0.train.opt==0)/sum(credit.train$default==1)
cost<- costfunc(obs = credit.train$default, pred.p = pred.glm.credittrain, pcut = optimal.pcut.glm0)  



# Out of sample prediction  -----------------------------------------------

pred.glm.credittest<-predict(credit.glm.back.BIC,newdata=credit.test,type='response')
class.glm0.test.opt<-(pred.glm.credittest>optimal.pcut.glm0)*1  #predicted class
table(credit.test$default,class.glm0.test.opt,dnn=c('True','Predicted'))

MR<- mean(credit.test$default!= class.glm0.test.opt)
FPR<- sum(credit.test$default==0 & class.glm0.test.opt==1)/sum(credit.test$default==0)
FNR<- sum(credit.test$default==1 & class.glm0.test.opt==0)/sum(credit.test$default==1)
cost<- costfunc(obs = credit.test$default, pred.p = pred.glm.credittest, pcut = optimal.pcut.glm0)  

pred <- prediction(pred.glm.credittest, credit.test$default)
#This function basically calculates many confusion matrices with different cut-off probability. Therefore, it requires two vectors as inputs - predicted probability and observed response (0/1). 
perf <- performance(pred, "tpr", "fpr")
#This line, performance() calculates TPR and FPR based all confusion matrices you get from previous step.
plot(perf, colorize=TRUE)
# Get AUC value -----------------------------------------------------------
unlist(slot(performance(pred, "auc"), "y.values"))


# Using cost function defined for AUC -------------------------------------
costfunc_AUC = function(obs, pred.p)
{
  pred=prediction(pred.p,obs)
  return(unlist(slot(performance(pred,"auc"),"y.values"))) 
}
credit.glm_full<-glm(default~.,family='binomial',data=credit.data)
cvglm2<-cv.glm(data=credit.data,glmfit =credit.glm_full,cost = costfunc_AUC,K=3)
cvglm2$delta[2]




# Classification tree -----------------------------------------------------

credit.rpart<-rpart(formula = default~.,data=credit.train,method = 'class',parms = list(loss=matrix(c(0,5,1,0),nrow=2)))
prp(credit.rpart,digits=4,extra=1)

#Insample
credit.tree.train.prob<-predict(credit.rpart,credit.train,type='prob')[,2]
pred<-prediction(credit.tree.train.prob,credit.train$default) #detach(package:neuralnet) or it will cause an error
perf = performance(pred,'tpr','fpr')
unlist(slot(performance(pred,"auc"),"y.values"))

credit.tree.train.class<-predict(credit.rpart,credit.train,type='class')
M.R_tree_train<-mean(credit.tree.train.class!=credit.train$default)


cost_tree <- function(r, pi){
  weight1 = 5
  weight0 = 1
  c1 = (r==1)&(pi==0) #logical vector - true if actual 1 but predict 0
  c0 = (r==0)&(pi==1) #logical vector - true if actual 0 but predict 1
  return(mean(weight1*c1+weight0*c0))
}

cost_tree(credit.train$default,credit.tree.train.class)


# Out of sample -----------------------------------------------------------
credit.tree.test.prob<-predict(credit.rpart,credit.test,type='prob')[,2]
pred.test<-prediction(credit.tree.test.prob,credit.test$default) #detach(package:neuralnet) or it will cause an error
perf = performance(pred.test,'tpr','fpr')
unlist(slot(performance(pred.test,"auc"),"y.values"))

credit.tree.test.class<-predict(credit.rpart,credit.test,type='class')
M.R_tree_test<-mean(credit.tree.test.class!=credit.test$default)
cost_tree(credit.test$default,credit.tree.test.class)



# Advanced Tree methods - Bagging, Boosting, Random forest ----------------

# Bagging  ----------------------------------------------------------------

credit.bagging<-randomForest(default~.,data = credit.train,ntree=100,mtry=ncol(credit.train)-1)
credit.bagging.pred<- predict(credit.bagging, type = "prob")[,2]


costfunc = function(obs, pred.p, pcut){
  weight1 = 5   # define the weight for "true=1 but pred=0" (FN)
  weight0 = 1    # define the weight for "true=0 but pred=1" (FP)
  c1 = (obs==1)&(pred.p<pcut)    # count for "true=1 but pred=0"   (FN)
  c0 = (obs==0)&(pred.p>=pcut)   # count for "true=0 but pred=1"   (FP)
  cost = mean(weight1*c1 + weight0*c0)  # misclassification with weight
  return(cost) # you have to return to a value when you write R functions
} 
p.seq = seq(0.01, 0.5, 0.01)
cost = rep(0, length(p.seq))  
for(i in 1:length(p.seq)){ 
  cost[i] = costfunc(obs = credit.train$default, pred.p = credit.bagging.pred, pcut = p.seq[i])  
}
plot(p.seq, cost)
optimal.pcut<-p.seq[which(cost==min(cost))]
cost<-costfunc(obs = credit.train$default, pred.p = credit.bagging.pred, pcut = optimal.pcut)


# Insample AUC and ROC ----------------------------------------------------


credit.bagging.pred<- predict(credit.bagging, type = "prob")[,2]
AMRcost.in<-costfunc(obs = credit.train$default, pred.p = credit.bagging.pred, pcut = optimal.pcut)

pred <- prediction(credit.bagging.pred, credit.train$default)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
unlist(slot(performance(pred, "auc"), "y.values"))


# Out of sample -----------------------------------------------------------

credit.bagging.outsample.pred<- predict(credit.bagging,newdata = credit.test, type = "prob")[,2]
AMRcost.out<-costfunc(obs = credit.test$default, pred.p = credit.bagging.outsample.pred, pcut = optimal.pcut)

pred.out <- prediction(credit.bagging.outsample.pred, credit.test$default)
perf.out <- performance(pred.out, "tpr", "fpr")
plot(perf.out, colorize=TRUE)
unlist(slot(performance(pred.out, "auc"), "y.values"))






# Random Forest ----------------------------------------------------------

credit.rf<-randomForest(default~.,data = credit.train,ntree=500,importance=TRUE,mtry=floor(sqrt(ncol(credit.train)-1)))
#mtry=sqrt(p)
plot(credit.rf, lwd=rep(2, 3))
legend("right", legend = c("OOB Error", "FPR", "FNR"), lwd=rep(2, 3), lty = c(1,2,3), col = c("black", "red", "green"))


# Insample prediction and AMR ---------------------------------------------

credit.rf.pred<-predict(credit.rf,newdata = credit.train,type='prob')[,2]
cost.rf.insample<-costfunc(obs = credit.train$default,pred.p =credit.rf.pred,pcut = optimal.pcut )

pred <- prediction(credit.rf.pred, credit.train$default)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
unlist(slot(performance(pred, "auc"), "y.values"))


# Out of sample prediction and AMR ----------------------------------------

credit.rf.outsample.pred<-predict(credit.rf,newdata = credit.test,type='prob')[,2]
cost.rf.outsample<-costfunc(obs = credit.test$default,pred.p =credit.rf.outsample.pred,pcut = optimal.pcut )

pred <- prediction(credit.rf.outsample.pred, credit.test$default)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
unlist(slot(performance(pred, "auc"), "y.values"))




# Boosting ----------------------------------------------------------------

credit.boost<-gbm(default~.,data=credit.train,distribution = "bernoulli",n.trees = 10000, shrinkage = 0.01, interaction.depth = 1)


# insample Prediction and AMR  ----------------------------------------------------------------
credit.boost.insample.pred<-predict(credit.boost,newdata=credit.train,type='response',n.trees = 10000)
credit.boost.insample.AMR<-costfunc(obs=credit.train$default,pred.p =credit.boost.insample.pred,pcut = optimal.pcut )

pred <- prediction(pred.cred.boost, credit.test$default.payment.next.month)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
#Get the AUC
unlist(slot(performance(pred, "auc"), "y.values"))


install.packages('adabag')
library(adabag)
credit.boost= boosting(default~., data = credit.train, boos = T)
save(credit.boost, file = "credit.boost.Rdata")

# Training AUC
pred.credit.boost= predict(credit.boost, newdata = credit.train)
pred <- prediction(pred.credit.boost$prob[,2], credit.train$default)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
#Get the AUC
unlist(slot(performance(pred, "auc"), "y.values"))

pred.credit.boost= predict(credit.boost, newdata = credit.test)
# Testing AUC
pred <- prediction(pred.credit.boost$prob[,2], credit.test$default)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
#Get the AUC
unlist(slot(performance(pred, "auc"), "y.values"))




# Generalized additive models -------------------------------------------------

head(credit.data)
#gam.formula.credit<-as.formula(paste("default~s(LIMIT_BAL)+s(AGE)+(PAY_0)+(PAY_2)+(PAY_3)+(PAY_4)+(PAY_5)+(PAY_6)+s(BILL_AMT1)+s(BILL_AMT2)+s(BILL_AMT3)+s(BILL_AMT4)+s(BILL_AMT5)+s(BILL_AMT6)+s(PAY_AMT1)+s(PAY_AMT2)+s(PAY_AMT3)+s(PAY_AMT4)+s(PAY_AMT5)+s(PAY_AMT6)+",paste(colnames(credit.data)[2:4],collapse='+')))

form<-paste("default~s(LIMIT_BAL)+s(AGE)+s(",paste(colnames(credit.data)[12:23],collapse=')+s('),")+",paste(colnames(credit.data)[2:4],collapse='+'),"+",paste(colnames(credit.data)[6:11],collapse='+'))
gam.formula.credit<-as.formula(form)
credit.gam<-gam(formula = gam.formula.credit,family='binomial',data=credit.train)
summary(credit.gam)


form2<-paste("default~(LIMIT_BAL)+PAY_AMT5+PAY_AMT6+BILL_AMT2+BILL_AMT3+BILL_AMT6+s(AGE)+s(",paste(colnames(credit.data)[12:21],collapse=')+s('),")+",paste(colnames(credit.data)[2:4],collapse='+'),"+",paste(colnames(credit.data)[6:11],collapse='+'))
gam.formula.credit_2<-as.formula(form2)
credit.gam<-gam(formula = gam.formula.credit_2,family='binomial',data=credit.train)
summary(credit.gam)


predict_mgcv.train<-predict(credit.gam,credit.train,type="response")

costfunc_mgcv = function(obs, pred.p, pcut)
{
  weight1 = 5   # define the weight for "true=1 but pred=0" (FN) placing weight on FN such that we need to avoid cases wherewe predict it wont default but it has in the actual data
  weight0 = 1    # define the weight for "true=0 but pred=1" (FP)
  c1 = (obs==1)&(pred.p<pcut) # count for "true=1 but pred=0"   (FN)
  #print(sum(c1))
  c0 = (obs==0)&(pred.p>=pcut)   # count for "true=0 but pred=1"   (FP)
  #print (sum(c0))
  cost = mean(weight1*c1 + weight0*c0)  # misclassification with weight
  return(cost) # you have to return to a value when you write R functions
} 

p.seq = seq(0.01, 1, 0.01) 
cost = rep(0, length(p.seq))  
for(i in 1:length(p.seq)){ 
  cost[i] = costfunc_mgcv(obs = credit.train$default, pred.p = predict_mgcv.train, pcut = p.seq[i])  
}

min(cost)
plot(p.seq, cost)
optimal.pcut_mgcv = p.seq[which(cost==min(cost))]
optimal.pcut_mgcv

# Insample performance ----------------------------------------------------

pred.gam.in<-(predict_mgcv.train>=optimal.pcut_mgcv)*1
table(credit.train$default,pred.gam.in,dnn=c("Observed","Predicted"))
mean(pred.gam.in!=credit.train$default)


# Out of sample -----------------------------------------------------------

predict_mgcv.test<-predict(credit.gam,newdata = credit.test,type='response')
pred.gam.out<-(predict_mgcv.test>=optimal.pcut_mgcv)*1
table(credit.test$default,pred.gam.out,dnn=c("Observed","Predicted"))
mean(pred.gam.out!=credit.test$default)


# Nueral Net --------------------------------------------------------------

credit.nnet <- nnet(default~.,data=credit.train, size=3, maxit=1000,decay=0.3)
credit.nnet$fitted.values

prob.nnet= predict(credit.nnet,credit.test)
head(prob.nnet)

pcut<-(1/6)
pred.nnet = (prob.nnet > pcut)*1
table(credit.test$default,pred.nnet, dnn=c("Observed","Predicted"))

mean(credit.test$default!= pred.nnet)
costfunc_mgcv(credit.test$default,pred.nnet,optimal.pcut_mgcv)
