
# Packages ----------------------------------------------------------------

install.packages('PerformanceAnalytics')
install.packages("ggcorrplot")
install.packages('ROCR')

# Bankruptcy ---------------------------------------------------------------

bankruptcy.data<-read.csv("C:/Users/Vishnu/Desktop/Spring 2020/Data Mining/bankruptcy.csv")
head(bankruptcy.data)
bankruptcy.data$CUSIP<-as.numeric(bankruptcy.data$CUSIP)


summary(bankruptcy.data$DLRSN)
length(bankruptcy.data$CUSIP)


# CUSIP is ID column ------------------------------------------------------

bankruptcy.data<-bankruptcy.data[,!names(bankruptcy.data) %in% c("FYEAR","CUSIP")]


library('tidyverse')
glimpse(bankruptcy.data$DLRSN)



hist(bankruptcy.data$DLRSN)
bankruptcy.data$DLRSN<-as.factor(bankruptcy.data$DLRSN)


ggplot(bankruptcy.data, aes(x = DLRSN)) + geom_histogram(aes(y = ..density..),color="green", fill="green") + geom_density(color="red")
ggplot(bankruptcy.data, aes(x = R1)) + geom_histogram(aes(y = ..density..),color="blue", fill="light blue") + geom_density(color="red")
ggplot(bankruptcy.data, aes(x = bankruptcy.data$R2)) + geom_histogram(aes(y = ..density..),color="blue", fill="light blue") + geom_density(color="red")
ggplot(bankruptcy.data, aes(x = R3)) + geom_histogram(aes(y = ..density..),color="blue", fill="light blue") + geom_density(color="red")
ggplot(bankruptcy.data, aes(x = R4)) + geom_histogram(aes(y = ..density..),color="blue", fill="light blue") + geom_density(color="red")
ggplot(bankruptcy.data, aes(x = R5,fill=DLRSN)) + geom_histogram(aes(y = ..density..),color="blue", fill="light blue") + geom_density(color="red")
ggplot(bankruptcy.data, aes(x = R6,fill=DLRSN)) + geom_histogram(aes(y = ..density..),color="blue", fill="light blue") + geom_density(color="red")


ggplot(bankruptcy.data, aes(x = R6,fill=DLRSN)) + geom_histogram(position = 'dodge') 
ggplot(bankruptcy.data, aes(x = R1,fill=DLRSN)) + geom_histogram(position = 'dodge') 
ggplot(bankruptcy.data, aes(x = R2,fill=DLRSN)) + geom_histogram(position = 'dodge') 
ggplot(bankruptcy.data, aes(x = R3,fill=DLRSN)) + geom_histogram(position = 'dodge') 
ggplot(bankruptcy.data, aes(x = R4,fill=DLRSN)) + geom_histogram(position = 'dodge') 
ggplot(bankruptcy.data, aes(x = R5,fill=DLRSN)) + geom_histogram(position = 'dodge') 
ggplot(bankruptcy.data, aes(x = R7,fill=DLRSN)) + geom_histogram(position = 'dodge') 
ggplot(bankruptcy.data, aes(x = R8,fill=DLRSN)) + geom_histogram(position = 'dodge') 
ggplot(bankruptcy.data, aes(x = R9,fill=DLRSN)) + geom_histogram(position = 'dodge') 
ggplot(bankruptcy.data, aes(x = R10,fill=DLRSN)) + geom_histogram(position = 'dodge') 
 
nums <- unlist(lapply(bankruptcy.data, is.numeric)) 
bankruptcy_numeric<-bankruptcy.data[,nums]
bankruptcy_corr<- bankruptcy_numeric[,!names(bankruptcy_numeric) %in% c("FYEAR","CUSIP")]
corr<-cor(bankruptcy_corr)


library("PerformanceAnalytics")
my_data <- bankruptcy_corr
chart.Correlation(my_data, histogram=TRUE, pch=19)


# Correlation between all numeric variables -------------------------------

library(ggcorrplot)
ggcorrplot(corr, hc.order = TRUE, type = "lower",lab = TRUE)


# Split data --------------------------------------------------------------
set.seed(13255870)
index <- sample(nrow(bankruptcy.data),nrow(bankruptcy.data)*0.70)
bankruptcy.train = bankruptcy.data[index,]
bankruptcy.test = bankruptcy.data[-index,]


names(bankruptcy.train)
#lapply(bankruptcy.train,class)



# Logistic model with all X variables  ------------------------------------

bankruptcy.glm<-glm(DLRSN~.,family = binomial,data = bankruptcy.train)
summary(bankruptcy.glm)
bankruptcy.glm$coefficients

bankruptcy.glm_probit<-glm(DLRSN~.,family = binomial(link = 'probit'),data = bankruptcy.train)
summary(bankruptcy.glm_probit)
bankruptcy.glm_probit$coefficients

bankruptcy.glm_cloglog<-glm(DLRSN~.,family = binomial(link = 'cloglog'),data = bankruptcy.train)
summary(bankruptcy.glm_cloglog)
bankruptcy.glm_cloglog$coefficients

# Best models with AIC, BIC and LASSO --------------------------------------------

bankruptcy.glm.back <- step(bankruptcy.glm)  #k=2
summary(bankruptcy.glm.back)

bankruptcy.glm.back.BIC <- step(bankruptcy.glm, k=log(nrow(bankruptcy.train))) 
Finalmodel<-summary(bankruptcy.glm.back.BIC)
Finalmodel$deviance
BIC(bankruptcy.glm.back.BIC)

library(car)
vif(bankruptcy.glm)

bankruptcy.glm.back$deviance
bankruptcy.glm.back.BIC$deviance


BIC(bankruptcy.glm.back)
BIC(bankruptcy.glm.back.BIC)

AIC(bankruptcy.glm.back)
AIC(bankruptcy.glm.back.BIC)


# LASSO -------------------------------------------------------------------

str(bankruptcy.train)
library('dplyr')
#index <- sample(nrow(bankruptcy.data),nrow(bankruptcy.data)*0.70)
bankruptcy.train.X = as.matrix(select(bankruptcy.data, -DLRSN)[index,])
bankruptcy.test.X = as.matrix(select(bankruptcy.data, -DLRSN)[-index,])
bankruptcy.train.Y = bankruptcy.data[index, "DLRSN"]
bankruptcy.test.Y = bankruptcy.data[-index, "DLRSN"]

library("glmnet")
bankruptcy.lasso<- glmnet(x=bankruptcy.train.X, y=bankruptcy.train.Y, family = "binomial")


# Perform cross validation to find shrinkage parameter --------------------

bankruptcy.lasso.cv<- cv.glmnet(x=bankruptcy.train.X, y=bankruptcy.train.Y, family = "binomial", type.measure = "class")
#for logistic regression we can specify type.measure = "class" sp that CV error will be Misclassification error

plot(bankruptcy.lasso.cv)
coef(bankruptcy.lasso, s=bankruptcy.lasso.cv$lambda.min)
coef(bankruptcy.lasso, s=bankruptcy.lasso.cv$lambda.1se)

# in-sample prediction
pred.lasso.train<- predict(bankruptcy.lasso, newx=bankruptcy.train.X, s=bankruptcy.lasso.cv$lambda.1se, type = "response")
# out-of-sample prediction
pred.lasso.test<- predict(bankruptcy.lasso, newx=bankruptcy.test.X, s=bankruptcy.lasso.cv$lambda.1se, type = "response")


# Lasso - insample --------------------------------------------------------

pred.lasso<-prediction(pred.lasso.train,bankruptcy.train.Y)
perf.lasso<-performance(pred.lasso,"tpr","fpr")
plot(perf.lasso,colorize=T)
unlist(slot(performance(pred.lasso,"auc"),"y.values"))

# Lasso - out of sample ---------------------------------------------------

pred.osample<-prediction(pred.lasso.test,bankruptcy.test.Y)
perfo<-performance(pred.osample,"tpr","fpr")
plot(perfo,colorize=T)
unlist(slot(performance(pred.osample,"auc"),"y.values"))










# Variable selection --  done with AIC--------------------------------------
pred.glm.bankruptcytrain_AIC<- predict(bankruptcy.glm.back, type="response")
library(ROCR)
pred_AIC <- prediction(pred.glm.bankruptcytrain_AIC, bankruptcy.train$DLRSN)
perf_AIC <- performance(pred_AIC, "tpr", "fpr")
plot(perf_AIC, colorize=TRUE)
unlist(slot(performance(pred_AIC, "auc"), "y.values"))





# Variable selection --  done with BIC--------------------------------------
pred.glm.bankruptcytrain<- predict(bankruptcy.glm.back.BIC, type="response")
library(ROCR)
pred <- prediction(pred.glm.bankruptcytrain, bankruptcy.train$DLRSN)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
unlist(slot(performance(pred, "auc"), "y.values"))
class.glm0.train<- (pred.glm.bankruptcytrain>(1/36))*1
MR.train<- mean(bankruptcy.train$DLRSN!= class.glm0.train)
table(bankruptcy.train$DLRSN,class.glm0.train,dnn=c("True","Predicted"))
FPR.train<- sum(bankruptcy.train$DLRSN==0 & class.glm0.train==1)/sum(bankruptcy.train$DLRSN==0)
FNR.train<- sum(bankruptcy.train$DLRSN==1 & class.glm0.train==0)/sum(bankruptcy.train$DLRSN==1)


# BIC- Out of sample ------------------------------------------------------

pred.glm.bankruptcytest<- predict(bankruptcy.glm.back.BIC,newdata = bankruptcy.test, type="response")
library(ROCR)
pred_test <- prediction(pred.glm.bankruptcytest, bankruptcy.test$DLRSN)
perf_test <- performance(pred_test, "tpr", "fpr")
plot(perf_test, colorize=TRUE)
unlist(slot(performance(pred_test, "auc"), "y.values"))
class.glm0.test<- (pred.glm.bankruptcytest>(1/36))*1
MR.test<- mean(bankruptcy.test$DLRSN!= class.glm0.test)
table(bankruptcy.test$DLRSN,class.glm0.test,dnn=c("True","Predicted"))
FPR.test<- sum(bankruptcy.test$DLRSN==0 & class.glm0.test==1)/sum(bankruptcy.test$DLRSN==0)
FNR.test<- sum(bankruptcy.test$DLRSN==1 & class.glm0.test==0)/sum(bankruptcy.test$DLRSN==1)

# Imbalanced data - Precision Recall curve  -------------------------------

#install.packages("PRROC")

library(PRROC)
score1= pred.glm.bankruptcytrain[bankruptcy.train$DLRSN==1]
score0= pred.glm.bankruptcytrain[bankruptcy.train$DLRSN==0]
roc= roc.curve(score1, score0, curve = T)
roc$auc


# Cost function for assymetric data / Grid search---------------------------------------
nrow(bankruptcy.train)

costfunc = function(obs, pred.p, pcut)
  {
    weight1 = 35   # define the weight for "true=1 but pred=0" (FN) placing weight on FN such that we need to avoid cases wherewe predict it wont default but it has in the actual data
    weight0 = 1    # define the weight for "true=0 but pred=1" (FP)
    c1 = (obs==1)&(pred.p<pcut) # count for "true=1 but pred=0"   (FN)
    #print(sum(c1))
    c0 = (obs==0)&(pred.p>=pcut)   # count for "true=0 but pred=1"   (FP)
    #print (sum(c0))
    cost = mean(weight1*c1 + weight0*c0)  # misclassification with weight
    return(cost) # you have to return to a value when you write R functions
  } 



# Define a seq, need to search for opt cut off from this ------------------

p.seq = seq(0.01, 1, 0.01) 
cost = rep(0, length(p.seq))  
for(i in 1:length(p.seq)){ 
    cost[i] = costfunc(obs = bankruptcy.train$DLRSN, pred.p = pred.glm.bankruptcytrain, pcut = p.seq[i])  
}

min(cost)
plot(p.seq, cost)

optimal.pcut.glm0 = p.seq[which(cost==min(cost))]

class.glm0.train.opt<- (pred.glm.bankruptcytrain>optimal.pcut.glm0)*1
table(bankruptcy.train$DLRSN, class.glm0.train.opt, dnn = c("True", "Predicted"))

MR<- mean(bankruptcy.train$DLRSN!= class.glm0.train.opt)
sum(bankruptcy.train$DLRSN!= class.glm0.train.opt)
FPR<- sum(bankruptcy.train$DLRSN==0 & class.glm0.train.opt==1)/sum(bankruptcy.train$DLRSN==0)
FNR<- sum(bankruptcy.train$DLRSN==1 & class.glm0.train.opt==0)/sum(bankruptcy.train$DLRSN==1)
cost<- costfunc(obs = bankruptcy.train$DLRSN, pred.p = pred.glm.bankruptcytrain, pcut = optimal.pcut.glm0)  



# Out of sample prediction  -----------------------------------------------

pred.glm.bankruptcytest<-predict(bankruptcy.glm.back.BIC,newdata=bankruptcy.test,type='response')
class.glm0.test.opt<-(pred.glm.bankruptcytest>optimal.pcut.glm0)*1
table(bankruptcy.test$DLRSN,class.glm0.test.opt,dnn=c('True','Predicted'))

MR<- mean(bankruptcy.test$DLRSN!= class.glm0.test.opt)
FPR<- sum(bankruptcy.test$DLRSN==0 & class.glm0.test.opt==1)/sum(bankruptcy.test$DLRSN==0)
FNR<- sum(bankruptcy.test$DLRSN==1 & class.glm0.test.opt==0)/sum(bankruptcy.test$DLRSN==1)
cost<- costfunc(obs = bankruptcy.test$DLRSN, pred.p = pred.glm.bankruptcytest, pcut = optimal.pcut.glm0)  

pred <- prediction(pred.glm.bankruptcytest, bankruptcy.test$DLRSN)
#This function basically calculates many confusion matrices with different cut-off probability. Therefore, it requires two vectors as inputs - predicted probability and observed response (0/1). 

perf <- performance(pred, "tpr", "fpr")
#This line, performance() calculates TPR and FPR based all confusion matrices you get from previous step.

plot(perf, colorize=TRUE)


# Get AUC value -----------------------------------------------------------
unlist(slot(performance(pred, "auc"), "y.values"))







# Cross validation --------------------------------------------------------

library('boot')
# Using cost function defined for Misclassification -----------------------


bankruptcy.glm_full<-glm(DLRSN~.,family = binomial,data=bankruptcy.data)
pcut <- 1/36
costfunc = function(obs, pred.p)
{
  weight1 = 35   # define the weight for "true=1 but pred=0" (FN)
  weight0 = 1    # define the weight for "true=0 but pred=1" (FP)
  c1 = (obs==1)&(pred.p<pcut)    # count for "true=1 but pred=0"   (FN)
  c0 = (obs==0)&(pred.p>=pcut)   # count for "true=0 but pred=1"   (FP)
  cost = mean(weight1*c1 + weight0*c0)  # misclassification with weight
  return(cost) # you have to return to a value when you write R functions
}

cvglm<-cv.glm(data=bankruptcy.data,glmfit =bankruptcy.glm_full,cost = costfunc,K=3)
cvglm$delta[2]



# Using cost function defined for AUC -------------------------------------

costfunc_AUC = function(obs, pred.p)
{
  pred=prediction(pred.p,obs)
  return(unlist(slot(performance(pred,"auc"),"y.values"))) 
}
cvglm2<-cv.glm(data=bankruptcy.data,glmfit =bankruptcy.glm_full,cost = costfunc_AUC,K=3)
cvglm2$delta[2]



# Decision Trees ----------------------------------------------------------

library(rpart)
library(rpart.plot)


# Building tree -----------------------------------------------------------

bankruptcy.rpart <- rpart(formula = DLRSN ~ ., data = bankruptcy.train)
bankruptcy.rpart

prp(bankruptcy.rpart,digits = 4, extra = 1)

bankruptcy.rpart1 <- rpart(formula = DLRSN ~ . , data = bankruptcy.train, method = "class", parms = list(loss=matrix(c(0,35,1,0), nrow = 2)))
bankruptcy.rpart1
prp(bankruptcy.rpart1,digits = 4, extra = 1)


# Predictions - In sample -------------------------------------------------

bankruptcy.train.pred.tree1<- predict(bankruptcy.rpart1, bankruptcy.train, type="prob")
bankruptcy.train.pred.tree1class<- predict(bankruptcy.rpart1, bankruptcy.train, type="class")
table(bankruptcy.train$DLRSN, bankruptcy.train.pred.tree1class, dnn=c("Truth","Predicted"))

pred.traintree = prediction(bankruptcy.train.pred.tree1[,2], bankruptcy.train$DLRSN)
perf = performance(pred.traintree, "tpr", "fpr")
plot(perf, colorize=TRUE)
unlist(slot(performance(pred.traintree,"auc"),"y.values"))

MR.treetrain<- mean(bankruptcy.train$DLRSN!= bankruptcy.train.pred.tree1class)


# Predictions - Out of sample ---------------------------------------------

bankruptcy.test.pred.tree1<-predict(bankruptcy.rpart1,newdata = bankruptcy.test,type='prob')
bankruptcy.test.pred.tree1class<-predict(bankruptcy.rpart1,newdata = bankruptcy.test,type='class')
table(bankruptcy.test$DLRSN,bankruptcy.test.pred.tree1class,dnn = c("Truth","Predicted"))


pred.testtree<-prediction(bankruptcy.test.pred.tree1[,2],bankruptcy.test$DLRSN)
perf.testtree<-performance(pred.testtree,"tpr","fpr")
unlist(slot(performance(pred.testtree,"auc"),"y.values"))

MR.treetest<- mean(bankruptcy.test$DLRSN!= bankruptcy.test.pred.tree1class)


# Pruning ---------------------------------------------------------------

#boston.largetree <- rpart(formula = medv ~ ., data = boston.train, cp = 0.001)
bankruptcy.largetree<-rpart(formula = DLRSN ~ . , data = bankruptcy.train, method = "class",cp = 0.001,parms = list(loss=matrix(c(0,35,1,0), nrow = 2)) )

prp(bankruptcy.largetree)
plotcp(bankruptcy.largetree)
printcp(bankruptcy.largetree)

bankruptcy.largetree<-prune(bankruptcy.largetree,cp=0.001)

bankruptcy.test.pred.tree1<-predict(bankruptcy.rpart1,newdata = bankruptcy.test,type='prob')
bankruptcy.test.pred.tree1class<-predict(bankruptcy.rpart1,newdata = bankruptcy.test,type='class')
table(bankruptcy.test$DLRSN,bankruptcy.test.pred.tree1class,dnn = c("Truth","Predicted"))


pred.testtree<-prediction(bankruptcy.test.pred.tree1[,2],bankruptcy.test$DLRSN)
perf.testtree<-performance(pred.testtree,"tpr","fpr")
unlist(slot(performance(pred.testtree,"auc"),"y.values"))







# Random sample 80,20 -----------------------------------------------------

set.seed(13255870)
index2<-sample(nrow(bankruptcy.data),nrow(bankruptcy.data)*0.9)
bankruptcy.train2<- bankruptcy.data[index2,]
bankruptcy.test2<- bankruptcy.data[-index2,]


bankruptcy.glm2<-glm(DLRSN~.,family = binomial,data = bankruptcy.train2)

#AC
bankruptcy.glm.back2 <- step(bankruptcy.glm2)

#BIC
bankruptcy.glm.back2.BIC <- step(bankruptcy.glm2,k=log(nrow(bankruptcy.train2)))

#LASSO
bankruptcy.train2.X = as.matrix(select(bankruptcy.data, -DLRSN)[index2,])
bankruptcy.test2.X = as.matrix(select(bankruptcy.data, -DLRSN)[-index2,])
bankruptcy.train2.Y = bankruptcy.data[index2, "DLRSN"]
bankruptcy.test2.Y = bankruptcy.data[-index2, "DLRSN"]
bankruptcy.lasso<- glmnet(x=bankruptcy.train2.X, y=bankruptcy.train2.Y, family = "binomial")


# Perform cross validation to find shrinkage parameter --------------------

bankruptcy.lasso.cv2<- cv.glmnet(x=bankruptcy.train2.X, y=bankruptcy.train2.Y, family = "binomial", type.measure = "class")
#for logistic regression we can specify type.measure = "class" sp that CV error will be Misclassification error

plot(bankruptcy.lasso.cv2)
coef(bankruptcy.lasso, s=bankruptcy.lasso.cv2$lambda.min)
coef(bankruptcy.lasso, s=bankruptcy.lasso.cv2$lambda.1se) #can use this for predicting by specifying "s" parameter in predict function



# Prediction- BIC model - In sample ---------------------------------------

pred.glm2.bankruptcytrainBIC<- predict(bankruptcy.glm.back2.BIC, type="response")
library(ROCR)
pred2_BIC <- prediction(pred.glm2.bankruptcytrainBIC, bankruptcy.train2$DLRSN)
perf2_BIC <- performance(pred2_BIC, "tpr", "fpr")
plot(perf2_BIC, colorize=TRUE)
unlist(slot(performance(pred2_BIC, "auc"), "y.values"))



# Out of sample -----------------------------------------------------------

pred.glm2.bankruptcytest<- predict(bankruptcy.glm.back2.BIC,newdata = bankruptcy.test2, type="response")
pred2_BIC.test <- prediction(pred.glm2.bankruptcytest, bankruptcy.test2$DLRSN)
perf2_BIC.test <- performance(pred2_BIC.test, "tpr", "fpr")
plot(perf2_BIC.test, colorize=TRUE)
unlist(slot(performance(pred2_BIC.test, "auc"), "y.values"))




# Cost function and MR ----------------------------------------------------

for(i in 1:length(p.seq)){ 
  cost[i] = costfunc(obs = bankruptcy.train2$DLRSN, pred.p = pred.glm2.bankruptcytrainBIC, pcut = p.seq[i])  
}

plot(p.seq, cost)
optimal.pcut.glm0 = p.seq[which(cost==min(cost))]
cost

class.glm0.train.opt<- (pred.glm2.bankruptcytrainBIC>optimal.pcut.glm0)*1
table(bankruptcy.train2$DLRSN, class.glm0.train.opt, dnn = c("True", "Predicted"))
MR.2<- mean(bankruptcy.train2$DLRSN!= class.glm0.train.opt)

class.glm0.test.opt<- (pred.glm2.bankruptcytest>optimal.pcut.glm0)*1
MR.2test<- mean(bankruptcy.test2$DLRSN!= class.glm0.test.opt)


# Classification Tree -----------------------------------------------------

bankruptcy.rpart2 <- rpart(formula = DLRSN ~ . , data = bankruptcy.train2, method = "class", parms = list(loss=matrix(c(0,35,1,0), nrow = 2)))
bankruptcy.rpart2
prp(bankruptcy.rpart2,digits = 4, extra = 1)

bankruptcy.largetree2<-rpart(formula = DLRSN ~ . , data = bankruptcy.train2,cp=0.001, method = "class",parms = list(loss=matrix(c(0,35,1,0), nrow = 2)) )

prp(bankruptcy.largetree2)
plotcp(bankruptcy.largetree2)
printcp(bankruptcy.largetree2)

bankruptcy.rpart3<-prune(bankruptcy.largetree,cp=0.0011876)


# Predictions - In sample -------------------------------------------------

bankruptcy.train.pred.tree2<- predict(bankruptcy.rpart3, bankruptcy.train2, type="prob")
bankruptcy.train.pred.tree2class<- predict(bankruptcy.rpart3, bankruptcy.train2, type="class")
table(bankruptcy.train2$DLRSN, bankruptcy.train.pred.tree2class, dnn=c("Truth","Predicted"))

pred.traintree2 = prediction(bankruptcy.train.pred.tree2[,2], bankruptcy.train2$DLRSN)
perf2 = performance(pred.traintree2, "tpr", "fpr")
plot(perf2, colorize=TRUE)
unlist(slot(performance(pred.traintree2,"auc"),"y.values"))

MR.treetrain2<- mean(bankruptcy.train2$DLRSN!= bankruptcy.train.pred.tree2class)


# Predictions - Out of sample ---------------------------------------------

bankruptcy.test.pred.tree2<-predict(bankruptcy.rpart3,newdata = bankruptcy.test2,type='prob')
bankruptcy.test.pred.tree2class<-predict(bankruptcy.rpart2,newdata = bankruptcy.test2,type='class')
table(bankruptcy.test2$DLRSN,bankruptcy.test.pred.tree2class,dnn = c("Truth","Predicted"))


pred.testtree2<-prediction(bankruptcy.test.pred.tree2[,2],bankruptcy.test2$DLRSN)
perf.testtree2<-performance(pred.testtree2,"tpr","fpr")
unlist(slot(performance(pred.testtree2,"auc"),"y.values"))

MR.treetest2<- mean(bankruptcy.test2$DLRSN!= bankruptcy.test.pred.tree2class)



