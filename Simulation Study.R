
# HW -2 Simulation Study --------------------------------------------------


# Mean function : E(Y|X)= 5 + 1.2*x1 + 3*x2  ------------------------------
set.seed(101)
x1 = rnorm(200,mean = 2, sd = 0.4)
x2 = rnorm(200,mean = -1, sd = 0.1)
e= rnorm(200, mean = 0,sd = 1)
x3=x1*x2

# True model --------------------------------------------------------------

y<- 5 + 1.2*x1 + 3*x2 + e


# Simple L.Regression  ----------------------------------------------------

lm.y <- lm(y~ x1+x2+x1*x2)
summary_basic<-summary(lm.y)

model<-predict(lm.y)

# Null model --------------------------------------------------------------

null_model_y=lm(y~1)


model_step_bck<- step(lm.y,direction = "backward")
summary_bck<-summary(model_step_bck)
summary_bck$sigma^2

model_step_forward<-step(null_model_y,scope=list(lower=null_model_y,upper=lm.y),direction = "forward")

# Stepwise Regression, N=200, sd=1 ----------------------------------------

model_stepwise<-step(null_model_y,scope=list(lower=null_model_y,upper=lm.y),direction = "both")
model_stepwise$coefficients

#(Intercept)          x1          x2 
#5.138847    1.119801    3.040376


# MAE ---------------------------------------------------------------------

mae <- function(error)
{
  round(mean(abs(error)),4)
}

# Case 1, sd=0.1, N=25,100,500,5000 ---------------------------------------

sample_size<- c(25,100,500,5000)
betaMatrix<-matrix(NA,4,3)
#betaMatrix<-matrix(NA,100,3)
set.seed(1234)
listMSE<- c()
Rsqrd<- c()
adj_Rsqrd<- c()
MAE<-c()
coeffs<-c(5,1.2,3)
L3norm<-c()

for (j in 1: length(sample_size))
{
  
  x1 = rnorm(sample_size[j],mean = 2, sd = 0.4)
  x2 = rnorm(sample_size[j],mean = -1, sd = 0.1)
  e= rnorm(sample_size[j], mean = 0,sd = 0.1)

  y<- 5 + 1.2*x1 + 3*x2 + e

  null_model_y1=lm(y~1)
  lm.y1 <- lm(y~ x1+x2+x1*x2)
  model_stepwise<-step(null_model_y1,scope=list(lower=null_model_y1,upper=lm.y1),direction = "both")
  betaMatrix[j,] <- model_stepwise$coefficients

  mysummary <- summary(model_stepwise)
  listMSE[j] <- mysummary$sigma^2 #get MSE per iteration
  Rsqrd[j] <- mysummary$r.squared
  adj_Rsqrd[j] <- mysummary$adj.r.squared
  MAE[j]<-mae(model_stepwise$residuals)
  

}

betaMatrix
L3norm
listMSE
Rsqrd
adj_Rsqrd
MAE



diff<-c()
L3norm<-c()

for (i in 1:4)
{
  for ( j in 1:3)
  {
    diff[j]<-abs(betaMatrix[i,j]-coeffs[j])
  }
    print(sum(diff)/3)
    L3norm[i]<- sum(diff)/3
}

L3norm


sum(betaMatrix[1,])
(abs(5-5.332)+abs(1.2-1.185)+abs(3-3.265))/3


# Case 2 : sd = 0.5 -------------------------------------------------------

sample_size<- c(25,100,500,5000)
betaMatrix<-matrix(NA,4,12)
#betaMatrix<-matrix(NA,100,3)
set.seed(1234)
listMSE<- c()
Rsqrd<- c()
adj_Rsqrd<- c()
MAE<-c()
coeffs<-c(5,1.2,3)

for (j in 1: length(sample_size))
{
  
  x1 = rnorm(sample_size[j],mean = 2, sd = 0.4)
  x2 = rnorm(sample_size[j],mean = -1, sd = 0.1)
  e= rnorm(sample_size[j], mean = 0,sd = 0.5)
  
  y<- 5 + 1.2*x1 + 3*x2 + e
  
  null_model_y1=lm(y~1)
  lm.y1 <- lm(y~ x1+x2+x1*x2)
  model_stepwise<-step(null_model_y1,scope=list(lower=null_model_y1,upper=lm.y1),direction = "both")
  print("next")
  betaMatrix[j,] <- model_stepwise$coefficients
  mysummary <- summary(model_stepwise)
  listMSE[j] <- mysummary$sigma^2 #get MSE per iteration
  Rsqrd[j] <- mysummary$r.squared
  adj_Rsqrd[j] <- mysummary$adj.r.squared
  MAE[j]<-mae(model_stepwise$residuals)
  
}

betaMatrix
listMSE
Rsqrd
adj_Rsqrd
MAE
diff<-c()
L3norm<-c()

for (i in 1:4)
{
  for ( j in 1:3)
  {
    diff[j]<-abs(betaMatrix[i,j]-coeffs[j])
  }
  print(diff)
  L3norm[i]<- sum(diff)/3
}

L3norm

# Case 3 : sd=1 -----------------------------------------------------------

sample_size<- c(25,100,500,5000)
betaMatrix<-matrix(NA,4,12)

#betaMatrix<-matrix(NA,100,3)
listMSE<- c()
Rsqrd<- c()
adj_Rsqrd<- c()
MAE<-c()
coeffs<-c(5,1.2,3)
set.seed(1234)
for (j in 1: length(sample_size))
{
  
  x1 = rnorm(sample_size[j],mean = 2, sd = 0.4)
  x2 = rnorm(sample_size[j],mean = -1, sd = 0.1)
  e= rnorm(sample_size[j], mean = 0,sd = 1)
  
  y<- 5 + 1.2*x1 + 3*x2 + e
  
  null_model_y1=lm(y~1)
  lm.y1 <- lm(y~ x1+x2+x1*x2)
  model_stepwise<-step(null_model_y1,scope=list(lower=null_model_y1,upper=lm.y1),direction = "both")
  print("next")
  betaMatrix[j,] <- model_stepwise$coefficients
  mysummary <- summary(model_stepwise)
  listMSE[j] <- mysummary$sigma^2 #get MSE per iteration
  Rsqrd[j] <- mysummary$r.squared
  adj_Rsqrd[j] <- mysummary$adj.r.squared
  MAE[j]<-mae(model_stepwise$residuals)
  
}



betaMatrix
listMSE
Rsqrd
adj_Rsqrd
MAE
diff<-c()
L3norm<-c()

for (i in 1:4)
{
  for ( j in 1:3)
  {
    diff[j]<-abs(betaMatrix[i,j]-coeffs[j])
  }
  print(diff)
  L3norm[i]<- sum(diff)/3
}

L3norm

(abs(5-4.999)+abs(1.2-1.212)+abs(3-3.019))/3

# Trial -------------------------------------------------------------------



set.seed(100)
x1 = rnorm(25,mean = 2, sd = 0.4)
x2 = rnorm(25,mean = -1, sd = 0.1)
#x3=x1*x2
e= rnorm(25, mean = 0,sd = 1)

y<- 5 + 1.2*x1 + 3*x2 + e

null_model_y1=lm(y~1)
lm.y1 <- lm(y~ x1+x2+x1*x2)
model_stepwise<-step(null_model_y1,scope=list(lower=null_model_y1,upper=lm.y1),direction = "both")
#model_step_bck<- step(lm.y1,direction = "backward")


set.seed(100)
x1 = rnorm(100,mean = 2, sd = 0.4)
x2 = rnorm(100,mean = -1, sd = 0.1)
x3=x1*x2
e= rnorm(100, mean = 0,sd = 1)

y<- 5 + 1.2*x1 + 3*x2 + e

null_model_y1=lm(y~1)
lm.y1 <- lm(y~ x1+x2+x3)
model_stepwise<-step(null_model_y1,scope=list(lower=null_model_y1,upper=lm.y1),direction = "both")
#model_step_bck<- step(lm.y1,direction = "backward")


set.seed(100)
x1 = rnorm(500,mean = 2, sd = 0.4)
x2 = rnorm(500,mean = -1, sd = 0.1)
x3=x1*x2
e= rnorm(500, mean = 0,sd = 1)

y<- 5 + 1.2*x1 + 3*x2 + e

null_model_y1=lm(y~1)
lm.y1 <- lm(y~ x1+x2+x3)
model_stepwise<-step(null_model_y1,scope=list(lower=null_model_y1,upper=lm.y1),direction = "both")
#model_step_bck<- step(lm.y1,direction = "backward")


set.seed(100)
x1 = rnorm(5000,mean = 2, sd = 0.4)
x2 = rnorm(5000,mean = -1, sd = 0.1)
x3=x1*x2
e= rnorm(5000, mean = 0,sd = 1)

y<- 5 + 1.2*x1 + 3*x2 + e

null_model_y1=lm(y~1)
lm.y1 <- lm(y~ x1+x2+x3)
#model_stepwise<-step(null_model_y1,scope=list(lower=null_model_y1,upper=lm.y1),direction = "both")
model_step_bck<- step(lm.y1,direction = "backward")











model_stepwise$coefficients
summary<-summary(model_stepwise)
summary$sigma^2

lm.y2<-lm(y~ x1+x2)
summ<-summary(lm.y2)

summ$sigma^2
