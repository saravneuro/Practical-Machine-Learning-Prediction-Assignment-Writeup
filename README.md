Practical-Machine-Learning-Prediction-Assignment-Writeup
========================================================
---
title: "Practical Machine Learning, Prediction Assignment Writeup"
author: "Saravanan Subramaniam"
date: "Sunday, November 23, 2014"
output: html_document
---

This is an R Markdown document. Markdown is a simple formatting syntax for authoring HTML, PDF, and MS Word documents. For more details on using R Markdown see <http://rmarkdown.rstudio.com>.

When you click the **Knit** button a document will be generated that includes both content as well as the output of any embedded R code chunks within the document. You can embed an R code chunk like this:

### Goal: 
* The goal of my project is to predict the manner in which the exercise was done. This is the "classe" variable in the training set. I am herewith giving a report describing how I built the model, and how I used cross validation, and the expected sample error, and the reason for the choices to use  prediction model to predict 20 different test cases.

### Steps followed:
* At first the variables having high missing values are eliminated. Variables with very high correlation, are identified that can lead to multicollinearity which can increase missclassification error rate. Finally only 32 Variables were segregated from the analysis and used for prediction.In the next step an appropriate algorithm which learns from the data is used.Hence, randomForest was used, which uses the bagging method that is of high variance, low bias technique.Finally the learned algorithm from this data was then used to predict the test set.


#### Loading the required libraries:

```{r}
require(lattice)
require(ggplot2)
require(reshape)
require(corrplot)
require(Hmisc)
require(AppliedPredictiveModeling)
require(caret)
require(randomForest)
```
#### Load data sets:
```{r}
file.loc<-"C:/Users/Dr.Saravanan/Desktop/R/"
setwd(file.loc)
training<-read.csv(("pml-training.csv"),stringsAsFactor=FALSE,skip = 0,fill=NA,comment.char="#")
testing<-read.csv(paste0("pml-testing.csv"))
```
#### Removing Missing data Variables:
*The train dataset consist of 160 variables and 19622 observations from which we have to learn.

```{r}
length(names(training)) 
```
```{r}
nrow(training)  
```
```{r}
var<-names(training)[apply(training,2,function(x) table(is.na(x))[1]==19622)]   
training2<-training[,var]
testing2<-testing[,var[-length(var)]]
```
* The variables are reduced to keep only the numeric variables giving HAR information.

```{r}
var2<-melt(apply(training2,2,function(x) sum(ifelse(x=="",1,0)))==0)
select.var<-rownames(var2)[var2$value==TRUE]
training3<-training2[,select.var]
testing3<-testing2[,select.var[-length(select.var)]]
training4<-training3[,names(training3[-c(1:7,length(training3))])] 
testing4<-testing3[,names(testing3[-c(1:7)])]
```
* The correlations within numeric variables are set to remove multicollinearity. At least 20 variables with high correlation (with a cutoff of 0.75)

* The following are the variables:

* [1] "accel_belt_z" "roll_belt" "accel_belt_y" "accel_arm_y"
* [5] "total_accel_belt" "accel_dumbbell_z" "accel_belt_x"  "pitch_belt"
* [9] "magnet_dumbbell_x" "accel_dumbbell_y" "magnet_dumbbell_y" "accel_arm_x"
* [13] "accel_dumbbell_x" "accel_arm_z" "magnet_arm_y" "magnet_belt_y"
* [17] "accel_forearm_y" "gyros_arm_y" "gyros_forearm_z" "gyros_dumbbell_x"
```{r}
correlations <- cor(training4)
corrplot(correlations,order = "hclust",tl.cex = .5) 

```

```{r}
highCorr <- findCorrelation(correlations, cutoff = .75) # finding variables with high correlation
predictor <- training4[, -highCorr]                        # dataframe of train predictors
filtered.testing4 <- testing4[, -highCorr]                    # dataframe of test predictors
classe<-training3$classe                                   # target variable
trainData<-cbind(classe,predictor)                      # training dataset ready for prediction
```
* Since the variables derived are good for prediction. Hence, iimplementing the Random Forest Algorithm

```{r}
rfModel <- randomForest(classe ~ .,data = trainData,importance = TRUE,ntrees = 10)
```
#### Confusion Matrix (cross validation matrix):
* With accuracy of the model trained
```{r}
print(rfModel) 
```
#### Error rate plot (At all 5 levels):

* The out of Bag error rate plot illustrates a Black line in the middle is the mean standard error, which is the average overall trees.

```{r}
par(mar=c(3,4,4,4))                               
plot(rfModel)         # Error rate plot for each class
```

```{r}
varImpPlot(rfModel,cex=.5)  # Importance of Variable on Gini Index
```

