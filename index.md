---
title: "Practical Machine Learning Course Project : predicting how an exercise is performed"
author: "Vanni de Clippele"
date: "27 april 2018"
output:
  html_document:
    df_print: paged
    keep_md: yes
  pdf_document: default
self_contained: yes
---




###Introduction 
  
6 test persons have been measured (at arms, belt, dumbbel) while performing a specific physical exercise.
The exercise is either done correctly, or with a common mistake. We will try to predict those classifications
by using the measurements as predictors. 
The classification is captured by the classe variable, which has 5 outcomes : A (correct), B,C,D or E. 
This is the variable we shall try to predict, based on the other variables.

The article describing the set up can be found [here](https://d396qusza40orc.cloudfront.net/repdata%2Fpeer2_doc%2Fpd01016005curr.pdf)
  
###Synopsis  

The project consists of choosing a prediction model, describing the process of model selection, its outcome,
and application to a test set of 20 measurements.

We will set our goal to 95% accuracy (RMSE, since we work with continuous data).
In a first step the high number of dimensions is reduced from 160 to 43.
The resulting dataset is then be randomly split into a training and a test dataset, reflecting the outcome diversity.
Using a random forest approach, a model is selected and applied to the test dataset for validation.
Prediction accuracy results high (> 99%). 

The caret library is used.


```r
library("caret")
```
### Data preprocessing

#### Loading the data  


```r
url_train <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(url=url_train,destfile = "training.csv")
train <-  read.csv("training.csv")

url_test <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url=url_test,destfile = "testing.csv")
test <-  read.csv("testing.csv")

train <-  read.csv("training.csv",stringsAsFactors = FALSE)
test <-  read.csv("testing.csv",stringsAsFactors = FALSE)

dim(train)
```

```
## [1] 19622   160
```

```r
dim(test)
```

```
## [1]  20 160
```

#### Feature selection  

The dataset is large with lots of variables, so we will try to reduce the set of relevant (predictor) variables as much as possible.

First we remove the 67 variables which are mostly NA's (> 97% NA's), all others have no NA's.
Next we remove the 34 variables with no variation.
Finally we remove the variables without a relation to the outcome. From the testset we can see that our model has to predict from isolated testpoints, in the sense that we cannot take into account a time-dependent component, even if looking at the data structure, num_window seems to define the boundaries of an event resulting in its classification. We remove the row identifier X, as well as the timestamps and windows. The model might depend on the username, but for the sake of usability of the model, I will remove it as well. 
As a last step highly correlated columns will be removed (by pairwise comparison, retaining one of each highly correlated pair (cor >0.9)).


```r
## calculate proportion of NA's in columns and remove unusable columns (with high number of NA's).
testNA <- data.frame(names=names(train),colMeans(is.na(train)))
dim(testNA[testNA$colMeans.is.na.train..> 0.97,])
```

```
## [1] 67  2
```

```r
dim(testNA[testNA$colMeans.is.na.train..== 0,])
```

```
## [1] 93  2
```

```r
train <- train[,colnames(train) %in% testNA[testNA$colMeans.is.na.train..== 0,]$names]

## remove columns without variation
NZV <- nearZeroVar(train, saveMetrics = TRUE)
train <- train[,!NZV$nzv]

## remove columns unrelated to the outcome
train <- train[,-c(1,2,3,4,5,6)]

## remove columns highly correlated to others, removing the classe variable from the correlation calculation.
FC <- findCorrelation(cor(train[,-53]))
train <- train[,-FC]

## now we set the classe variable as a factor.
train$classe <- as.factor(train$classe)
```

### Data slicing

We create the test and training partitions (as the provided test set serves evaluation purposes)
the partitioning is balanced on the outcome classe and globally about 70% of the full dataset.


```r
set.seed(5873)
inTrain <- createDataPartition(train$classe,p=0.7,list=FALSE)
training <- train[inTrain,]
testing <- train[-inTrain,]
dim(training)
```

```
## [1] 13737    46
```

```r
dim(testing)
```

```
## [1] 5885   46
```

### Modelling

We still have a lot of variables and a lot of data (noisy, since based on real measurements), 
so a random forest approach might be a good idea. To limit computation time, I will not use the default resampling settings (25 Bootstrapped resamples) but instead start with a k=3-fold cross-validation, and lower the number of trees from 500 to 100. This should give me a good initial idea about the computational abilities of my PC for this problem, as well as how well this approach might perform. Alternatively, I could preprocess the data with principal components analysis in order to further reduce the number of dimensions, before applying any model.


```r
  tc <- trainControl(method = "cv", 3)
  modFitRF <- train(classe ~., method = "rf", data = training,trControl = tc,ntree = 100,allowParallel=TRUE, importance=TRUE) 
  modFitRF
```

```
## Random Forest 
## 
## 13737 samples
##    45 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (3 fold) 
## Summary of sample sizes: 9157, 9159, 9158 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9863142  0.9826859
##   23    0.9894446  0.9866474
##   45    0.9817290  0.9768859
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 23.
```

First results are satisfactory (accuracy>95%). Moreover the plot confirms that 100 trees was more than enough,
as prediction accuracy for all 5 classes exceeds 98% and does not improve significantly from 60 trees onwards.


```r
  plot(modFitRF$finalModel,log="y",main="final model converging over added # of trees")
```

![](index_files/figure-html/fig1-1.png)<!-- -->

### Model validation on the testing dataset

Let us validate with the testing dataset.


```r
  pred <- predict(modFitRF,testing)
  confusionMatrix(testing$classe,pred)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B   10 1121    4    4    0
##          C    0    7 1016    1    2
##          D    0    0    7  954    3
##          E    0    0    2    3 1077
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9927          
##                  95% CI : (0.9902, 0.9947)
##     No Information Rate : 0.2862          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9908          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9941   0.9938   0.9874   0.9917   0.9954
## Specificity            1.0000   0.9962   0.9979   0.9980   0.9990
## Pos Pred Value         1.0000   0.9842   0.9903   0.9896   0.9954
## Neg Pred Value         0.9976   0.9985   0.9973   0.9984   0.9990
## Prevalence             0.2862   0.1917   0.1749   0.1635   0.1839
## Detection Rate         0.2845   0.1905   0.1726   0.1621   0.1830
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9970   0.9950   0.9927   0.9948   0.9972
```

Our out-of-sample error, estimated by applying our model on the new testing dataset, is thus 0.63 % (= 1 - overall accuracy).

Random forests are difficult to interpret as they result in an averaged forest of trees, hence the predicting algorithm as such cannot be plotted in a directly interpretable way. We might however get some insight by looking at the importance of the predictors, defined as their influence on the overall accuracy (what happens to the accuracy when leaving the variable out). Because we limited the # of trees in each forest to 100, and the # of randomly picked variables for each tree is sqrt(43), i.e. 6 or 7 out of 43, its interpretation may be limited.


```r
  varImpt <- varImp(modFitRF)$importance
  head(varImpt)
```

<div data-pagedtable="false">
  <script data-pagedtable-source type="application/json">
{"columns":[{"label":[""],"name":["_rn_"],"type":[""],"align":["left"]},{"label":["A"],"name":[1],"type":["dbl"],"align":["right"]},{"label":["B"],"name":[2],"type":["dbl"],"align":["right"]},{"label":["C"],"name":[3],"type":["dbl"],"align":["right"]},{"label":["D"],"name":[4],"type":["dbl"],"align":["right"]},{"label":["E"],"name":[5],"type":["dbl"],"align":["right"]}],"data":[{"1":"31.3963915","2":"100.00000","3":"64.31966","4":"48.43277","5":"43.65120","_rn_":"pitch_belt"},{"1":"95.3060159","2":"78.45284","3":"75.13368","4":"92.52127","5":"67.44123","_rn_":"yaw_belt"},{"1":"16.4385659","2":"23.99052","3":"18.18258","4":"19.00030","5":"16.98067","_rn_":"total_accel_belt"},{"1":"37.7236470","2":"11.43215","3":"13.58691","4":"10.81521","5":"10.21723","_rn_":"gyros_belt_x"},{"1":"0.8383835","2":"12.77266","3":"13.96355","4":"15.04158","5":"18.24381","_rn_":"gyros_belt_y"},{"1":"30.4006164","2":"47.35116","3":"46.06583","4":"31.07137","5":"40.92749","_rn_":"gyros_belt_z"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
  </script>
</div>

We can however plot how well the model fits the testing data : 


```r
qplot(classe, pred, data=testing,  colour= classe, geom = c("boxplot", "jitter"), main = "predicted vs. observed on testing data", xlab = "observed classe", ylab = "predicted classe")
```

![](index_files/figure-html/fig2-1.png)<!-- -->

### Applying validated model to the 20 provided datapoints


```r
predict(modFitRF, test)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
