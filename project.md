# PracticalMachineLearningProject
Danilo Lofaro  
14 febbraio 2016  

### The Project

#### Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 


#### Data

The goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of the 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: [linked phrase](http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

The training data for this project are available at [linked phrase](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) 

The test data are available at [linked phrase](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)

The data for this project come from this source: [linked phrase](http://groupware.les.inf.puc-rio.br/har)


#### Goal

Goal of the project is to predict the manner in which the participants did the exercise. This is the "classe" variable in the training set, using any of the other variables to predict with. 
A report describing how the model is built, how cross validation was used, what is the expected out of sample error. The prediction model will be used to predict 20 different test cases. 

#### Loading, Getting and Pre-processing data


```r
set.seed(12345)

trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

training <- read.csv(url(trainUrl), na.strings=c("NA","#DIV/0!",""))
testing <- read.csv(url(testUrl), na.strings=c("NA","#DIV/0!",""))

dim(training)
```

```
## [1] 19622   160
```

```r
dim(testing)
```

```
## [1]  20 160
```

<br>
Backup of original data

```r
training_original <- training
testing_original <- testing
```

<br>
Remove 1st (`ID`) variable 

```r
training <- training[,-1]
```

<br>
Remove NearZeroVariance variables

```r
library(caret)

nzv <- nearZeroVar(training, saveMetrics=TRUE)
training <- training[,nzv$nzv==FALSE]
```

<br>
Remove variables with > 60% `NA` values 

```r
training <- training[,colSums(is.na(training))<.4*NROW(training)]
```

<br>
Partioning the training set into training and test datasets

```r
inTrain <- createDataPartition(training$classe, p=0.6, list=FALSE)

myTraining <- training[inTrain, ]
myTesting <- training[-inTrain, ]

dim(myTraining)
```

```
## [1] 11776    58
```

```r
dim(myTesting)
```

```
## [1] 7846   58
```

#### Prediction Model

Using the dataset `myTraining` and Linear SVM we create the prediction model. Since the time needed to run the model we will test a single prediction model.


```r
library(parallel)
library(doParallel)

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

reg_Control <- trainControl("cv", number = 5, verboseIter = TRUE)
    
fit <- train(myTraining$classe ~ .,data = myTraining, method="svmLinear2", tuneLength = 3,preProc=c('scale','center'),trControl = reg_Control)
```

```
## Aggregating results
## Selecting tuning parameters
## Fitting cost = 1, gamma = 2 on full training set
```

```r
stopCluster(cl)
```

Accuracy Plot

```r
plot(fit)
```

![](project_files/figure-html/plot_fit-1.png)

Confusion Matrix on `myTraining` set. 

```r
predFit<- predict(fit, myTraining)
confusionMatrix(predFit, myTraining$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3272  246   13    0    0
##          B   64 1903  140   13    0
##          C   12  130 1886  136    5
##          D    0    0   14 1688  103
##          E    0    0    1   93 2057
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9176          
##                  95% CI : (0.9125, 0.9225)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8956          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9773   0.8350   0.9182   0.8746   0.9501
## Specificity            0.9693   0.9772   0.9709   0.9881   0.9902
## Pos Pred Value         0.9266   0.8976   0.8695   0.9352   0.9563
## Neg Pred Value         0.9908   0.9611   0.9825   0.9757   0.9888
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2779   0.1616   0.1602   0.1433   0.1747
## Detection Prevalence   0.2998   0.1800   0.1842   0.1533   0.1827
## Balanced Accuracy      0.9733   0.9061   0.9445   0.9314   0.9702
```

<br> 
Accuracy on `myTraining` set is aroung 90%. Check the performance on `myTesting`

```r
predTest<- predict(fit, myTesting)
confusionMatrix(predTest, myTesting$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2170  152    8    0    0
##          B   52 1244   87    9    0
##          C   10  122 1257  100    5
##          D    0    0   16 1102   74
##          E    0    0    0   75 1363
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9095          
##                  95% CI : (0.9029, 0.9158)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8854          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9722   0.8195   0.9189   0.8569   0.9452
## Specificity            0.9715   0.9766   0.9634   0.9863   0.9883
## Pos Pred Value         0.9313   0.8937   0.8414   0.9245   0.9478
## Neg Pred Value         0.9888   0.9575   0.9825   0.9723   0.9877
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2766   0.1586   0.1602   0.1405   0.1737
## Detection Prevalence   0.2970   0.1774   0.1904   0.1519   0.1833
## Balanced Accuracy      0.9719   0.8981   0.9411   0.9216   0.9668
```

<br>
Accuracy on `myTesting` remain aroung 90%. Now let's check the prediction on the 20 test cases 

```r
predCases <- predict(fit, testing)
predCases
```

```
##  [1] B A B A A E D B A B B C B A E E A B B B
## Levels: A B C D E
```

<br>
#### Conclusions

The model predicted the 20 test cases with 100% accuracy. All 20 points were awarded after submitting the 20 test files.
