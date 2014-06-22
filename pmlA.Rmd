---
title: "Human Activity Recognition. Results Verification"
---
This repository contain reproduction of analysis from Groupware <http://groupware.les.inf.puc-rio.br/har>

The purpose of exercise is to create machine learning algorithm to predict activity quality from activity monitors.

Here is a short overview of machine learning program:

### Data preparation

As an input for our learning algorithm we have csv file called pml-training.csv. We load csv into R and removing non sensor features from data. Also we switch class defenitive columnt to be first.

```{r}
trainData <- read.csv("pml-training.csv", na.strings=c("NA",""));
trainData <- trainData[, c(ncol(trainData), 1:ncol(trainData)-1)]

# removing non sensor features from data
uselessData <- grep("timestamp|X|user_name|new_window|num_window", names(trainData))
trainData <- trainData[,-uselessData]
```

In order to simplify the input for our algorithm we reduce number of features by removing all columns containing at least one NA.

```{r}
nonNAFeatures <- colSums(is.na(trainData))==0;
trainData <- trainData[, nonNAFeatures];
```

Then we split our training data into 3 different sets: training set, cross validation set, and test set. We use default **60% train/ 20% cv/ 20% test** proportions.

```{r}
library(caret)

inTrain <- createDataPartition(y = trainData[,1], p = 0.6, list=FALSE)
training <- trainData[inTrain,]

t_cv <- trainData[-inTrain,]
inTrain <- createDataPartition(y = t_cv[,1],  p =  0.5, list=FALSE)
  
cv <- t_cv[inTrain,]
  
test<- t_cv[-inTrain,]

```

Also we reduce the number of features with Dimensionality reduction procedure:
```{r, eval=FALSE }
preproc <- preProcess(trainData[,-1], method='pca', thresh=0.99)
trainData.pca <- predict(preproc, trainData[,-1])     
cvData.pca <- predict(preproc, cvData[,-1])    
testData.pca <- predict(preproc, testData[,-1])    
```

After all this steps we have datasets with the most meaningful 37 features.

# Training

In this analysis  we use 4 different optimization models: random forest, SVM, GBM and LDA.

For training we using our trainData optimized with pca.

```{r, eval=FALSE }
# Train Random Forest
trainControl <- trainControl(method = "cv", number = 10)
modelFitRF <- train(trainData$classe ~., data=trainData.pca, method='rf', trControl = trainControl)
		
# Train SVM
modelFitSVM <- train(trainData$classe ~., data=trainData.pca, method='svmRadial')
	
# Train GBM
modelFitGBM <- train(trainData$classe ~., data=trainData.pca, method="gbm", verbose=FALSE)
	
# Train LDA
modelFitLDA <- train(trainData$classe ~., data=trainData.pca, method="lda")
```

Now we validate the result of our algorithm on cross validation set:

```{r, eval=FALSE }
predictionRF <- predict(modelFitRF,  cvData.pca)
predictionSVM <- predict(modelFitSVM,  cvData.pca)
predictionGBM <- predict(modelFitGBM,  cvData.pca)
predictionLDA <- predict(modelFitLDA,  cvData.pca)
```

Algorithm accuracy for different models:

```{r, eval=FALSE }
> confusionMatrix(predictionRF, cvData$classe)$overall["Accuracy"]
Accuracy 
0.994647 
> confusionMatrix(predictionSVM, cvData$classe)$overall["Accuracy"]
 Accuracy 
0.9469794 
> confusionMatrix(predictionGBM, cvData$classe)$overall["Accuracy"]
 Accuracy 
0.8801937 
> confusionMatrix(predictionLDA, cvData$classe)$overall["Accuracy"]
 Accuracy 
0.5893449 
```
As we see Random Forest did a really good job. It gives us 99% accuracy on cross validation set.

In order to increase accuracy we try to build assemble models.

```{r, eval=FALSE }

# Combine models together
dataCombined <- data.frame(predictionRF,predictionSVM,predictionGBM,predictionLDA,classe=cvData$classe)
modelFitCombined <- train(classe ~.,method="rf", data=dataCombined)
	
predictionCombined <- predict(modelFitCombined,  dataCombined)
confusionMatrix(predictionCombined, dataCombined$classe)
	
# Check final model on test set
testDataCombined <- data.frame(predict(modelFitRF,  testData.pca),predict(modelFitSVM,  testData.pca),predict(modelFitGBM,  testData.pca),predict(modelFitLDA,  testData.pca))

testPrediction <- predict(modelFitCombined, testDataCombined)
confusionMatrix(testPrediction, testData$classe)
Accuracy 
0.9954
```
We gained only additional 0.1% accuracy. Not worth doing assemble for this problem.

### Results
Given csv with proper features we can almost surely predict the class of activity for this problem. In our case random forest gave best result with accuracy 0.994647 on cross validation set.

