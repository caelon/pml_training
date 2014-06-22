pmlA <- function()
{
	library(caret)

	# Prepare data
	pmlData <- prepareData();
	trainData <- pmlData$trainData;
	cvData <- pmlData$cvData;
	testData <- pmlData$testData;
	submissionData <- pmlData$submissionData;
	
	# Dimensionality reduction
	preproc <- preProcess(trainData[,-1], method='pca', thresh=0.99)
	trainData.pca <- predict(preproc, trainData[,-1])     
	cvData.pca <- predict(preproc, cvData[,-1])    
	testData.pca <- predict(preproc, testData[,-1])    
	
	# Train Random Forest
	trainControl <- trainControl(method = "cv", number = 10)
	modelFitRF <- train(trainData$classe ~., data=trainData.pca, method='rf', trControl = trainControl)
		
	# Train SVM
	modelFitSVM <- train(trainData$classe ~., data=trainData.pca, method='svmRadial')
	
	# Train GBM
	modelFitGBM <- train(trainData$classe ~., data=trainData.pca, method="gbm", verbose=FALSE)
	
	# Train LDA
	modelFitLDA <- train(trainData$classe ~., data=trainData.pca, method="lda")
		
	predictionRF <- predict(modelFitRF,  cvData.pca)
	predictionSVM <- predict(modelFitSVM,  cvData.pca)
	predictionGBM <- predict(modelFitGBM,  cvData.pca)
	predictionLDA <- predict(modelFitLDA,  cvData.pca)
	
	# Combine models together
	dataCombined <- data.frame(predictionRF,predictionSVM,predictionGBM,predictionLDA,classe=cvData$classe)
	modelFitCombined <- train(classe ~.,method="rf", data=dataCombined)
	
	predictionCombined <- predict(modelFitCombined,  dataCombined)
	confusionMatrix(predictionCombined, dataCombined$classe)
	
	# Check final model on test set
	testDataCombined <- data.frame(predict(modelFitRF,  testData.pca),predict(modelFitSVM,  testData.pca),predict(modelFitGBM,  testData.pca),predict(modelFitLDA,  testData.pca))
	
	testPrediction <- predict(modelFitCombined, testDataCombined)
	confusionMatrix(testPrediction, testData$classe)
	
	save(modelFitRF, modelFitSVM, modelFitGBM, modelFitLDA, pmlData, preproc, file="pmlA.RData")
	
	modelFitCombined
}

prepareData <- function(trainProportion = 0.6, cvProportion=0.5)
{
	trainData <- read.csv("pml-training.csv", na.strings=c("NA",""));
	trainData <- trainData[, c(ncol(trainData), 1:ncol(trainData)-1)]
	
	submissionData <- read.csv("pml-testing.csv", na.strings=c("NA",""));
	submissionData <- submissionData[, c(ncol(submissionData), 1:ncol(submissionData)-1)]
	
	# removing non sensor features from data
	uselessData <- grep("timestamp|X|user_name|new_window|num_window", names(trainData))
	trainData <- trainData[,-uselessData]
	submissionData <- submissionData[, -uselessData]
	
	# removing all columns with at least one na in them
	nonNAFeatures <- colSums(is.na(trainData))==0;
	trainData <- trainData[, nonNAFeatures];
	submissionData <- submissionData[, nonNAFeatures]
	
	dataSplit <- splitData(trainData, trainProportion, cvProportion);
	
	list(trainData = dataSplit$training, cvData = dataSplit$cv, testData = dataSplit$test, submissionData = submissionData)
}


splitData <- function (dataset, t=0.6, cv =0.5) 
{
	inTrain <- createDataPartition(y = dataset[,1], p = t, list=FALSE)
	training <- dataset[inTrain,]
	
	t_cv <- dataset[-inTrain,]
	inTrain <- createDataPartition(y = t_cv[,1],  p =  cv, list=FALSE)
	
	cv <- t_cv[inTrain,]
	
	test<- t_cv[-inTrain,]
	
	list(training = training, cv = cv, test = test)
}
