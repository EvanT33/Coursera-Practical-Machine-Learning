#Practical Machine Learning Course Project
library(caret)
library(RANN)
library(tictoc)

set.seed(1)

#import
pml_training <- read.csv("~/Coursera Data Science Specialization/Practical Machine Learning/Coursera-Practical-Machine-Learning/pml-training.csv")
validation <- read.csv("~/Coursera Data Science Specialization/Practical Machine Learning/pml-testing.csv")


#partition
inTrain <- createDataPartition(y=pml_training$classe, p=0.6, list = FALSE)
training <- pml_training[inTrain,]
testing <- pml_training[-inTrain,]


#preprocess
training <- training[ ,colSums(is.na(training)) < nrow(training)*0.95]#remove cols with many NAs
training <- training[,-c(1:5)]#do not predict on name or observations number
testing <- testing[,-c(1:5)]#do not predict on name or observations number
validation <- validation[,-c(1:5,160)]#do not predict on name or observations number
nzv <- nearZeroVar(training, saveMetrics = TRUE)#remove low variance predictors
training <- training[, which(nzv$nzv==FALSE)]


#model (random forest)
modFit_rf <- train(classe~., 
                   method = "rf",
                   trControl = trainControl(method = "cv", number = 3), 
                   metric = "Accuracy",
                   data = training)
print(modFit_rf) # Accuracy (in-sample) = 0.9942
pred_rf <- predict(modFit_rf, testing)
confusionMatrix(pred_rf, testing$classe) # Accuracy (out-of-sample) = 0.9978


#apply model to validation set
val_predictions <- rbind(validation$problem_id, as.character(predict(modFit_rf, validation)))
val_predictions


