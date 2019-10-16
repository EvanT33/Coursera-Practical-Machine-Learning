#Practical Machine Learning Course Project
library(caret)
library(RANN)
library(tictoc)

set.seed(1)

#import
pml_training <- read.csv("~/Coursera Data Science Specialization/Practical Machine Learning/Coursera-Practical-Machine-Learning/pml-training.csv")
validation <- read.csv("~/Coursera Data Science Specialization/Practical Machine Learning/Coursera-Practical-Machine-Learning/pml-testing.csv")


#partition
inTrain <- createDataPartition(y=pml_training$classe, p=0.6, list = FALSE)
training <- pml_training[inTrain,]
testing <- pml_training[-inTrain,]
dim(training)
dim(testing)


#preprocess
preObj <- preProcess(training, method = c("center", "scale", "knnImpute")) 
#subtract mean of each var and divide by standard dev
#also can try method = "BoxCox", which takes continuous data and attempts to fit it to a normal distribution
#knnImpute to deal with missing values. Find nearest neighbors and make a guess at the missing value
train <- predict(preObj, training)
test <- predict(preObj, testing)
val <- predict(preObj, validation)

#remove low var variables
nzv <- nearZeroVar(train, saveMetrics = TRUE)
train <- train[, which(nzv$nzv==FALSE)]


#model (random forest)
modFit_rf <- train(classe~., 
                   method = "rf", 
                   trControl = trainControl(method = "cv", number = 3), 
                   metric = "Accuracy",
                   data = train)

print(modFit_rf) # Accuracy (in-sample) = .9998302

pred_rf <- predict(modFit_rf, test)

confusionMatrix(pred_rf, test$classe) # Accuracy (out-of-sample) = .9997






