library(tidyverse) 
library(ggplot2)
library(stringr)
library(caret)
library(quanteda)
library(doSNOW)
library(e1071)
library(irlba)
library(tidytext)
library(textdata)
library(keras)
library(wordcloud)
library(reshape2)
library(tm)
library(ROCR)
library(plotROC)

input_file <- "data/Gungor_2018_VictorianAuthorAttribution_data-train.csv"

# expermentation with our bag-of-words model without tf-idf proved to be more successful in later iterations

# loading in the dataset. This was done to accomodate my system processing prowess of the dataset 
read_data <- function(num) {
  dataset <- read.csv(input_file, header = T, stringsAsFactors = FALSE, nrows = num)
}

# sampling only 2 authors because of the size of the data
df <- read_data(1294)

# rename author column to label and convert to factor
df <- df %>% rename(label = author) 
df$label <- as.factor(df$label)

# stratified sampling
split <- function(data) {
  splits <- sample(1:3, size = nrow(df), prob = c(.5, .2, .3), replace = T)
  training <<- df[splits == 1,]
  validation <<- df[splits == 2,]
  testing <<- df[splits == 3,]
}
split(df)

# shuffle data: in order to reduce variance
shuffleRows <- function(df){
  return(df[sample(nrow(df)),])
}

# Probability distribution
prob_dist <- function(x) {
  prop.table(summary(x))
}
prob_dist(df$label)

training <- shuffleRows(training)
validation <- shuffleRows(validation)
testing <- shuffleRows(testing)

#-----------------------------------------------------------------------------------
# Bag-of-Words Model 
# data preprocessing
bag_of_words <- function(data, word, ngram) {
  corpus <- tokens(data, what = word, remove_numbers = TRUE,remove_punct = TRUE, 
                   remove_symbols = TRUE, ngrams = ngram, remove_url = TRUE)
  corpus <- tokens_tolower(corpus)
  corpus <- tokens_select(corpus, stopwords(), selection = "remove")
  corpus <- tokens_wordstem(corpus, language = quanteda_options("language_stemmer"))
  corpus_dfm <- dfm(corpus, tolower = F)
}

corpus_prep <- function(training) {
  corpus_dfm <- bag_of_words(training$text, "word",  ngram = 1L)
  corpus <- as.matrix(corpus_dfm)
  corpus <- cbind(label = training$label, data.frame(corpus))
}

corpus <- corpus_prep(training)
names(corpus) <-  make.names(names(corpus))

corpus_process <- function(training) {
  corpus_dfm <- bag_of_words(training$text, "word",  ngram = 1L)
  corpus_tfidf <- dfm_tfidf(corpus_dfm, scheme_tf = "count", scheme_df = "inverse", base = 10)
  corpus_tfidf <- cbind(label = training$label, convert(corpus_tfidf, to = "data.frame"))
}

corpus_tfidf <- corpus_process(training)
names(corpus_tfidf) <-  make.names(names(corpus_tfidf))

# 10 fold cross validation 
cv_folds <- function() {
  multifolds <- createMultiFolds(validation$label, k = 10, times = 2)
  control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, 
                          index = multifolds, verboseIter = T)
}
cv <- cv_folds()


#================= Random Forest Model =============================
# further cleaning
corpus_tfidf <- droplevels(corpus_tfidf) 
colnames(corpus_tfidf) <- make.names(colnames(corpus_tfidf))
corpus_tfidf <- corpus_tfidf[, !duplicated(colnames(corpus_tfidf))]

# Random Forest Model 
model <- function(label, ., corpus, method, cv) {
  cm <- train(label~., data = corpus, method = method, trControl = cv, tuneLength = 7)
}

# Training the model 
cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)
start_time <- Sys.time()
rf_model <- model(label, ., corpus, "rf", cv)
stopCluster(cl)
Sys.time() - start_time

rf_model

# Checking var importance
var_imp <- varImp(rf_model)

svm_imp <- varImp(svm_model)
tibble(var_imp)
ggplot(rf_model, top = dim(data$importance)[1]) +
  labs(title="Random Forest Model")
ggsave("var_imp.png")

#============ BOW test set prepation ==============
test_preprocess <- function(data, word, ngram, corpus) {
  
  corpus_test <- tokens(data, what = word, remove_numbers = TRUE,remove_punct = TRUE, 
                        remove_symbols = TRUE, ngrams = ngram, remove_url = TRUE)
  corpus_test <- tokens_tolower(corpus_test)
  corpus_test <- tokens_select(corpus_test, stopwords(), selection = "remove")
  corpus_test <- tokens_wordstem(corpus_test, language = quanteda_options("language_stemmer"))
  corpus_test <- dfm(corpus_test, tolower = F)
  corpus_test <- dfm_select(corpus_test, pattern = corpus, selection = "keep")
}

 test_prep <- function(testing, corpus_dfm) {
  corpus_test <<- test_preprocess(testing$text, "word",  ngram = 1L, corpus_dfm)
  test_tfidf <- dfm_tfidf(corpus_test, scheme_tf = "count", scheme_df = "inverse", base = 10)
  test_tfidf <- cbind(label = testing$label, convert(test_tfidf, to = "data.frame"))
}


# TFIDF testing corpus
test_tfidf <- test_prep(testing, corpus_dfm)
names(test_tfidf) <-  make.names(names(test_tfidf))

# Without tfifdf 
corpus_test <- dfm(corpus_test)
corpus_test <- as.matrix(corpus_test)
corpus_test <- cbind(label = testing$label, data.frame(corpus_test))

#===========Predictions on our preprocessed test set===================== 
names(test_tfidf) <- make.names(colnames(test_tfidf))
rf_predictions_1 <- predict(rf_model, test_tfidf)
rf_predictions_1

#=======Predictions on non-tfidf corpus===========
names(corpus_test) <- make.names(colnames(corpus_test))
rf_predictions_2 <- predict(rf_model, corpus_test)
rf_predictions_2

#========= Model Evaulation ==============
# Misclassifcation rate 
1 - sum(diag(tab))/sum(tab)
rf_pred <- predict(rf_model, corpus_test, type = "prob")
rf_pred <- prediction(rf_pred$"2", corpus_test$label)
evaluation <- performance(rf_pred, "acc")
plot(evaluation, xlab = "Cutoff", ylab = "Accuracy")
abline(h = 1, v = 0.408)
dev.off()

# Reciever Operating Characteristics (ROC) curve  
roc <- performance(rf_pred, "tpr", "fpr")
plot(roc, main = "ROC Curve", ylab = "Sensitivity", xlab = "1 - Specificity", colorize = T )
abline(a = 0, b = 1)

# AUC (Area under the curve) embeded 
auc <- performance(rf_pred, "auc")
auc <- unlist(slot(auc, "y.values"))
auc <- round(auc, 3)
legend(.7, .3, auc, title = "AUC", cex = 1.3)


# Confusion matrix 
(rftest_cm <- confusionMatrix(rf_predictions, corpus_test$label, mode = "sens_spec"))
confusion_matrix(rftest_cm, "Random Forest") 

# Save random forest model
saveRDS(rf_model, file = "./rf_model.rds")
# Load random forest model 
rf_model <- readRDS("./rf_model.rds")

# Cross validation sampling
plot(rf_model)

#=========================Support Vector Machines-linear kernel =========================

# svm 10 fold cross validation
cv_folds2 <- function() {
  multifolds <- createMultiFolds(validation$label, k = 10, times = 2)
  control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, verboseIter = T,
                          index = multifolds, classProbs = TRUE)
}
cv_svm <- cv_folds2()
  
svm_model <- function(corpus_tfidf, method, cv_svm) {
  svm <- train(make.names(label)~., data = corpus_tfidf, method = method, preProc = c("center","scale"), metric="Accuracy",
                 trControl = cv_svm, tuneLength = 10)
}

# further cleaning 
corpus_tfidf <- droplevels(corpus_tfidf) 
colnames(corpus_tfidf) <- make.names(colnames(corpus_tfidf))
corpus_tfidf <- corpus_tfidf[, !duplicated(colnames(corpus_tfidf))]


# Training the nodel 
cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)
start_time <- Sys.time()
svm_model <- svm_model(corpus_tfidf, "svmLinear", cv_svm)
stopCluster(cl)
Sys.time() - start_time

# Saving svm model
saveRDS(svm_model, file = "./svm_model.rds")
# Loading svm model
svm_model <- readRDS("./svm_model.rds")

#===========Predictions on our preprocessed test set===================== 
colnames(test_tfidf) <- make.names(colnames(test_tfidf))
svm_predictions <- predict(svm_model, corpus_test)
svm_predictions

# Confusion matrix 
cm_prep <- function(svm_predictions, testing) {
  svm_pred_df <- as.data.frame(svm_predictions)
  svm_pred_df <- svm_pred_df %>% mutate(svm_predictions = ifelse(svm_predictions == "X1", 1, 2))
  svm_pred_df$svm_predictions <- as.factor(svm_pred_df$svm_predictions)
  svm_cm <- confusionMatrix(svm_pred_df$svm_predictions, testing$label)
}

svm_cm <- cm_prep(svm_predictions, testing)

# Visualizations
confusion_matrix(svm_cm, "SVM Model") 

#====================== Naive Bayes =======================================
# using kernel density estimate and laplace smoother

search_grid <- expand.grid(usekernel = c(TRUE, FALSE), fL = 0:5, adjust = seq(0, 5, by = 1))

nb_model <- function(label, ., corpus_tfidf, method, cv, search_grid) {
  nb_train <- train(make.names(label)~., data = corpus_tfidf, method = method, preProc = c("BoxCox", "center","scale", "pca"),
               trControl = cv, tuneLength = 10, tuneGrid = search_grid)
}

cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)
start_time <- Sys.time()
nb_model <- model(label, ., corpus, "nb", cv)
stopCluster(cl)
Sys.time() - start_time

# Checking results
nb_model

# Save random forest model
saveRDS(nb_model, file = "./nb_model.rds")
# Load random forest model 
nb_model <- readRDS("./nb_model.rds")


#=== Predictions =======
nb_predictions <- predict(nb_model, corpus_test)
nb_predictions