library(readr)
mimic_iii <- read_csv("Desktop/case_study/mimic_iii.csv")
mimic_iv <- read_csv("Desktop/case_study/mimic_iv.csv")

library(SuperLearner)
library(glmnet)
library(arm)
library(nnet)
library(polspline)
library(randomForest)
library(gam)
library(ipred)
library(gbm)
library(dbarts)
library(e1071)
library(rpart)


x_vars <- c(
"sapsii_prob",
"age_score",
"hr_score",
"sysbp_score",
"temp_score",
"pao2fio2_score",
"uo_score",
"bun_score",
"wbc_score",
"potassium_score",
"sodium_score",
"bicarbonate_score",
"bilirubin_score",
"gcs_score",
"comorbidity_score",
"admissiontype_score")

X <- mimic_iii[, x_vars]

# Shorthand function to calculate the mode
calculate_mode <- function(x) {
  uniq_x <- unique(na.omit(x))
  uniq_x[which.max(tabulate(match(x, uniq_x)))]
}

# Impute missing values with the mode
X <- as.data.frame(lapply(X, function(x) ifelse(is.na(x), calculate_mode(x), x)))

# Store the modes of each variable from MIMIC III
X_modes <- sapply(X, calculate_mode)

# Get the outcome variable
Y <- mimic_iii$hospital_mortality

# Fit the SuperLearner
SL.library <- c("SL.svm", "SL.glmnet", "SL.bayesglm", "SL.glm", "SL.stepAIC", "SL.nnet", #"SL.bart",
                "SL.polymars", "SL.randomForest", "SL.gam", "SL.ipredbagg", 
                "SL.gbm", "SL.rpartPrune")

fitSL <- SuperLearner(Y = Y, X = X, family = binomial(),
                      SL.library = SL.library, method = "method.NNLS")

# Get relevant features from MIMIC IV
X_new <- mimic_iv[, x_vars]

# Impute missing values in MIMIC IV with the modes from MIMIC III
X_new <- sapply(seq_along(X_new), function(i) {
  ifelse(is.na(X_new[[i]]), X_modes[i], X_new[[i]])
})
X_new <- as.data.frame(X_new)
colnames(X_new) <- colnames(X)

# Predict with SICULA on MIMIC IV
predictions <- predict(fitSL, newdata = X_new, onlySL = TRUE)

pilot_test_set <- as.data.frame(cbind(mimic_iv$hospital_mortality, mimic_iv$sapsii_prob, predictions$pred[,1]))
names(pilot_test_set) <- c("label", "sapsii_prob", "sicula_prob")

# Export the pilot test set as a CSV
write.csv(pilot_test_set, "Desktop/case_study/pilot_test_set.csv", row.names = FALSE)

# Get the AUROC for the two models
roc_obj_sicula <- pROC::roc(pilot_test_set$label, pilot_test_set$sicula_prob)
roc_obj_sapsii <- pROC::roc(pilot_test_set$label, pilot_test_set$sapsii_prob)

# Mean & variance of predictions on cases for both models
sapply(pilot_test_set[pilot_test_set$label==1, ], mean)
sapply(pilot_test_set[pilot_test_set$label==1, ], var)

# Mean & variance of predictions on controls for both models
sapply(pilot_test_set[pilot_test_set$label==0, ], mean)
sapply(pilot_test_set[pilot_test_set$label==0, ], var)
