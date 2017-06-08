#########################################################################################
######################### Read in data and library assembling ###########################
setwd("~/Desktop/Bit_tiger/数据科学家/wk1")
train <- read.csv("train.csv", stringsAsFactors = FALSE)
#install.packages("xgboost")
#install.packages("drat", repos="https://cran.rstudio.com")
#drat:::addRepo("dmlc")
#install.packages("xgboost", repos="http://dmlc.ml/drat/", type = "source")
#install.packages("DiagrammeR")
install.packages("devtools")
library(devtools)
install_version("DiagrammeR", version = "0.8.1", repos = "http://cran.us.r-project.org")
install.packages("Ckmeans.1d.dp")
library(Ckmeans.1d.dp)
library(lattice)
library(corrplot)
library(tabplot)
library(data.table)
library(glmnet)
library(mice)
library(car)
library(rpart)
library(randomForest)
library(xgboost)
library(DiagrammeR)

#########################################################################################
############################## Understand the data set ##################################
##Overall look
dim(train)
class(train)      #all the dataset is data.frame, no need for class transformation
str(train)        #int(continuous, category), character(> facotrs for modeling)
names(train)      #frequent strings:Bsmt, SF, Garage, Bath, #SalePrice as response
length(unique(train$Id)) #But Id might be removed for modeling, useless for prediction
summary(train)    #look at SalePrice, max is much bigger than 3rd quantile
                  #need transformation?
head(train)       #observe a lot of zero and NA
head(rownames(train))
##check response var SalePrice
summary(train$SalePrice)   #assume skewed distribution
str(train$SalePrice)
quantile(train$SalePrice, c(0.1, 0.25, 0.5, 0.75, 0.9, 1))   #1:=max
sort(table(train$SalePrice), decreasing=TRUE)
plot(density(train$SalePrice))   #visualize distribution
log_SalePrice <- log(train$SalePrice)    #just see the transformation, not make changes to dataset
plot(density(log_SalePrice)) #it becomes more normal
                             #So we consier act log transformation on SalePrice when modeling
boxplot(train$SalePrice)
boxplot(log(train$SalePrice))


#########################################################################################
######################### Data exploration and attibute selection #######################
## 1.Intuitions of categorical related features
#OverallQual, consider bin
table(train$OverallQual)
barplot(table(train$OverallQual)) #plot their frequency plot
bwplot(OverallQual ~ SalePrice, data = train)  #SalePrice increse as OverallQual level incre 
boxplot(subset(train, OverallQual <= 5)$SalePrice, 
        subset(train, OverallQual > 5)$SalePrice,
        xlab = "Overall Quality", ylab = "SalesPrice", names = c("Low Qual", "High Qual"), col = c("red", "green"))
boxplot(log(subset(train, OverallQual <= 5)$SalePrice + 1), #So we consider bin this feature
        log(subset(train, OverallQual > 5)$SalePrice + 1),
        xlab = "Overall Quality", ylab = "log(SalesPrice)", names = c("Low Qual", "High Qual"), col = c("red", "green"))
#OverallCond, consider bin
table(train$OverallCond)
barplot(table(train$OverallCond)) #plot their frequency plot
bwplot(OverallCond ~ SalePrice, data = train)
boxplot(log(subset(train, OverallCond <= 5)$SalePrice + 1), #So we consider bin this feature
        log(subset(train, OverallCond > 5)$SalePrice + 1),
        xlab = "Overall Condition", ylab = "log(SalesPrice)", names = c("Low Qual", "High Qual"), col = c("blue", "red"))
#Neighborhood, consider to bin
table(train$Neighborhood)
barplot(table(train$Neighborhood))
bwplot(Neighborhood ~ SalePrice, data = train)  #Not that good, but c("StoneBr","NridgHt","Noridge")  might be combined as excellent neibor
#GarageType,
table(train$GarageType)
barplot(table(train$GarageType))
bwplot(GarageType ~ SalePrice, data = train)

## 2.Intuitions of continuous realted feature
#for cont, we can plot their relationship with SalePrice, or just lm() to see significance
with(train, plot(X1stFlrSF, SalePrice))
with(train, plot(X1stFlrSF, log(SalePrice))) 
with(train, plot(LotArea, SalePrice))
with(train, plot(LotArea, log(SalePrice))) 
lm1 <- lm(log(SalePrice) ~ X1stFlrSF, data = train)
summary(lm1)
lm2 <- lm(log(SalePrice) ~ LotArea, data = train)
summary(lm2)
lm3 <- lm(log(SalePrice) ~ TotalBsmtSF, data = train)
summary(lm3)
#or see it using library(tabplot)
tableplot(train, select = c(SalePrice, LotArea))  #feature sort by 100bins of SalePrice

## 3. Analysis of correlations
with(train, cor(LotArea,SalePrice))
cor(train)
#select out numeric features for correlation calculation
numVar <- names(train)[which(sapply(train, is.numeric))]
numVar   #toally 38 numeric out of 81, this is a vector of col names
str(numVar) 
train_num <- train[, numVar]
cor(train_num)   #find a lot of NA because of unpaired data, and need to delete Id
correlations <- cor(train_num[, -1], use = "pairwise.complete.obs")
correlations
corrplot(correlations, method = "square")  #library(corrplot)
#select features with high corr, >0.5, check the value with SalePrice
#要留下的feature是与至少另外一个feature有较强的cor的(condition set as 0.5),
#so use sum 来算出每个feature和多少个其他feature存在较强的cor and check if it is >1,因为每个feature与自身的correlation都是1
rowInd=apply(correlations, 1, function(x) return(sum(x>0.5|x< -0.5)>1))
rowInd #returns T or F that vars have high corr with others,we want to keep those Ts feature
cor <- correlations[rowInd, rowInd]
dim(cor) #totally 17 vars have >0.5 corr with at least one other features
corrplot(cor, method = "square") #corrplot(correlations[rowInd, rowInd], method = "square")
#or use: labrary(lattice)
levelplot(correlations)
levelplot(cor)

## 4. missing data check
sapply(train, function(x) {length(which(is.na(x)))})
sort(sapply(train, function(x) {length(which(is.na(x)))}), decreasing=TRUE)
#or use: sort(sapply(train, function(x) {sum(is.na(x))}), decreasing=TRUE)
#c("Alley", "PoolQC", "Fence", "MiscFeature") features with missing values over 1000
#c("FireplaceQu", "LotFrontage") features with missing values over 200
sum(is.na(train)) / (nrow(train) *ncol(train))  #0.0589, percentage of total missings in "train".


#########################################################################################
############################# Pre-processing of data ####################################
## 1. Large missing data column
sort(sapply(train, function(x) {length(which(is.na(x)))}), decreasing=TRUE)
#Alley: NA means no Alley not missing, we need to reassign value to NA,
train1 <- as.data.table(train)
is.data.table(train1)
trian1 <- train1[,Alley:=ifelse(is.na(Alley),'noAlley', Alley)]
train1$Alley
bwplot(Alley ~ SalePrice, data = train1) #delete
#PoolQC: NA means no Pool not missing
trian1 <- train1[,PoolQC:=ifelse(is.na(PoolQC),'noPool', PoolQC)]
train1$PoolQC
bwplot(PoolQC ~ SalePrice, data = train1) #reassign NA and keep
#Fence
trian1 <- train1[,Fence:=ifelse(is.na(Fence),'noFence', Fence)]
train1$Fence
bwplot(Fence ~ SalePrice, data = train1) #delete
#MiscFeature
trian1 <- train1[,MiscFeature:=ifelse(is.na(MiscFeature),'None', MiscFeature)]
train1$MiscFeature
bwplot(MiscFeature ~ SalePrice, data = train1) #delete
#FireplaceQu
trian1 <- train1[,FireplaceQu:=ifelse(is.na(FireplaceQu),'NoFP', FireplaceQu)]
train1$FireplaceQu
bwplot(FireplaceQu ~ SalePrice, data = train1) #reassign NA and keep
#LotFrontage has missing records, delete
head(train1)
class(train1)
train2 <- train1[, c("LotFrontage", "MiscFeature", "Fence", "Alley","Id") := NULL]  #in library(data,table)
dim(train2)

## 2. Missing data columns with Bsmt
a <- sort(sapply(train2, function(x) {length(which(is.na(x)))}), decreasing=TRUE)
missVar <- names(a)[sapply(a, function(x) {all(x)>0})]
str(missVar)
train_miss <- train[, missVar]
colnames(train_miss)[which(grepl("Bsmt", colnames(train_miss)))]
with(subset(train2, is.na(BsmtExposure)), summary(BsmtUnfSF))##"unfinish"的变量有值，说明这些obs的basement都没有finish
#we notive all the corresponding records by BsmtExposure is 38, mathced, they should be from same records
#They all should be unfinished
with(subset(train2, is.na(BsmtExposure)), summary(BsmtFinType1))
with(subset(train2, is.na(BsmtExposure)), summary(BsmtFinType2))
with(subset(train2, is.na(BsmtExposure)), summary(BsmtQual))
with(subset(train2, is.na(BsmtExposure)), summary(BsmtCond))
#So we mark those houses(records) all as unfinished
train2$BsmtExposure[which(is.na(train2$BsmtExposure))] <- 'Unf'
train2$BsmtFinType1[which(is.na(train2$BsmtFinType1))] <- 'Unf'
train2$BsmtFinType2[which(is.na(train2$BsmtFinType2))] <- 'Unf'
train2$BsmtQual[which(is.na(train2$BsmtQual))] <- 'Unf'
train2$BsmtCond[which(is.na(train2$BsmtCond))] <- 'Unf'
sort(sapply(train2, function(x) {length(which(is.na(x)))}), decreasing=TRUE)

## 3. Missing data columns with Garage, they all have the same missings as well
#the missing obs mean no garage in those houses
train2$GarageType[which(is.na(train2$GarageType))] <- 'noGar'
train2$GarageFinish[which(is.na(train2$GarageFinish))] <- 'noGar'
train2$GarageQual[which(is.na(train2$GarageQual))] <- 'noGar'
train2$GarageCond[which(is.na(train2$GarageCond))] <- 'noGar'
sort(sapply(train2, function(x) {length(which(is.na(x)))}), decreasing=TRUE)
#for GarageYrBlt, we find it is the same as YearBuilt
#just delete it
class(train2)
train3 <- train2[, "GarageYrBlt":= NULL]
dim(train3)
sort(sapply(train3, function(x) {length(which(is.na(x)))}), decreasing=TRUE) #now only 3 features with NA

## 4. Remove the records with tiny percentage of missing values
train4 <- subset(train3,!is.na(MasVnrType))
train4 <- subset(train4,!is.na(MasVnrArea))
train4 <- subset(train4,!is.na(Electrical))
sort(sapply(train4, function(x) {length(which(is.na(x)))}), decreasing=TRUE)
dim(train4)
#train4 here is the dataset after directly removing records method
##############or we do imputation in library(MICE)
##############
#train4 <- mice(train3, m=1, printFlag=FALSE)
#sort(sapply(complete(train4), function(x) { sum(is.na(x)) }), decreasing=TRUE)
#train3$MasVnrType <- as.factor(train3$MasVnrType)
#train3$Electrical <- as.factor(train3$Electrical)
#train4 <- mice(train3, m=1, method='cart', printFlag=FALSE)
#sort(sapply(complete(train3), function(x) { sum(is.na(x)) }), decreasing=TRUE)
##############
##############

## 5. Bin vars to reduce levels
#check zero_var features
zero_variance_variables <- Filter(function(x) {length(unique(x)) == 1}, train4)
summary(zero_variance_variables) #no features with all same values for all records
table(train4$OverallCond)
train4$SimOverallCond <- with(train4, ifelse(OverallCond <= 3, "low",
                                           ifelse(OverallCond <= 6, "med", "high")))
names(train3) #check the new var has been added to data set
train4 <- train4[, "OverallCond":= NULL]
dim(train4)
table(train4$OverallQual)
train4$SimOverallQual <- with(train4, ifelse(OverallQual <= 3, "low",
                                             ifelse(OverallQual <= 7, "med", "high")))
names(train4)
train4 <- train4[, "OverallQual":= NULL]
dim(train4)
table(train4$Neighborhood)
bwplot(Neighborhood ~ SalePrice, data = train4) 
train4$Neighborhood[train4$Neighborhood=="StoneBr"] <- "advanced"
train4$Neighborhood[train4$Neighborhood=="NridgHt"] <- "advanced"
train4$Neighborhood[train4$Neighborhood=="NoRidge"] <- "advanced"
train4$Neighborhood[train4$Neighborhood=="Blmngtn"] <- "ordinary"
train4$Neighborhood[train4$Neighborhood=="Blueste"] <- "ordinary"
train4$Neighborhood[train4$Neighborhood=="BrDale"] <- "ordinary"
train4$Neighborhood[train4$Neighborhood=="BrkSide"] <- "ordinary"
train4$Neighborhood[train4$Neighborhood=="ClearCr"] <- "ordinary"
train4$Neighborhood[train4$Neighborhood=="CollgCr"] <- "ordinary"
train4$Neighborhood[train4$Neighborhood=="Crawfor"] <- "ordinary"
train4$Neighborhood[train4$Neighborhood=="Edwards"] <- "ordinary"
train4$Neighborhood[train4$Neighborhood=="Gilbert"] <- "ordinary"
train4$Neighborhood[train4$Neighborhood=="IDOTRR"] <- "ordinary"
train4$Neighborhood[train4$Neighborhood=="MeadowV"] <- "ordinary"
train4$Neighborhood[train4$Neighborhood=="Mitchel"] <- "ordinary"
train4$Neighborhood[train4$Neighborhood=="Names"] <- "ordinary"
train4$Neighborhood[train4$Neighborhood=="NAmes"] <- "ordinary"
train4$Neighborhood[train4$Neighborhood=="NPkVill"] <- "ordinary"
train4$Neighborhood[train4$Neighborhood=="NWAmes"] <- "ordinary"
train4$Neighborhood[train4$Neighborhood=="OldTown"] <- "ordinary"
train4$Neighborhood[train4$Neighborhood=="SWISU"] <- "ordinary"
train4$Neighborhood[train4$Neighborhood=="Sawyer"] <- "ordinary"
train4$Neighborhood[train4$Neighborhood=="SawyerW"] <- "ordinary"
train4$Neighborhood[train4$Neighborhood=="Somerst"] <- "ordinary"
train4$Neighborhood[train4$Neighborhood=="Timber"] <- "ordinary"
train4$Neighborhood[train4$Neighborhood=="Veenker"] <- "ordinary"
with(train4,table(Neighborhood))
table(train4$GarageType)
bwplot(GarageType ~ SalePrice, data = train4)

## 6. From correlations, combine and adjust features
#find: 1stFlrSF + 2ndFlrSF = GrLivArea, just remove 1stFlrSF and 2ndFlrSF
train4 <- train4[, c("X1stFlrSF", "X2ndFlrSF"):= NULL]
dim(train4)
names(train4)
train4$AllSF <- with(train4, GrLivArea + TotalBsmtSF) #add AllSF to the dataset
with(train4, cor(log(SalePrice), AllSF))
with(train4, cor(log(SalePrice), X1stFlrSF))
with(train4, cor(log(SalePrice), X2ndFlrSF))
with(train4, cor(log(SalePrice), GrLivArea))
train4 <- train4[, c("GrLivArea", "TotalBsmtSF"):= NULL]
dim(train4)
# Total number of bathrooms
train4$TotalBath <- with(train4, BsmtFullBath + 0.5 * BsmtHalfBath + FullBath + 0.5 * HalfBath)
train4 <- train4[, c("BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath"):= NULL]
dim(train4)
#Notice house remodel, we think remodel should have influence on saleprice, we need to check our intuition
train4$YearBuilt
train4$YearRemodAdd #We can find houses without remodel have same year records for the two features
#builtyear and remodel year should have relations, we check using plot
with(train4, plot(YearBuilt, YearRemodAdd))
#to see builtyear not equal to remodel years records
dim(subset(train4, YearBuilt != YearRemodAdd))
train4$remod <- with(train4, ifelse(YearBuilt != YearRemodAdd, 1, 0))
table(train4$remod) # 692 remodeled
##way to check response become more normal
boxplot(log(subset(train4, remod == 1)$SalePrice),
        log(subset(train4, remod == 0)$SalePrice)) #more symmetric
names(train4)
train4 <- train4[, c("YearRemodAdd"):= NULL]
dim(train4)
tableplot(train4, select = c(SalePrice, GarageCars))
tableplot(train4, select = c(SalePrice, GarageArea))
#check correlations again for numeric vars
numTrain4 <- names(train4)[which(sapply(train4, is.numeric))]
numTrain4
train5 <- as.data.frame(train4)
train5_num <- train5[, numTrain4]
corr1 <- cor(train5_num)
corrplot(corr1, method = "square")
highcorr=apply(corr1, 1, function(x) return(sum(x>0.85|x< -0.85)>1))
highcorr #returns T or F that vars have high corr with others,we want to keep those Ts feature
corhigh <- corr1[highcorr, highcorr]
corrplot(corhigh)
lm0 <-lm(GarageArea ~ GarageCars, data=train4)
summary(lm0)
dim(train4)

##Create dummy var
numcols <- seq(train6)[which(sapply(train6, is.numeric))] #select numeric features out
train6 <- as.data.frame(train6)
train7 <- train6[, -c(3,15,21,29,31,32,37,38,39,41,43,47,48,52,53,54,55,56,57,59,60,61,66,67,69)] #train7 are all categories for dummy
for(i in 1:dim(train7)[2]) {
  if(is.character(train7[, i])) {
    train7[, i] <- as.factor(train7[, i])
  }
}
for(i in 1:dim(train7)[2]) {
  if(is.numeric(train7[, i])) {
    train7[, i] <- as.factor(train7[, i])
  }
}
str(train7)
length(train7)
ind <- model.matrix( ~., train7) #ind: category data matrix after getting dummy var
#We need to rescale numeric vars:
numer <- train6[, c(3,15,21,29,31,32,37,38,39,41,43,47,48,52,53,54,55,56,57,59,60,61,66,67)]
numer <- as.matrix(numer)
numer <- scale(numer)
class(numer)
pre <- cbind(numer, ind, train6$Log_Price)
dim(pre)

#########################################################################################
######################################  Modeling  #######################################
##Simplest lm(), train4 should be the dataset for modeling
pre <- as.data.frame(pre)
class(pre)
mod1 <- lm(V233 ~ ., data=pre)
summary(mod1) #R-square:0.9228
rmse1 <- sqrt(mean(mod1$residuals^2))
sort(summary(mod1)$coeff[-1,1]) #select most influential var
#ignore the intercept pval
summary(mod1)$coeff[-1,4] < 0.05
toselect.x <- summary(mod1)$coeff[-1,4] < 0.05 # credit to kith
# select sig. variables
sig.x <- names(toselect.x)[toselect.x == TRUE]
# formula with only sig variables
sig.formula <-as.formula(paste("V233 ~",paste(sig.x, collapse= "+")))
sig.mod1 <- lm(formula=sig.formula,data=pre)
summary(sig.mod1)
sig.rmse1 <- sqrt(mean(sig.mod1$residuals^2))

##glmnet, add regularization on model
pre1 <- pre[, "V233":= NULL]
pre1 <- as.matrix(pre1) #Log_Price removed matrix
dim(pre1)
pre <- cbind(numer, ind, train6$Log_Price)
pre <- as.data.table(pre) #data table after generating dummy var including Log_Price
                          #after scaling
dim(pre)
#LASSO Regularization
mod2 <- glmnet(x=pre1, y=pre$V233) # default is alpha = 1, lasso
plot(mod2)
# Understand the plot
# The y axis indicates the number of nonzero coefficients at the current λ, 
# which is the effective degrees of freedom (df) for the lasso.
plot(mod2, label = T) #label shows the num of predictor for each line
plot(mod2, xvar = "lambda", label = T)
# Cross Validation: We need to find the optimum lambda,which is s in the function
cvmod2 <- cv.glmnet(pre1, pre$V233)
summary(cvmod2)
plot(cvmod2)
# Two selected lambdas are shown, 
cvmod2$lambda.min # value of lambda gives minimal mean cross validated error
cvmod2$lambda.1se# most regularized model such that error is within one std err of the minimum
sqrt(cvmod2$cvm[cvmod2$lambda == cvmod2$lambda.1se])
cvfit = cv.glmnet(pre1, pre$V233, type.measure = "mse", nfolds = 20)
sqrt(cvfit$cvm[cvfit$lambda == cvfit$lambda.1se])

x1 = coef(cvmod2, s = "lambda.min")
x2 = coef(cvmod2, s = "lambda.1se")
?cv.glmnet
alphas<-seq(0,1,by=0.1)
elasticnet<-lapply(alphas, function(a) {cv.glmnet(pre1, pre$V233, alpha=a)})
alpha <- for (i in 1:11) {print(min(elasticnet[[i]]$cvm))}
sqrt(0.02444)
# how to select alpha? Still use cross validation
foldid = sample(1:3, size=length(train6$Log_Price), replace=TRUE)
cv1 = cv.glmnet(pre, train6$Log_Price, foldid = foldid, alpha=1)
cv.5 = cv.glmnet(pre1, pre$V233, foldid = 20, alpha=.5)
par(mfrow = c(1,2))
plot(cv1);plot(cv.5)
plot(log(cv1$lambda),cv1$cvm,pch=19,col="red",xlab="log(Lambda)",ylab=cv1$name)
points(log(cv.5$lambda),cv.5$cvm,pch=19,col="grey")
legend("topleft",legend=c("alpha= 1","alpha= .5"),pch=19,col=c("red","grey","blue"))

##Decision Tree, plit test data into test & train
library(rpart)#, ?rpart
set.seed(1)
pre <- as.data.frame(pre) #Log_Price included in the data frame
dim(pre)
#should add Log_PRICE first
train.ind <- sample(1:dim(pre)[1], dim(pre)[1] * 0.7)
length(train.ind)
train.data <- pre[train.ind, ]
dim(train.data)
test.data <- pre[-train.ind, ]
dim(test.data)
# formula <- paste("train6$Log_Price ~ ", paste(colnames(train)[-length(colnames(train))], collapse = " + "))
formula <- paste("V233 ~.-V233 ")
set.seed(1)
tree1 <- rpart(formula, method = 'anova', data = train.data, # method = 'class' for classification 
               control=rpart.control(cp=0.0020748)) # cp = 1, 0.1, 0.01, 0
tree1    # cp:complexity parameter method = 'anova' for regression
#deviance, sum of square error, * means terminal node
plot(tree1)
text(tree1)
#We need to optimized cp value: use corss validation
printcp(tree1)
plotcp(tree1)
# Hence we want the cp value (with a simpler tree) that minimizes the xerror(cross vali error)
bestcp <- tree1$cptable[which.min(tree1$cptable[,"xerror"]), "CP"]
#Prune the tree using the best cp.
tree.pruned <- prune(tree1, cp = bestcp)
tree.pruned
test.pred <- predict(tree.pruned, test.data)
plot(tree.pruned)
plot(tree.pruned, uniform = TRUE) 
# Since labels often extend outside the plot region it can be helpful to specify xpd = TRUE
text(tree.pruned, cex = 0.5, use.n = TRUE, xpd = TRUE)
sum((test.pred - test.data$V233)^2)  #16.23828
sqrt(sum((test.pred - test.data$V233)^2)/dim(test.data)[1])
sqrt(0.03724)
##library(randomForest)
set.seed(2)
rf.formula <- paste("V233 ~.-V233")
rf.formula
names(train.data) <- make.names(names(train.data))
rf <- randomForest(as.formula(rf.formula), data = train.data, importance = TRUE, ntree = 2000)
dim(train.data)
getTree(rf, k = 1, labelVar = TRUE)
#to know which feature is important
varImpPlot(rf)
#%IncMSE shows if a variable is assigned values by random permutation, how much will the MSE increase?
order(importance(rf, type=1))
importanceOrder= order(-rf$importance[, "%IncMSE"])
names=rownames(rf$importance)[importanceOrder]
for (name in names[1:1]) {
  partialPlot(rf, train.data, eval(name), main=name, xlab=name)
}
plot(rf)
names(test.data) <- make.names(names(test.data))
test.pred1<- predict(rf, test.data)
sqrt((sum((test.pred1 - test.data$V233)^2)/dim(test.data)[1])) #0.1442



##xgboosting
library(xgboost) #http://xgboost.readthedocs.io/en/latest/
train.label <- train.data$V233
test.label <- test.data$V233
gbt <- xgboost(data = train.data[, -dim(train.data)[2]], 
               label = train.label, 
               max_depth = 8, 
               nround = 20,
               objective = "reg:linear",
               nthread = 3,
               verbose = 2)
# Error bcuz Xgboost manages only numeric vectors.
# one hot encoding
# The purpose is to transform each value of each categorical feature into a binary feature {0, 1}.
feature.matrix <- model.matrix( ~ ., data = train.data[, -dim(train.data)[2]])
set.seed(1)
gbt <- xgboost(data =  feature.matrix, 
               label = train.label, 
               max_depth = 8, # for each tree, how deep it goes
               nround = 20, # number of trees
               objective = "reg:linear",
               nthread = 3,
               verbose = 2)
importance <- xgb.importance(feature_names = colnames(feature.matrix), model = gbt)
head(importance)
# Gain: contribution of each feature to the model. improvement in accuracy brought by a feature to the branches it is on.
# For boosted tree model, gain of each feature in each branch of each tree is taken into account, 
#    then average per feature to give a vision of the entire model.
#    Highest percentage means important feature to predict the label used for the training (only available for tree models);
# Cover: the number of observation through a branch using this feature as split feature 
# Frequency: counts the number of times a feature is used in all generated trees (often we don't use it).
# https://kaggle2.blob.core.windows.net/forum-message-attachments/76715/2435/Understanding%20XGBoost%20Model%20on%20Otto%20Dataset.html?sv=2015-12-11&sr=b&sig=Vk8PO2jTJ34csLNWepZ3VFMeF4Jw2h5waRJ9Pft73rA%3D&se=2017-02-22T01%3A15%3A01Z&sp=r
xgb.plot.importance(importance[1:6,])
# error due to library not installed
library(Ckmeans.1d.dp)
#xgb.plot.importance(importance_matrix[1:6,])
xgb.plot.tree(model = gbt)
library("DiagrammeR")
xgb.plot.tree(feature_names = colnames(feature.matrix), model = gbt, n_first_tree = 1)

# what's the optimal parameter, for example, number of trees?
par <- list( max_depth = 8,
             objective = "reg:linear",
             nthread = 3,
             verbose = 2)
gbt.cv <- xgb.cv(params = par,
                 data = feature.matrix, label = train.label,
                 nfold = 5, nrounds = 100) #we set nrounds to 100 for every tree
# gbt.cv is to choose best nrounds based on certain parameters set. 
# Because, too small nrounds is underfitting, and too large nrounds is overfitting
# But what about the other parameters? 
# See: http://stackoverflow.com/questions/35050846/xgboost-in-r-how-does-xgb-cv-pass-the-optimal-parameters-into-xgb-train
#We want to plot the train and test RMSE
par(mfrow=c(1, 1))
str(gbt.cv)
plot(gbt.cv$evaluation_log$train_rmse_mean, type = 'l')
lines(gbt.cv$evaluation_log$test_rmse_mean, col = 'red')
nround = which.min(gbt.cv$evaluation_log$test_rmse_mean) #52
gbt.cv2 <- xgb.cv(data = feature.matrix, 
               label = train.label,
               nround = nround,
               params = par, nfold=50)
plot(gbt.cv2$evaluation_log$train_rmse_mean, type = 'l')
lines(gbt.cv2$evaluation_log$test_rmse_mean, col = 'red')
#grid searching for parameters.
all_param = NULL
all_test_rmse = NULL
all_train_rmse = NULL
best_param = list()
best_seednumber = 1234
best_rmse = Inf
best_rmse_index = 0
for (iter in 1:20) {
  param <- list(objective = "reg:linear",
                max_depth = sample(5:12, 1),
                eta = runif(1, .01, .3),
                gamma = runif(1, 0.0, 0.2), 
                subsample = runif(1, .6, .9),
                colsample_bytree = runif(1, .5, .8), 
                min_child_weight = sample(1:40, 1),
                max_delta_step = sample(1:10, 1)
  )
  cv.nround = 52
  cv.nfold = 5
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  mdcv <- xgb.cv(data=feature.matrix, label = train.label, params = param, nthread=6, 
                 nfold=cv.nfold, nrounds=cv.nround,
                 verbose = F, early_stop_round=8, 
                 maximize=FALSE)
  min_train_rmse = min(mdcv$evaluation_log$train_rmse_mean)
  min_test_rmse = min(mdcv$evaluation_log$test_rmse_mean)
  all_param <- rbind(all_param, unlist(param)[-1])
  all_train_rmse <- c(all_train_rmse, min_train_rmse)
  all_test_rmse <- c(all_test_rmse, min_test_rmse)
  #if (min_rmse < best_rmse) {
  #best_rmse = min_rmse
  #best_rmse_index = min_rmes_index
  #best_seednumber = seed.number
  #best_param = param
  #}
}
all_param <- as.data.frame(as.numeric(all_param))
best.gbt <- xgboost(data = feature.matrix, 
                    label = train.label,
                    objective = "reg:linear",
                    nround = 52,
                    max_depth = 12,
                    eta = 0.1482157,       
                    gamma = 0.0751971,
                    subsample = 0.799658,
                    colsample_bytree = 0.547264,
                    min_child_weight = 4,
                    max_delta_step = 6)
best.xgb<- xgb.cv(data = feature.matrix, 
                  label = train.label,
                  nround = 52,
                  params = param, nfold=50)

# prediction
prediction <- predict(best.xgb, model.matrix( ~ ., data = test.data[, -dim(test.data)[2]]))
RMSE.xgb <-sqrt(sum((prediction - test.data$V233)^2)/dim(test.data)[1])
importance <- xgb.importance(feature_names = colnames(feature.matrix), model = best.gbt)
plot(best.gbt$evaluation_log$train_rmse, type = 'l')
lines(gbt.cv$evaluation_log$test_rmse_mean, col = 'red')
xgb.plot.importance(importance[1:6,])


ind <- model.matrix( ~., train.data[, -c(dim(train.data)[2])])
dep <- log(train.data$SalePrice)
cvfit <- cv.glmnet(ind, dep)
plot(cvfit)
x = coef(cvfit, s = "lambda.min") # not matrix type, but dgCmatrix type, still have dim
dim(x) # 190 * 1
nonzero.feat <- rownames(x)[which(x!=0)]
in.nonzero.feat <- sapply(colnames(train.data), function(x) {return(sum(grepl(x, nonzero.feat)) > 0)} )
keep.feat <- names(in.nonzero.feat)[which(in.nonzero.feat == TRUE)]



