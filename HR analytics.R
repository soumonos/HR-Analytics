getwd()
setwd(dir = "C:/Users/User/Documents/Predictive analyitics")
getwd()
df <- read.csv("HR-Attrition.csv")
summary(df)

names<- c("Education","Attrition","EnvironmentSatisfaction","JobInvolvement","JobLevel","JobSatisfaction","NumCompaniesWorked","PerformanceRating",
  "RelationshipSatisfaction","StockOptionLevel","TrainingTimesLastYear","WorkLifeBalance")
df[,names]<- lapply(df[,names],factor)
summary(df)

dfn<- select(df,-c("StandardHours","Over18","EmployeeNumber","EmployeeCount"))
dffeatures<- select(dfn,-c("Attrition"))

#Train-test split
set.seed(1234)
ind<- sample(2, nrow(dfn), replace=TRUE, prob=c(0.8,0.2)) 
training <- dfn[ind==1,]
testing <- dfn[ind==2,]
y_train <- training[,2]
x_train <- training[,-2]
x_test <- testing[,-2]
y_test<- testing[,2]
#Model-1 : Random Forest (Tuned mtry value by RF)

#control<- trainControl(method='repeatedcv', number=10, repeats= 3, search='random')
set.seed(1)

bestmtry <- tuneRF(x_train, y_train,stepFactor = 1.5,improve = 1e-5,ntree=500)
bestmtry
modelrf <- randomForest(x_train,y_train,ntree = 500, mtry = bestmtry, importance = TRUE)
modelrf

pred= predict(modelrf,training)
pred1= predict(modelrf,testing)
confusionMatrix(testing$Attrition,pred1)

importancerf <-round(importance(modelrf),2)
newimp <-data.frame(importancerf)
plot(newimp$MeanDecreaseAccuracy)
postResample(pred,training$Attrition)
postResample(pred1,testing$Attrition)
plot(modelrf)

#Random Forest with Grid Search
control <- trainControl(method='repeatedcv',number=10, repeats=3,search='grid')
tunegrid <- expand.grid(.mtry=(1:15))
rf_grid <- train(Attrition ~.,data = training,method='rf',metric='Accuracy',tuneGrid=tunegrid)
print(rf_grid)
plot(rf_grid)

#Random Forest with Random search:
controlrand <- trainControl(method='repeatedcv',number=10, repeats=3,search='random')
set.seed(2)
mtry <- sqrt(ncol(x_train))
rf_random <- train(Attrition ~.,data = training,method='rf',metric='Accuracy',tuneLength=15,trControl=controlrand)
print(rf_random)
plot(rf_random)

#Model-2: Logistic Regression

modellogr= glm(formula=Attrition ~.,data = training,family = binomial)
summary(modellogr)
anova(modellogr,test = "Chisq")
predlr= predict(modellogr, testing, type = "response")
binpred= ifelse(predlr>0.5,"Yes","No")
mean(binpred == testing$Attrition)
postResample(binpred,testing$Attrition)
plot(modellogr)
pr<- prediction(predlr, testing$Attrition)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc
#Model-3: Gradient Boosted Tree (XG Boost)
controlxg <- trainControl(method='cv',number=10)
modelxgcv <- train(Attrition ~., data=training, method='xgbTree',trControl=controlxg)
plot(modelxgcv)
predxg <- predict(modelxgcv,testing)
mean(predxg==testing$Attrition)
postResample(predxg,testing$Attrition)
impxg <-varImp(modelxgcv)
plot(impxg)

# We have identified that at eta=0.4 and nrounds=150, it performs the best with 0.41 kappa and 0.865 accuracy. Hence:
#dtrain<- xgb.DMatrix(data=new_tr,label=label_train)
new_tr <- model.matrix(~.+0,data = x_train)
new_test <- model.matrix(~.+0,data=x_test)
label_train <- ifelse(y_train=="Yes",1,0)
label_test<- ifelse(y_test=="Yes",1,0)
dtrain<- xgb.DMatrix(data=new_tr,label=label_train)
dtest <- xgb.DMatrix(data=new_test,label=label_test)
paramsxgb<- list(booster = "gbtree", objective = "binary:logistic", eta=0.4, gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)
xgbop <- xgb.train(params = paramsxgb,data=dtrain,nrounds = 150 )
predxgop <- predict(xgbop,dtest)
binpredxg= ifelse(predxgop>0.5,"Yes","No")
postResample(binpredxg,testing$Attrition)
imp<- xgb.importance(feature_names= colnames(new_tr),model=xgbop)
xgb.plot.importance (importance_matrix = imp) 

#Model-4: ADABOOST (ADA bag boosting):
adacv <- boosting.cv(Attrition ~., data=training,boos=TRUE,)

