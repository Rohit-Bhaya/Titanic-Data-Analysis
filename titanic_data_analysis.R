## Applying Machine Learning Techniques on Titanic Data

## Loading Libraries

library(MASS)
library(ggplot2)
library(dplyr)

## Data exploration and Data Cleaning

### Loading the data from the directory

train <- read.csv("train.csv")
test <- read.csv("test.csv")

test$Survived <- NA
all <- rbind(train, test)

glimpse(all)

colSums(is.na(all))

space <- function (x) {sum(x=="") }
apply(all, 2, space)


### Predicting missing age
ggplot(data = all, aes(x = Fare, y = Age, col = factor(Pclass))) +
  geom_smooth()

index <- which(is.na(all$Age))
predict_age <- all[index,]
train_age <- all[-index,]

lm_age <- lm(Age ~ Pclass + Fare + Pclass:Fare, data = train_age)
summary(lm_age)

predict_age$Age <- as.integer(predict(lm_age, newdata = predict_age))

all <- rbind(train_age, predict_age)
colSums(is.na(all))

all$family <- all$SibSp + all$Parch + 1


### Splitting the data for predictive modeling

#Convert categorical variable to factor type
all$Survived <- as.factor(all$Survived)
all$Pclass <- as.factor(all$Pclass)


index <- which(is.na(all$Survived)) 
train <- all[-index,]
test <- all[index,]

str(all)

index <- sample(1:nrow(train), nrow(train)*0.8)
train_t <- train[index,]
test_t <- train[-index,]


## Logistic Regression

### Initial Model

log.fit <- glm(Survived ~ Pclass + Sex + Age + Fare + Embarked + family, family = "binomial", data = train_t)
summary(log.fit)

### Model with important variables
log.fit2 <- glm(Survived ~ Pclass +Sex + Age + family , family = "binomial", data = train_t)
summary(log.fit2)

### Prediction on the test data
test_t$prob <- predict(log.fit2, newdata = test_t, type = "response")
test_t$pred <- ifelse(test_t$prob > 0.5, 1 , 0)
table(test_t$pred, test_t$Survived)


## Linear Discriminant Analysis

### LDA Modeling 

lda.model = lda (Survived ~ Pclass + Sex + Age + Parch, data=train_t)
lda.model

plot(lda.model)

### Prediction on the test data
lda_prob <- predict(lda.model, newdata = test_t)
table(Predicted = lda_prob$class, test_t$Survived)

## Quadratic Discriminant Analysis

### QDA Modeling 

qda.model <- qda (Survived ~ Pclass + Sex + Age + Parch, data=train_t)
qda.model


### Prediction on the test data

qda_prob <- predict(qda.model, newdata = test_t)
table(Predicted = qda_prob$class, test_t$Survived)

