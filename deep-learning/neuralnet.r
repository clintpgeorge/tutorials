library(neuralnet)
library(caret)

set.seed(500)

## MNIST data ----------------------

displayDigit <- function(X) {
  m <- matrix(unlist(X), nrow = 28, byrow = T)
  m <- t(apply(m, 2, rev))
  image(m, col = grey.colors(255))
}


load("MNIST-train.rda")
displayDigit(train[18, -1])

## Data preprocessing --------

pixels <- train[, -1] # 784 labels
labels <- train[, "label"]

nzv <- nearZeroVar(pixels) # near zero values
pixels.norm <- pixels[, -nzv] # remove near zero values
pixels.norm <- pixels.norm / max(pixels.norm) # scale

train.index <- createDataPartition(y = labels, p = 0.7, list = F)
X.train <- pixels.norm[train.index, ]
y.train <- labels[train.index] # class labels
X.valid <- pixels.norm[-train.index, ]
y.valid <- labels[-train.index] # class labels


## Appending the labels to the training data ------
# neuralnet package

output <- class2ind(as.factor(y.train))
colnames(output) <- paste0('out.', colnames(output))
output.names <- colnames(output)
input.names <- colnames(X.train)
X.train <- cbind(output, X.train)
dim(X.train)

## Model training ------

trainsize = 500 # considering only 500 rows
library(neuralnet)
form <-
  as.formula(paste(
    paste(output.names, collapse = '+'),
    '~',
    paste(input.names, collapse = '+')
  ))
nnet.mdl <- neuralnet(
  formula = form,
  data = X.train[1:trainsize, ],
  hidden = 10,
  algorithm = 'rprop+',
  learningrate = 0.01,
  rep = 1
)

## Accuracy on training data ---------

res <- compute(nnet.mdl, X.train[1:trainsize, input.names])
picks <- (0:9)[apply(res$net.result, 1, which.max)]
prop.table(table(y.train[1:trainsize] == picks))

## Accuracy on validation data --------

predict.res <- compute(nnet.mdl, X.valid)
predict.picks <- (0:9)[apply(predict.res$net.result, 1, which.max)]
prop.table(table(y.valid == predict.picks))


## Predict a sample --------

X.valid.orig <- pixels[-train.index, ] # data before preprocessing
sindex <- sample(1:nrow(X.valid.orig), 1)
displayDigit(X.valid.orig[sindex,])
s.res <- compute(nnet.mdl, t(X.valid[sindex,]))
(0:9)[which.max(s.res$net.result)]




## Boston data -----
## References:
##  https://datascienceplus.com/fitting-neural-network-in-r/


library(MASS)
data <- Boston
apply(data, 2, function(x)sum(is.na(x)))

index <- sample(1:nrow(data), round(0.75 * nrow(data)))
train <- data[index, ]
test <- data[-index, ]
lm.fit <- glm(medv ~ ., data = train)
summary(lm.fit)
pr.lm <- predict(lm.fit, test)
MSE.lm <- sum((pr.lm - test$medv) ^ 2) / nrow(test)

maxs <- apply(data, 2, max)
mins <- apply(data, 2, min)
scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))

train_ <- scaled[index, ]
test_ <- scaled[-index, ]

library(neuralnet)
n <- names(train_)
f <- as.formula(paste("medv ~", paste(n[!n %in% "medv"], collapse = " + ")))
nn <- neuralnet(f,
                data = train_,
                hidden = c(5, 3),
                linear.output = T)


plot(nn)


pr.nn <- compute(nn, test_[, 1:13])
pr.nn_ <- pr.nn$net.result * (max(data$medv) - min(data$medv)) + min(data$medv)
test.r <- (test_$medv) * (max(data$medv) - min(data$medv)) + min(data$medv)

MSE.nn <- sum((test.r - pr.nn_) ^ 2) / nrow(test_)

MSE.lm; MSE.nn



par(mfrow = c(1, 2))

plot(
  test$medv,
  pr.nn_,
  col = 'red',
  main = 'Real vs predicted NN',
  pch = 18,
  cex = 0.7
)
abline(0, 1, lwd = 2)
legend(
  'bottomright',
  legend = 'NN',
  pch = 18,
  col = 'red',
  bty = 'n'
)

plot(
  test$medv,
  pr.lm,
  col = 'blue',
  main = 'Real vs predicted lm',
  pch = 18,
  cex = 0.7
)
abline(0, 1, lwd = 2)
legend(
  'bottomright',
  legend = 'LM',
  pch = 18,
  col = 'blue',
  bty = 'n',
  cex = .95
)


plot(
  test$medv,
  pr.nn_,
  col = 'red',
  main = 'Real vs predicted NN',
  pch = 18,
  cex = 0.7
)
points(test$medv,
       pr.lm,
       col = 'blue',
       pch = 18,
       cex = 0.7)
abline(0, 1, lwd = 2)
legend(
  'bottomright',
  legend = c('NN', 'LM'),
  pch = 18,
  col = c('red', 'blue')
)
