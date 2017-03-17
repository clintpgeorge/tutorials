## Multilayer neural network and logistic regression on Boston data ----- 
##
## Adapted from: https://datascienceplus.com/fitting-neural-network-in-r


library(neuralnet)
library(caret)
library(MASS)

set.seed(500)

data <- Boston # ?Boston
apply(data, 2, function(x) sum(is.na(x))) # check NAs  

index <- sample(1:nrow(data), round(0.75 * nrow(data)))
train <- data[index,]
test <- data[-index,]

## Logistic regression

lm.fit <- glm(medv ~ ., data = train)
summary(lm.fit)


## Multilayer neural network

maxs <- apply(data, 2, max)
mins <- apply(data, 2, min)
scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))
train_ <- scaled[index,]
test_ <- scaled[-index,]

n <- names(train_)
f <- as.formula(paste("medv ~", paste(n[!n %in% "medv"], collapse = " + ")))
nn <- neuralnet(f,
                data = train_,
                hidden = c(5, 3),
                linear.output = T)

## Visualize the network weights 

plot(nn)


## Compute MSE

pr.lm <- predict(lm.fit, test)
MSE.lm <- sum((pr.lm - test$medv) ^ 2) / nrow(test)

pr.nn <- compute(nn, test_[, 1:13])
pr.nn_ <- pr.nn$net.result * (max(data$medv) - min(data$medv)) + min(data$medv)
test.r <- (test_$medv) * (max(data$medv) - min(data$medv)) + min(data$medv)
MSE.nn <- sum((test.r - pr.nn_) ^ 2) / nrow(test_)

MSE.lm; MSE.nn


## Plot predicted values 

par(mfrow = c(1, 2))

plot(
  test$medv,
  pr.nn_,
  col = 'red',
  main = 'Real vs predicted (ANN)',
  pch = 3,
  ylab = "Predicted values", 
  xlab = "True values"
)
abline(0, 1, lwd = 2)
legend(
  'bottomright',
  legend = 'NN',
  pch = 3,
  col = 'red',
  bty = 'n'
)

plot(
  test$medv,
  pr.lm,
  col = 'blue',
  main = 'Real vs predicted (GLM)',
  pch = 4,
  ylab = "Predicted values", 
  xlab = "True values"
)
abline(0, 1, lwd = 2)
legend(
  'bottomright',
  legend = 'LM',
  pch = 4,
  col = 'blue',
  bty = 'n',
  cex = .95
)


plot(
  test$medv,
  pr.nn_,
  col = 'red',
  main = 'Real vs predicted (ANN and GLM)',
  pch = 3,
  ylab = "Predicted values", 
  xlab = "True values"
)
points(test$medv,
       pr.lm,
       col = 'blue',
       pch = 4)
abline(0, 1, lwd = 2)
legend(
  'bottomright',
  legend = c('ANN', 'GLM'),
  pch = c(3, 4),
  col = c('red', 'blue')
)
