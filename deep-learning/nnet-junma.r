# Multilayer feed-forward networks: examples
#
# Reference:
# http://junma5.weebly.com/data-blog/build-your-own-neural-network-classifier-in-r

library(ggplot2)
library(caret)

## Utility functions ----

nnet <-
  function(X,
           Y,
           step_size = 0.5,
           reg = 0.001,
           h = 10,
           niteration) {

    # get dim of input
    N <- nrow(X) # number of examples
    K <- ncol(Y) # number of classes
    D <- ncol(X) # dimensionality

    # initialize parameters randomly
    W <- 0.01 * matrix(rnorm(D * h), nrow = D)
    b <- matrix(0, nrow = 1, ncol = h)
    W2 <- 0.01 * matrix(rnorm(h * K), nrow = h)
    b2 <- matrix(0, nrow = 1, ncol = K)

    # gradient descent loop to update weight and bias
    for (i in 0:niteration) {
      # hidden layer, ReLU activation
      hidden_layer <- pmax(0, X %*% W + matrix(rep(b, N), nrow = N, byrow = T))
      hidden_layer <- matrix(hidden_layer, nrow = N)
      # class score
      scores <- hidden_layer %*% W2 + matrix(rep(b2, N), nrow = N, byrow = T)

      # compute and normalize class probabilities
      exp_scores <- exp(scores)
      probs <- exp_scores / rowSums(exp_scores)

      # compute the loss: sofmax and regularization
      data_loss <- sum(-log(probs) * Y) / N
      reg_loss <- 0.5 * reg * sum(W * W) + 0.5 * reg * sum(W2 * W2)
      loss <- data_loss + reg_loss
      # check progress
      if (i %% 1000 == 0 | i == niteration) {
        print(paste("iteration", i, ': loss', loss))
      }

      # compute the gradient on scores
      dscores <- (probs - Y) / N

      # backpropate the gradient to the parameters
      dW2 <- t(hidden_layer) %*% dscores
      db2 <- colSums(dscores)

      # finally into W,b
      dhidden <- dscores %*% t(W2) # next backprop into hidden layer
      dhidden[hidden_layer <= 0] <- 0 # backprop the ReLU non-linearity
      dW <- t(X) %*% dhidden
      db <- colSums(dhidden)

      # add regularization gradient contribution
      dW2 <- dW2 + reg * W2
      dW <- dW + reg * W

      # update parameter
      W2 <- W2 - step_size * dW2
      b2 <- b2 - step_size * db2
      W <- W - step_size * dW
      b <- b - step_size * db
    }

    return(list(W, b, W2, b2))
  }


nnetPred <- function(X, para = list()) {
  W <- para[[1]]
  b <- para[[2]]
  W2 <- para[[3]]
  b2 <- para[[4]]

  N <- nrow(X)
  hidden_layer <- pmax(0, X %*% W + matrix(rep(b, N), nrow = N, byrow = T))
  hidden_layer <- matrix(hidden_layer, nrow = N)
  scores <- hidden_layer %*% W2 + matrix(rep(b2, N), nrow = N, byrow = T)
  predicted_class <- apply(scores, 1, which.max)

  return(predicted_class)
}


set.seed(2017)


## Generate spiral data -----

N <- 200 # number of points per class
D <- 2 # dimensionality
K <- 4 # number of classes

X <- data.frame() # data matrix (each row = single example)
y <- data.frame() # class labels
for (j in (1:K)) {
  r <- seq(0.05, 1, length.out = N) # radius
  t <- seq((j - 1) * 4.7, j * 4.7, length.out = N) + rnorm(N, sd = 0.3) # theta
  Xtemp <- data.frame(x = r * sin(t) , y = r * cos(t))
  ytemp <- data.frame(matrix(j, N, 1))
  X <- rbind(X, Xtemp)
  y <- rbind(y, ytemp)
}
data <- cbind(X, y)
colnames(data) <- c(colnames(X), 'label')
X <- as.matrix(X)
Y <- matrix(0, N * K, K)
for (i in 1:(N * K)) {
  Y[i, y[i, ]] <- 1
}


## Visualize the data ----

x_min <- min(X[, 1]) - 0.2
x_max <- max(X[, 1]) + 0.2
y_min <- min(X[, 2]) - 0.2
y_max <- max(X[, 2]) + 0.2

ggplot(data) +
  geom_point(aes(
    x = x,
    y = y,
    color = as.character(label)
  ), size = 2) +
  xlim(x_min, x_max) +
  ylim(y_min, y_max) +
  ggtitle('Spiral Data Visulization') +
  theme(
    axis.ticks = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text = element_blank(),
    axis.title = element_blank(),
    legend.position = 'none'
  )





## Model training ----

nnet.model <-
  nnet(
    X,
    Y,
    step_size = 0.4,
    reg = 0.0002,
    h = 50,
    niteration = 6000
  )

predicted_class <- nnetPred(X, nnet.model)
print(paste('Training Accuracy:', mean(predicted_class == (y))))

## Visualize the classifier ----

hs <- 0.01
grid <- as.matrix(expand.grid(seq(x_min, x_max, by = hs), seq(y_min, y_max, by = hs)))
Z <- nnetPred(grid, nnet.model)

ggplot() +
  geom_tile(
    aes(x = grid[, 1], y = grid[, 2], fill = as.character(Z)),
    alpha = 0.3,
    show.legend = F
  ) +
  geom_point(
    data = data,
    aes(x = x, y = y, color = as.character(label)),
    size = 2
  ) +
  ggtitle('Neural Network Decision Boundary') +
  theme(
    axis.ticks = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text = element_blank(),
    axis.title = element_blank(),
    legend.position = 'none'
  )



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

K <- length(unique(y.train)) # number of classes
N <- nrow(X.train) # number of examples
Y.train <- matrix(0, N, K)
for (i in 1:N) {
  Y.train[i, y.train[i] + 1] <- 1
}

nnet.mnist <- nnet(X.train,
                   Y.train,
                   step_size = 0.3,
                   reg = 0.0001,
                   niteration = 3500)

predicted_class <- nnetPred(X.train, nnet.mnist)
print(paste('Training set accuracy:', mean(predicted_class == (y.train + 1))))

predicted_class <- nnetPred(X.valid, nnet.mnist)
print(paste('Validation set accuracy:', mean(predicted_class == (y.valid + 1))))


sindex <- sample(1:nrow(X.valid), 1)
predicted_test <- nnetPred(t(X.valid[sindex,]), nnet.mnist)
print(paste('The predicted digit is:', predicted_test - 1))
displayDigit(pixels[-train.index,][sindex,])
