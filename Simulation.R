rm(list = ls())

library(glmnet)
library(Metrics)

set.seed(160625)
# Simulationsstudie mit n > k
# Parameter festlegen
n <- 100
k <- 50

# X simulieren
mean <- 0
sd <- 3
x.data <- rnorm(n * k, mean, sd)
X <- matrix(x.data, nrow = n, ncol = k)

# hat X vollen Spaltenrang
qr(X)$rank == k

# wahres beta simulieren
mean <- 0
sd <- 3

beta <- rnorm(k, mean, sd)

# Fehler simulieren
mean <- 0
sd <- 15

u <- rnorm(n, mean, sd)

# Y berechnen
Y <- X %*% beta + u

# in Training und Testdaten teilen mit 80:20 (klassisch)
anz.train <- n * 0.8
Y.train <- Y[1:anz.train]
Y.test <- Y[(anz.train + 1):n]

X.train <- X[1:anz.train,]
X.test <- X[(anz.train + 1):n,]

# Y mit linearem Modell schaetzen
lm.y <- lm(Y.train ~ X.train)
summary(lm.y)

# MSE der Trainingsdaten
MSE.lm.train <- mse(Y.train, lm.y$fitted.values)
MSE.lm.train

# MSE der Testdaten
fitted.test <- cbind(1, X.test) %*% as.matrix(lm.y$coefficients, ncol = 1)
MSE.lm.test <- mse(Y.test, fitted.test)
MSE.lm.test

# Ridge Regression
# optmimales lambda bestimmen
ridge.cv <- cv.glmnet(X.train, Y.train, family = "gaussian", alpha = 0)
plot(ridge.cv, xvar = "lambda", label = TRUE)
lambda.ridge <- ridge.cv$lambda.min
lambda.ridge

# Ridge Regression schaetzen und betas zeigen
Ridge.y <- glmnet(X.train, Y.train, lambda = lambda.ridge, alpha = 0)
Ridge.y$beta

# MSE der Trainingsdaten
fitted.train <- X.train %*% as.matrix(Ridge.y$beta, ncol = 1)
MSE.Ridge.train <- mse(Y.train, fitted.train)
MSE.Ridge.train

# MSE der Testdaten
fitted.test <- X.test %*% as.matrix(Ridge.y$beta, ncol = 1)
MSE.Ridge.test <- mse(Y.test, fitted.test)
MSE.Ridge.test

# Lasso Regression
# optimales lambda bestimmen
lasso.cv <- cv.glmnet(X.train, Y.train, family = "gaussian", alpha = 1)
plot(lasso.cv, xvar = "lambda", label = TRUE)
lambda.lasso <- lasso.cv$lambda.min
lambda.lasso

# Lasso schaetzen und betas ausgeben
Lasso.y <- glmnet(X.train, Y.train, family = "gaussian", lambda = lambda.lasso, alpha = 1)
Lasso.y$beta

# Anzahl der Parameter ungleich 0 berechnen
sum(Lasso.y$beta != 0)

# MSE der Trainingsdaten
fitted.train <- X.train %*% as.matrix(Lasso.y$beta, ncol = 1)
MSE.Lasso.train <- mse(Y.train, fitted.train)
MSE.Lasso.train

# MSE der Testdaten
fitted.test <- X.test %*% as.matrix(Lasso.y$beta, ncol = 1)
MSE.Lasso.test <- mse(Y.test, fitted.test)
MSE.Lasso.test

# alles in eine schöne Matrix
ergebnisse <- matrix(c(MSE.lm.train, MSE.lm.test, MSE.Ridge.train, MSE.Ridge.test, 
                       MSE.Lasso.train, MSE.Lasso.test), nrow = 2)

rownames(ergebnisse) <- c("Trainingsdaten", "Testdaten")
colnames(ergebnisse) <- c("LM", "Ridge", "Lasso")

ergebnisse