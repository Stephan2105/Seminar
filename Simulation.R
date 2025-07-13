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

##### Ergaenzung Bachelorarbeit
### Anzahl der Kovariablen flexibel gestalten
# Parameter festlegen
n <- 100
k.range <- 2:150

# Fehler und andere interessante Kenngroessen indizieren
MSE.lm.train <- rep(NA, length(k.range))
MSE.lm.test <- rep(NA, length(k.range))
MSE.Ridge.train <- rep(NA, length(k.range))
MSE.Ridge.test <- rep(NA, length(k.range))
MSE.Lasso.train <- rep(NA, length(k.range))
MSE.Lasso.test <- rep(NA, length(k.range))
lambda.ridge <- rep(NA, length(k.range))
lambda.lasso <- rep(NA, length(k.range))
anz.parameter.lasso <- rep(NA, length(k.range))

# Fehler fuer moegliche k berechnen
for(k in k.range){
  # immer den seed resten
  set.seed(160625)
  
  # X simulieren
  mean <- 0
  sd <- 3
  x.data <- rnorm(n * k, mean, sd)
  X <- matrix(x.data, nrow = n, ncol = k)
  
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
  anz.train <- floor(n * 0.8)
  Y.train <- Y[1:anz.train]
  Y.test <- Y[(anz.train + 1):n]
  
  X.train <- X[1:anz.train,]
  X.test <- X[(anz.train + 1):n,]
  
  # Y mit linearem Modell schaetzen
  lm.y <- lm(Y.train ~ X.train)
  
  # MSE der Trainingsdaten
  MSE.lm.train[k - 1] <- mse(Y.train, lm.y$fitted.values)
  
  # MSE der Testdaten
  fitted.test <- cbind(1, X.test) %*% as.matrix(lm.y$coefficients, ncol = 1)
  MSE.lm.test[k - 1] <- mse(Y.test, fitted.test)
  
  # Ridge Regression
  # optmimales lambda bestimmen
  ridge.cv <- cv.glmnet(X.train, Y.train, family = "gaussian", alpha = 0)
  lambda.ridge[k - 1] <- ridge.cv$lambda.min
  
  # Ridge Regression schaetzen
  Ridge.y <- glmnet(X.train, Y.train, lambda = lambda.ridge[k - 1], alpha = 0)
  
  # MSE der Trainingsdaten
  fitted.train <- X.train %*% as.matrix(Ridge.y$beta, ncol = 1)
  MSE.Ridge.train[k - 1] <- mse(Y.train, fitted.train)
  
  # MSE der Testdaten
  fitted.test <- X.test %*% as.matrix(Ridge.y$beta, ncol = 1)
  MSE.Ridge.test[k - 1] <- mse(Y.test, fitted.test)
  
  # Lasso Regression
  # optimales lambda bestimmen
  lasso.cv <- cv.glmnet(X.train, Y.train, family = "gaussian", alpha = 1)
  lambda.lasso[k - 1] <- lasso.cv$lambda.min
  
  # Lasso schaetzen
  Lasso.y <- glmnet(X.train, Y.train, family = "gaussian", 
                    lambda = lambda.lasso[k - 1], alpha = 1)
  
  # Anzahl der Parameter ungleich 0 berechnen
  anz.parameter.lasso[k - 1] <- sum(Lasso.y$beta != 0)
  
  # MSE der Trainingsdaten
  fitted.train <- X.train %*% as.matrix(Lasso.y$beta, ncol = 1)
  MSE.Lasso.train[k - 1] <- mse(Y.train, fitted.train)
  
  # MSE der Testdaten
  fitted.test <- X.test %*% as.matrix(Lasso.y$beta, ncol = 1)
  MSE.Lasso.test[k - 1] <- mse(Y.test, fitted.test)
}

# Hilfsvektoren fuer Legenden
farben <- c("seagreen", "firebrick", "navy")
Modelle <- c("Lasso", "Ridge", "KQ")

# wenn MSE.lm.train 0 ist, liegt das daran, dass das Modell dort aufgrund der Anzahl
# an Kovariablen nicht mehr funktioniert, daher durch NA ersetzen
MSE.lm.train <- ifelse(MSE.lm.train == 0, NA, MSE.lm.train)

# Trainingsfehler plotten
plot(k.range, MSE.Lasso.train, col = farben[1], type = "l", lwd = 2, 
     ylim = c(0, max(pmax(MSE.Lasso.train, MSE.Ridge.train))),
     xlab = "Anzahl Kovariablen", ylab = "MSE der Trainingsdaten")
lines(k.range, MSE.Ridge.train, col = farben[2], lwd = 2)
lines(k.range, MSE.lm.train, col = farben[3], lwd = 2)
legend("topleft", Modelle, lwd = 2, col = farben)

# Testfehler plotten
plot(k.range, MSE.Lasso.test, col = farben[1], type = "l", lwd = 2, 
     ylim = c(0, max(pmax(MSE.Lasso.test, MSE.Ridge.test, 
                          ifelse(is.na(MSE.lm.test), 0, MSE.lm.test)))),
     xlab = "Anzahl Kovariablen", ylab = "MSE der Testdaten")
lines(k.range, MSE.Ridge.test, col = farben[2], lwd = 2)
lines(k.range, MSE.lm.test, col = farben[3], lwd = 2)
legend("topleft", Modelle, lwd = 2, col = farben)

# da lm Testfehler so gross nur Bereich bis 20 000 ansehen
plot(k.range, MSE.Lasso.test, col = farben[1], type = "l", lwd = 2, 
     ylim = c(0, 20000), xlab = "Anzahl Kovariablen", 
     ylab = "MSE der Testdaten")
lines(k.range, MSE.Ridge.test, col = farben[2], lwd = 2)
lines(k.range, MSE.lm.test, col = farben[3], lwd = 2)
legend("topleft", Modelle, lwd = 2, col = farben)

# lambdas anschauen
lambda.ridge
lambda.lasso

# Anzahl berücksichtigter Koeffizienten von lasso
anz.parameter.lasso
# anzahl nicht berücksichtiger Koeffizienten von lasso
nb <- k.range - anz.parameter.lasso
nb

# diese infos plotten
plot(k.range, lambda.lasso, type = "l", col = farben[1], lwd = 2,
     xlab = "Anzahl Kovariablen", ylab = "Wert des Starfparameters")
lines(k.range, lambda.ridge, col = farben[2], lwd = 2)
legend("topleft", Modelle[1:2], col = farben[1:2], lwd = 2)

plot(k.range, nb, type = "l", col = farben[1], lwd = 2,
     xlab = "Anzahl Kovariablen", ylab = "Anzahl nicht berücksichtigter Kovariablen")

plot(k.range, anz.parameter.lasso, type = "l", col = farben[1], lwd = 2, 
     xlab = "Anzahl Kovariablen", ylab = "Anzahl berücksichtigter Kovariablen")
# bei 60 linie einzeichnen, weil es sich dort einpendelt
abline(h = 60, col = "red", lwd = 2)
legend("topleft", c("Berücksichtigte Kovariablen", "Linie bei 60 Kovariablen"),
       col = c(farben[1], "red"), lwd = 2)

### Nun alles fuer 50 Kovariablen aber steigende Anzahl von Beobachtungen von 10 bis 200
# Parameter festlegen
n.range <- 10:200
k <- 50

# Fehler und andere interessante Kenngroessen indizieren
MSE.lm.train <- rep(NA, length(n.range))
MSE.lm.test <- rep(NA, length(n.range))
MSE.Ridge.train <- rep(NA, length(n.range))
MSE.Ridge.test <- rep(NA, length(n.range))
MSE.Lasso.train <- rep(NA, length(n.range))
MSE.Lasso.test <- rep(NA, length(n.range))
lambda.ridge <- rep(NA, length(n.range))
lambda.lasso <- rep(NA, length(n.range))
anz.parameter.lasso <- rep(NA, length(n.range))

# Fehler fuer moegliche k berechnen
for(n in n.range){
  # seed resten
  set.seed(160625)
  
  # X simulieren
  mean <- 0
  sd <- 3
  x.data <- rnorm(n * k, mean, sd)
  X <- matrix(x.data, nrow = n, ncol = k)
  
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
  anz.train <- floor(n * 0.8)
  Y.train <- Y[1:anz.train]
  Y.test <- Y[(anz.train + 1):n]
  
  X.train <- X[1:anz.train,]
  X.test <- X[(anz.train + 1):n,]
  
  # Y mit linearem Modell schaetzen
  lm.y <- lm(Y.train ~ X.train)
  
  # MSE der Trainingsdaten
  MSE.lm.train[n - 9] <- mse(Y.train, lm.y$fitted.values)
  
  # MSE der Testdaten
  fitted.test <- cbind(1, X.test) %*% as.matrix(lm.y$coefficients, ncol = 1)
  MSE.lm.test[n - 9] <- mse(Y.test, fitted.test)
  
  # Ridge Regression
  # optmimales lambda bestimmen
  ridge.cv <- cv.glmnet(X.train, Y.train, family = "gaussian", alpha = 0)
  lambda.ridge[n - 9] <- ridge.cv$lambda.min
  
  # Ridge Regression schaetzen
  Ridge.y <- glmnet(X.train, Y.train, lambda = lambda.ridge[n - 9], alpha = 0)
  
  # MSE der Trainingsdaten
  fitted.train <- X.train %*% as.matrix(Ridge.y$beta, ncol = 1)
  MSE.Ridge.train[n - 9] <- mse(Y.train, fitted.train)
  
  # MSE der Testdaten
  fitted.test <- X.test %*% as.matrix(Ridge.y$beta, ncol = 1)
  MSE.Ridge.test[n - 9] <- mse(Y.test, fitted.test)
  
  # Lasso Regression
  # optimales lambda bestimmen
  lasso.cv <- cv.glmnet(X.train, Y.train, family = "gaussian", alpha = 1)
  lambda.lasso[n - 9] <- lasso.cv$lambda.min
  
  # Lasso schaetzen
  Lasso.y <- glmnet(X.train, Y.train, family = "gaussian",
                    lambda = lambda.lasso[n - 9], alpha = 1)
  
  # Anzahl der Parameter ungleich 0 berechnen
  anz.parameter.lasso[n - 9] <- sum(Lasso.y$beta != 0)
  
  # MSE der Trainingsdaten
  fitted.train <- X.train %*% as.matrix(Lasso.y$beta, ncol = 1)
  MSE.Lasso.train[n - 9] <- mse(Y.train, fitted.train)
  
  # MSE der Testdaten
  fitted.test <- X.test %*% as.matrix(Lasso.y$beta, ncol = 1)
  MSE.Lasso.test[n - 9] <- mse(Y.test, fitted.test)
}

# wenn MSE.lm.train 0 ist, liegt das daran, dass das Modell dort aufgrund der Anzahl
# an Kovariablen nicht mehr funktioniert, daher durch NA ersetzen
MSE.lm.train <- ifelse(MSE.lm.train == 0, NA, MSE.lm.train)

# Hilfsvektoren fuer Legenden
farben <- c("seagreen", "firebrick", "navy")
Modelle <- c("Lasso", "Ridge", "KQ")

# Trainingsfehler plotten
plot(n.range, MSE.Lasso.train, col = farben[1], type = "l", lwd = 2, 
     ylim = c(0, max(pmax(MSE.Lasso.train, MSE.Ridge.train))),
     xlab = "Anzahl Beobachtungen", ylab = "MSE der Trainingsdaten")
lines(n.range, MSE.Ridge.train, col = farben[2], lwd = 2)
lines(n.range, MSE.lm.train, col = farben[3], lwd = 2)
legend("topright", Modelle, lwd = 2, col = farben)

# Testfehler plotten
plot(n.range, MSE.Lasso.test, col = farben[1], type = "l", lwd = 2, 
     ylim = c(0, max(pmax(MSE.Lasso.test, MSE.Ridge.test, 
                          ifelse(is.na(MSE.lm.test), 0, MSE.lm.test)))),
     xlab = "Anzahl Beobachtungen", ylab = "MSE der Testdaten")
lines(n.range, MSE.Ridge.test, col = farben[2], lwd = 2)
lines(n.range, MSE.lm.test, col = farben[3], lwd = 2)
legend("topright", Modelle, lwd = 2, col = farben)

# da lm Testfehler so gross nur Bereich bis 11 000 ansehen
plot(n.range, MSE.Lasso.test, col = farben[1], type = "l", lwd = 2, 
     ylim = c(0, 11000), xlab = "Anzahl Beobachtungen", 
     ylab = "MSE der Testdaten")
lines(n.range, MSE.Ridge.test, col = farben[2], lwd = 2)
lines(n.range, MSE.lm.test, col = farben[3], lwd = 2)
legend("topright", Modelle, lwd = 2, col = farben)

# lambdas anschauen
lambda.ridge
lambda.lasso

# Anzahl berücksichtigter Koeffizienten von lasso
anz.parameter.lasso
# anzahl nicht berücksichtiger Koeffizienten von lasso
nb <- k - anz.parameter.lasso
nb

# alles plotten
plot(n.range, lambda.lasso, type = "l", col = farben[1], lwd = 2,
     xlab = "Anzahl Beobachtungen", ylab = "Wert des Strafparameters")
lines(n.range, lambda.ridge, col = farben[2], lwd = 2)
legend("topright", Modelle[1:2], lwd = 2, col = farben[1:2])

plot(n.range, nb, type = "l", lwd = 2, col = farben[1], 
     xlab = "Anzahl Beobachtungen", ylab = "Anzahl nicht berücksichtigter Kovariablen")

plot(n.range, anz.parameter.lasso, type = "l", lwd = 2, col = farben[1], 
     xlab = "Anzahl Beobachtungen", ylab = "Anzahl berücksichtigter Kovariablen")

### nun mit bestimmten Eigenschaften von beta
## zu beginn lassen wir den Mittelwert von beta durchlaufen bei n = 100 und k = 50
# Parameter festlegen
n <- 100
k <- 50
beta.range <- -50:50

# Fehler und andere interessante Kenngroessen indizieren
MSE.lm.train <- rep(NA, length(beta.range))
MSE.lm.test <- rep(NA, length(beta.range))
MSE.Ridge.train <- rep(NA, length(beta.range))
MSE.Ridge.test <- rep(NA, length(beta.range))
MSE.Lasso.train <- rep(NA, length(beta.range))
MSE.Lasso.test <- rep(NA, length(beta.range))
lambda.ridge <- rep(NA, length(beta.range))
lambda.lasso <- rep(NA, length(beta.range))
anz.parameter.lasso <- rep(NA, length(beta.range))

index <- 0

# Fehler fuer moegliche k berechnen
for(mu in beta.range){
  index <- index + 1
  # immer den seed resten
  set.seed(160625)
  
  # X simulieren
  mean <- 0
  sd <- 3
  x.data <- rnorm(n * k, mean, sd)
  X <- matrix(x.data, nrow = n, ncol = k)
  
  # wahres beta simulieren
  mean <- mu
  sd <- 3
  
  beta <- rnorm(k, mean, sd)
  
  # Fehler simulieren
  mean <- 0
  sd <- 15
  
  u <- rnorm(n, mean, sd)
  
  # Y berechnen
  Y <- X %*% beta + u
  
  # in Training und Testdaten teilen mit 80:20 (klassisch)
  anz.train <- floor(n * 0.8)
  Y.train <- Y[1:anz.train]
  Y.test <- Y[(anz.train + 1):n]
  
  X.train <- X[1:anz.train,]
  X.test <- X[(anz.train + 1):n,]
  
  # Y mit linearem Modell schaetzen
  lm.y <- lm(Y.train ~ X.train)
  
  # MSE der Trainingsdaten
  MSE.lm.train[index] <- mse(Y.train, lm.y$fitted.values)
  
  # MSE der Testdaten
  fitted.test <- cbind(1, X.test) %*% as.matrix(lm.y$coefficients, ncol = 1)
  MSE.lm.test[index] <- mse(Y.test, fitted.test)
  
  # Ridge Regression
  # optmimales lambda bestimmen
  ridge.cv <- cv.glmnet(X.train, Y.train, family = "gaussian", alpha = 0)
  lambda.ridge[index] <- ridge.cv$lambda.min
  
  # Ridge Regression schaetzen
  Ridge.y <- glmnet(X.train, Y.train, lambda = lambda.ridge[index], alpha = 0)
  
  # MSE der Trainingsdaten
  fitted.train <- X.train %*% as.matrix(Ridge.y$beta, ncol = 1)
  MSE.Ridge.train[index] <- mse(Y.train, fitted.train)
  
  # MSE der Testdaten
  fitted.test <- X.test %*% as.matrix(Ridge.y$beta, ncol = 1)
  MSE.Ridge.test[index] <- mse(Y.test, fitted.test)
  
  # Lasso Regression
  # optimales lambda bestimmen
  lasso.cv <- cv.glmnet(X.train, Y.train, family = "gaussian", alpha = 1)
  lambda.lasso[index] <- lasso.cv$lambda.min
  
  # Lasso schaetzen
  Lasso.y <- glmnet(X.train, Y.train, family = "gaussian", 
                    lambda = lambda.lasso[index], alpha = 1)
  
  # Anzahl der Parameter ungleich 0 berechnen
  anz.parameter.lasso[index] <- sum(Lasso.y$beta != 0)
  
  # MSE der Trainingsdaten
  fitted.train <- X.train %*% as.matrix(Lasso.y$beta, ncol = 1)
  MSE.Lasso.train[index] <- mse(Y.train, fitted.train)
  
  # MSE der Testdaten
  fitted.test <- X.test %*% as.matrix(Lasso.y$beta, ncol = 1)
  MSE.Lasso.test[index] <- mse(Y.test, fitted.test)
}

# Hilfsvektoren fuer Legenden
farben <- c("seagreen", "firebrick", "navy")
Modelle <- c("Lasso", "Ridge", "KQ")

# Trainingsfehler plotten
plot(beta.range, MSE.Lasso.train, col = farben[1], type = "l", lwd = 2, 
     ylim = c(0, max(pmax(MSE.Lasso.train, MSE.Ridge.train, MSE.lm.train))),
     xlab = "Mittelwert der Regressionskoeffizienten", ylab = "MSE der Trainingsdaten")
lines(beta.range, MSE.Ridge.train, col = farben[2], lwd = 2)
lines(beta.range, MSE.lm.train, col = farben[3], lwd = 2)
legend("top", Modelle, lwd = 2, col = farben)

# Testfehler plotten
plot(beta.range, MSE.Lasso.test, col = farben[1], type = "l", lwd = 2, 
     ylim = c(0, max(pmax(MSE.Lasso.test, MSE.Ridge.test, 
                          ifelse(is.na(MSE.lm.test), 0, MSE.lm.test)))),
     xlab = "Mittelwert der Regressionskoeffizienten", ylab = "MSE der Testdaten")
lines(beta.range, MSE.Ridge.test, col = farben[2], lwd = 2)
lines(beta.range, MSE.lm.test, col = farben[3], lwd = 2)
legend("top", Modelle, lwd = 2, col = farben)

# lambdas anschauen
lambda.ridge
lambda.lasso

# Anzahl berücksichtigter Koeffizienten von lasso
anz.parameter.lasso
# anzahl nicht berücksichtiger Koeffizienten von lasso
nb <- k - anz.parameter.lasso 
nb

# diese infos plotten
plot(beta.range, lambda.lasso, type = "l", col = farben[1], lwd = 2, 
     ylim = c(0, max(pmax(lambda.lasso, lambda.ridge))),
     xlab = "Mittelwert der Regressionskoeffizienten", 
     ylab = "Wert des Starfparameters")
lines(beta.range, lambda.ridge, col = farben[2], lwd = 2)
legend("top", Modelle[1:2], col = farben[1:2], lwd = 2)

plot(beta.range, nb, type = "l", col = farben[1], lwd = 2,
     xlab = "Mittelwert der Regressionskoeffizienten", 
     ylab = "Anzahl nicht berücksichtigter Kovariablen")

plot(beta.range, anz.parameter.lasso, type = "l", col = farben[1], lwd = 2,
     xlab = "Mittelwert der Regressionskoeffizienten", 
     ylab = "Anzahl berücksichtigter Kovariablen")