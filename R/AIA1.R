# AIA1.R

library(tensorflow)
library(reticulate)
library(feather)
source("/home/thernandez/solar-forecast/R/AIA2.R")
# Create Data
# lf <- list.files("/home/thernandez/FeatherAIA2014/")
y.mat <- read.csv("/home/thernandez/solar-forecast/datasets/Flux_FeatherAIA2014.csv")
y.mat$Time <- as.POSIXct(y.mat$Time, tz = "UTC")
# Creating file list from pre-approved y.mat times indices
lf <- paste0("AIA", as.character(format(y.mat$Time, "%Y%m%d_%H%M")), ".feather")
if(length(lf) == nrow(y.mat)){cat("Flux and AIA data matches.")}

last.train.ind <- round(length(lf) * 4 / 5)
#############################################################
#############################################################
sess <- tf$InteractiveSession()
# sess = tf$Session()

x <- tf$placeholder(tf$float32, shape(NULL, as.integer(1048576)))
y <- tf$placeholder(tf$float32, shape(NULL, 1L))
x_image <- tf$reshape(x, shape(-1L, 1024L, 1024L, 1L))

CV.mat <- as.data.frame(matrix(0, nrow = 25, ncol = 22))
colnames(CV.mat) <- c("LogRMSE", "RMSE",
                      "log.y", "learning.rate", "fcl.num.units", "batch.size",
                      "conv.window.size1", "conv.depth1",
                      "conv.window.size2", "conv.depth2",
                      "strideXconv2D1", "strideYconv2D1",
                      "ksizeX1", "ksizeY1", "strideX1", "strideY1",
                      "strideXconv2D2", "strideYconv2D2", "ksizeX2", "ksizeY2",
                      "strideX2", "strideY2")
set.seed(1)
CV.mat[, "log.y"] <- sample(0:1, nrow(CV.mat), replace = T)
CV.mat[, "learning.rate"] <- 2 ^ -sample(8:15, nrow(CV.mat), replace = T)
CV.mat[, "fcl.num.units"] <- sample(32:256, nrow(CV.mat), replace = T)
CV.mat[, "batch.size"] <- sample(16:64, nrow(CV.mat), replace = T)
CV.mat[, "conv.window.size1"] <- sample(32:64, nrow(CV.mat), replace = T)
CV.mat[, "conv.depth1"] <- sample(8:64, nrow(CV.mat), replace = T)
CV.mat[, "conv.window.size2"] <- sample(16:128, nrow(CV.mat), replace = T)
CV.mat[, "conv.depth2"] <- sample(32:128, nrow(CV.mat), replace = T)
CV.mat[, "strideXconv2D1"] <- sample(2:64, nrow(CV.mat), replace = T)
CV.mat[, "strideYconv2D1"] <- sample(8:128, nrow(CV.mat), replace = T)
CV.mat[, "ksizeX1"] = sample(8:128, nrow(CV.mat), replace = T)
CV.mat[, "ksizeY1"] = sample(16:128, nrow(CV.mat), replace = T)
CV.mat[, "strideX1"] = sample(16:128, nrow(CV.mat), replace = T)
CV.mat[, "strideY1"] = sample(2:64, nrow(CV.mat), replace = T)
CV.mat[, "strideXconv2D2"] <- sample(8:128, nrow(CV.mat), replace = T)
CV.mat[, "strideYconv2D2"] <- sample(2:64, nrow(CV.mat), replace = T)
CV.mat[, "ksizeX2"] = sample(2:64, nrow(CV.mat), replace = T)
CV.mat[, "ksizeY2"] = sample(8:64, nrow(CV.mat), replace = T)
CV.mat[, "strideX2"] = sample(2:64, nrow(CV.mat), replace = T)
CV.mat[, "strideY2"] = sample(2:64, nrow(CV.mat), replace = T)

for(i in 1:nrow(CV.mat)){
  
  cat(paste0(colnames(CV.mat), ":", CV.mat[i, ]), "\n")
  set.seed(1)
  log.y <- as.logical(CV.mat$log.y[i])
  conv.window.size1 <- as.integer(CV.mat$conv.window.size1[i])
  conv.depth1 <- as.integer(CV.mat$conv.depth1[i])
  conv.window.size2 <- as.integer(CV.mat$conv.window.size2[i])
  conv.depth2 <- as.integer(CV.mat$conv.depth2[i])
  strideXconv2D1 <- as.integer(CV.mat$strideXconv2D1[i])
  strideYconv2D1 <- as.integer(CV.mat$strideYconv2D1[i])
  ksizeX1 = as.integer(CV.mat$ksizeX1[i])
  ksizeY1 = as.integer(CV.mat$ksizeY1[i])
  strideX1 = as.integer(CV.mat$strideX1[i])
  strideY1 = as.integer(CV.mat$strideY1[i])
  strideXconv2D2 <- as.integer(CV.mat$strideXconv2D2[i])
  strideYconv2D2 <- as.integer(CV.mat$strideYconv2D2[i])
  ksizeX2 = as.integer(CV.mat$ksizeX2[i])
  ksizeY2 = as.integer(CV.mat$ksizeY2[i])
  strideX2 = as.integer(CV.mat$strideX2[i])
  strideY2 = as.integer(CV.mat$strideY2[i])
  fcl.num.units <- as.integer(CV.mat$fcl.num.units[i])
  batch.size <- as.integer(CV.mat$batch.size[i])

  W_conv1 <- weight_variable(shape(conv.window.size1, conv.window.size1, 1L,
                                   conv.depth1))
  b_conv1 <- bias_variable(shape(conv.depth1))
  
  h_conv1 <- tf$nn$relu(conv2d(x_image, W_conv1,
                               strideX = strideXconv2D1,
                               strideY = strideYconv2D1) + b_conv1)
  h_pool1 <- max_pool_2x2(h_conv1, ksizeX = ksizeX1, ksizeY = ksizeY1,
                          strideX = strideX1, strideY = strideY1)
  
  W_conv2 <- weight_variable(shape = shape(conv.window.size2, conv.window.size2,
                                           conv.depth1, conv.depth2))
  b_conv2 <- bias_variable(shape = shape(conv.depth2))
  
  h_conv2 <- tf$nn$relu(conv2d(h_pool1, W_conv2,
                               strideX = strideXconv2D2,
                               strideY = strideYconv2D2) + b_conv2)
  h_pool2 <- max_pool_2x2(h_conv2, ksizeX = ksizeX2, ksizeY = ksizeY2,
                          strideX = strideX2, strideY = strideY2)
  
  wfc1.shape <- as.integer(as.character(h_pool2$get_shape()[[1]]))
  wfc2.shape <- as.integer(as.character(h_pool2$get_shape()[[2]]))
  wfc3.shape <- as.integer(as.character(h_pool2$get_shape()[[3]]))
  
  W_fc1 <- weight_variable(shape(wfc1.shape * wfc2.shape * wfc3.shape, fcl.num.units))
  b_fc1 <- bias_variable(shape(fcl.num.units))
  
  h_pool2_flat <- tf$reshape(h_pool2, shape(-1L, wfc1.shape *
                                              wfc2.shape *
                                              wfc3.shape))
  h_fc1 <- tf$nn$relu(tf$matmul(h_pool2_flat, W_fc1) + b_fc1)
  
  keep_prob <- tf$placeholder(tf$float32)
  h_fc1_drop <- tf$nn$dropout(h_fc1, keep_prob)
  
  W_fc2 <- weight_variable(shape(fcl.num.units, 1L))
  b_fc2 <- bias_variable(shape(1L))
  
  y_conv <- tf$matmul(h_fc1_drop, W_fc2) + b_fc2
  
  loss <- tf$reduce_mean((y - y_conv) ^ 2)
  optimizer <- tf$train$AdamOptimizer(learning_rate = CV.mat[i, "learning.rate"]) # GradientDescentOptimizer(0.05) #
  train <- optimizer$minimize(loss)
  
  sess$run(tf$global_variables_initializer())
  
  train.mat <- matrix(0, nrow = ceiling(last.train.ind/batch.size), ncol = 2)
  colnames(train.mat) <- c("LogRMSE", "RMSE")
  test.err.calc <- TRUE

  # Iterate through mini-batches
  for (j in 1:ceiling(last.train.ind/batch.size)) {
    inds <- (j - 1) * batch.size + 1:batch.size
    if(inds[length(inds)] > last.train.ind){
      inds <- inds[-which(inds > last.train.ind)]
    }
    batch <- mini.batch(indices = inds, lf = lf,
                        target.y = "Flux", log.y = log.y)

    train$run(feed_dict = dict(x = batch[[1]], y = batch[[2]], keep_prob = 0.5))

    train_accuracy <- loss$eval(feed_dict = dict(x = batch[[1]],
                                                 y = batch[[2]],
                                                 keep_prob = 1.0))
    train_accuracy_exp <- y_conv$eval(feed_dict = dict(x = batch[[1]],
                                                       y = batch[[2]],
                                                       keep_prob = 1.0))
    if(log.y == TRUE){
      train.mat[j, "LogRMSE"] <- round(sqrt(train_accuracy), 8)
      train.mat[j, "RMSE"] <- round(sqrt(mean((exp(batch[[2]]) -
                                                 exp(train_accuracy_exp)) ^ 2)), 8)
    } else {
      train.mat[j, "LogRMSE"] <- round(sqrt(mean((log(batch[[2]]) -
                                                    log(train_accuracy_exp)) ^ 2)), 8)
      train.mat[j, "RMSE"] <- round(sqrt(train_accuracy), 8)
    }

    cat(paste0("Step ", j, " LogRMSE: ", train.mat[j, "LogRMSE"],
               " | RMSE: ", train.mat[j, "RMSE"], "\n"))
    if(round(train_accuracy, 8) == Inf | is.na(train_accuracy)){
      test.err.calc <- FALSE
      break()
    }
  }
  write.csv(train.mat, paste0("CNNtrainPars", i, ".csv"))
  #######################################################
  # Test against test set
  # Need batches because of memory
  #######################################################
  if(round(train_accuracy, 8) == Inf | is.na(train_accuracy)){
    cat("This network did not converge.")
  } else {
    if(!exists("test.batch")){
      test.batch <- mini.batch(indices = last.train.ind:length(lf), lf = lf,
                               target.y = "Flux", log.y = log.y)
    }
    
    num.test.samples <- nrow(test.batch[[1]])
    y_hat <- c()
    for(j in 1:ceiling(num.test.samples / 200)){
      cat("Testing batch", j, "of", ceiling(num.test.samples / 200), "\n")
      inds <- 1:200 + (j - 1) * 200
      if(inds[length(inds)] > nrow(test.batch[[1]])){
        inds <- inds[-which(inds > num.test.samples)]
      }
      testY <- y_conv$eval(feed_dict = dict(x = test.batch[[1]][inds, ],
                                            y = test.batch[[2]][inds, 1,
                                                                drop = F],
                                            keep_prob = 1.0))
      y_hat <- c(y_hat, testY)
    }
    # plot(test.batch[[2]], y_hat)
    if(log.y == TRUE){
      Diffs <- exp(testY) - exp(test.batch[[2]])
      logDiffs <- testY - test.batch[[2]]
    } else {
      Diffs <- testY - test.batch[[2]]
      logDiffs <- log(testY) - log(test.batch[[2]])
    }
    
    CV.mat[i, "RMSE"] <- sqrt(mean(expDiffs ^ 2))
    CV.mat[i, "LogRMSE"] <- sqrt(mean(Diffs ^ 2))
    cat("RMSE: ", CV.mat[i, "RMSE"], "LogRMSE: ", CV.mat[i, "LogRMSE"], "\n" )
  }
}

write.csv(CV.mat, paste0("CNNcvPars_", Sys.Date(), ".csv"), row.names = FALSE)

################# CV.mat Analysis #################################

CV.mat2 <- as.data.frame(CV.mat[order(CV.mat[, "TestMSE"]), -2])
# CV.mat2$ProxyMSE <- 0
# CV.mat2$ProxyMSE[which(CV.mat2$TestMSE > 1 & CV.mat2$TestMSE < 1.797693e+308)] <- 1
# CV.mat2$ProxyMSE[which(CV.mat2$TestMSE == 1.797693e+308)] <- 2
# CV.mat3 <- CV.mat2[, -1]
fit <- lm(TestMSE~., data = CV.mat2)
sort(fit$coefficients)
