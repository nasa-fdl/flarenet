# AIA2.R

library(tensorflow)
library(reticulate)
library(feather)

# astropy <- import("astropy")
weight_variable <- function(shape) {
  initial <- tf$truncated_normal(shape, stddev=0.1)
  tf$Variable(initial)
}

bias_variable <- function(shape) {
  initial <- tf$constant(0.1, shape=shape)
  tf$Variable(initial)
}

conv2d <- function(x, W, strideX = 50, strideY = 50) {
  tf$nn$conv2d(x, W,
               strides=c(1L, as.integer(strideX), as.integer(strideY), 1L),
               padding='SAME')
}

max_pool_2x2 <- function(x, ksizeX = 50, ksizeY = 50,
                         strideX = 50, strideY = 50) {
  tf$nn$max_pool(
    x, 
    ksize=c(1L, as.integer(ksizeX), as.integer(ksizeY), 1L),
    strides=c(1L, as.integer(strideX), as.integer(strideY), 1L), 
    padding='SAME')
}


# Extracts matrices from fits file
fits2mat <- function(filename){
  temp <- astropy$io$fits$open(filename)
  temp$verify("fix")
  exp.time <- as.numeric(substring(strsplit(as.character(temp[[1]]$header),
                                            "EXPTIME")[[1]][2], 4, 12))
  temp.mat <- temp[[1]]$data
  temp.mat[temp.mat <= 0] <- 1
  log(t(temp.mat / exp.time))
}

###### Example usage:
# temp.x <- fits2mat(filename = paste0("/home/thernandez/AIA2014/", lf[2]))
# image(temp.x, zlim = c(-1, 5))
######

# Creates a 3D matrix of all 8 channels
indexTo3Dmat <- function(channel.index, channels.used = c(3), feather = TRUE){
  # array(., dim = c(1024, 1024, length(channels.used)))
  if(feather == TRUE){
    temp.mat <- as.matrix(read_feather(paste0("/home/thernandez/FeatherAIA2014/",
                                              channel.index)))[, channels.used]
  } else {
    temp.mat <- matrix(NA, nrow = 1024 ^ 2, ncol = length(channels.used))
    for(i in 1:length(channels.used)){
      temp.mat[, i] <- unlist(fits2mat(paste0("/home/thernandez/AIA2014/AIA",
                                              channel.index[channels.used],
                                              ".fits")))
    }
  }
  c(temp.mat)
}

###### Example usage:
# temp = indexTo3Dmat(channel.index = all.channel.index[1, ])
# temp2 = apply(temp, c(1, 2), mean)
# image(temp2, zlim = c(10, 100))
######
# Creates collection of 3D input matrices and corresponding Flux outputs
# Use for each to parallelize
mini.batch <- function(indices = 1:5,
                       lf = list.files("/home/thernandez/FeatherAIA2014/"),
                       y.mat = read.csv("/home/thernandez/solar-forecast/datasets/Flux_FeatherAIA2014.csv"),
                       target.y = c("Flux", "Flux_1h_1h12m",
                                    "Flux_1h_2h", "Flux_1h_1d1h_top1"),
                       normalization.x = TRUE, normalization.y = FALSE,
                       log.x = TRUE, log.y = TRUE,
                       channels.used = c(3), feather = TRUE){
  x = matrix(NA, nrow = length(indices),
             ncol = 1024 ^ 2 * length(channels.used))
  y = as.matrix(y.mat[indices, target.y])
  # cat("\n", dim(x))
  for(i in 1:length(indices)){
    x[i, ] <- c(indexTo3Dmat(channel.index = lf[indices[i]], #format(all.channel.index.POSIX, "%Y%m%d_%H%M"),
                             channels.used = channels.used))
  }
  if(log.x == TRUE){
    x <- log(x)
  }
  if(normalization.x == TRUE){
    x <- matrix(scale(c(x)), nrow = length(indices))
  }
  if(log.y == TRUE){
    y <- log(y)
  }
  if(normalization.y == TRUE){
    y <- matrix(scale(y), nrow = length(indices))
  }
  mb <- list(x = x, y = y)
}

###### Example usage:
# temp = mini.batch(1:5, "Flux")
######################################################
######################################################
############ Old functions from before Feather files ########################
# Creates an index of date-times with all 8 channels and lists them as a matrix
# One row for time, one col for channel
# all.channel.index.POSIX <- as.POSIXct(read.csv("/home/thernandez/AIA_index_allChannels.csv",
#                                                stringsAsFactors = FALSE)[[1]],
#                                       tz = "UTC")
# 
# all.channel.index <- outer(format(all.channel.index.POSIX, "%Y%m%d_%H%M"),
#                            c("_0094", "_0131", "_0171", "_0193",
#                              "_0211", "_0304", "_0335", "_1600"),
#                            paste, sep = "")
# 
# y.mat <- read.csv("/home/thernandez/Flux_2010_2017_allY.csv", stringsAsFactors = F)
# y.mat$Time <- as.POSIXct(y.mat$Time, tz = "UTC")
# good.inds <- which(all.channel.index.POSIX %in% y.mat$Time)
# y.mat <- y.mat[good.inds, ]
# good.inds2 <- which(!is.na(y.mat[, "Flux"]))
# y.mat <- y.mat[good.inds2, ]

# FIXX!!!!!!!!
# Double check things all line up.
# if(length(good.inds) != nrow(all.channel.index)){stop("Not same!!!")}