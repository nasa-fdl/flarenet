# create_AIA_index.R

###### Moving files #######################
filename <- "/home/thernandez/AIA/AIA201401*"

cat(
paste0("scp ", filename, " thernandez@",
       c("10.184.143.161", "10.184.143.134", "10.184.143.148", "10.184.143.183"), #,
         # "10.184.143.136", "10.184.143.174", "10.184.143.182", "10.184.143.178"),
       ":/home/thernandez/AIA2014/\n")[1:3], "exit\n",
paste0("nasa", 9:11, "\nsudo cp ", filename, " /data/sw/AIA2014/\nexit\n"))
#######################################
# Download AIA data

# http://jsoc.stanford.edu/data/aia/synoptic/2014/06/01/H0000/
# download.file(url, destfile, method)
base.url <- "http://jsoc.stanford.edu/data/aia/synoptic/2014/"
months <- sprintf("%02.f", 1:12)
max.days <- c(31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
hours <- sprintf("%02.f",0:23)
minutes <- sprintf("%02.f",c(00, 12, 24, 36, 48))
wl <- c("0094", "0131", "0171", "0193", "0211", "0304", "0335", "1600")

target.file.list <- target.urls <- rep("", sum(max.days) * 5 * 24 * 8)
ind <- 1

for(i in 1:length(months)){
  cat(i, "\n")
  for(j in 1:(max.days[i])){
    for(k in 1:length(hours)){
      for(l in 1:length(minutes)){
        for(m in 1:length(wl)){
          aia.url <- paste0(base.url, months[i], "/", sprintf("%02.f", j),
                            "/H", hours[k], "00/")
          aia.file <- paste0("AIA2014", months[i], sprintf("%02.f", j), "_",
                             hours[k], minutes[l], "_", wl[m], ".fits")
          # http://jsoc.stanford.edu/data/aia/synoptic/2014/06/01/H0000/AIA20140601_0000_0094.fits
          target.file.list[ind] <- aia.file
          target.urls[ind] <- paste0(aia.url, aia.file)
          ind <- ind + 1
        }
      }
    }
  }
}

lf <- list.files("AIA2014")
done.list <- which(target.file.list %in% lf)
todo.list <- c(1:length(target.file.list))[-done.list]

for(i in 1:length(todo.list)){
  download.file(url = target.urls[todo.list[i]],
                destfile = paste0("AIA2014/", target.file.list[todo.list[i]]),
                method = "libcurl")
}

# Checking progress
lf <- list.files("AIA2014")
length(lf)
# [1] 117132

########################## AIA file indexing ###########################
# setwd("/data/sw/AIA0171/")
lf <- list.files("~/AIA2014/")
# 
year <- as.numeric(substr(lf, 4, 7))
month <- as.numeric(substr(lf, 8, 9))
day <-as.numeric(substr(lf, 10, 11))
hour <- as.numeric(substr(lf, 13, 14))
minute <- as.numeric(substr(lf, 15, 16))
write.csv(AIAindex <- cbind(year, month, day, hour, minute),
          "~/AIA_index.csv",
          row.names = F)

time.stamp <- substr(lf, 1, 16)
wave.length <- substr(lf, 18, 21)
time.wave.tbl <- table(time.stamp, wave.length)
time.wave.tbl8 <- time.wave.tbl[, 1:8]
num.channels <- apply(time.wave.tbl8, 1, sum)
all.channels <- num.channels == 8
sum(all.channels)
all.channels.times <- rownames(time.wave.tbl8)[all.channels]
year <- as.numeric(substr(all.channels.times, 4, 7))
month <- as.numeric(substr(all.channels.times, 8, 9))
day <-as.numeric(substr(all.channels.times, 10, 11))
hour <- as.numeric(substr(all.channels.times, 13, 14))
minute <- as.numeric(substr(all.channels.times, 15, 16))
AIAindex.allchannels <- cbind(year, month, day, hour, minute)
AIAindex.allchannels.POSIX <- as.POSIXct(paste(
  paste(AIAindex.allchannels[, 1],
        sprintf("%02.f", AIAindex.allchannels[, 2]),
        sprintf("%02.f", AIAindex.allchannels[, 3]), sep = "-"),
  paste(sprintf("%02.f", AIAindex.allchannels[, 4]),
        sprintf("%02.f", AIAindex.allchannels[, 5]), "00", sep = ":"),
  sep = " "), tz = "UTC")
write.csv(AIAindex.allchannels.POSIX, "~/AIA_index_allChannels.csv",
          row.names = F)

TimeDiffs <- AIAindex.allchannels.POSIX[-1] -
  AIAindex.allchannels.POSIX[-length(AIAindex.allchannels.POSIX)]
table(TimeDiffs)
####################### Checking completeness of data #################################

datetime.all <- read.csv("AIAfilelist.txt", stringsAsFactors = F)[[1]]
aia.bool <- substr(datetime.all, 32, 34)
datetime.all <- datetime.all[which(aia.bool == "AIA")]

year <- as.numeric(substr(datetime.all, 35, 38))
month <- as.numeric(substr(datetime.all, 39, 40))
day <-as.numeric(substr(datetime.all, 41, 42))
hour <- as.numeric(substr(datetime.all, 44, 45))
minute <- as.numeric(substr(datetime.all, 46, 47))
write.csv(AIAindex <- cbind(year, month, day, hour, minute),
          "~/AIA_index.csv",
          row.names = F)

# AIAindex <- read.csv("~/solar-forecast/preprocessing/AIA0171_index.csv")
# AIAindex <- read.csv("/data/sw/AIA0171_index.csv")
day.time <- as.POSIXct(paste(
  paste(AIAindex[, 1], sprintf("%02.f", AIAindex[, 2]),
        sprintf("%02.f", AIAindex[, 3]), sep = "-"),
  paste(sprintf("%02.f", AIAindex[, 4]),
        sprintf("%02.f", AIAindex[, 5]), sep = ":"),
  sep = " "), tz = "UTC")
# write.csv(day.time, "/data/sw/AIA0171_POSIX.csv", row.names = F)
write.csv(day.time, "~/AIA_POSIX.csv", row.names = F)

day.time <- read.csv("~/AIA_POSIX.csv", stringsAsFactors = F)[[1]]

########################### Flux data processing ############################
filepath <- "Flux_data_2010_2017.csv"

# processFile = function(filepath) {
date.times <- seq(from = as.POSIXct("2010-01-01 00:02", "UTC"),
                  to = as.POSIXct("2017-06-30 23:58", "UTC"), by = 120)
flux <- data.frame(matrix(NA, nrow = length(date.times), ncol = 3))
colnames(flux) <- c("Time", "Flux", "Count")
flux$Time <- date.times

current.minute <- date.times[1]
vec <- rep(NA, 60)
i <- 1
j <- 1
con = file(filepath, "r")
# Burn the header
line <- readLines(con, n = 1)

while ( TRUE ) {
  
  line <- readLines(con, n = 1)
  if ( length(line) == 0 ) {
    break
  } else {
    line <- gsub("\"", "", line)
    line <- strsplit(line, ",")[[1]][c(1, 4)]
    
    minute <- as.POSIXct(line[1], "UTC")
    
    if(minute <= current.minute){
      vec[j] <- as.numeric(line[2])
      j <- j + 1
    } else {
      flux[i, "Flux"] <- max(vec, na.rm = TRUE)
      flux[i, "Count"] <- sum(!is.na(vec))
      print(flux[i, ])
      
      vec <- rep(NA, 60)
      i <- i + 1
      while(minute > flux[i, "Time"]){
        i <- i + 1
      }
      current.minute <- flux[i, "Time"]
      vec[1] <- as.numeric(line[2])
      j <- 2
    }
  }
}
close(con)
write.csv(flux, "Flux_2010_2017_max.csv", row.names = F)
# }

# filepath <- "flux_data2.csv"
# processFile(filepath)
##################################################################
################## Sample head of data #######################
# # Every two seconds! not every two minutes!
# filepath <- "flux_data2.csv"
# con = file(filepath, "r")
# head.of.file <- readLines(con, n = 100000)
# write.csv(head.of.file, "flux_data_head.csv", row.names = F)

################# Creating additional Y vars #####################

# day.time <- read.csv("AIA0171_POSIX.csv", stringsAsFactors = F)[[1]]
# day.time <- as.POSIXct(day.time, tz = "UTC")

# Adding additional cols
flux <- read.csv("Flux_2010_2017_max.csv")
flux[, 1] <- as.POSIXct(flux[, 1], tz = "UTC")
flux$Flux_1h_1h12m <- NA
flux$Flux_1h_2h <- NA
flux$Flux_1h_1d1h_top1 <- NA
flux$Flux_1h_1d1h_top2 <- NA
flux$Flux_1h_1d1h_top3 <- NA
flux$Flux_1h_1d1h_top4 <- NA
flux$Flux_1h_1d1h_top5 <- NA
flux$Flux_1h_1d1h_top6 <- NA
flux$Flux_1h_1d1h_top7 <- NA
flux$Flux_1h_1d1h_top8 <- NA
flux$Flux_1h_1d1h_top9 <- NA
flux$Flux_1h_1d1h_top10 <- NA
flux$Flux_1h_1d1h_top10sum <- NA
top10.cols <- c(which(colnames(flux) == "Flux_1h_1d1h_top1"):which(colnames(flux) == "Flux_1h_1d1h_top10"))

library(zoo)
window.len <- 6
temp.rollmax <- rollmax(flux$Flux, k = window.len)
flux$Flux_1h_1h12m <- c(temp.rollmax[31:length(temp.rollmax)],
                        rep(NA, 30 + window.len - 1))
window.len <- 30
temp.rollmax <- rollmax(flux$Flux, k = window.len)
flux$Flux_1h_2h <- c(temp.rollmax[31:length(temp.rollmax)],
                     rep(NA, 30 + window.len - 1))

sort.top10 <- function(x){sort(x, decreasing = TRUE)[1:10]}
temp.top10 <- rollapply(flux$Flux, 720, sort, decreasing = TRUE)[, 1:10]

flux[, top10.cols] <- rbind(temp.top10[31:nrow(temp.top10), ],
                            matrix(NA, nrow = 30 + 720 - 1, ncol = 10))
flux$Flux_1h_1d1h_top10sum <- c(apply(temp.top10, 1, sum), rep(NA, 30 + 690 - 1))
write.csv(flux, "Flux_2010_2017_allY.csv", row.names = FALSE)

# Comparing flux data sets
flux <- read.csv("Flux_2010_2017_allY.csv", stringsAsFactors = FALSE)
flux$Time <- as.POSIXct(flux$Time, tz = "UTC")

flux2 <- read.csv("Y_GOES_XRAY_20140101_20141231.csv", stringsAsFactors = FALSE,
                  header = FALSE)
################# Plot #####################


Plot_Monthly <- function(flux){
  y_m <- format(flux[, 1], "%y_%m")
  # table(y_m)
  y_m_names <- names(table(y_m))[49:60] # Doing only 2014
  
  for(i in 1:length(y_m_names)){
    png(file = paste0("XrayFlux_20", y_m_names[i],".png")) #, width = nrow(temp.mat)/100)
    par(mfrow=c(2, 1))
    
    cex.sz <- .5
    temp.mat <- flux[which(y_m == y_m_names[i]), ]
    # plot(x = timestamps, y = xrayflux, type = "l")
    # if(log.flux == F){
    plot(x = temp.mat[, 1], y = temp.mat[, 2], type = "b",
         col = heat.colors(90)[temp.mat[, 3]],
         xlab = "Date", ylab = "X-ray Flux", cex = cex.sz,
         main = paste0("XrayFlux 20", y_m_names[i]),
         ylim = c(min(flux[, 2], na.rm = T), 10e-4))
    abline(10e-04, 0)
    abline(10e-05, 0, col = 2)
    abline(10e-06, 0, col = 3)
    abline(10e-07, 0, col = 4)
    which.na <- which(is.na(temp.mat$Flux))
    points(x = temp.mat$Time[which.na], y = rep(0, length(which.na)),
           cex = cex.sz)
    legend("topright", legend = c("X-class", "M-class", "C-class", "B-class"),
           lty = 1, col = 1:4, cex = .75)
    legend("topleft", legend = c("Missing"), pch = 1, col = 1, cex = .75)
    
    # } else {
    plot(x = temp.mat[, 1], y = temp.mat[, 2], type = "b", log = "y",
         col = heat.colors(90)[temp.mat[, 3]],
         xlab = "Date", ylab = "X-ray Flux", cex = cex.sz,
         main = paste0("XrayFlux 20", y_m_names[i]),
         ylim = c(min(flux[, 2], na.rm = T), 2*10e-4))
    abline(10e-04, 0, untf = T)
    abline(10e-05, 0, col = 2, untf = T)
    abline(10e-06, 0, col = 3, untf = T)
    abline(10e-07, 0, col = 4, untf = T)
    which.na <- which(is.na(temp.mat$Flux))
    points(x = temp.mat$Time[which.na], y = rep(log(min(flux[, 2],
                                                        na.rm = T) + 1),
                                                length(which.na)),
           cex = cex.sz)
    covered <- intersect(day.time, temp.mat$Time)
    covered.col <- rep(2, nrow(temp.mat))
    if(length(covered) > 0){
      covered.col[which(covered %in% temp.mat$Time)] <- 1
    }
    
    points(x = temp.mat$Time, y = rep(2 * 10e-4, nrow(temp.mat)),
           col = covered.col, pch = 7, cex = cex.sz)
    legend("right", legend = c("sample", "missing"),
           pch = 7, col = 1:2, cex = .75)
    
    # }
    dev.off()
  }
}

Plot_Monthly(flux = flux)

############## Convert to feather #############################
library(reticulate)
astropy <- import("astropy")
lf <- list.files("/home/thernandez/AIA2014")

all.channel.index.POSIX <- as.POSIXct(read.csv("/home/thernandez/AIA_index_allChannels.csv",
                                               stringsAsFactors = FALSE)[[1]],
                                      tz = "UTC")

all.channel.index <- outer(format(all.channel.index.POSIX, "%Y%m%d_%H%M"),
                           c("_0094", "_0131", "_0171", "_0193",
                             "_0211", "_0304", "_0335", "_1600"),
                           paste, sep = "")

library(feather)

# Could parallelize if needed
# Need to do error checking
# Need to delete 343kb files
for(i in 3562:3720){ #nrow(all.channel.index)){
  if(i %% 10 == 0){
    cat("Num", i, "of 3720\n")
  }
  big.mat <- matrix(NA, nrow = 1024*1024, ncol = 8)
  for(j in 1:8){
    temp <- astropy$io$fits$open(paste0("/home/thernandez/AIA2014/AIA", all.channel.index[i, j], ".fits"))
    temp$verify("fix")
    exp.time <- as.numeric(substring(strsplit(as.character(temp[[1]]$header),
                                              "EXPTIME")[[1]][2], 4, 12))
    temp.mat <- temp[[1]]$data
    temp.mat[temp.mat <= 0] <- 0
    temp.mat <- temp.mat + 1
    big.mat[, j] <- c(t(temp.mat / exp.time))
    if(j == 3){
      header <- as.character(temp[[1]]$header)
      write.table(header, paste0("/home/thernandez/HeaderAIA2014/AIA",
                                 all.channel.index[i, j], ".txt"),
                  quote = FALSE, row.names = FALSE, col.names = FALSE)
    }
  }
  write_feather(as.data.frame(big.mat),
                paste0("/home/thernandez/FeatherAIA2014/AIA",
                       format(all.channel.index.POSIX[i], "%Y%m%d_%H%M"),
                       ".feather"))
}

# big.mat <- read_feather(paste0("/home/thernandez/FeatherAIA2014/AIA",
#                                format(all.channel.index.POSIX[i], "%Y%m%d_%H%M"),
#                                ".feather"))

errors <- c(95, 715, 1662, 1769:1793, 2502, 2503, 3133, 3342, 3436:3443, 3561)

################################################################################
#################### Synthesizing y.mat and feather files ####################

lf <- list.files("/home/thernandez/FeatherAIA2014/")
y.mat <- read.csv("/home/thernandez/Flux_2010_2017_allY.csv", stringsAsFactors = F)
y.mat$Time <- as.POSIXct(y.mat$Time, tz = "UTC")
bad.y <- which(is.na(y.mat$Flux))
y.mat <- y.mat[-bad.y, ]

lf.str <- substring(lf, 4, 16)
y.mat.str <- as.character(format(y.mat$Time, "%Y%m%d_%H%M"))
good.times <- intersect(lf.str, y.mat.str)
good.ind <- which(y.mat.str %in% good.times)
write.csv(y.mat[good.ind, ],
          "~/solar-forecast/datasets/Flux_FeatherAIA2014.csv",
          row.names = FALSE)
