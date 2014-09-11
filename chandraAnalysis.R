library(boot)
library(arm)
require(lattice)
library(splines)
source( "/Users/bvegetabile/_ucirvine/Misc/stat211gillennotes/Stat211Functions.q" )

#Loading in the chandra data
chandra <- read.csv('chandra/development/acq_data.csv', header=FALSE, 
                    col.names=c('obsid','obi','tstart','tstop','slot',
                                'idx','cat_pos','type','agasc_id','obc_id',
                                'yang','zang','mag','color','halfw','mag_obs',
                                'yang_obs','zang_obs','y_offset','z_offset',
                                'd_mag','d_yang','d_zang','revision','warm_pix',
                                'red_spoiler','yellow_spoiler','bad_pixel',
                                'common_column','known_bad_star','tstart_jyear',
                                'tstart_quarter','mag_floor','year','failure'))

oldmodel <- function(mags, warmpix){
  m <- mags - 10.
  scale <- 10**(0.18 + 0.99*m - 0.49*m*m)
  offset <- 10**(-1.49 + 0.89*m + 0.28*m*m)
  p_fail <- (warmpix * scale) + offset
  return(p_fail)
}

oldmodel2 <- function(mags, warmpix){
  m <- mags - 10.
  scale <- 10**(0.14380254580909044 
                + 1.1120795606754033*m 
                - 0.23828842036054182*m*m)
  offset <- 10**(-1.5420568527434402
                + 0.6924898542529816*m 
                + 0.1414809635388084*m*m)
  p_fail <- (warmpix * scale) + offset
  return(p_fail)
}

#Setting the range of the training data to be subsetted
train_lower = 2000.
train_upper = 2014.0

trainingdata <- subset(chandra, tstart_jyear>=train_lower & tstart_jyear <= train_upper)

#Fitting the models based on this training data
probit_fit <- glm(failure ~ 
                    poly(mag,2) + poly(mag,2)*log(warm_pix)
                  + sin(tstart_jyear) + cos(tstart_jyear) + tstart_jyear,
                  data=trainingdata, family ='binomial'(link='probit'))
logit_fit <- glm(failure ~ 
                   poly(mag,2) + poly(mag,2)*log(warm_pix)
                 + sin(tstart_jyear) + cos(tstart_jyear) + tstart_jyear,
                 data=trainingdata, family ='binomial'(link='logit'))
probit_fit_bayes <- bayesglm(failure ~ 
                               poly(mag,2) + poly(mag,2)*warm_pix
                             + sin(tstart_jyear) + cos(tstart_jyear),
                             data=trainingdata, family ='binomial'(link='probit'))
summary(probit_fit_bayes)
summary(probit_fit)
summary(logit_fit)

#Setting up visualizations for out of sample test data
test_lower = 2014.0
test_upper = 2014.5
wptest = 0.14

testdata <- subset(chandra, tstart_jyear>=test_lower & tstart_jyear <= test_upper)

failures <- subset(testdata, failure==1
                   & warm_pix >= wptest - 0.02
                   & warm_pix <= wptest + 0.02)$mag
totalobs <- subset(testdata,
                   warm_pix >= wptest - 0.02
                   & warm_pix <= wptest + 0.02)$mag

counts_fails <- hist(failures, breaks=seq(5.0,13.,0.25))
counts_totals <- hist(totalobs, breaks=seq(5.,13.,0.25))

percents <- counts_fails$counts / counts_totals$counts

# Plotting and setting up confidence intervals
mag <- seq(5.0,13.,0.25)
warm_pix <- rep(wptest,length(mag))
tstart_jyear <- rep((test_lower + test_upper)/2.0, length(mag))
xtest <- data.frame(mag, warm_pix, tstart_jyear)

critval <- 2 ## approx 95% CI

preds <- predict(probit_fit, newdata=xtest, type='link', se.fit=TRUE)
upr <- preds$fit + (critval * preds$se.fit)
lwr <- preds$fit - (critval * preds$se.fit)
fit <- preds$fit

preds_logit <- predict(logit_fit, newdata=xtest, type='link', se.fit=TRUE)
upr2 <- preds_logit$fit + (critval * preds_logit$se.fit)
lwr2 <- preds_logit$fit - (critval * preds_logit$se.fit)
fit2 <- preds_logit$fit

preds_bayes <- predict(probit_fit_bayes, newdata=xtest, type='link', se.fit=TRUE)
upr_b <- preds_bayes$fit + (critval * preds_bayes$se.fit)
lwr_b <- preds_bayes$fit - (critval * preds_bayes$se.fit)
fit_b <- preds_bayes$fit

# png(paste('modelfit_wpf_',as.integer(100*wptest),'.png', sep=''), 
#     width=10, height=6, units='in', res=400)
plot(mag[2:length(mag)], percents, 
     type='l', xlab='Star Magnitude', ylab='Probability of Failure', xlim=c(8,11))
lines(mag, pnorm(fit), col='red', lty=1, lwd=2)
lines(mag, pnorm(upr), col='red', lty=2, lwd=2)
lines(mag, pnorm(lwr), col='red', lty=2, lwd=2)
lines(mag, oldmodel(mag, wptest), col='green', lty=1, lwd=2)
lines(mag, oldmodel2(mag, wptest), col='blue', lty=1, lwd=2)
title(paste("Warm Pixel Fraction: ", wptest))
# dev.off()

binary.gof(probit_fit, ngrp=25)
max(chandra["tstart_jyear"])

# 
# dpt <- data.frame(mag=10.0, warm_pix=0.12, tstart_jyear=2014.25)
# 
# wps = seq(0.08,0.18,0.005)
# mags = seq(5.0,12.0,0.05)
# t = 2014.25
# 
# grid = c()
# for (m in mags){
#     row = c()
#     for (wp in wps){
#       dpt <- data.frame(mag=m, warm_pix=wp, tstart_jyear=t)
#       prob <- pnorm(predict(probit_fit, newdata=dpt))
#       row <- cbind(row, prob)
#     }
#     grid <- rbind(grid, row)
# }
# print(grid)
# 
# success = subset(chandra, tstart_jyear>= 2014.0 &obc_id==" 'ID'")
# fails = subset(chandra, tstart_jyear>= 2014.0 &obc_id==" 'NOID'")
# 
# png('/Users/bvegetabile/git/aca_stats/2d_contour_overlay.png',
#     width=10, height=6, units='in', res=400)
# image(x=mags, y=wps, grid, lwd=2, 
#       col=topo.colors(51, alpha=.05), 
#       xlab='Star Magnitude', ylab='Warm Pixel Fraction',
#       xlim=c(6.0,11.25), ylim=c(.08,.18))
# points(success$mag, success$warm_pix, 
#       col='blue', cex=4, pch='.')
# points(fails$mag, fails$warm_pix, col='red', cex=2, pch='*')
# contour(x=mags, y=wps, grid, lwd=2, 
#         drawlabels=TRUE, add=TRUE, alpha=.5)
# dev.off()
