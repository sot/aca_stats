library(boot)
library(arm)

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

#Setting the range of the training data to be subsetted
train_lower = 2000.0
train_upper = 2014.0

trainingdata <- subset(chandra, year>=train_lower & year <= train_upper)

#Setting up visualizations for test data

test_lower = 2000.0
test_upper = 2014.25
wptest = 0.1

testdata <- subset(chandra, year>=test_lower & year <= test_upper)

successes <- subset(testdata, failure==0 
                    & warm_pix >= wptest - 0.02
                    & warm_pix <= wptest + 0.02)$mag
failures <- subset(testdata, failure==1
                   & warm_pix >= wptest - 0.02
                   & warm_pix <= wptest + 0.02)$mag
totalobs <- subset(testdata,
                   warm_pix >= wptest - 0.02
                   & warm_pix <= wptest + 0.02)$mag

counts_fails <- hist(failures, breaks=seq(5.0,13.,0.25))
counts_totals <- hist(totalobs, breaks=seq(5.,13.,0.25))

percents <- counts_fails$counts / counts_totals$counts

probit_fit <- glm(failure ~ 
                    poly(mag,2) + warm_pix + poly(mag,2)*warm_pix
                    + poly(mag,2)*tstart_jyear + warm_pix*tstart_jyear,
                    data=trainingdata, family ='binomial'(link='probit'))
logit_fit <- glm(failure ~ 
                   poly(mag,2) + warm_pix + poly(mag,2)*warm_pix
                 + poly(mag,2)*tstart_jyear + warm_pix*tstart_jyear,
                 data=trainingdata, family ='binomial'(link='logit'))
summary(probit_fit)
summary(logit_fit)

# Plotting and setting up confidence intervals
mag <- seq(5.0,13.,0.25)
# warm_pix <- rep(mean(testdata$warm_pix),length(mag))
warm_pix <- rep(wptest,length(mag))
tstart_jyear <- rep((test_lower + test_upper)/2.0, length(mag))
xtest <- data.frame(mag, warm_pix, tstart_jyear)

preds <- predict(probit_fit, newdata=xtest, type='link', se.fit=TRUE)
preds_logit <- predict(logit_fit, newdata=xtest, type='link', se.fit=TRUE)
preds
critval <- 2 ## approx 95% CI
upr <- preds$fit + (critval * preds$se.fit)
lwr <- preds$fit - (critval * preds$se.fit)
fit <- preds$fit

upr2 <- preds_logit$fit + (critval * preds_logit$se.fit)
lwr2 <- preds_logit$fit - (critval * preds_logit$se.fit)
fit2 <- preds_logit$fit

#pdf('logit_v_probit.pdf', width=10, height=6.0)
plot(mag[2:length(mag)], percents, 
     type='l', xlab='Star Magnitude', ylab='Probability of Failure', 
     ylim=c(0,0.8), xlim=c(8,11))
lines(mag, pnorm(fit), col='red', lty=2, lwd=2)
lines(mag, pnorm(fit2), col='red', lty=3, lwd=2)
#lines(skX, sklearn, lty=2, lwd=2, col='blue')
#lines(mag, pnorm(as.matrix(xtest2)%*%bayes), lty=3, lwd=2, col='blue')
title('Fits from Different Methods')
legend(8,0.8,c("Observed", "R-Probit", "R-Logit"), 
       lty=c(1,2,3), lwd=c(1,2,2), 
       col=c("Black", "Red", "Red"))
#dev.off()
print(percents)

plot(mag[2:length(mag)], percents, 
     type='l', xlab='Star Magnitude', ylab='Probability of Failure', 
     ylim=c(0,1.0), xlim=c(5,13))
lines(mag, pnorm(fit), col='red', lty=2, lwd=2)
lines(mag, pnorm(upr), col='red', lty=2, lwd=2)
lines(mag, pnorm(lwr), col='red', lty=2, lwd=2)


# probit_fit_bayes <- bayesglm(failure ~ 
#                         poly(mag,2) + warm_pix + poly(mag,2)*warm_pix
#                         + poly(mag,2)*tstart_jyear + warm_pix*tstart_jyear,
#                         data=trainingdata, family ='binomial'(link='probit'))
# simmed <- sim(probit_fit_bayes,10)

