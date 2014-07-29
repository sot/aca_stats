SA<-read.csv('data/southAfrica.csv')
fit <- glm(chd ~ sbp+tobacco+ldl+factor(famhist)+obesity+alcohol+age, family='binomial', data=SA)
summary(fit)

f47<-read.csv('data/finney1947.csv')
fit2 <- glm(Y ~ Volume+Rate, family='binomial', data=f47, trace=TRUE)
summary(fit2)

pima<-read.csv('data/PimaIndians.csv')
fit3 <- glm(type ~ npreg+glu+bp+skin+bmi+ped, family='binomial', data=pima, trace=TRUE)
summary(fit3)