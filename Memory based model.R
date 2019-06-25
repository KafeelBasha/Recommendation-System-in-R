setwd("D:\\ML\\Recommendation System")
library(recommenderlab)
data=read.csv("sample_data.csv")
names(data)
# First data needs to be converted to sparse format
#dims=c(max(i),max(j)) default
data=sparseMatrix(i=data$user,j=data$item,x=data$rating,dims = c(length(unique(data$user)), length(unique(data$item))))
data
#realRatingMatrix object
sample_rating=new("realRatingMatrix",data=data)
sample_rating
as(sample_rating,"matrix") #Missed rating

rating=read.csv("rating_sample.csv")[-4]
colnames(rating)<-c("user","movie","rating")
max(rating$user) #i
max(rating$movie) #j
length(unique(rating$movie))
rating$nmovie=factor(rating$movie,labels=1:length(unique(rating$movie)))
rating$nmovie=as.numeric(rating$nmovie)
head(rating)
length(unique(rating$nmovie))
max(rating$nmovie) #new j
tail(sort(unique(rating$nmovie)))
length(unique(rating$user))
length(unique(rating$movie))
length(unique(rating$nmovie))
write.csv(rating,"ratings_sample.csv",row.names=FALSE)
library(dplyr)
rating%>%filter(movie==553)%>%summarise(n())
rating%>%filter(nmovie==441)%>%summarise(n())

rating%>%filter(movie==2424)%>%summarise(n())
rating%>%filter(nmovie==1649)%>%summarise(n())

rating=sparseMatrix(i=rating$user,j=rating$nmovie,x=rating$rating)

#realRatingMatrix object
rating=new("realRatingMatrix",data=rating)
rating
#User based collaborative filtering(UBCF)
UB<-Recommender(rating,method="UBCF",param=list(normalize="center",method="Cosine",nn=40))

#Predictions of missed ratings
predicted=predict(UB,rating,type="ratings")

#Extract the predicted rating for user1 with movie=1200(nmovie=816)
as(predicted[1,816],"matrix")

#Item based collaborative filtering(IBCF)
IB<-Recommender(rating,method="IBCF",param=list(normalize="center",method="Cosine",k=40))

#Predictions of missed ratings 
predi=predict(IB,rating,type="ratings")

#Extract the predicted rating for user3 with movie=2269(nmovie=5769)
as(predi,"matrix")[4,1002]

#Item based collaborative filtering(IBCF)
IBp<-Recommender(rating,method="IBCF",param=list(normalize="center",method="pearson",k=40))

#Predictions of missed ratings 
predp=predict(IBp,rating,type="ratings")

#Extract the predicted rating for user3 with movie=7000(nmovie=3002)
as(predp,"matrix")[5,5000]

##Instead of using just one train set,we can split the data into k-parts
#and then evaluate the model performance out of sample.
n_folds=3
ratings_movies <-rating[rowCounts(rating)>50,  colCounts(rating) > 20]
min(rowCounts(ratings_movies))
eval_sets=evaluationScheme(data=ratings_movies,method="cross-validation",k=n_folds,given=5,goodRating=2.75)
eval_sets

#Training set
getData(eval_sets, "train")
nrow(getData(eval_sets, "train")) / nrow(ratings_movies) 

#Test sets
getData(eval_sets, "known")
getData(eval_sets, "unknown")

nrow(getData(eval_sets, "known")) / nrow(ratings_movies)
#Number of items in known set should be equal to 'given'
unique(rowCounts(getData(eval_sets, "known")))

library(ggplot2)
qplot(rowCounts(getData(eval_sets, "unknown"))) + geom_histogram(binwidth = 10) + ggtitle("unknown items by the users")

eval_ib=Recommender(data=getData(eval_sets,"train"),method="IBCF")
eval_pred=predict(eval_ib,newdata=getData(eval_sets,"known"),n=5,type="ratings")
eval_acc=calcPredictionAccuracy(x=eval_pred,data=getData(eval_sets,"unknown"),byUser=TRUE)
str(eval_acc)
head(eval_acc)
dim(eval_acc)

#Evaluate the model as whole by setting byUser=FALSE
eval_acc=calcPredictionAccuracy(x=eval_pred,data=getData(eval_sets,"unknown"),byUser=FALSE)
eval_acc



#Comapring Models
#Algorithms
algorithms<-list(model1=list(name="UBCF",param=list(normalize="center",method="Cosine",nn=20)),
                 model2=list(name="UBCF",param=list(normalize="center",method="Cosine",nn=30)),
                 model3=list(name="UBCF",param=list(normalize="center",method="pearson",nn=20)),
                 model4=list(name="UBCF",param=list(normalize="center",method="pearson",nn=30)))
