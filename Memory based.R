setwd("D:\\ML\\Recommendation System")
library(recommenderlab)
data=read.csv("sample_data.csv")
names(data)
# First data needs to be converted to sparse(will have lots of empty field) format
#dims=c(max(i),max(j)) default
data=sparseMatrix(i=data$user,j=data$item,x=data$rating,dims = c(length(unique(data$user)), length(unique(data$item))))
data
#realRatingMatrix object
sample_rating=new("realRatingMatrix",data=data)
sample_rating
as(sample_rating,"matrix") #Missed rating

rating=read.csv("ratings_sample.csv")
colnames(rating)
max(rating$user) #i
max(rating$movie) #j
length(unique(rating$movie))

rating=sparseMatrix(i=rating$user,j=rating$movie,x=rating$rating,dimnames = list(paste("u", 1:length(unique(rating$user)), sep = ""), 
                                                                                 paste("m", 1:length(unique(rating$movie)), sep = "")))
#realRatingMatrix object
rating=new("realRatingMatrix",data=rating)
rating

#User based collaborative filtering(UBCF)
UB<-Recommender(rating,method="UBCF",param=list(normalize="center",method="Cosine",nn=40))

#Predictions of missed ratings
predu=predict(UB,rating,type="ratings")
predu

#Extract the predicted rating for user1 with movie=816
as(predu[5,500],"matrix")

#Item based collaborative filtering(IBCF)
IB<-Recommender(rating,method="IBCF",param=list(normalize="center",method="Cosine",k=40))

#Predictions of missed ratings 
predi=predict(IB,rating,type="ratings")

#Extract the predicted rating for user5 with movie=500
as(predi[5,500],"matrix")

#Most Popular or Item Average
pAvg<-Recommender(rating,method="POPULAR",param=list(normalize="center"))

#Predictions of missed ratings 
predp=predict(pAvg,rating,type="ratings")

#Extract the predicted rating for user with movie=500
as(predp[5,500],"matrix")

##Instead of using just one train set,we can split the data into k-parts
#and then evaluate the model performance out of sample.
dim(rating)
set.seed(1234)
n_folds=3
#ratings_movies <-rating[rowCounts(rating)>50,  colCounts(rating) > 20]
min(rowCounts(rating))
eval_sets=evaluationScheme(data=rating,method="cross-validation",
                           k=n_folds,given=2,goodRating=2.75)
eval_sets

#train: Training set
#known: Test set with known ratings used for prediction
#unknown: Test set with ratings used for evaluation

#Number of Items in each set
size_sets <- sapply(eval_sets@runsTrain,length)
size_sets

#Distribution of row counts
library(ggplot2)
qplot(rowCounts(getData(eval_sets, "unknown"))) + geom_histogram(binwidth = 10) + ggtitle("unknown items by the users")

eval_ib=Recommender(data=getData(eval_sets,"train"),method="UBCF")
eval_pred=predict(eval_ib,newdata=getData(eval_sets,"known"),n=5,type="ratings")

#how many movies we are recommending to each user
qplot(rowCounts(eval_pred)) + geom_histogram(binwidth = 10) + ggtitle("Distribution of movies per user")

eval_acc=calcPredictionAccuracy(x=eval_pred,data=getData(eval_sets,"unknown"),byUser=TRUE)
head(eval_acc)

#Evaluate the model as whole by setting byUser=FALSE
eval_acc=calcPredictionAccuracy(x=eval_pred,data=getData(eval_sets,"unknown"),byUser=FALSE)
eval_acc

#Optimizing a numeric parameter/model comparison
nn_values=c(5,10,20,30,40)

models<-lapply(nn_values,function(nn){
  list(name="UBCF",param=list(method="Cosine",nn=nn))
})
names(models) <- paste0("UBCF_nn_", nn_values)

#Predict missing ratings and return RMSE, MSE and MAE only if type="ratings"
result<-evaluate(eval_sets,models,type="ratings")
avg(result)

#Sampling
set.seed(1234)
index=sample(nrow(rating),nrow(rating)*0.8)
train=rating[index,]
test=rating[-index,]
#UBCF with nn=30 and method="Cosine"
UB_opt=Recommender(train,method="UBCF",param=list(normalize="center",method="Cosine",nn=30))

#Top 5 recommendations for next user
recomendation=predict(UB_opt,test,n=5)
class(recomendation)
head(as(recomendation,"list"))
