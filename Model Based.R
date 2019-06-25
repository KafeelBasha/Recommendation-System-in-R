setwd("D:\\ML\\Recommendation System")
library(recommenderlab)

rating=read.csv("ratings_sample.csv")

#Sparse matrix
rating=sparseMatrix(i=rating$user,j=rating$movie,x=rating$rating,dimnames = list(paste("u", 1:length(unique(rating$user)), sep = ""), 
                                                                                 paste("m", 1:length(unique(rating$movie)), sep = "")))
#realRatingMatrix object
rating=new("realRatingMatrix",data=rating)
rating

#SVD
s_svd=funkSVD(rating,k=15) # k number of features (5mins)

#Prediction
pred=predict(s_svd,rating)
as(rating[1,31],"matrix") #actual
pred[1,31] #predicted

#Cross Validation
library(rrecsys)
data=defineData(as(rating,"matrix"))
class(data)
cv=evalModel(data,folds=3)
class(cv)
svd_pred=evalPred(cv,"FunkSVD",k=15)
svd_pred

#Tune number of features and evaluate model performance
k=c(10,15,20)

models<-lapply(k,function(k){
  list(evalPred(cv,"FunkSVD",k=k))
})
names(models) <- paste0("Simon Funk SVD:k=", k)
models

dim(rating)
#Non negative matrix factorization
library(NMF)
rating_matrix <-rating[rowCounts(rating)>40,rowCounts(rating)>40]
rating_matrix =as(rating_matrix ,"matrix")
rating_matrix[is.na(rating_matrix)]<-0

#Shuffle the original data to avoid overfitting
data_rnd=randomize(rating_matrix)
model_nmf=nmf(rating_matrix,rank=20,'lee') #Minimize ||V-WH||^2
Pred_mat=fitted(model_nmf) #Predicted ratings

#W=basis(model_nmf) and H=coef(model_nmf)
#Performance
summary(model_nmf,target=rating_matrix)

#Sample data for tuning with NMF
rating_tune=read.csv("sample_tune.csv",header=FALSE)
rating_tune
nmf_tune=nmf(rating_tune,rank=c(4,5,6),'lee')
plot(nmf_tune)
