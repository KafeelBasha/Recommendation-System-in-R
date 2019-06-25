setwd("D:\\ML\\Recommendation System")
data=read.csv("UI.csv",na.strings=c("?"))
data$rating=as.numeric(data$rating)
library(recommenderlab)
# First data needs to be converted to sparse format
data=sparseMatrix(i=data$user,j=data$item,x=data$rating)

#realRatingMatrix object
rating=new("realRatingMatrix",data=data)
as(rating,"matrix")

#User based collaborative filtering(UBCF)
model<-Recommender(rating,method="UBCF",param=list(normalize="center",method="Cosine",nn=2))
model

pred=predict(model,rating[1,],type="ratings")
as(pred,"matrix")
