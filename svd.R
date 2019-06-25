#Find SVD of A. ie A= U*S*V^T
A=matrix(c(3,2,2,2,3,-2),byrow=TRUE,nrow=2) #dim(n,p)

#Step 1: Find AA^T and its eigen value
options(scipen=999)
aat=A%*%t(A)
aat

eigen(aat)$values

#Step 2: Find U ie eigen vectors of AA^T dim(n,n)
U=eigen(aat)$vectors
U

#Step 3: Find A^TA and its eigen value
ata=t(A)%*%A
ata

eata=eigen(ata)$values
eata

#Step 4: Find V^T ie eigen vectors of A^TA dim(pxp)
vt=eigen(ata)$vectors #orthonormal
vt

crossprod(vt[,1],vt[,2]) #orthogonal

#Step 5: S is the square root of the eigenvalues from AAT or ATA. dim(nxp)
s=eigen(aat)$values
s=sqrt(s)
S=diag(s,nrow=2,ncol=3)
S

svd(A)