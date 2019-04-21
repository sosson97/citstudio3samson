### reall example, on ictus data
### Preparing the data
library(kmlShape)
set.seed(1)
data(ictusShort)
myClds <- cldsWide(ictusShort)
### Reducing the data size
reduceTraj(myClds,nbSenators=64,nbTimes=5)
### Clustering using shape
kmlShape(myClds,4)
plotMeans(myClds)
