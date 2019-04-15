###1. Load csv file as a matrix in R
dat <- read.csv(file.choose())
mat <- as.matrix(dat)
 
###2. Extract WAR by time data from matrix
###Note: mat[, 3:3+(n-1)] should be used when using career_until_n_years.csv
wmat <- mat[, 3:6]
class(wmat) <- "numeric"
 
###3. Create clusterLondData from matrix
ld <- clusterLongData(wmat)
 
###4. Execute kml algorithm then get clustering result
###Note: 2nd argument to kml function determines how many clusters you use.
kml(ld, 10:15)
out <- getClusters(ld,15)
 
###5. Dump the result
write.csv(out, file=fileName)
