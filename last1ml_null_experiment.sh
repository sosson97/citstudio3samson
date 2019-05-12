#!/bin/bash
for ((i=2018;i>2012;i--)); do
    path4=/home/sam95/CD3/simple/output/last1ml_exp/last1ml_null_$i.csv
    python3.6 predictor/last1ml_null.py $i $path4
done
