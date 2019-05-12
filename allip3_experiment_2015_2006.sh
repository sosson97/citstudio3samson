#!/bin/bash
for ((i=2018;i>2017;i--)); do
    path4=/home/sam95/CD3/simple/output/allip3_exp/allip3_kml4_strp_$i.csv
    path3=/home/sam95/CD3/simple/output/allip3_exp/allip3_kml3_strp_$i.csv
    path2=/home/sam95/CD3/simple/output/allip3_exp/allip3_kml2_strp_$i.csv
    python3.6 predictor/kml4_strp.py $i $path4
    python3.6 predictor/kml3_strp.py $i $path3
    python3.6 predictor/kml2_strp.py $i $path2
    python3.6 kml234_join.py $path4 $path3 $path2 /home/sam95/CD3/simple/output/allip3_kml234_strp_joined_$i.csv  
done

for ((i=2014;i>2005;i--)); do
    path4=/home/sam95/CD3/simple/output/allip3_exp/allip3_kml4_strp_$i.csv
    path3=/home/sam95/CD3/simple/output/allip3_exp/allip3_kml3_strp_$i.csv
    path2=/home/sam95/CD3/simple/output/allip3_exp/allip3_kml2_strp_$i.csv
    python3.6 predictor/kml4_strp.py $i $path4
    python3.6 predictor/kml3_strp.py $i $path3
    python3.6 predictor/kml2_strp.py $i $path2
    python3.6 kml234_join.py $path4 $path3 $path2 /home/sam95/CD3/simple/output/allip3_kml234_strp_joined_$i.csv  
done
