#!/bin/bash

path174=/home/sam95/CD3/simple/output/allip3_cut_exp/allip3_kml4_strp_2017.csv
path173=/home/sam95/CD3/simple/output/allip3_cut_exp/allip3_kml3_strp_2017.csv
path172=/home/sam95/CD3/simple/output/allip3_cut_exp/allip3_kml2_strp_2017.csv
path164=/home/sam95/CD3/simple/output/allip3_cut_exp/allip3_kml4_strp_2016.csv
path163=/home/sam95/CD3/simple/output/allip3_cut_exp/allip3_kml3_strp_2016.csv
path162=/home/sam95/CD3/simple/output/allip3_cut_exp/allip3_kml2_strp_2016.csv
path154=/home/sam95/CD3/simple/output/allip3_cut_exp/allip3_kml4_strp_2015.csv
path153=/home/sam95/CD3/simple/output/allip3_cut_exp/allip3_kml3_strp_2015.csv
path152=/home/sam95/CD3/simple/output/allip3_cut_exp/allip3_kml2_strp_2015.csv

python3.6 predictor/kml4_strp.py 2017 $path174
python3.6 predictor/kml3_strp.py 2017 $path173
python3.6 predictor/kml2_strp.py 2017 $path172
python3.6 predictor/kml4_strp.py 2016 $path164
python3.6 predictor/kml3_strp.py 2016 $path163
python3.6 predictor/kml2_strp.py 2016 $path162
python3.6 predictor/kml4_strp.py 2015 $path154
python3.6 predictor/kml3_strp.py 2015 $path153
python3.6 predictor/kml2_strp.py 2015 $path152

python3.6 kml234_join.py $path172 $path173 $path174 /home/sam95/CD3/simple/output/allip3_cut_kml234_strp_joined_2017.csv  
python3.6 kml234_join.py $path162 $path163 $path164 /home/sam95/CD3/simple/output/allip3_cut_kml234_strp_joined_2016.csv  
python3.6 kml234_join.py $path152 $path153 $path154 /home/sam95/CD3/simple/output/allip3_cut_kml234_strp_joined_2015.csv  
