import subprocess

for i in range(1,16):
    num = i
    infile = "/home/sam95/CD3/simple/raw/players_until_career_" + str(num) + ".csv" 
    outfile = "/home/sam95/CD3/simple/raw/clusters_players_until_career_" + str(num) + ".csv"
    subprocess.call(['Rscript', 'kml.R', str(num), infile, outfile], shell=False)

