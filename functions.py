#functions.py
"""
A collection of custom functions and split functions for FeatrueExtractor class
"""

##################
#Custom Functions#
##################
def WAR2014to2016(spark, df):
    df.createOrReplaceTempView('pitcher')
    df = spark.sql('''SELECT Name, playerid, 
                                        sum(CASE WHEN Season = "2014" THEN WAR ELSE 0 END) 2014WAR,
                                        sum(CASE WHEN Season = "2015" THEN WAR ELSE 0 END) 2015WAR,
                                        sum(CASE WHEN Season = "2016" THEN WAR ELSE 0 END) 2016WAR,
                                        avg(WAR) as last3WAR, max(Age) as Age
                                        FROM pitcher
                                        GROUP BY Name, playerid''')
    return df

def join_with_2017(spark, df):
    df_2017 = spark.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("FanGraphs_Leaderboard_2017_Pitcher_Leader.csv")
    df_2017.createOrReplaceTempView('2017pitcher')
    df.createOrReplaceTempView('pitcher')
    df = spark.sql('''SELECT pitcher.2014WAR as 2014WAR, pitcher.2015WAR as 2015WAR, pitcher.2016WAR as 2016WAR,
                                                     pitcher.Age as Age, 2017pitcher.WAR as 2017WAR
                                        FROM pitcher, 2017pitcher
                                        WHERE pitcher.playerid = 2017pitcher.playerid
                                        ''')

    #from pyspark.ml.feature import VectorAssembler
    #all_features = ["2014WAR", "2015WAR", "2016WAR"]
    #assembler = VectorAssembler( inputCols = all_features, outputCol = "features")
    df = df.select("Age", "2014WAR", "2015WAR", "2016WAR", "2017WAR")
    return df


#################
#Split Functions#
#################
def random_split(spark, df):
    train_df, test_df = df.randomSplit([0.9, 0.1], seed=42)
    return train_df, test_df

