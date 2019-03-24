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

def join_2014to2016_with_2017(spark, df):
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

def join_with_2017(spark, df):
    df_2017 = spark.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema",
                "true").load("raw/2017_WAR.csv")
    df_2017.createOrReplaceTempView('w2017')
    df.createOrReplaceTempView('df')
    df = spark.sql('''SELECT df.*, w2017.WAR
                      FROM df, w2017
                      WHERE df.Player = w2017.Name
                      ''')
    return df

def rescaling(spark, df):
    columns = df.columns[2:-1] # except label, name, playerid
    from pyspark.sql.functions import col,min,max
    for column in columns:
        col_min = df.agg(min(column)).head()[0]
        col_max = df.agg(max(column)).head()[0]
        df = df.withColumn(column, (col(column)-col_min)/(col_max-col_min))
    return df
        
    
def clustering(spark, df, cluster_num=3):
    all_columns = df.columns
    columns = df.columns[2:-1]
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.clustering import KMeans

    vecAssembler = VectorAssembler(inputCols=columns, outputCol="features")
    vector_df = vecAssembler.transform(df)
    kmeans = KMeans().setK(cluster_num).setSeed(42)
    model = kmeans.fit(vector_df)
    predictions = model.transform(vector_df)
   
    all_columns.append("prediction")
    df = predictions.select(all_columns)
    df.show(10) 
    return df


#################
#Split Functions#
#################
from pyspark.sql.functions import col

def random_split(spark, df):
    train_df, test_df = df.randomSplit([0.9, 0.1], seed=42)
    return [train_df, test_df]


def test_2017_train_less2017_split(spark, df):
    test_df = df.where(col("1ySeason") == 1.0)
    train_df = df.where(col("1ySeason") < 1.0)
    return [train_df, test_df]

def cluster_split(spark, df, cluster_num):
    out = []
    for i in range(cluster_num):
        out.append(df.where(col("prediction") == i))
    return out
    
