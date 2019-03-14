#main.py

#This code contains simple demo and implementaion of abstract classes in this directory.


from raw_crawl import CrawlerDemo
from feature import FeatureExtractorDemo, OutputType
from feature import WAR2014to2016, join_with_2017
from models import NN, XGBoostModel
from trainer import Trainer
from tester import Tester

import time
import env

test_model = "XGB"

if __name__ == "__main__":
	"""
	#1. crwaling
	print("\033[31m" + "#1. Crwaling")
	print("-------------" + "\033[0m")
	crd = CrawlerDemo("FanGraphs_Leaderboard_2014-2016_Pitcher_Leader_Board_IP_GE_30_plus_Age.csv", "fangraph")
	crd.crawl()
	crd.dump_output("raw")
	print("\033[31m" + "-------------\n" + "\033[0m")

	time.sleep(1)
	#2. cleaning, feature extarction
	print("\033[31m" + "#2. Cleaing, feature extraction")
	print("-------------" + "\033[0m")
	fed = FeatureExtractorDemo("simple schema", "raw/crawled_data.txt")
	fed.raw_to_df()
	fed.df_update(WAR2014to2016)
	fed.df_update(join_with_2017)
	fed.dump_output("train_input", "test_input")
	print("\033[31m" + "-------------\n" + "\033[0m")
	"""
	#model by model

	if(test_model == "NN"):
		"""
		#3. creating model
		print("\033[31m" + "#3. Creating Model")
		print("-------------" + "\033[0m")
		
		nn = NN("parameters")

		print("\033[31m" + "-------------\n" + "\033[0m")
		time.sleep(1)

		#4. training
		print("\033[31m" + "#4. Training Model")
		print("-------------" + "\033[0m")	
		
		trainer = Trainer("parameters", "adam", "abs_diff", nn)
		trainer.train("train_input", 1000)
		trainer.dump_model("model", "nn_model")
		print("\033[31m" + "-------------\n" + "\033[0m")
		time.sleep(1)
		"""

		#5. testing and get result
		print("\033[31m" + "#5. Testing Model")
		print("-------------" + "\033[0m")	
		
		tester = Tester("parameters", None)
		tester.load_model("model/nn_model")
		tester.test("test_input")
		#tester.dump_output("output")
		
		print("\033[31m" + "-------------\n" + "\033[0m")	

	if(test_model == "XGB"):
		#3. creating model
		#4. training
		seed_list = [42, 56, 100, 3, 15]
		f = open("xgb_result.txt", "a")
		f.write(env.test_input_name + " testing result, " + time.ctime(time.time()) + "\n")
		f.close()
		for seed in seed_list:
			print("\033[31m" + "#4. Training Model")
			print("-------------" + "\033[0m")	
			xgbm = XGBoostModel("parameters")	
			xgbm.train("parameters", 1000, seed)
			print("\033[31m" + "-------------\n" + "\033[0m")
			time.sleep(1)

			#5. testing and get result
			print("\033[31m" + "#5. Testing Model")
			print("-------------" + "\033[0m")	
			xgbm.test("paramerters", " ")	
			xgbm.dump_output("output", env.test_input_name[11:-4] + "_output_" + str(seed) + ".csv")	
			print("\033[31m" + "-------------\n" + "\033[0m")	


