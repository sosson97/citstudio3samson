#main.py

#This code contains simple demo and implementaion of abstract classes in this directory.


from raw_crawl import CrawlerDemo
from feature import FeatureExtractorDemo, OutputType
from models import CNNDemo
from trainer import TrainerDemo
from tester import TesterDemo


if __name__ == "__main__":
	#1. crwaling
	crd = CrawlerDemo("fake_web.txt", "fangraph")
	crd.crawl()
	crd.dump_output("raw")

	#2. cleaning, feature extarction
	fed = FeatureExtractorDemo("simple schema", "raw/crawled_data.txt")
	fed.raw_to_df()
	fed.dump_output("train_input")


	#3. creating model
	cnn = CNNDemo("parameters")

	#4. training
	trainer = TrainerDemo("parameters", "adam", "abs_diff", cnn)
	trainer.train("train_input")
	trainer.dump_model("model", "cnn_toy_model")

	#5. testing and get result
	tester = TesterDemo("parameters", None)
	tester.load_model("model/cnn_toy_model")
	tester.test("test_input")
	tester.dump_output("output")
	
	
	
