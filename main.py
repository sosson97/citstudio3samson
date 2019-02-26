#main.py

#This code contains simple demo and implementaion of abstract classes in this directory.


from raw_crawl import CrawlerDemo
from feature import FeatureExtractorDemo, OutputType
from models import CNNDemo
from trainer import TrainerDemo
#from tester import TesterDemo


if __name__ == "__main__":
	crd = CrawlerDemo("fake_web.txt", "fangraph")
	crd.crawl()
	crd.dump_output("raw")
	
	
#model = CNNDemo(None)
#model.forward()
	
	
