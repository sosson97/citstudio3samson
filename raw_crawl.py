#raw_crwal.py

#Abstract Implementation
from abc import ABC, abstractmethod
import time
#import selemnium


#class: Crawler
#abstract class
class Crawler(ABC):
    #Internal
    def __init__(self, filename_player_info, website_type):
        #super.__init__()
        self.filename_player_info = filename_player_info
        self.website_type = website_type

    def parsing_table_(self, table_text):
        pass


    #API
    @abstractmethod
    def crawl(self):
        print("crawling starts")
        time.sleep(1.5)
        print(". . .")
        time.sleep(1.5)
        print("crawling done")
    
    @abstractmethod
    def dump_output(self, dirname_output):
        print("Dumped crwaled result in " + dirname_output)
        time.sleep(1.0)

class CrawlerDemo(Crawler):
    def crawl(self):
        super().crawl()

    def dump_output(self, dirname_output):
        super().dump_output(dirname_output)
        infile = open(self.filename_player_info, "r")
        outfile = open(dirname_output + "/crawled_data.txt", "w")
        outfile.write(infile.read())

