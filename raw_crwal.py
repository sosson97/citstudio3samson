#raw_crwal.py

#Abstract Implementation
from abc import ABC, abstractmethod

#import selemnium


#class: Crawler
#abstract class
class Crawler(ABC):
	#Internal
	def __init__(self, filename_player_info, website_type):
		super.__init__()
		self.filename_player_info = filename_player_info
		self.website_type = website_type

	def parsing_table_(table_text):
		pass


	#API
	@abstractmethod
	def crwal():
		print("crwaling starts")
	
	@abstractmethod
	def dump_output(dirname_output):
		print("Dumped crwaled result in " + dirname_output)



