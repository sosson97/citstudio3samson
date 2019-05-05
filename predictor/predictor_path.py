import os 
main_path = '/home/sam95/CD3/simple'
raw_dir = 'raw'
raw_file = '1960-2018_edited2.csv'
internal_dir = 'internal'
train_dir = 'internal/train' 
test_dir = 'internal/test'
output_dir = 'output' 

raw_path = os.path.join(main_path, raw_dir)
raw_file_path = os.path.join(raw_path, raw_file)
internal_path = os.path.join(main_path, internal_dir)
train_path = os.path.join(main_path, train_dir) 
test_path = os.path.join(main_path, test_dir)
output_path = os.path.join(main_path, output_dir)
