
# from jpg dir
cd /home/walml/data/desi

# split main file into 50k chunks, named like temp/gz_desi_* where * is an int
split -d -l 50000 /home/walml/repos/galaxy-datasets/derived_data/gz_desi_all_files.txt gz_desi_chunks/gz_desi_chunk_

# get the paths to those files and then use xargs to start 24 processes, each using tar to read the paths in a chunkfile (-T) and save to path.tar.gz
ls gz_desi_chunks/gz_desi_chunk_* | xargs -n 1 -P 24 -i tar -czf {}_archive.tar.gz -T {} 
