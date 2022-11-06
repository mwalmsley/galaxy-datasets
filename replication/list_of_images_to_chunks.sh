#!/bin/bash
#SBATCH --job-name=copy                    # Job name
#SBATCH --output=copy_%A.log 
#SBATCH --mem=0                                     # "reserve all the available memory on each node assigned to the job"
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=23:00:00                                # Time limit hrs:min:sec
#SBATCH --exclusive   # only one task per node
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=24
pwd; hostname; date

# this script takes a list of images (already saved from elsewhere e.g.) and groups those images into .tar.gz "chunks" for easy upload/download

# from jpg dir
cd /share/nas2/walml/galaxy_zoo/decals/dr8/jpg

# gz desi

# make sure all_files list is up-to-date (run locally)
# rsync -avz /nvme1/scratch/walml/repos/pytorch-galaxy-datasets/derived_data/gz_desi_all_files.txt walml@galahad.ast.man.ac.uk:/share/nas2/walml/repos/pytorch-galaxy-datasets/derived_data

# split main file into 50k chunks, named like temp/gz_desi_* where * is an int
split -d -l 50000 /share/nas2/walml/repos/pytorch-galaxy-datasets/derived_data/gz_desi_all_files.txt gz_desi_chunks/gz_desi_chunk_

# get the paths to those files (temp/...), and then use xargs to start 24 processes, each using tar to read the paths in a chunkfile (-T) and save to path.tar.gz
ls gz_desi_chunks/gz_desi_chunk_* | xargs -n 1 -P 24 -i tar -czvf {}_archive.tar.gz -T {} 

# then rsync them back as usual






# low z legs (deprecated for now)

# # split main file into 200k chunks, named like temp/low_z_jpg_* where * is an int
# split -d -l 200000 /share/nas2/walml/repos/pytorch-galaxy-datasets/notebooks/temp_legs_z_below_0p1_all_files.txt low_z_jpg_chunks/low_z_jpg_chunk_

# # get the paths to those files (low_z_jpg_chunks/...), and then use xargs to start 24 processes, each using tar to read the paths in a chunkfile (-T) and save to path.tar.gz
# ls low_z_jpg_chunks/low_z_jpg_chunk_* | xargs -n 1 -P 24 -i tar -czvf {}_archive.tar.gz -T {} 
