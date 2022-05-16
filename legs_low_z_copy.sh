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

# srun head -n 10 /share/nas2/walml/repos/pytorch-galaxy-datasets/notebooks/temp_legs_z_below_0p1_all_files.txt | xargs -I X -n 1 -P 24 echo X

# srun head -n 10 /share/nas2/walml/repos/pytorch-galaxy-datasets/notebooks/temp_legs_z_below_0p1_all_files.txt | xargs -I X -n 1 -P 24 -t -d " " rsync --relative /share/nas2/walml/galaxy_zoo/decals/dr8/jpg/./X /share/nas2/walml/galaxy_zoo/decals/dr8/low_z_jpg

# rsync --version
# xargs --version

# xargs -a /share/nas2/walml/repos/pytorch-galaxy-datasets/notebooks/temp_legs_z_below_0p1_all_files.txt -I X -n 1 -P 24 -t rsync --relative /share/nas2/walml/galaxy_zoo/decals/dr8/jpg/./X /share/nas2/walml/galaxy_zoo/decals/dr8/low_z_jpg

# xargs -a /share/nas2/walml/repos/pytorch-galaxy-datasets/notebooks/temp_legs_z_below_0p1_all_files.txt -I X -n 1 -P 24 mkdir -p /share/nas2/walml/galaxy_zoo/decals/dr8/low_z_jpg/X
# xargs -a /share/nas2/walml/repos/pytorch-galaxy-datasets/notebooks/temp_legs_z_below_0p1_all_files.txt -I X -n 1 -P 24 cp /share/nas2/walml/galaxy_zoo/decals/dr8/jpg/X /share/nas2/walml/galaxy_zoo/decals/dr8/low_z_jpg/X

# xargs -a /share/nas2/walml/repos/pytorch-galaxy-datasets/notebooks/temp_legs_z_below_0p1_all_files.txt -I X -n 1 -P 24 cp /share/nas2/walml/galaxy_zoo/decals/dr8/jpg/X /share/nas2/walml/galaxy_zoo/decals/dr8/low_z_jpg/X

# timestamp() {
#   "%T" # current time
# }

# xargs -a /share/nas2/walml/repos/pytorch-galaxy-datasets/notebooks/temp_legs_z_below_0p1_all_files.txt -n 50000 -P 1 -t -i tar -czvf /share/nas2/walml/galaxy_zoo/decals/dr8/archive_{timestamp}.tar.gz {}

# tar -cf -a --files-from /share/nas2/walml/repos/pytorch-galaxy-datasets/notebooks/temp_legs_z_below_0p1_all_files.txt /share/nas2/walml/galaxy_zoo/decals/dr8/archive_test.tar.gz

# head -n 10 /share/nas2/walml/repos/pytorch-galaxy-datasets/notebooks/temp_legs_z_below_0p1_all_files.txt | tar -cvf --auto-compress  /share/nas2/walml/galaxy_zoo/decals/dr8/archive_test.tar.gz 

# tar -cvfz --directory /share/nas2/walml/galaxy_zoo/decals/dr8/jpg /share/nas2/walml/galaxy_zoo/decals/dr8/archive_test.tar.gz 42832/42832_4476.jpg


# tar -cvfz /share/nas2/walml/galaxy_zoo/decals/dr8/archive_test.tar.gz /share/nas2/walml/galaxy_zoo/decals/dr8/jpg/42832/42832_4476.jpg

# tar -czvf /share/nas2/walml/galaxy_zoo/decals/dr8/archive_test.tar.gz jpg/42832/42832_4476.jpg jpg/42832/42832_4476.jpg

# tar -czvf /share/nas2/walml/checks_debug.tar.gz checks_old/100000_100499.parquet checks_old/6452500_6452999.parquet

# random not re-triggered
# head -n 10 /share/nas2/walml/repos/pytorch-galaxy-datasets/notebooks/temp_legs_z_below_0p1_all_files.txt | xargs -n 2 -P 1 -t -I @ tar -czvf /share/nas2/walml/galaxy_zoo/decals/dr8/archive_$RANDOM.tar.gz @

# head -n 10 /share/nas2/walml/repos/pytorch-galaxy-datasets/notebooks/temp_legs_z_below_0p1_all_files.txt | split -l 2

# ls low_z_jpg_* | xargs -P 1 -t -i tar -czvf -T {} /share/nas2/walml/galaxy_zoo/decals/dr8/{}.tar.gz

# tar -czvf /share/nas2/walml/galaxy_zoo/decals/dr8/low_z_jpg_00.tar.gz
# tar -czvf /share/nas2/walml/galaxy_zoo/decals/dr8/jpg/temp/debug.tar.gz -T temp/low_z_jpg_00 



# from jpg dir
cd /share/nas2/walml/galaxy_zoo/decals/dr8/jpg

# split main file into 200k chunks, named like temp/low_z_jpg_* where * is an int
split -d -l 200000 /share/nas2/walml/repos/pytorch-galaxy-datasets/notebooks/temp_legs_z_below_0p1_all_files.txt temp/low_z_jpg_chunk_

# get the paths to those files (temp/...), and then use xargs to start 24 processes, each using tar to read the paths in a chunkfile (-T) and save to path.tar.gz
ls temp/low_z_jpg_chunk_* | xargs -n 1 -P 24 -i tar -czvf {}_archive.tar.gz -T {} 