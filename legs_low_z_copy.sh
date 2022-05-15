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

rsync --version

srun xargs -a /share/nas2/walml/repos/pytorch-galaxy-datasets/notebooks/temp_legs_z_below_0p1_all_files.txt -I X -n 1 -P 24 -t rsync --relative /share/nas2/walml/galaxy_zoo/decals/dr8/jpg/./X /share/nas2/walml/galaxy_zoo/decals/dr8/low_z_jpg

