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



# rsync -R -a /share/nas2/walml/galaxy_zoo/decals/dr8/jpg/397101/397101_4203.jpg /share/nas2/walml/galaxy_zoo/decals/dr8/low_z_jpg

head -n 1 /share/nas2/walml/repos/pytorch-galaxy-datasets/notebooks/temp_legs_z_below_0p1_all_files.txt | rsync --relative /share/nas2/walml/galaxy_zoo/decals/dr8/jpg/./{} /share/nas2/walml/galaxy_zoo/decals/dr8/low_z_jpg

# head -n 1 /share/nas2/walml/repos/pytorch-galaxy-datasets/notebooks/temp_legs_z_below_0p1_all_files.txt | xargs -I{} -n 1 -P 1 echo /share/nas2/walml/galaxy_zoo/decals/dr8/jpg/{} /share/nas2/walml/galaxy_zoo/decals/dr8/low_z_jpg/{}

head -n 1 /share/nas2/walml/repos/pytorch-galaxy-datasets/notebooks/temp_legs_z_below_0p1_all_files.txt | xargs -I{} -n 1 -P 1 rsync --relative /share/nas2/walml/galaxy_zoo/decals/dr8/jpg/./{} /share/nas2/walml/galaxy_zoo/decals/dr8/low_z_jpg
