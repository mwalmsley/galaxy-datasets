#!/bin/bash
#SBATCH --job-name=test-pgd                     # Job name
#SBATCH --output=test-pgd_%A.log 
#SBATCH --mem=10gb                                     # "reserve all the available memory on each node assigned to the job"
#SBATCH --no-requeue                                    # Do not resubmit a failed job
#SBATCH --time=23:00:00                                # Time limit hrs:min:sec
#SBATCH --exclusive   # only one task per node
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=24
pwd; hostname; date

REPO_DIR=/share/nas2/walml/repos/pytorch-galaxy-datasets
PYTHON=/share/nas2/walml/miniconda3/envs/zoobot/bin/python

# will download every dataset and hence be really slow
# https://docs.pytest.org/en/6.2.x/tmpdir.html#base-temporary-directory as no access to /tmp
$PYTHON -m pytest --basetemp=/share/nas2/walml/pytest_tmp_dir
