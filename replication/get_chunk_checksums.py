import os
import hashlib
import glob


if __name__ == '__main__':    

    root_to_checksum = 'gz_desi'
    locs_to_checksum = glob.glob(f'/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/roots/{root_to_checksum}/*_chunk_0*.tar.gz')

    for loc in locs_to_checksum:
        # print hash
        with open(loc, 'rb') as f:
            md5_checksum = hashlib.md5(f.read()).hexdigest()

        print(os.path.basename(loc), md5_checksum)
