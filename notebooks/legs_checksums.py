import os
import hashlib


if __name__ == '__main__':    

    for loc in ['/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/roots/legs/low_z_jpg_chunk_0{}_archive.tar.gz'.format(n) for n in range(8)]:
        # print hash
        with open(loc, 'rb') as f:
            md5_checksum = hashlib.md5(f.read()).hexdigest()

        print(os.path.basename(loc), md5_checksum)
