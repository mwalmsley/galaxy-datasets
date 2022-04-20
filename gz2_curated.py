import os
import logging

import pandas as pd

from galaxy_zoo_2 import GZ2DataModule, GZ2Dataset


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    data_dir = '/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/tests/gz2_root'
    catalog = pd.read_parquet('/nvme1/scratch/walml/repos/curation-datasets/gz2_downloadable_catalog.parquet')
    catalog['file_loc'] = catalog['filename'].apply(lambda x: os.path.join(data_dir, 'images', x))

    catalog = catalog.query('label >= 0')  # -1 indicates could not be divided into a class, regardless of confidence/curation



    # datamodule = GZ2DataModule(
    #     dataset_class=GZ2Dataset,
    #     label_cols=['label'],
    #     data_dir=data_dir,
    #     catalog=catalog,
    # )

    # datamodule.prepare_data()
    # datamodule.setup()

    # for images, labels in datamodule.train_dataloader():
    #     # print(images.shape, labels.shape, labels[:5])
    #     # break
    #     pass  # to iterate through them all
