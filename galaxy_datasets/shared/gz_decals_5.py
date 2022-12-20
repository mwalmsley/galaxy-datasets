import os

import pandas as pd

from galaxy_datasets.shared import download_utils, label_metadata

# DR8 will be basically the same

def gz_decals_5(root, train, download):
    resources = [
        ('https://dl.dropboxusercontent.com/s/1tuehajonhgv8a2/decals_dr5_ortho_train_catalog.parquet', 'a0cd74edc073fdff068370f6eefeb802'),  # train catalog
        ('https://dl.dropboxusercontent.com/s/3vo6hjlnbqgzuxz/decals_dr5_ortho_test_catalog.parquet', '55820e3712b22e587f6971e4b6c73dfe'),  # test catalog
        ('https://dl.dropboxusercontent.com/s/bs6jp0mkgiekhww/decals_dr5_images.tar.gz', '1347de4c8df4ec579d5a58241c1f280b')  # the images
    ]
    images_to_spotcheck = ['J073/J073013.60+242930.0.jpg']

    downloader = download_utils.DatasetDownloader(root, resources, images_to_spotcheck)
    if download is True:
        downloader.download()

    # useful_columns = label_cols + ['subfolder', 'filename']
    if train:
        train_catalog_loc = os.path.join(root, 'decals_dr5_ortho_train_catalog.parquet')
        catalog = pd.read_parquet(train_catalog_loc)
    else:
        test_catalog_loc = os.path.join(root, 'decals_dr5_ortho_test_catalog.parquet')
        catalog = pd.read_parquet(test_catalog_loc)

    # removed 'root' from here as downloader.image_dir already includes root
    catalog['file_loc'] = catalog.apply(lambda x: os.path.join(downloader.image_dir, x['subfolder'], x['filename']), axis=1)

    label_cols = label_metadata.decals_dr5_ortho_label_cols

    return catalog, label_cols

