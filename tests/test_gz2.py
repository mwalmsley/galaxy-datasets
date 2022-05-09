# import pytest

# import os
# from pytorch_galaxy_datasets import galaxy_datamodule, galaxy_dataset
# from pytorch_galaxy_datasets.prepared_datasets import gz2

# """
# Each prespecified dataset can be created in canonical form

#     dataset = gz2.GZ2Dataset(
#         root='/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/tests/gz2_root',
#         download=True
#     )

# From it, I can create GalaxyDataset if I want to make my own datamodule

#     train_dataset = galaxy_dataset.GalaxyDataset(catalog.sample(1000), label_cols=['smooth-or-featured_smooth'])
#     val_dataset = ...
# Define datamodule using this pattern - see byol_main/datasets/galaxy_zoo_2.py

# Or I can use the default datamodule, which creates these datasets internally
# (also sets dataloader specs and associated transforms with sensible choices for all for supervised training)

#     gz2_datamodule = gz2.GZ2DataModule(catalog)
#     gz2_datamodule = gz2.GZ2DataModule(train_catalog, val_catalog, test_catalog)
# """


# def get_gz2_dataset():
#     # this dataset is always the canonical GZ2 dataset, full catalog, no split
#     dataset = gz2.GZ2Dataset(
#         root='/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/tests/gz2_root',
#         download=True
#     )
#     for image, label in dataset:
#         print(image.shape, label)
#         break




#     root = '/nvme1/scratch/walml/repos/pytorch-galaxy-datasets/tests/gz2_root'
#     catalog = pd.read_parquet(
#         '/nvme1/scratch/walml/repos/curation-datasets/gz2_downloadable_catalog.parquet')
#     catalog['file_loc'] = catalog['filename'].apply(
#         lambda x: os.path.join(root, 'images', x))

#     datamodule = GZ2DataModule(
#         root=root,
#         catalog=catalog,
#     )

#     datamodule.prepare_data()
#     datamodule.setup()

#     for images, labels in datamodule.train_dataloader():
#         print(images.shape, labels.shape)
#         break
