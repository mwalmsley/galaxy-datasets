
import logging

import galaxy_datamodule
import galaxy_dataset


class DECALSDR8DataModule(galaxy_datamodule.GalaxyDataModule):
    """
    Currently identical to GalaxyDataModule - see that description
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Currently identical to GalaxyDataModule - see that description
        """
        super().__init__(*args, **kwargs)

    def prepare_data(self):
        logging.warning(
            'DR8 is too large to download dynamically - you had better already have it prepared!')
        # TODO include some basic checks?


class DECALSDR8Dataset(galaxy_dataset.GalaxyDataset):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


"""
I played with an in-memory version for speed, but it wasn't particularly faster - cpu is the limiting factor at Manchester
"""
# class DECALSDR8DatasetMemory(DECALSDR8Dataset):
#     # compressed data will be loaded into memory
#     # use cpu/simplejpeg to decode as needed, can't store decoded all in memory

#     def __init__(self, catalog: pd.DataFrame, schema, transform=None, target_transform=None):
#         super().__init__(catalog=catalog, schema=schema, transform=transform, target_transform=target_transform)

#         logging.info('Loading encoded jpegs into memory: {}'.format(len(self.catalog)))

#         self.catalog = self.catalog.sort_values('file_loc')  # for sequential -> faster hddreading. Shuffle later.
#         logging.warning('In-Memory loading will shuffle for faster reading - outputs will not align with earlier/later reads')

#         # assume I/O limited so use pool
#         pool = Pool(processes=int(os.cpu_count()/2))
#         self.encoded_galaxies = pool.map(load_encoded_jpeg, self.catalog['file_loc'])  # list not generator
#         logging.info('Loading complete: {}'.format(len(self.encoded_galaxies)))

#         shuffle_indices = list(range(len(self.catalog)))
#         random.shuffle(shuffle_indices)

#         self.catalog = self.catalog.iloc[shuffle_indices].reset_index()
#         self.encoded_galaxies = list({self.encoded_galaxies[idx] for idx in shuffle_indices})
#         logging.info('Shuffling complete')

#     def __getitem__(self, idx):
#         galaxy = self.catalog.iloc[idx]
#         label = get_galaxy_label(galaxy, self.schema)
#         image = decode_jpeg(self.encoded_galaxies[idx])
#         # Read image as torch array for consistency

#         # logging.info(image.shape)
#         if self.transform:
#             # image = np.asarray(image).transpose(2,0,1)  # not needed simplejpeg gives np array HWC
#             # logging.info(type(image))
#             image = self.transform(image=image)['image']  # assumed to output torch
#             # image = self.transform(image)
#         else:
#             image = torch.from_numpy(image)

#         if self.target_transform:
#             label = self.target_transform(label)

#         return image, label
