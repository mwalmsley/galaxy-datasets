from galaxy_datasets.prepared_datasets import gz_candels, gz_decals, gz_desi,  gz_hubble, gz_rings, gz2, tidal
from galaxy_datasets.pytorch import galaxy_dataset

# TODO could refactor these into same class if needed



class GZCandelsDataset(galaxy_dataset.GalaxyDataset):
    
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):

        catalog, label_cols = gz_candels.setup(root, train, download)

        super().__init__(catalog, label_cols, transform, target_transform)


class GZDecalsDataset(galaxy_dataset.GalaxyDataset):
    
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):

        catalog, label_cols = gz_decals.setup(root, train, download)

        super().__init__(catalog, label_cols, transform, target_transform)

class GZDesiDataset(galaxy_dataset.GalaxyDataset):
    
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):

        catalog, label_cols = gz_desi.setup(root, train, download)

        super().__init__(catalog, label_cols, transform, target_transform)


class GZ2Dataset(galaxy_dataset.GalaxyDataset):
    
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):

        catalog, label_cols = gz2.setup(root, train, download)  # no train arg

        super().__init__(catalog, label_cols, transform, target_transform)



class GZHubbleDataset(galaxy_dataset.GalaxyDataset):
    
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):

        catalog, label_cols = gz_hubble.setup(root, train, download)

        super().__init__(catalog, label_cols, transform, target_transform)

# temporarily deprecated
# class LegsDataset(galaxy_dataset.GalaxyDataset):
    
#     # based on https://pytorch.org/vision/stable/generated/torchvision.datasets.STL10.html
#     def __init__(self, root=None, split='train', download=False, transform=None, target_transform=None, train=None):
#         # train=None is just an exception-raising parameter to avoid confused users using the train=False api

#         catalog, label_cols = legs.setup(root, split, download, train)

#         # paths are not adjusted as cannot be downloaded
#         # catalog = _temp_adjust_catalog_paths(catalog)
#         # catalog = adjust_catalog_dtypes(catalog, label_cols)

#         super().__init__(catalog, label_cols, transform, target_transform)


class GZRingsDataset(galaxy_dataset.GalaxyDataset):
    
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):

        catalog, label_cols = gz_rings.setup(root, train, download)

        super().__init__(catalog, label_cols, transform, target_transform)


class TidalDataset(galaxy_dataset.GalaxyDataset):
    
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None, label_mode='coarse'):

        catalog, label_cols = tidal.setup(root, train, download, label_mode=label_mode)

        super().__init__(catalog, label_cols, transform, target_transform)
