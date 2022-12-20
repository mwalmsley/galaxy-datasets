from galaxy_datasets.shared import gz2, gz_candels, gz_decals_5, gz_hubble, demo_rings, tidal

from galaxy_datasets.pytorch import galaxy_dataset


# TODO could refactor these into same class if needed

class GZCandels(galaxy_dataset.GalaxyDataset):
    
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):

        catalog, label_cols = gz_candels(root, train, download)

        super().__init__(catalog, label_cols, transform, target_transform)


class GZDecals5(galaxy_dataset.GalaxyDataset):
    
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):

        catalog, label_cols = gz_decals_5(root, train, download)

        super().__init__(catalog, label_cols, transform, target_transform)



class GZ2(galaxy_dataset.GalaxyDataset):
    
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):

        catalog, label_cols = gz2(root, train, download)  # no train arg

        super().__init__(catalog, label_cols, transform, target_transform)


class GZHubble(galaxy_dataset.GalaxyDataset):
    
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):

        catalog, label_cols = gz_hubble(root, train, download)

        super().__init__(catalog, label_cols, transform, target_transform)


class DemoRings(galaxy_dataset.GalaxyDataset):
    
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):

        catalog, label_cols = demo_rings(root, train, download)

        super().__init__(catalog, label_cols, transform, target_transform)

class Tidal(galaxy_dataset.GalaxyDataset):
    
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None, label_mode='coarse'):

        catalog, label_cols = tidal(root, train, download, label_mode=label_mode)

        super().__init__(catalog, label_cols, transform, target_transform)



from galaxy_datasets import check_internal_urls
if check_internal_urls.INTERNAL_URLS_EXIST:
    from galaxy_datasets.shared import gz_desi, gz_rings

    class GZDesi(galaxy_dataset.GalaxyDataset):
        
        def __init__(self, root, train=True, download=False, transform=None, target_transform=None):

            catalog, label_cols = gz_desi(root, train, download)

            super().__init__(catalog, label_cols, transform, target_transform)

    class GZRings(galaxy_dataset.GalaxyDataset):
        
        def __init__(self, root, train=True, download=False, transform=None, target_transform=None):

            catalog, label_cols = gz_rings(root, train, download)

            super().__init__(catalog, label_cols, transform, target_transform)



# temporarily deprecated
# class Legs(galaxy_dataset.GalaxyDataset):
    
#     # based on https://pytorch.org/vision/stable/generated/torchvision.datasets.STL10.html
#     def __init__(self, root=None, split='train', download=False, transform=None, target_transform=None, train=None):
#         # train=None is just an exception-raising parameter to avoid confused users using the train=False api

#         catalog, label_cols = legs(root, split, download, train)

#         # paths are not adjusted as cannot be downloaded
#         # catalog = _temp_adjust_catalog_paths(catalog)
#         # catalog = adjust_catalog_dtypes(catalog, label_cols)

#         super().__init__(catalog, label_cols, transform, target_transform)
