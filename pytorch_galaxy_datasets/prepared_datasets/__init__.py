from pytorch_galaxy_datasets.prepared_datasets.gz2 import GZ2Dataset, gz2_setup
from pytorch_galaxy_datasets.prepared_datasets.dr5 import DecalsDR5Dataset, decals_dr5_setup

import os
# i.e. if internal_urls.py is available in the folder of this script
if os.path.isfile(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'internal_urls.py')):  
    # import the internal datasets
    from pytorch_galaxy_datasets.prepared_datasets.legs import LegsDataset, legs_setup
    from pytorch_galaxy_datasets.prepared_datasets.rings import RingsDataset, rings_setup

# allows for imports like
# from pytorch_galaxy_datasets.prepared_datasets import GZ2Dataset