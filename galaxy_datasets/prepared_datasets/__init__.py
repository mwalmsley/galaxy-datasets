from galaxy_datasets.prepared_datasets import gz_candels, gz_decals, gz_rings, gz_hubble,  gz2, tidal

import os
if os.path.isfile(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'internal_urls.py')):
    from galaxy_datasets.prepared_datasets import gz_desi
