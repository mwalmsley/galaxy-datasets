from galaxy_datasets.shared.gz_candels import gz_candels
from galaxy_datasets.shared.gz_decals_5 import gz_decals_5
from galaxy_datasets.shared.gz_hubble import gz_hubble
from galaxy_datasets.shared.gz2 import gz2
from galaxy_datasets.shared.tidal import tidal

import os
if os.path.isfile(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'internal_urls.py')):
    from galaxy_datasets.shared.gz_desi import gz_desi
    from galaxy_datasets.shared.gz_rings import gz_rings
