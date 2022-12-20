from galaxy_datasets.shared.gz_candels import gz_candels
from galaxy_datasets.shared.gz_decals_5 import gz_decals_5
from galaxy_datasets.shared.gz_hubble import gz_hubble
from galaxy_datasets.shared.gz2 import gz2
from galaxy_datasets.shared.demo_rings import demo_rings
from galaxy_datasets.shared.tidal import tidal

from galaxy_datasets.check_internal_urls import INTERNAL_URLS_EXIST
if INTERNAL_URLS_EXIST:
    from galaxy_datasets.shared.gz_desi import gz_desi
    from galaxy_datasets.shared.gz_rings import gz_rings
