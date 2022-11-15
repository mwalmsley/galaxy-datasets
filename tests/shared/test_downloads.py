import pytest

import os
import numpy as np
import pandas as pd
from galaxy_datasets.shared import gz_candels, gz_hubble, gz_decals_5, gz2, gz_desi, gz_rings, tidal

# https://docs.pytest.org/en/6.2.x/fixture.html#using-marks-with-parametrized-fixtures
# in order of dataset size
# pytest.param(gz_desi.setup, marks=pytest.mark.skip)
@pytest.fixture(params=[tidal, gz_rings, gz_candels, gz_hubble, gz2, gz_decals_5, gz_desi])
# @pytest.fixture(params=[gz_desi.setup])
def setup_method(request):
    return request.param  # param is the func, param() calls the func (which breaks here as no args)

def test_download_dataset(setup_method, tmp_path):
    catalog, label_cols = setup_method(
        train=True,
        root=tmp_path,
        download=True
    )

    assert isinstance(catalog, pd.DataFrame)
    assert 'file_loc' in catalog.columns.values
    assert len(catalog) > 0
    
    assert isinstance(label_cols, list)
    assert len(label_cols) > 0
    for col in label_cols:
        assert col in catalog.columns.values

    for loc in catalog.sample(10, random_state=42)['file_loc']:
        assert os.path.isfile(loc)
