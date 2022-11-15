from galaxy_datasets.pytorch.datasets import GZCandels, GZDecals5, GZHubble, GZ2, Tidal

try:
    from galaxy_datasets.pytorch.datasets import GZDesi, GZRings
except ImportError:
    # not using logging in case config still required
    print('GZDESI and GZRings not available from galaxy_datasets.pytorch.datasets - skipping')    
