from galaxy_datasets.pytorch.our_datasets import GZCandels, GZDecals5, GZHubble, GZ2, Tidal

try:
    from galaxy_datasets.pytorch.our_datasets import GZDesi, GZRings, GZH2O
except ImportError:
    # not using logging in case config still required
    print('GZDESI/GZRings/GZCD not available from galaxy_datasets.pytorch.datasets - skipping')    
