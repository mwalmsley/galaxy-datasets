from galaxy_datasets.pytorch.datasets import GZCandels, GZDecals5, GZRings, GZHubble, GZ2, Tidal

try:
    from galaxy_datasets.pytorch.datasets import GZDesi
except ImportError:
    # not using logging in case config still required
    print('GZ DESI not available from galaxy_datasets.pytorch.datasets - skipping')    
