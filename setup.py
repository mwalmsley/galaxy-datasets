import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="galaxy-datasets",
    version="0.0.10",
    author="Mike Walmsley",
    author_email="walmsleymk1@gmail.com",
    description="Galaxy Zoo datasets for PyTorch/TensorFlow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mwalmsley/galaxy-datasets",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Environment :: GPU :: NVIDIA CUDA"
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.7",  # zoobot has 3.8
    extras_require={
        # these are lower than zoobot's reqs
        'pytorch': [
            'torch >= 1.10.1',
            'torchvision >= 0.11.2',
            'torchaudio >= 0.10.1',
            'pytorch-lightning >=1.6.5',  # 1.7 requires protobuf version incompatible with tensorflow/tensorboard. Otherwise works.
            'simplejpeg',
            'albumentations'
        ],
        'tensorflow': [
            'tensorflow >= 2.10.0',
            'protobuf <= 3.19'  # tensorflow incompatible above this (usually resolved by pip automatically)
        ]
        # if you need both, specify both
    }
)
