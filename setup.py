import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch_galaxy_datasets",
    version="0.0.1",
    author="Mike Walmsley",
    author_email="walmsleymk1@gmail.com",
    description="Galaxy Zoo datasets for PyTorch/Lightning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mwalmsley/pytorch-galaxy-datasets",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Environment :: GPU :: NVIDIA CUDA"
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.7"
)
