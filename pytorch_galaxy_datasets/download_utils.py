import os

from urllib.error import URLError
from torchvision.datasets.utils import download_and_extract_archive, check_integrity


class DatasetDownloader():
    # responsible for downloading a prespecified set of images/catalogs to a directory
    # supports GalaxyDataset via composition

    def __init__(self, root, resources, images_to_spotcheck=None):
        self.root = root
        self.image_dir = os.path.join(self.root, 'images')
        self.resources = resources
        self.images_to_spotcheck = images_to_spotcheck

    def download(self) -> None:
        """Download the data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.root, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = os.path.basename(url)
            try:
                print(f"Downloading {url}")
                download_and_extract_archive(
                    url, download_root=self.root, filename=filename, md5=md5)
            except URLError as error:
                print(f"Failed to download (trying next):\n{error}")
                continue


    def _check_exists(self) -> bool:
        # takes a few seconds for the image .zip
        resources_downloaded = all([
            check_integrity(
                os.path.join(self.root, os.path.basename(res)),
                md5
            )
            for res, md5 in self.resources])

        images_unpacked = all([
            os.path.isfile(os.path.join(self.image_dir, image_loc) for image_loc in self.images_to_spotcheck),
        ])

        return resources_downloaded & images_unpacked



