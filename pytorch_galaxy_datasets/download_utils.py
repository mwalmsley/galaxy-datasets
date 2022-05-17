import os
import logging

from urllib.error import URLError
from torchvision.datasets.utils import download_and_extract_archive, download_url, check_integrity


class DatasetDownloader():
    # responsible for downloading a prespecified set of images/catalogs to a directory
    # supports GalaxyDataset via composition

    def __init__(self, root, resources, images_to_spotcheck=None, image_dirname='images'):
        self.root = root
        self.image_dir = os.path.join(self.root, image_dirname)
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
                logging.info(f"Downloading {url}")
                if url.endswith('.tar.gz') or url.endswith('.zip'):
                    download_and_extract_archive(
                        url, download_root=self.root, filename=filename, md5=md5)
                else:  # don't try to extract archive, just download
                    download_url(url, root=self.root, filename=filename, md5=md5)
            except URLError as error:
                logging.info(f"Failed to download (trying next):\n{error}")
                continue


    def _check_exists(self) -> bool:
        # takes a few seconds for the image .zip
        logging.info('Checking integrity of resources')
        resources_downloaded = all([
            check_integrity(
                os.path.join(self.root, os.path.basename(res)),
                md5
            )
            for res, md5 in self.resources])
        logging.info('Resources downloaded: {}'.format(resources_downloaded))

        images_unpacked = all([
            os.path.isfile(os.path.join(self.image_dir, image_loc)) for image_loc in self.images_to_spotcheck
        ])

        logging.info('Images unpacked: {} ({}, {})'.format(images_unpacked, self.image_dir, self.images_to_spotcheck))

        return resources_downloaded & images_unpacked



