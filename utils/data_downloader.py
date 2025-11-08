import os
import requests
import zipfile
import tarfile
import time
from pathlib import Path
from typing import List, Union
from tqdm import tqdm

class DataDownloader:
    def __init__(self):
        self.config = {
            'download_urls': [],    # List of URLs to download
            'output_dir': '../data',  # Default output directory (project root/data)
            'extract_to': None,    # Default directory name after extraction
            'extract_to_map': {},  # URL to extract directory mapping
            'verify_ssl': True,    # SSL verification
            'force_download': False  # Force re-download even if exists
        }

    def set_config(self, **kwargs):
        """Update download configuration"""
        self.config.update(kwargs)
        if not self.config['download_urls']:
            raise ValueError("At least one download URL must be provided")

        # Set extraction directory names if not provided
        if not self.config['extract_to'] and not self.config['extract_to_map'] and self.config['download_urls']:
            url = self.config['download_urls'][0]
            filename = url.split('/')[-1].split('?')[0]
            self.config['extract_to'] = Path(filename).stem

    def download(self):
        """Download and extract data from all configured URLs"""
        output_dir = Path(self.config['output_dir'])
        extract_to = self.config['extract_to']

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        for url in self.config['download_urls']:
            try:
                # Determine extract directory
                current_extract_to = self.config['extract_to_map'].get(url, extract_to)
                extract_path = output_dir / current_extract_to

                # Skip if already exists and not forced
                if extract_path.exists() and not self.config['force_download']:
                    print(f"Skipping download (already exists): {extract_path}")
                    continue

                print(f"Downloading: {url}")
                response = requests.get(url, stream=True, verify=self.config['verify_ssl'])
                response.raise_for_status()

                # Get file size from headers
                total_size = int(response.headers.get('content-length', 0))

                # Temporary file path
                temp_path = output_dir / f"temp_download_{self.config['download_urls'].index(url)}"

                # Download with progress bar
                with open(temp_path, 'wb') as f:
                    with tqdm(
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        desc=f"Downloading {Path(url).name}",
                        miniters=1
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))

                print(f"Extracting to: {extract_path}")

                if zipfile.is_zipfile(temp_path):
                    with zipfile.ZipFile(temp_path) as zip_ref:
                        zip_ref.extractall(extract_path)
                elif tarfile.is_tarfile(temp_path):
                    with tarfile.open(temp_path) as tar_ref:
                        tar_ref.extractall(extract_path)
                else:
                    # If not an archive, move directly
                    extract_path.mkdir(exist_ok=True)
                    os.rename(temp_path, extract_path / Path(url).name)

                # Remove temporary file
                os.remove(temp_path)
                print(f"Download and extraction completed for: {url}")

            except Exception as e:
                print(f"Failed to download {url}: {str(e)}")
                continue

if __name__ == '__main__':
    # Example usage
    downloader = DataDownloader()
    downloader.set_config(
        download_urls=[
            "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
        ],
        extract_to="data",  # Default folder name
        extract_to_map={
            "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz": "oxford_pets"
        },
        force_download=False  # Set to True to force re-download
    )
    downloader.download()