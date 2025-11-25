"""
Data Downloader Utility
Project: PetBuddy
Author: Bright wang
Description:
A configurable data downloader that supports:
- Multiple URL downloads with progress display
- Automatic extraction of zip/tar archives
- Configurable output directories per URL
- Skip existing downloads to avoid duplicates
- SSL verification and error handling
"""

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

    def _is_grouped_download(self, url):
        """Check if a URL is part of a grouped download (multiple files for same dataset)"""
        # Check if this URL has the same extract directory as other URLs
        current_extract_dir = self.config['extract_to_map'].get(url, self.config['extract_to'])

        # Count how many URLs share the same extract directory
        same_dir_count = 0
        for other_url in self.config['download_urls']:
            other_extract_dir = self.config['extract_to_map'].get(other_url, self.config['extract_to'])
            if other_extract_dir == current_extract_dir:
                same_dir_count += 1

        return same_dir_count > 1

    def _get_expected_files_for_group(self, url):
        """Get expected files for a grouped download (placeholder implementation)"""
        # This is a simplified implementation - in a real scenario you might want
        # to return actual expected filenames based on the dataset
        return ["images", "annotations", "labels"]  # Common dataset file types

    def _get_extract_dir(self, temp_path):
        """Determine extraction directory based on archive contents"""
        extract_to = self.config['extract_to']
        extract_map = self.config['extract_to_map']

        if zipfile.is_zipfile(temp_path):
            with zipfile.ZipFile(temp_path) as zip_ref:
                # Check if archive contains a single directory
                namelist = zip_ref.namelist()
                if len(namelist) > 0 and all(name.startswith(namelist[0]) for name in namelist[1:]):
                    return Path(namelist[0]).parts[0]

        elif tarfile.is_tarfile(temp_path):
            with tarfile.open(temp_path) as tar_ref:
                # Check if archive contains a single directory
                members = tar_ref.getmembers()
                if len(members) > 0 and all(member.name.startswith(members[0].name) for member in members[1:]):
                    return Path(members[0].name).parts[0]

        # If no obvious single directory, use configured name or filename stem
        return extract_to or Path(temp_path).stem

    def download(self):
        """Download and extract data from all configured URLs with resume support"""
        output_dir = Path(self.config['output_dir'])
        extract_to = self.config['extract_to']

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        for url in self.config['download_urls']:
            try:
                # Determine extract directory from mapping or default
                current_extract_to = self.config['extract_to_map'].get(url, extract_to)

                # Skip only if this is the first file in a grouped download and directory exists
                if (output_dir / current_extract_to).exists() and not self.config['force_download']:
                    if not self._is_grouped_download(url):
                        print(f"Skipping download (already exists): {output_dir / current_extract_to}")
                        continue
                    else:
                        print(f"Continuing grouped download to existing directory: {output_dir / current_extract_to}")

                # Get filename from URL
                filename = url.split('/')[-1].split('?')[0]
                temp_path = output_dir / filename

                # Check if file already exists for resume support
                resume_header = {}
                if temp_path.exists() and not self.config['force_download']:
                    file_size = temp_path.stat().st_size
                    resume_header = {'Range': f'bytes={file_size}-'}
                    print(f"Resuming download from byte {file_size}: {url}")

                print(f"Downloading: {url}")
                response = requests.get(url, stream=True, verify=self.config['verify_ssl'],
                                      headers=resume_header)

                # Handle HTTP 416 (Requested Range Not Satisfiable) - file already complete
                if response.status_code == 416:
                    print(f"File already complete: {url}")
                    # Check if file exists and has expected size
                    if temp_path.exists():
                        # Try to get file size from headers if available
                        try:
                            head_response = requests.head(url, verify=self.config['verify_ssl'])
                            if head_response.status_code == 200:
                                expected_size = int(head_response.headers.get('content-length', 0))
                                actual_size = temp_path.stat().st_size
                                if actual_size >= expected_size:
                                    print(f"File {filename} is already complete ({actual_size} bytes)")
                                    continue
                        except:
                            # If HEAD request fails, assume file is complete
                            print(f"Assuming file {filename} is complete, continuing...")
                            continue
                    else:
                        # If file doesn't exist but server says range not satisfiable, reset and retry
                        print(f"Server returned 416 but file doesn't exist, retrying from beginning...")
                        if temp_path.exists():
                            os.remove(temp_path)
                        resume_header = {}
                        response = requests.get(url, stream=True, verify=self.config['verify_ssl'],
                                              headers=resume_header)
                        response.raise_for_status()

                response.raise_for_status()

                # Handle partial content (206) vs full content (200)
                if response.status_code == 206:  # Partial Content
                    total_size = int(response.headers.get('content-range').split('/')[-1])
                    mode = 'ab'  # Append mode for resume
                else:
                    total_size = int(response.headers.get('content-length', 0))
                    mode = 'wb'  # Write mode for new download

                # Download with progress bar and resume support
                with open(temp_path, mode) as f:
                    with tqdm(
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        desc=f"Downloading {filename}",
                        initial=temp_path.stat().st_size if temp_path.exists() else 0,
                        miniters=1
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))

                # Determine extraction directory - use configured mapping first
                if url in self.config['extract_to_map']:
                    actual_extract_dir = self.config['extract_to_map'][url]
                else:
                    actual_extract_dir = self._get_extract_dir(temp_path)

                extract_path = output_dir / actual_extract_dir
                extract_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
                print(f"Extracting to: {extract_path}")

                # Extract based on file type - extract to the specific extract_path
                if zipfile.is_zipfile(temp_path):
                    with zipfile.ZipFile(temp_path) as zip_ref:
                        zip_ref.extractall(extract_path)
                elif tarfile.is_tarfile(temp_path):
                    with tarfile.open(temp_path) as tar_ref:
                        tar_ref.extractall(extract_path)
                else:
                    # If not an archive, move to target directory
                    target_file = extract_path / filename
                    os.rename(temp_path, target_file)

                # Remove temporary file if it's not the final file
                if temp_path.name != filename or (extract_path / filename).exists():
                    os.remove(temp_path)

                print(f"Download and extraction completed for: {url}")

            except Exception as e:
                print(f"Failed to download {url}: {str(e)}")
                continue

            # For grouped downloads, check if all expected files exist
            if self._is_grouped_download(url):
                expected_files = self._get_expected_files_for_group(url)
                if not all((output_dir / current_extract_to / f).exists() for f in expected_files):
                    print(f"Some files missing in grouped download, continuing...")

if __name__ == '__main__':
    # Example usage - WARNING: COCO dataset is very large (~20GB)
    # Comment out COCO URLs if you have limited disk space or bandwidth
    downloader = DataDownloader()
    downloader.set_config(
        download_urls=[
            "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar",
            # WARNING: The following COCO dataset files are very large (~20GB total)
            # Only uncomment if you have sufficient disk space and fast internet connection
            "http://images.cocodataset.org/zips/train2017.zip",  # ~18GB
            "http://images.cocodataset.org/zips/val2017.zip",    # ~1GB
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"  # ~241MB
        ],
        extract_to="data",  # Default folder name
        extract_to_map={
            "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz": "oxford_pets",
            "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar":"stanford_dogs",
            # COCO dataset mapping (comment out if URLs above are commented)
             "http://images.cocodataset.org/zips/train2017.zip":"coco",
             "http://images.cocodataset.org/zips/val2017.zip":"coco",
             "http://images.cocodataset.org/annotations/annotations_trainval2017.zip":"coco"
        },
        force_download=False  # Set to True to force re-download
    )
    downloader.download()