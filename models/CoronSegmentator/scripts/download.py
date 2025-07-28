import yaml
import hashlib
import os
from pathlib import Path
import gdown

BUNDLE_ROOT = Path(__file__).parent.parent
DOWNLOAD_CONFIG = os.path.join(BUNDLE_ROOT, "large_file.yml")

def _calculate_md5(filepath: Path, block_size: int = 65536) -> str:
    """
    Calculate md5 value of file
    """
    md5 = hashlib.md5()

    with open(filepath, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            md5.update(block)
    return md5.hexdigest()

def _download(url, destination_path):
    try:
        gdown.download(url=url, output=str(destination_path), quiet=False, fuzzy=True)
    except Exception as e:
        raise e

def download_and_verify():
    """
    Download the files accordding to DOWNLOAD_CONFIG and verify md5 value of every file
    """
    with open(DOWNLOAD_CONFIG, 'r') as file:
        downloadConfig = yaml.safe_load(file)

    for fileInfo in downloadConfig['large_files']:
        rel_path = fileInfo.get("path")
        url = fileInfo.get("url")
        expected_hash = fileInfo.get("hash_val")
        hash_type = fileInfo.get("hash_type")
        if hash_type != 'md5':
            raise ValueError(f'hash_type must be md5, but got {hash_type}')

        destination_path = BUNDLE_ROOT / rel_path
        destination_dir = Path(destination_path).parent

        if destination_path.exists():
            actual_hash = _calculate_md5(destination_path)
            if actual_hash == expected_hash: # If md5 correct, don't download again
                continue

        destination_dir.mkdir(parents=True, exist_ok=True)
        _download(url, destination_path) # download file

        final_hash = _calculate_md5(destination_path)
        if final_hash != expected_hash:
            raise RuntimeError(f"final_hash donesn't match expected_hash")
