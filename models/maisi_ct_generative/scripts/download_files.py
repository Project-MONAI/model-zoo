import yaml
import os
from monai.apps import download_url

# Load YAML file
with open('large_files.yml', 'r') as file:
    data = yaml.safe_load(file)

# Iterate over each file in the YAML and download it
for file in data['large_files']:
    download_url(url=file["url"], filepath=file["path"])
