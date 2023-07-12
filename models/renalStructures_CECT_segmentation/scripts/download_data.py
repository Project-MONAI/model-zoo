import json
import logging
import os
import sys
import threading
import time
import zipfile
from urllib.parse import urlencode
from urllib.request import urlopen, urlretrieve


# https://stackoverflow.com/a/39504463
class Spinner:
    busy = False
    delay = 0.1

    @staticmethod
    def spinning_cursor():
        while 1:
            for cursor in "|/-\\":
                yield cursor

    def __init__(self, delay=None):
        self.spinner_generator = self.spinning_cursor()
        if delay and float(delay):
            self.delay = delay

    def spinner_task(self):
        while self.busy:
            sys.stdout.write(next(self.spinner_generator))
            sys.stdout.flush()
            time.sleep(self.delay)
            sys.stdout.write("\b")
            sys.stdout.flush()

    def __enter__(self):
        self.busy = True
        threading.Thread(target=self.spinner_task).start()

    def __exit__(self, exception, value, tb):
        self.busy = False
        time.sleep(self.delay)
        if exception is not None:
            return False


def download_cect_data(bundle_root: str = "."):
    base_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?"
    public_key = "https://disk.yandex.ru/d/pWEKt6D3qi3-aw"

    url = base_url + urlencode(dict(public_key=public_key))
    response = urlopen(url)
    filepath = "AVUCTK_cases.zip"
    if not os.path.exists(os.path.join(bundle_root, filepath)):
        with urlopen(url) as response:
            code = response.getcode()
            if code == 200:
                download_url = json.loads(response.read())["href"]
                print("Downloading file...")
                with Spinner():
                    urlretrieve(download_url, filepath)
            else:
                raise RuntimeError(
                    f"Download of file from {url} to {filepath} failed due to network issue or denied permission."
                )
    else:
        logging.info("zipfile exists, skipping download")
    with zipfile.ZipFile(os.path.join(bundle_root, filepath), "r") as zip_ref:
        zip_ref.extractall()
