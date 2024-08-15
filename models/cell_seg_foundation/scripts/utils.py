# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import warnings
from logging.config import fileConfig
from pathlib import Path

import numpy as np
from monai.apps import get_logger
from monai.apps.utils import DEFAULT_FMT
from monai.bundle import ConfigParser
from monai.utils import RankFilter, ensure_tuple

logger = get_logger("VistaCell")

np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
logging.getLogger("torch.nn.parallel.distributed").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*Divide by zero.*")  # intensity transform divide by zero warning

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"monai_default": {"format": DEFAULT_FMT}},
    "loggers": {
        "VistaCell": {"handlers": ["file", "console"], "level": "DEBUG", "propagate": False},
    },
    "filters": {"rank_filter": {"()": RankFilter}},
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "filename": "default.log",
            "mode": "a",  # append or overwrite
            "level": "DEBUG",
            "formatter": "monai_default",
            "filters": ["rank_filter"],
        },
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "monai_default",
            "filters": ["rank_filter"],
        },
    },
}


def parsing_bundle_config(config_file, logging_file=None, meta_file=None):
    if config_file is not None:
        _config_files = ensure_tuple(config_file)
        config_root_path = Path(_config_files[0]).parent
        for _config_file in _config_files:
            _config_file = Path(_config_file)
            if _config_file.parent != config_root_path:
                logger.warning(
                    f"Not all config files are in '{config_root_path}'. If logging_file and meta_file are"
                    f"not specified, '{config_root_path}' will be used as the default config root directory."
                )
            if not _config_file.is_file():
                raise FileNotFoundError(f"Cannot find the config file: {_config_file}.")
    else:
        config_root_path = Path("configs")

    logging_file = str(config_root_path / "logging.conf") if logging_file is None else logging_file
    if os.path.exists(logging_file):
        fileConfig(logging_file, disable_existing_loggers=False)

    parser = ConfigParser()
    parser.read_config(config_file)
    meta_file = str(config_root_path / "metadata.json") if meta_file is None else meta_file
    if isinstance(meta_file, str) and not os.path.exists(meta_file):
        logger.error(
            f"Cannot find the metadata config file: {meta_file}. "
            "Please see: https://docs.monai.io/en/stable/mb_specification.html"
        )
    else:
        parser.read_meta(f=meta_file)

    return parser
