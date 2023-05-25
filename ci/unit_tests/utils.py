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


import subprocess

from monai.bundle import ConfigParser


def run_command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    while process.poll() is None:
        line = process.stdout.readline()
        line = line.rstrip()
        if line:
            print(line, flush=True)
    print((f"Return code: {process.returncode}"))
    process.stdout.close()


def export_overrided_config(config_file, override_dict, output_path):
    parser = ConfigParser()
    parser.read_config(config_file)
    for k, v in override_dict.items():
        parser.set(v, k)
    ConfigParser.export_config_file(parser.config, output_path, indent=4)


def produce_mgpu_cmd(config_file, meta_file, logging_file, nnodes=1, nproc_per_node=2):
    cmd = [
        "torchrun",
        "--standalone",
        f"--nnodes={nnodes}",
        f"--nproc_per_node={nproc_per_node}",
        "-m",
        "monai.bundle",
        "run",
        "--config_file",
        config_file,
        "--logging_file",
        logging_file,
        "--meta_file",
        meta_file,
    ]
    return cmd


def export_config_and_run_mgpu_cmd(config_file, meta_file, logging_file, override_dict, output_path, nnode=1, ngpu=2):
    """
    step 1: export a config file with overrided components
    step 2: produce multi-gpu running command
    step 3: run produced command
    """
    export_overrided_config(config_file=config_file, override_dict=override_dict, output_path=output_path)
    cmd = produce_mgpu_cmd(
        config_file=output_path, meta_file=meta_file, logging_file=logging_file, nnodes=nnode, nproc_per_node=ngpu
    )
    run_command(cmd)
