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


import os
import subprocess

from monai.bundle import ConfigParser, ConfigWorkflow


def export_overrided_config(config_file, override_dict, output_path):
    parser = ConfigParser()
    parser.read_config(config_file)
    parser.update(pairs=override_dict)
    ConfigParser.export_config_file(parser.config, output_path, indent=4)


def produce_mgpu_cmd(config_file, meta_file, logging_file=None, nnodes=1, nproc_per_node=2):
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
        "--meta_file",
        meta_file,
    ]
    if logging_file is not None:
        cmd.extend(["--logging_file", logging_file])
    return cmd


def produce_custom_workflow_mgpu_cmd(
    custom_workflow, config_file, meta_file, logging_file=None, nnodes=1, nproc_per_node=2
):
    cmd = [
        "torchrun",
        "--standalone",
        f"--nnodes={nnodes}",
        f"--nproc_per_node={nproc_per_node}",
        "-m",
        "monai.bundle",
        "run_workflow",
        custom_workflow,
        "--config_file",
        config_file,
        "--meta_file",
        meta_file,
    ]
    if logging_file is not None:
        cmd.extend(["--logging_file", logging_file])
    return cmd


def export_config_and_run_mgpu_cmd(
    config_file,
    meta_file,
    override_dict,
    output_path,
    custom_workflow=None,
    logging_file=None,
    workflow_type="train",
    nnode=1,
    ngpu=2,
    check_config=False,
):
    """
    step 1: override the config file and export it
    step 2: (optional) check the exported config file
    step 3: produce multi-gpu running command
    step 4: run produced command
    """
    export_overrided_config(config_file=config_file, override_dict=override_dict, output_path=output_path)
    if check_config is True:
        engine = ConfigWorkflow(
            workflow_type=workflow_type, config_file=output_path, logging_file=logging_file, meta_file=meta_file
        )
        engine.initialize()
        check_result = engine.check_properties()
        if check_result is not None and len(check_result) > 0:
            raise ValueError(f"check properties for overrided mgpu configs failed: {check_result}")
    if custom_workflow is None:
        cmd = produce_mgpu_cmd(
            config_file=output_path, meta_file=meta_file, logging_file=logging_file, nnodes=nnode, nproc_per_node=ngpu
        )
    else:
        cmd = produce_custom_workflow_mgpu_cmd(
            custom_workflow=custom_workflow,
            config_file=output_path,
            meta_file=meta_file,
            logging_file=logging_file,
            nnodes=nnode,
            nproc_per_node=ngpu,
        )
    env = os.environ.copy()
    # ensure customized library can be loaded in subprocess
    env["PYTHONPATH"] = override_dict.get("bundle_root", ".")
    subprocess.check_call(cmd, env=env)


def check_workflow(workflow: ConfigWorkflow, check_properties: bool = False):
    workflow.initialize()
    if check_properties is True:
        check_result = workflow.check_properties()
        if check_result is not None and len(check_result) > 0:
            raise ValueError(f"check properties for workflow failed: {check_result}")
    workflow.run()
    workflow.finalize()
