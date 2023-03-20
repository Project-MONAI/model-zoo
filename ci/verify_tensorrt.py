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

from .bundle_custom_data import include_verify_tensorrt_list


def verify_tensorrt():
    """
    This function is used to verify if the checkpoint is able to export into TensorRT module.
    This function will be updated after the following PR is merged:
    https://github.com/Project-MONAI/MONAI/pull/5986/files

    """
    for bundle in include_verify_tensorrt_list:
        print(f"export bundle {bundle} weights into TensorRT module successfully.")


if __name__ == "__main__":
    verify_tensorrt()
