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


TEST_CASES = {
    "spleen_ct_segmentation": {
        "dataset_size": 16,
        "input_shape": (224, 224, 224),
        "check_determinism": True,
        "train_override": {
            "images": "$list(sorted(glob.glob(@dataset_dir + '/image_*.nii.gz')))",
            "labels": "$list(sorted(glob.glob(@dataset_dir + '/label_*.nii.gz')))",
            "epochs": 1,
            "train#dataset#cache_rate": 0.0,
            "validate#dataset#cache_rate": 0.0,
            "validate#handlers#-1#key_metric_filename": "test_model.pt",
        },
        "inference_override": {
            "datalist": "$list(sorted(glob.glob(@dataset_dir + '/image_*.nii.gz')))",
            "handlers#0#load_path": "$@bundle_root + '/models/test_model.pt'",
        },
    }
}
