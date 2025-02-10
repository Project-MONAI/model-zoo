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


# This list is used for our CI tests to determine whether a bundle needs to be tested with
# the `verify_data_shape` function in `verify_bundle.py`.
# If a bundle does not need to be tested, please add the bundle name into the list.
exclude_verify_shape_list = [
    "mednist_gan",
    "mednist_ddpm",
    "lung_nodule_ct_detection",
    "pathology_nuclei_segmentation_classification",
    "brats_mri_generative_diffusion",
    "brats_mri_axial_slices_generative_diffusion",
    "maisi_ct_generative",
    "cxr_image_synthesis_latent_diffusion_model",
    "brain_image_synthesis_latent_diffusion_model",
]

# This list is used for our CI tests to determine whether a bundle contains the preferred files.
# If a bundle does not have any of the preferred files, please add the bundle name into the list.
exclude_verify_preferred_files_list = ["pediatric_abdominal_ct_segmentation", "maisi_ct_generative"]

# This list is used for our CI tests to determine whether a bundle needs to be tested with
# the `verify_export_torchscript` function in `verify_bundle.py`.
# If a bundle does not support TorchScript, please add the bundle name into the list.
exclude_verify_torchscript_list = [
    "swin_unetr_btcv_segmentation",
    "renalStructures_UNEST_segmentation",
    "renalStructures_CECT_segmentation",
    "wholeBrainSeg_Large_UNEST_segmentation",
    "breast_density_classification",
    "mednist_reg",
    "brats_mri_axial_slices_generative_diffusion",
    "brats_mri_generative_diffusion",
    "vista3d",
    "maisi_ct_generative",
    "vista2d",
    "mednist_ddpm",
    "cxr_image_synthesis_latent_diffusion_model",
    "brain_image_synthesis_latent_diffusion_model",
]

# This list is used for our CI tests to determine whether a bundle needs to be tested after downloading
# the large files.
# If a bundle contains large files that are too lage (like > 10GB), please add the bundle name into the list.
# For bundles in this list, related tests will be skipped in Github Actions, but they will still be tested in blossom CI.
exclude_download_large_file_list = ["maisi_ct_generative"]

# This dict is used for our CI tests to install required dependencies that cannot be installed by `pip install` directly.
# If a bundle has this kind of dependencies, please add the bundle name (key), and the path of the install script (value)
# into the dict.
install_dependency_dict = {"maisi_ct_generative": "ci/install_scripts/install_maisi_ct_generative_dependency.sh"}

# This list is used for our CI tests to determine whether a bundle supports TensorRT export. Related
# test will be employed for bundles in the dict.
include_verify_tensorrt_dict = {
    "spleen_ct_segmentation": {},
    "endoscopic_tool_segmentation": {},
    "pathology_tumor_detection": {},
    "pathology_nuclei_classification": {},
    "pathology_nuclick_annotation": {"use_trace": True},
    "wholeBody_ct_segmentation": {"use_trace": True},
    "pancreas_ct_dints_segmentation": {
        "use_trace": True,
        "converter_kwargs": {"truncate_long_and_double": True, "torch_executed_ops": ["aten::upsample_trilinear3d"]},
    },
}

# This list is used for our CI tests to determine whether a bundle supports ONNX-TensorRT export. Related
# test will be employed for bundles in the dict.
include_verify_onnx_tensorrt_dict = {
    "brats_mri_segmentation": {},
    "endoscopic_inbody_classification": {},
    "spleen_deepedit_annotation": {},
    "spleen_ct_segmentation": {},
    "lung_nodule_ct_detection": {
        "input_shape": [1, 1, 512, 512, 192],
        "onnx_output_names": ["output_0", "output_1", "output_2", "output_3", "output_4", "output_5"],
        "network_def#use_list_output": True,
    },
}
