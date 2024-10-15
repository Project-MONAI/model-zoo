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
import sys

import torch

from monai.apps import get_logger
from monai.bundle import BundleWorkflow
from monai.data import Dataset
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import UNet
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged
)
from monai.utils.enums import CommonKeys


class InferenceWorkflow(BundleWorkflow):
    """
    Test class simulates the bundle workflow defined by Python script directly.

    Typical usage:
        from monai.bundle import create_workflow
        from monai.transforms import LoadImaged
        
        workflow = create_workflow("inference.InferenceWorkflow", dataset_dir="/workspace/Data/Task09_Spleen")
        workflow.initialize()
        input_loader = LoadImaged(keys="image")
        workflow.dataflow.update(input_loader({"image": "/workspace/Data/Task09_Spleen/imagesTr/spleen_46.nii.gz"}))
        workflow.run()
        
        # update dataflow
        workflow.dataflow.clear()
        workflow.dataflow.update({"image": "/workspace/Data/Task09_Spleen/imagesTr/spleen_38.nii.gz"})
        workflow.run()

    """

    def __init__(self, dataset_dir: str = "./infer", bundle_root: str = "/workspace/Code/model-zoo/models/spleen_ct_segmentation_real_time_support"):
        super().__init__(workflow="inference", properties_path="/workspace/Code/model-zoo/models/spleen_ct_segmentation_real_time_support/scripts/properties.json")
        # set root log level to INFO and init a evaluation logger, will be used in `StatsHandler`
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        get_logger("eval_log")

        self.dataset_dir = dataset_dir
        self.bundle_root = bundle_root
        self.dataflow = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2,
        ).to(self.device)

        self.preprocessing = Compose(
            [
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityd(keys="image"),
                Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear")),
                ScaleIntensityRanged(keys="image", a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            ]
        )
    def initialize(self):
        
        self.dataset = Dataset(
            data=[self.dataflow],
            transform=self.preprocessing,
        )
        self.postprocessing = Compose(
            [
                Activationsd(keys="pred", softmax=True),
                AsDiscreted(keys="pred", argmax=True),
            ]
        )
        self.inferer = SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=1, overlap=0)

    def run(self):
        data = self.dataset[0]
        inputs = data[CommonKeys.IMAGE].unsqueeze(0).to(self.device)
        # define sliding window size and batch size for windows inference
        data[CommonKeys.PRED] = self.inferer(inputs, self.net)
        self.dataflow.update({CommonKeys.PRED: self.postprocessing(data)[CommonKeys.PRED]})

    def finalize(self):
        pass
