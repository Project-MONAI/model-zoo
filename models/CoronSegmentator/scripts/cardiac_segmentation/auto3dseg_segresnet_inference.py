"""
Author: John Y. Ke, MC. Chen, TY. Lin, YC. Chan
Copyright © 2025 Hon Hai Precision Industry Co.,Ltd. All rights reserved.
License: Apache License 2.0
"""

import os
import time
from collections import OrderedDict

import fire
import nrrd
import numpy as np
import torch
from monai.bundle import ConfigParser
from monai.data import decollate_batch, list_data_collate
from monai.inferers import SlidingWindowInfererAdapt
from monai.transforms import (
    Compose,
    ConcatItemsd,
    CropForegroundd,
    EnsureTyped,
    Invertd,
    KeepLargestConnectedComponentd,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Resized,
    ScaleIntensityRanged,
    Spacingd,
)

# from monai.utils import MetaKeys, convert_to_dst_type
from monai.utils import convert_to_dst_type
from torch.cuda.amp import autocast


def logits2pred(logits, sigmoid=False, dim=1):
    if isinstance(logits, (list, tuple)):
        logits = logits[0]

    if sigmoid:
        pred = torch.sigmoid(logits)
        pred = pred >= 0.5
    else:
        pred = torch.softmax(logits, dim=dim)
        pred = torch.argmax(pred, dim=dim, keepdim=True).to(dtype=torch.uint8)

    return pred


@torch.no_grad()
def auto3dseg_inference(
    model_file,
    image_file,
    result_file,
    save_mode=None,
    image_file_2=None,
    image_file_3=None,
    image_file_4=None,
    **kwargs,
):
    timing_checkpoints = []  # list of (operation, time) tuples

    # Checking for model file

    if not os.path.exists(model_file):
        raise ValueError("Cannot find model file:" + str(model_file))

    checkpoint = torch.load(model_file, map_location="cpu")

    if "config" not in checkpoint:
        raise ValueError("Config not found in checkpoint (not a auto3dseg/segresnet model):" + str(model_file))

    config = checkpoint["config"]

    state_dict = checkpoint["state_dict"]
    sigmoid = config.get("sigmoid", False)

    model = ConfigParser(config["network"]).get_parsed_content()
    model.load_state_dict(state_dict, strict=True)

    # print(f"Model epoch {epoch} metric {best_metric}")

    device = torch.device("cpu") if torch.cuda.device_count() == 0 else torch.device(0)
    model = model.to(device=device, memory_format=torch.channels_last_3d)  # gpu
    model.eval()

    image_files = {}
    for index, img in enumerate([image_file, image_file_2, image_file_3, image_file_4]):
        if img is not None:
            image_files[f"image{index + 1}"] = img

    keys = list(image_files.keys())

    for img in image_files.keys():
        if image_files[img] is None or not os.path.exists(image_files[img]):
            raise ValueError(f'Incorrect image filename for {img}: "{image_files[img]}"')

    # Loading volumes
    loader = LoadImaged(keys=keys, ensure_channel_first=True, dtype=None, allow_missing_keys=True, image_only=False)
    images_loaded = loader(image_files)
    timing_checkpoints.append(("Loading volumes", time.time()))

    if len(keys) > 1:
        # Loading size of image 1
        image1_shape = images_loaded[keys[0]].shape[1:]
        # Resizing the other volumes if needed
        for _, img in enumerate(keys[1:]):
            temp_shape = images_loaded[img].shape[-len(image1_shape) :]
            if np.any(np.not_equal(image1_shape, temp_shape)):
                # print(f"Volumes do not have the same size - Resizing volume {img}")
                resizer = Resized(keys=img, spatial_size=image1_shape, mode="bilinear")
                images_loaded = resizer(images_loaded)
                timing_checkpoints.append((f"Resizing volume {img}", time.time()))

    # make input Transform chain
    main_normalize_mode = config["normalize_mode"]
    intensity_bounds = config["intensity_bounds"]
    if len(keys) == 1:  # only one input image
        ts = [
            ConcatItemsd(keys=keys, name="image", dim=0),
            EnsureTyped(keys="image", data_type="tensor", dtype=torch.float, allow_missing_keys=True),
        ]
        _add_normalization_transforms(ts, "image", main_normalize_mode, intensity_bounds)
    else:  # multiple input images
        ts = []

        extra_modalities = OrderedDict(config["extra_modalities"])
        normalize_modes = [main_normalize_mode] + list(extra_modalities.values())
        for key, normalize_mode in zip(keys, normalize_modes):
            _add_normalization_transforms(ts, key, normalize_mode, intensity_bounds)
        ts.extend(
            [
                ConcatItemsd(keys=keys, name="image", dim=0),
                EnsureTyped(keys="image", data_type="tensor", dtype=torch.float, allow_missing_keys=True),
            ]
        )

    if config.get("orientation_ras", False):
        print("Using orientation_ras")
        # we assume LPS physical coordinate system orientation
        # This code is only tested with NRRD files that use LPS space
        ts.append(Orientationd(keys="image", axcodes="RAS"))  # reorient #
    if config.get("crop_foreground", True):
        print("Using crop_foreground")
        ts.append(CropForegroundd(keys="image", source_key="image1", margin=10, allow_smaller=True))  # subcrop

    if config.get("resample_resolution", None) is not None:
        pixdim = list(config["resample_resolution"])
        print(f"Using resample with  resample_resolution {pixdim}")

        ts.append(
            Spacingd(
                keys=["image"],
                pixdim=list(pixdim),
                mode=["bilinear"],
                dtype=torch.float,
                min_pixdim=np.array(pixdim) * 0.75,
                max_pixdim=np.array(pixdim) * 1.25,
                allow_missing_keys=True,
            )
        )

    inf_transform = Compose(ts)

    # sliding_inferrer
    roi_size = config["roi_size"]
    # roi_size = [224, 224, 144]
    sliding_inferrer = SlidingWindowInfererAdapt(
        roi_size=roi_size, sw_batch_size=1, overlap=0.625, mode="gaussian", cache_roi_weight_map=False, progress=True
    )

    # process DATA
    batch_data = inf_transform([images_loaded])
    batch_data = list_data_collate([batch_data])
    data = batch_data["image"].as_subclass(torch.Tensor).to(memory_format=torch.channels_last_3d, device=device)
    timing_checkpoints.append(("Preprocessing", time.time()))

    # print("Running Inference ...")
    with autocast(enabled=True):
        logits = sliding_inferrer(inputs=data, network=model)
    timing_checkpoints.append(("Inference", time.time()))

    # print(f"Logits {logits.shape}")
    # logits -> preds
    # print("Converting logits into predictions")
    try:
        pred = logits2pred(logits, sigmoid=sigmoid)
    except RuntimeError as e:
        if not logits.is_cuda:
            raise e
        print(f"logits2pred failed on GPU pred retrying on CPU {logits.shape}")
        logits = logits.cpu()
        pred = logits2pred(logits, sigmoid=sigmoid)
    # print(f"preds {pred.shape}")
    timing_checkpoints.append(("Logits", time.time()))
    logits = None

    # pred = pred.cpu() # convert to CPU if the next step (reverse interpolation) is OOM on GPU
    # invert loading transforms (uncrop, reverse-resample, etc)
    post_transforms_list = [Invertd(keys="pred", orig_keys="image", transform=inf_transform, nearest_interp=True)]
    (
        post_transforms_list.append(KeepLargestConnectedComponentd(keys="pred", num_components=2))
        if "whole-head" in model_file
        else post_transforms_list
    )
    post_transforms = Compose(post_transforms_list)

    batch_data["pred"] = convert_to_dst_type(pred, batch_data["image"], dtype=pred.dtype, device=pred.device)[
        0
    ]  # make Meta tensor
    pred = [post_transforms(x)["pred"] for x in decollate_batch(batch_data)]
    seg = pred[0][0]

    # print(f"preds inverted {seg.shape}")
    timing_checkpoints.append(("Preds", time.time()))

    seg = seg.cpu().numpy().astype(np.uint8)
    timing_checkpoints.append(("Convert to array", time.time()))

    # save result by copying all image metadata from the input, just replacing the voxel data
    nrrd_header = nrrd.read_header(image_file)
    nrrd.write(result_file, seg, nrrd_header)
    timing_checkpoints.append(("Save", time.time()))


def _add_normalization_transforms(ts, key, normalize_mode, intensity_bounds):
    if normalize_mode == "none":
        pass
    elif normalize_mode in ["range", "ct"]:
        ts.append(
            ScaleIntensityRanged(
                keys=key, a_min=intensity_bounds[0], a_max=intensity_bounds[1], b_min=-1, b_max=1, clip=False
            )
        )
        ts.append(Lambdad(keys=key, func=lambda x: torch.sigmoid(x)))
    elif normalize_mode in ["meanstd", "mri"]:
        ts.append(NormalizeIntensityd(keys=key, nonzero=True, channel_wise=True))
    elif normalize_mode in ["meanstdtanh"]:
        ts.append(NormalizeIntensityd(keys=key, nonzero=True, channel_wise=True))
        ts.append(Lambdad(keys=key, func=lambda x: 3 * torch.tanh(x / 3)))
    elif normalize_mode in ["pet"]:
        ts.append(Lambdad(keys=key, func=lambda x: torch.sigmoid((x - x.min()) / x.std())))
    else:
        raise ValueError("Unsupported normalize_mode" + str(normalize_mode))


if __name__ == "__main__":
    fire.Fire(auto3dseg_inference)
