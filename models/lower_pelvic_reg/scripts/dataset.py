import os
from ctypes import Union
from typing import Optional, Sequence, Tuple

import numpy as np

import torch
from monai.transforms import (
    LoadImaged,
    AddChanneld,
    Spacingd,
    NormalizeIntensityd,
    ScaleIntensityd,
    ToTensord,
    RandAffined,
    CenterSpatialCropd,
    SpatialPadd,
    Compose
)

from torch.nn import functional as F
from torch.utils.data import Dataset


def get_institution_patient_dict(dataset_dir, train):
    """
    divide images by institution, take 3/4 for training and 1/4 for inference
    :param dataset_dir: str
    :param train: bool, specify training or not
    :return: dict
    """
    if os.path.exists(f'{dataset_dir}/institution.txt'):
        # divide images by institution
        institution_patient_dict = {i: [] for i in range(1, 8)}
        with open(f'{dataset_dir}/institution.txt') as f:
            patient_ins_list = f.readlines()
        for patient_ins in patient_ins_list:
            patient, ins = patient_ins[:-1].split(" ")
            institution_patient_dict[int(ins)].append(patient)
    else:
        # if no institution info, consider all patients as from the same institution
        patient_list = [p.replace("_mask.nii", "") for p in os.listdir(f'{dataset_dir}/data') if "mask" in p]
        institution_patient_dict = {1: patient_list}

    # take 3/4 for training and 1/4 for inference
    for k, v in institution_patient_dict.items():
        if train:
            institution_patient_dict[k] = v[:-len(v)//4]
        else:
            institution_patient_dict[k] = v[-len(v)//4:]
    return institution_patient_dict


def sample_pair(idx, img_list_len):
    """
    given query index, sample a support index
    :param idx: int, query index
    :param img_list_len: int, number of training images
    :return: int
    """
    out = idx
    while out == idx:
        out = np.random.randint(img_list_len)
    return out


class RegDataset(Dataset):

    def __init__(self,
                 train: bool,
                 dataset_dir: str,
                 pixdim: Sequence[float],
                 spatial_size,#: Optional[Union[Sequence[int], int]] = None,
                 rotate_range,#: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
                 translate_range,#: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
                 scale_range,#: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None
                 ) -> None:
        """
        Args:
            train: bool, specify if training or not
            dataset_dir: directory storing the t2w images and segmentations.
            pixdim: output voxel spacing. if providing a single number, will use it for the first dimension.
                items of the pixdim sequence map to the spatial dimensions of input image, if length
                of pixdim sequence is longer than image spatial dimensions, will ignore the longer part,
                if shorter, will pad with `1.0`.
                if the components of the `pixdim` are non-positive values, the transform will use the
                corresponding components of the original pixdim, which is computed from the `affine`
                matrix of input image.
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if some components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            rotate_range: angle range in radians. If element `i` is a pair of (min, max) values, then
                `uniform[-rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the `i`th spatial dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used.
                This can be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be
                in range `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]`
                for dim0 and nothing for the remaining dimensions.
            translate_range: translate range with format matching `rotate_range`, it defines the range to randomly
                select pixel/voxel to translate for every spatial dims.
            scale_range: scaling range with format matching `rotate_range`. it defines the range to randomly select
                the scale factor to translate for every spatial dims. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1.0).
        """
        super(RegDataset, self).__init__()
        self.train = train

        # divide images by institution, take 3/4 for training and 1/4 for inference
        institution_patient_dict = get_institution_patient_dict(
            dataset_dir=dataset_dir,
            train=train,
        )
        self.img_list = []
        for ins, patient_list in institution_patient_dict.items():
            self.img_list.extend([(p, ins) for p in patient_list])

        # sample inference pairs if not training
        if not train:
            self.val_pair = []
            # for each query image
            for moving_p, moving_ins in self.img_list:
                # for each institution
                for fixed_ins, patient_list in institution_patient_dict.items():
                    while True:
                        fixed_p = patient_list[np.random.randint(0, len(patient_list))]
                        if fixed_p != moving_p:
                            break
                    self.val_pair.append([(moving_p, moving_ins), (fixed_p, fixed_ins)])

        # initialise transformation
        self.image_loader = LoadImages(
            dataset_dir=dataset_dir,
            augmentation=train,
            spatial_size=spatial_size,
            pixdim=pixdim,
            rotate_range=rotate_range,
            translate_range=translate_range,
            scale_range=scale_range
        )

    def __len__(self):
        return len(self.img_list) if self.train else len(self.val_pair)

    def __getitem__(self, idx):
        if self.train:
            moving = idx
            fixed = sample_pair(idx, len(self.img_list))
            moving, fixed = self.img_list[moving], self.img_list[fixed]
        else:
            moving, fixed = self.val_pair[idx]

        moving = self.image_loader(moving)
        fixed = self.image_loader(fixed)

        return moving, fixed


def get_transform(augmentation, spatial_size, pixdim, rotate_range, translate_range, scale_range):
    """
    Args:
        pixdim: output voxel spacing. if providing a single number, will use it for the first dimension.
            items of the pixdim sequence map to the spatial dimensions of input image, if length
            of pixdim sequence is longer than image spatial dimensions, will ignore the longer part,
            if shorter, will pad with `1.0`.
            if the components of the `pixdim` are non-positive values, the transform will use the
            corresponding components of the original pixdim, which is computed from the `affine`
            matrix of input image.
        spatial_size: output image spatial size.
            if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
            the transform will use the spatial size of `img`.
            if some components of the `spatial_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        augmentation: bool, specifying apply augmentation or not.
        rotate_range: angle range in radians. If element `i` is a pair of (min, max) values, then
            `uniform[-rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
            for the `i`th spatial dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used.
            This can be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be
            in range `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]`
            for dim0 and nothing for the remaining dimensions.
        translate_range: translate range with format matching `rotate_range`, it defines the range to randomly
            select pixel/voxel to translate for every spatial dims.
        scale_range: scaling range with format matching `rotate_range`. it defines the range to randomly select
            the scale factor to translate for every spatial dims. A value of 1.0 is added to the result.
            This allows 0 to correspond to no change (i.e., a scaling of 1.0).
    """
    pre_augmentation = [
        LoadImaged(keys=["t2w", "seg"]),
        AddChanneld(keys=["t2w", "seg"]),
        Spacingd(
            keys=["t2w", "seg"],
            pixdim=pixdim,
            mode=("bilinear", "nearest"),
        ),
    ]

    post_augmentation = [
        NormalizeIntensityd(keys=["t2w"]),
        ScaleIntensityd(keys=["t2w"]),
        ToTensord(keys=["t2w", "seg"])
    ]

    if augmentation:
        middle_transform = [
            RandAffined(
                keys=["t2w", "seg"],
                spatial_size=spatial_size,
                prob=1.0,
                rotate_range=(rotate_range, rotate_range, rotate_range),
                shear_range=None,
                translate_range=translate_range,
                scale_range=scale_range,
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
                as_tensor_output=False,
                device=torch.device('cpu'),
                allow_missing_keys=False
            )
        ]
    else:
        middle_transform = [
            CenterSpatialCropd(keys=["t2w", "seg"], roi_size=spatial_size),
            SpatialPadd(
                keys=["t2w", "seg"],
                spatial_size=spatial_size,
                method='symmetric',
                mode='constant',
                allow_missing_keys=False
            )
        ]

    return Compose(pre_augmentation + middle_transform + post_augmentation)


class LoadImages:
    """
    Transform customised for registration
    given a dictionary specifying moving and fixed image names, output a dictionary of dictionaries each containing the
    following keys:
    - "t2w": tensor of shape (1, ...) the t2w image
    - "seg": tensor of shape (1, ...) the segmentation of the corresponding image
    - "image_name": the name of the image
    """

    def __init__(self,
                 dataset_dir,#: str,
                 pixdim,#: Union[Sequence[float], float],
                 spatial_size,#: Optional[Union[Sequence[int], int]] = None,
                 augmentation,#: bool = False,
                 rotate_range,#: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
                 translate_range,#: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
                 scale_range,#: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
                 ) -> None:
        """
        Args:
            dataset_dir: directory storing the t2w images and segmentations.
            pixdim: output voxel spacing. if providing a single number, will use it for the first dimension.
                items of the pixdim sequence map to the spatial dimensions of input image, if length
                of pixdim sequence is longer than image spatial dimensions, will ignore the longer part,
                if shorter, will pad with `1.0`.
                if the components of the `pixdim` are non-positive values, the transform will use the
                corresponding components of the original pixdim, which is computed from the `affine`
                matrix of input image.
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if some components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            augmentation: bool, specifying apply augmentation or not.
            rotate_range: angle range in radians. If element `i` is a pair of (min, max) values, then
                `uniform[-rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the `i`th spatial dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used.
                This can be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be
                in range `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]`
                for dim0 and nothing for the remaining dimensions.
            translate_range: translate range with format matching `rotate_range`, it defines the range to randomly
                select pixel/voxel to translate for every spatial dims.
            scale_range: scaling range with format matching `rotate_range`. it defines the range to randomly select
                the scale factor to translate for every spatial dims. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1.0).
        """
        self.dataset_dir = dataset_dir
        self.spatial_size = spatial_size
        self.transform = get_transform(
            pixdim=pixdim,
            spatial_size=spatial_size,
            augmentation=augmentation,
            rotate_range=rotate_range,
            translate_range=translate_range,
            scale_range=scale_range
        )

    def __call__(self, patient_ins_tuple):
        patient_name, ins = patient_ins_tuple
        print({
            "t2w": f"{self.dataset_dir}/data/{patient_name}_img.nii",
            "seg": f"{self.dataset_dir}/data/{patient_name}_mask.nii",
            "name": patient_name,
        })
        x = self.transform({
            "t2w": f"{self.dataset_dir}/data/{patient_name}_img.nii",
            "seg": f"{self.dataset_dir}/data/{patient_name}_mask.nii",
            "name": patient_name,
        })
        # crop and resize foregrounds depth-wise
        target_slice = torch.sum(x["seg"], dim=(0, 1, 2)) != 0
        for k in ["t2w", "seg"]:
            x[k] = x[k][..., target_slice]
            x[k] = F.interpolate(
                x[k].unsqueeze(0).to(torch.float),
                size=self.spatial_size,
                mode="trilinear" if k == "t2w" else "nearest"
            ).squeeze(0)
        return x

