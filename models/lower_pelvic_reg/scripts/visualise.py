import numpy as np
from pathlib import Path
from typing import Optional, Type, Union, Sequence
import nibabel as nib
import torch

from monai.config import DtypeLike, KeysCollection
from monai.data import image_writer
from monai.transforms.io.array import SaveImage
from monai.transforms.transform import MapTransform
from monai.utils import GridSamplePadMode, ensure_tuple_rep
from monai.utils.enums import PostFix

DEFAULT_POST_FIX = PostFix.meta()


class SaveRegd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.SaveImage`.

    Note:
        Image should be channel-first shape: [C,H,W,[D]].
        If the data is a patch of an image, the patch index will be appended to the filename.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        pixdim: output voxel spacing. if providing a single number, will use it for the first dimension.
            items of the pixdim sequence map to the spatial dimensions of input image, if length
            of pixdim sequence is longer than image spatial dimensions, will ignore the longer part,
            if shorter, will pad with `1.0`.
            if the components of the `pixdim` are non-positive values, the transform will use the
            corresponding components of the original pixdim, which is computed from the `affine`
            matrix of input image.
        spatial_size: output image spatial size.
            if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
            the transform will use the spatial size of `image`.
            if some components of the `spatial_size` are non-positive values, the transform will use the
            corresponding components of image size. For example, `spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of image is `64`.
        output_dir: output image directory.
        allow_missing_keys: don't raise exception if key is missing.

    """

    def __init__(
        self,
        keys: KeysCollection,
        pixdim: Sequence[float],
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        output_dir: Union[Path, str] = "./",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.affine = torch.diag(torch.tensor([*pixdim, 1]))
        self.spatial_size = spatial_size
        self.output_dir = output_dir

    def __call__(self, data):
        """
        A dictionary with following items:
            - moving_image: image Tensor data for model input, already moved to device.
            - moving_label: label Tensor data corresponding to the image, already moved to device.
            - fixed_image: image Tensor data for model input, already moved to device.
            - fixed_label: label Tensor data corresponding to the image, already moved to device.
            - ddf: dense displacement field which registers the moving towards fixed.
            - warped_image: moving image warped by the predicted ddf
            - warped_label: moving label warped by the predicted ddf
        """
        for k in self.keys:
            print(f"{k}: {torch.tensor(data[k]).shape}")
        for key in self.keys:
            print(type(data[key]))
            print(torch.tensor(data[key]).shape)
            img = nib.Nifti1Image(
                torch.tensor(data[key]).reshape(*self.spatial_size).detach().cpu().numpy().astype(dtype=np.float32),
                affine=self.affine
            )
            name = data["moving_name"] + "_" + data["fixed_name"]
            print(name)
            nib.save(img, f"{self.output_dir}/{name}_{key}.nii")
