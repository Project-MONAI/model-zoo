# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
from typing import Union

import SimpleITK as sitk  # noqa N813

parser = argparse.ArgumentParser(description="Center crop a 3d volume")
parser.add_argument("--file_name", type=str, required=True, help="Path to the input file to center crop.")
parser.add_argument(
    "--margin",
    type=Union[int, float],
    required=False,
    default=0.2,
    help="Crop margins applied to EACH side in the axial plane. "
    "If given a float value, will perform percentage crop. "
    "If given an int value, will perform absolute crop.",
)
parser.add_argument("--out_name", type=str, required=True, help="Path and filename for the cropped volume")

args = parser.parse_args()


def _flatten(t):
    return [item for sublist in t for item in sublist]


def crop(image: sitk.Image, margin: Union[int, float], interpolator=sitk.sitkLinear):
    """
    Crops a sitk.Image while retaining correct spacing. Negative margins will lead to zero padding

    Args:
        image:  a sitk.Image
        margin: margins to crop. Single integer or float (percentage crop),
                lists of int/float or nestes lists are supported.
    """
    if isinstance(margin, (list, tuple)):
        assert len(margin) == 3, "expected margin to be of length 3"
    else:
        assert isinstance(margin, (int, float)), "expected margin to be a float value"
        margin = [margin, margin, margin]

    margin = [m if isinstance(m, (tuple, list)) else [m, m] for m in margin]
    old_size = image.GetSize()

    # calculate new origin and new image size
    if all([isinstance(m, float) for m in _flatten(margin)]):
        assert all([m >= 0 and m < 0.5 for m in _flatten(margin)]), "margins must be between 0 and 0.5"
        to_crop = [[int(sz * _m) for _m in m] for sz, m in zip(old_size, margin)]
    elif all([isinstance(m, int) for m in _flatten(margin)]):
        to_crop = margin
    else:
        raise ValueError("Wrong format of margins.")

    new_size = [sz - sum(c) for sz, c in zip(old_size, to_crop)]

    # origin has Index (0,0,0)
    # new origin has Index (to_crop[0][0], to_crop[2][0], to_crop[2][0])
    new_origin = image.TransformIndexToPhysicalPoint([c[0] for c in to_crop])

    # create reference plane to resample image
    ref_image = sitk.Image(new_size, image.GetPixelIDValue())
    ref_image.SetSpacing(image.GetSpacing())
    ref_image.SetOrigin(new_origin)
    ref_image.SetDirection(image.GetDirection())

    return sitk.Resample(image, ref_image, interpolator=interpolator)


if __name__ == "__main__":
    image = sitk.ReadImage(args.file_name)
    cropped = crop(image, [args.margin, args.margin, 0.0])
    sitk.WriteImage(cropped, args.out_name)
