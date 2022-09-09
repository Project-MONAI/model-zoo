# Copyright (c) 2022 Eric Kerfoot under MIT license, see license.txt

import os
from typing import Any, Callable, Sequence

import monai
import monai.transforms as mt
import numpy as np
import torch
import torch.nn as nn
from monai.data.meta_obj import get_track_meta
from monai.networks.blocks import ConvDenseBlock, Convolution
from monai.networks.layers import Flatten, Reshape
from monai.networks.nets import Regressor
from monai.networks.utils import meshgrid_ij
from monai.utils import CommonKeys
from monai.utils import ImageMetaKey as Key
from monai.utils import convert_to_numpy, convert_to_tensor

# relates the label in training images to index of landmark point
LM_INDICES = {
    10: 0,  # mitral anterior 2CH
    15: 1,  # mitral posterior 2CH
    20: 2,  # mitral septal 3CH
    25: 3,  # mitral free wall 3CH
    30: 4,  # mitral septal 4CH
    35: 5,  # mitral free wall 4CH
    100: 6,  # aortic septal
    150: 7,  # aortic free wall
    200: 8,  # tricuspid septal
    250: 9,  # tricuspid free wall
}

output_trans = monai.handlers.from_engine(["pred", "label"])


def _output_lm_trans(data):
    pred, label = output_trans(data)
    return [p.permute(1, 0) for p in pred], [l.permute(1, 0) for l in label]


def convert_lm_image_t(lm_image):
    """Convert a landmark image into a (2,N) tensor of landmark coordinates."""
    lmarray = torch.zeros((2, len(LM_INDICES)), dtype=torch.float32).to(lm_image.device)

    for _, y, x in np.argwhere(lm_image.cpu().numpy() != 0):
        im_id = int(lm_image[0, y, x])
        lm_index = LM_INDICES[im_id]

        lmarray[0, lm_index] = y
        lmarray[1, lm_index] = x

    return lmarray


class ParallelCat(nn.Module):
    """
    Apply the same input to each of the given modules and concatenate their results together.

    Args:
        catmodules: sequence of nn.Module objects to apply inputs to
        cat_dim: dimension to concatenate along when combining outputs
    """

    def __init__(self, catmodules: Sequence[nn.Module], cat_dim: int = 1):
        super().__init__()
        self.cat_dim = cat_dim

        for i, s in enumerate(catmodules):
            self.add_module(f"catmodule_{i}", s)

    def forward(self, x):
        tensors = [s(x) for s in self.children()]
        return torch.cat(tensors, self.cat_dim)


class PointRegressor(Regressor):
    """Regressor defined as a sequence of dense blocks followed by convolution/linear layers for each landmark."""

    def _get_layer(self, in_channels, out_channels, strides, is_last):
        dout = out_channels - in_channels
        dilations = [1, 2, 4]
        dchannels = [dout // 3, dout // 3, dout // 3 + dout % 3]

        db = ConvDenseBlock(
            spatial_dims=self.dimensions,
            in_channels=in_channels,
            channels=dchannels,
            dilations=dilations,
            kernel_size=self.kernel_size,
            num_res_units=self.num_res_units,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
        )

        conv = Convolution(
            spatial_dims=self.dimensions,
            in_channels=out_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_last,
        )

        return nn.Sequential(db, conv)

    def _get_final_layer(self, in_shape):
        point_paths = []

        for _ in range(self.out_shape[1]):
            conv = Convolution(
                spatial_dims=self.dimensions,
                in_channels=in_shape[0],
                out_channels=in_shape[0] * 2,
                strides=2,
                kernel_size=self.kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                conv_only=True,
            )
            linear = nn.Linear(int(np.product(in_shape)) // 2, self.out_shape[0])
            point_paths.append(nn.Sequential(conv, Flatten(), linear))

        return torch.nn.Sequential(ParallelCat(point_paths), Reshape(*self.out_shape))


class LandmarkInferer(monai.inferers.Inferer):
    """Applies inference on 2D slices from 3D volumes."""

    def __init__(self, spatial_dim=0, stack_dim=-1):
        self.spatial_dim = spatial_dim
        self.stack_dim = stack_dim

    def __call__(self, inputs: torch.Tensor, network: Callable[..., torch.Tensor], *args: Any, **kwargs: Any):
        if inputs.ndim != 5:
            raise ValueError(f"Input volume to inferer must have shape BCDHW, input shape is {inputs.shape}")

        results = []
        input_slices = [slice(None) for _ in range(inputs.ndim)]

        for idx in range(inputs.shape[self.spatial_dim + 2]):
            input_slices[self.spatial_dim + 2] = idx
            input_2d = inputs[input_slices]  # BCDHW -> BCHW by iterating over one of DHW

            result = network(input_2d, *args, **kwargs)
            results.append(result)

        result = torch.stack(results, self.stack_dim)
        return result


class NpySaverd(mt.MapTransform):
    """Saves tensors/arrays to Numpy npy files."""

    def __init__(self, keys, output_dir, data_root_dir):
        super().__init__(keys)
        self.output_dir = output_dir
        self.data_root_dir = data_root_dir
        self.folder_layout = monai.data.FolderLayout(
            self.output_dir, extension=".npy", data_root_dir=self.data_root_dir
        )

    def __call__(self, d):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        for key in self.key_iterator(d):
            orig_filename = d[key].meta[Key.FILENAME_OR_OBJ]
            if isinstance(orig_filename, (list, tuple)):
                orig_filename = orig_filename[0]

            out_filename = self.folder_layout.filename(orig_filename, key=key)

            np.save(out_filename, convert_to_numpy(d[key]))

        return d


class FourierDropout(mt.Transform, mt.Fourier):
    """
    Apply dropout in Fourier space to corrupt images. This works by zeroing out pixels with greater probability the
    farther from the centre they are. All pixels closer than `min_dist` to the center are preserved, all beyond
    `max_dist` become 0. Distances from the centre to an edge in a given dimension are defined as 1.0.

    Args:
        min_dist: minimum distance to apply dropout, must be >0, smaller values will cause greater corruption
        max_dist: maximal distance to apply dropout, must be greater than `min_dist`, all pixels beyond become 0
    """

    def __init__(self, min_dist: float = 0.1, max_dist: float = 0.9):
        super().__init__()
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.prob_field = None
        self.field_shape = None

    def _get_prob_field(self, shape):
        shape = tuple(shape)
        if shape != self.field_shape:
            self.field_shape = shape
            spaces = [torch.linspace(-1, 1, s) for s in shape[1:]]
            grids = meshgrid_ij(*spaces)
            # middle is 0, mid edges 1, corners sqrt(2)
            self.prob_field = torch.stack(grids).pow_(2).sum(axis=0).sqrt_()

        return self.prob_field

    def __call__(self, im):
        probfield = self._get_prob_field(im.shape).to(im.device)

        # rand range from min_dist to max_dist
        dropout = torch.rand_like(im).mul_(self.max_dist - self.min_dist).add_(self.min_dist)
        # keep pixel if dropout value is greater than distance from center, so less likely farther from center
        dropout = dropout.ge_(probfield)

        result = self.shift_fourier(im, im.ndim - 1)
        result.mul_(dropout)
        result = self.inv_shift_fourier(result, im.ndim - 1)

        return convert_to_tensor(result, track_meta=get_track_meta())


class RandFourierDropout(mt.RandomizableTransform):
    def __init__(self, min_dist=0.1, max_dist=0.9, prob=0.1):
        mt.RandomizableTransform.__init__(self, prob)
        self.dropper = FourierDropout(min_dist, max_dist)

    def __call__(self, im, randomize: bool = True):
        if randomize:
            self.randomize(None)

        if self._do_transform:
            im = self.dropper(im)
        else:
            im = convert_to_tensor(im, track_meta=get_track_meta())

        return im


class RandFourierDropoutd(mt.RandomizableTransform, mt.MapTransform):
    def __init__(self, keys, min_dist=0.1, max_dist=0.9, prob=0.1):
        mt.RandomizableTransform.__init__(self, prob)
        mt.MapTransform.__init__(self, keys)
        self.dropper = FourierDropout(min_dist, max_dist)

    def __call__(self, data, randomize: bool = True):
        d = dict(data)

        if randomize:
            self.randomize(None)

        for key in self.key_iterator(d):
            if self._do_transform:
                d[key] = self.dropper(d[key])
            else:
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())

        return d


class RandImageLMDeformd(mt.RandSmoothDeform):
    """Apply smooth random deformation to the image and landmark locations."""

    def __call__(self, d):
        d = dict(d)
        old_label = d[CommonKeys.LABEL]
        new_label = torch.zeros_like(old_label)

        d[CommonKeys.IMAGE] = super().__call__(d[CommonKeys.IMAGE])

        if self._do_transform:
            field = self.sfield()
            labels = np.argwhere(d[CommonKeys.LABEL][0].cpu().numpy() > 0)

            # moving the landmarks this way prevents losing some to
            # interpolation errors if deformation were applied the landmark image
            for y, x in labels:
                dy = int(field[0, y, x] * new_label.shape[1] / 2)
                dx = int(field[1, y, x] * new_label.shape[2] / 2)

                new_label[:, y - dy, x - dx] = old_label[:, y, x]

            d[CommonKeys.LABEL] = new_label

        return d


class RandLMShiftd(mt.RandomizableTransform, mt.MapTransform):
    """Randomly shift the image and landmark image in either direction in integer amounts."""

    def __init__(self, keys, spatial_size, max_shift=0, prob=0.1):
        mt.RandomizableTransform.__init__(self, prob=prob)
        mt.MapTransform.__init__(self, keys=keys)

        self.spatial_size = tuple(spatial_size)
        self.max_shift = max_shift
        self.padder = mt.BorderPad(self.max_shift)
        self.unpadder = mt.CenterSpatialCrop(self.spatial_size)
        self.shift = (0,) * len(self.spatial_size)
        self.roll_dims = list(range(1, len(self.spatial_size) + 1))

    def randomize(self, data):
        super().randomize(None)
        if self._do_transform:
            rs = torch.randint(-self.max_shift, self.max_shift, (len(self.spatial_size),), dtype=torch.int32)
            self.shift = tuple(rs.tolist())

    def __call__(self, d, randomize: bool = True):
        d = dict(d)

        if randomize:
            self.randomize(None)

        if self._do_transform:
            for key in self.key_iterator(d):
                imp = self.padder(d[key])
                ims = torch.roll(imp, self.shift, self.roll_dims)  # prevents interpolation of landmark image
                d[key] = self.unpadder(ims)

        return d
