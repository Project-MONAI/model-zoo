from typing import Dict

import numpy as np
from einops import rearrange
from monai.transforms.transform import Transform


class OrientationGuidanceMultipleLabelDeepEditd(Transform):
    def __init__(self, ref_image="image", label_names=None):
        """
        Convert the guidance to the RAS orientation
        """
        self.ref_image = ref_image
        self.label_names = label_names

    def transform_points(self, point, affine):
        """transform point to the coordinates of the transformed image
        point: numpy array [bs, N, 3]
        """
        bs, n = point.shape[:2]
        point = np.concatenate((point, np.ones((bs, n, 1))), axis=-1)
        point = rearrange(point, "b n d -> d (b n)")
        point = affine @ point
        point = rearrange(point, "d (b n)-> b n d", b=bs)[:, :, :3]
        return point

    def __call__(self, data):
        d: Dict = dict(data)
        for key_label in self.label_names.keys():
            points = d.get(key_label, [])
            if len(points) < 1:
                continue
            reoriented_points = self.transform_points(
                np.array(points)[None],
                np.linalg.inv(d[self.ref_image].meta["affine"].numpy()) @ d[self.ref_image].meta["original_affine"],
            )
            d[key_label] = reoriented_points[0]
        return d
