import copy
import random

import monai
import numpy as np
import torch
import torch.nn.functional as F
from monai.utils import ensure_tuple_rep

ENABLE_SPECIAL = True
SPECIAL_INDEX = (23, 24, 25, 26, 27, 57, 128)
MERGE_LIST = {
    1: [25, 26],  # hepatic tumor and vessel merge into liver
    4: [24],  # pancreatic tumor merge into pancreas
    132: [57],  # overlap with trachea merge into airway
}


def get_point_label(id):
    # [B, N]
    if id in SPECIAL_INDEX and ENABLE_SPECIAL:
        return 2, 3
    else:
        return 0, 1


def convert_point_label(point_label, label_set=None):
    if label_set is None or not ENABLE_SPECIAL:
        return point_label
    assert point_label.shape[0] == len(label_set)
    for i in range(len(label_set)):
        if label_set[i] in SPECIAL_INDEX:
            for j in range(len(point_label[i])):
                point_label[i, j] = point_label[i, j] + 2 if point_label[i, j] > -1 else point_label[i, j]
    return point_label


def sample_points_patch_val(
    labels,
    patch_coords,
    label_set,
    prev_mask,
    class_vector,
    use_center=True,
    mapped_label_set=None,
    max_ppoint=1,
    max_npoint=0,
    **kwargs,
):
    """
    Sample points for patch during sliding window validation. The prev_mask is only used for auto + interactive.
    This function is called within vista3d.py and will use largested cc combine, do not use for iterative point evaluation.
    """
    # only in validation when labels of the whole image is provided, sample points for every position
    _, point_coords, point_labels, _ = generate_prompt_pairs_val(
        labels[patch_coords],
        label_set,
        max_ppoint=max_ppoint,
        max_npoint=max_npoint,
        device=labels.device,
        use_center=use_center,
    )
    point_labels = convert_point_label(point_labels, label_set)
    return point_coords, point_labels, torch.tensor(label_set).to(point_coords.device).unsqueeze(-1)


def erode3d(input_tensor, erosion=3):
    # Define the structuring element
    erosion = ensure_tuple_rep(erosion, 3)
    structuring_element = torch.ones(1, 1, erosion[0], erosion[1], erosion[2]).to(input_tensor.device)

    # Pad the input tensor to handle border pixels
    input_padded = F.pad(
        input_tensor.float().unsqueeze(0).unsqueeze(0),
        (erosion[2] // 2, erosion[2] // 2, erosion[1] // 2, erosion[1] // 2, erosion[0] // 2, erosion[0] // 2),
        mode="constant",
        value=1.0,
    )

    # Apply erosion operation
    output = F.conv3d(input_padded, structuring_element, padding=0)

    # Set output values based on the minimum value within the structuring element
    output = torch.where(output == torch.sum(structuring_element), 1.0, 0.0)

    return output.squeeze(0).squeeze(0)


def generate_prompt_pairs_val(labels, label_set=None, max_ppoint=1, max_npoint=0, device="cpu", use_center=False):
    """
    Args:
        labels: torch.tensor from dataload, [1,1,H,W,D]
        label_set: the label list for the specific dataset
    Returns:
        label_prompt: [b, 1]
        point: [b, N, 3]
        point_label: [b, N]
        prompt_class: [b, 1], exactly the same with label_prompt for label indexing for training lloss.

    """
    # class label number
    assert labels.shape[0] == 1, "only support batch size 1"
    labels = labels[0, 0]
    label_prompt = torch.tensor(label_set).to(device).unsqueeze(-1)
    unique_labels = labels.unique().cpu().numpy().tolist()
    _point = []
    _point_label = []
    num_n = max_npoint
    num_p = max_ppoint
    for id in label_set:
        if id in unique_labels:
            plabels = labels == int(id)
            nlabels = ~plabels
            _plabels = erode3d(plabels)
            # _plabels = monai.transforms.utils.get_largest_connected_component_mask(_plabels)
            plabelpoints = torch.nonzero(_plabels).to(device)
            if len(plabelpoints) == 0:
                plabelpoints = torch.nonzero(plabels).to(device)
            nlabelpoints = torch.nonzero(nlabels).to(device)
            if use_center:
                pmean = plabelpoints.float().mean(0)
                pdis = ((plabelpoints - pmean) ** 2).sum(-1)
                _, sorted_indices = torch.sort(pdis)
                _point.append(
                    torch.stack(
                        [plabelpoints[sorted_indices[i]] for i in range(min(len(plabelpoints), num_p))]
                        + random.choices(nlabelpoints, k=min(len(nlabelpoints), num_n))
                        + [torch.tensor([0, 0, 0], device=device)]
                        * (num_p + num_n - min(len(plabelpoints), num_p) - min(len(nlabelpoints), num_n))
                    )
                )
                _point_label.append(
                    torch.tensor(
                        [1] * min(len(plabelpoints), num_p)
                        + [0.0] * min(len(nlabelpoints), num_n)
                        + [-1] * (num_p + num_n - min(len(plabelpoints), num_p) - min(len(nlabelpoints), num_n))
                    ).to(device)
                )

            else:
                _point.append(
                    torch.stack(
                        random.choices(plabelpoints, k=min(len(plabelpoints), num_p))
                        + random.choices(nlabelpoints, k=min(len(nlabelpoints), num_n))
                        + [torch.tensor([0, 0, 0], device=device)]
                        * (num_p + num_n - min(len(plabelpoints), num_p) - min(len(nlabelpoints), num_n))
                    )
                )
                _point_label.append(
                    torch.tensor(
                        [1] * min(len(plabelpoints), num_p)
                        + [0.0] * min(len(nlabelpoints), num_n)
                        + [-1] * (num_p + num_n - min(len(plabelpoints), num_p) - min(len(nlabelpoints), num_n))
                    ).to(device)
                )
        else:
            # pad the background labels
            _point.append(torch.zeros(num_p + num_n, 3).to(device))  # all 0
            _point_label.append(torch.zeros(num_p + num_n).to(device) - 1)  # -1 not a point
    point = torch.stack(_point)
    point_label = torch.stack(_point_label)
    prompt_class = copy.deepcopy(label_prompt)
    return label_prompt, point, point_label, prompt_class


def generate_prompt_pairs(
    labels,
    label_set=None,
    image_size=None,
    max_prompt=None,
    max_foreprompt=None,
    max_backprompt=1,
    max_point=20,
    include_background=True,
    drop_label_prob=0.2,
    drop_point_prob=0.2,
    convert_to_disc=False,
    radius=2,
    metric_class=None,
    ignore_labelset=False,
    point_sampler=None,
):
    """
    Args:
        labels: torch.tensor from dataload, [1,1,H,W,D]
        label_set: the label list for the specific dataset
        total_prompt: int, number of total prompt
        max_point: maximum number of points for each object
        include_background: if include label=0 into training prompt. May casue issue in partial label
                            trainig.
        metric_class: validation dice of each class. Must be the same dim with label_set
    Returns:
        label_prompt: [b, 1]
        point: [b, N, 3]
        point_label: [b, N]
        prompt_class: [b, 1], exactly the same with label_prompt for label indexing for training lloss.

    """
    # class label number
    assert labels.shape[0] == 1, "only support batch size 1"
    labels = labels[0, 0]
    point_mask = None
    device = labels.device
    unique_labels = labels.unique()
    if include_background:
        unique_labels = list(set(unique_labels) - (set(unique_labels) - set(label_set)))
    else:
        unique_labels = list(set(unique_labels) - (set(unique_labels) - set(label_set)) - {0})
    background_labels = list(set(label_set) - set(unique_labels))
    # during training, balance background and foreground prompts
    if max_backprompt is not None:
        if len(background_labels) > max_backprompt:
            random.shuffle(background_labels)
            background_labels = background_labels[:max_backprompt]

    if max_foreprompt is not None:
        if len(unique_labels) > max_foreprompt:
            random.shuffle(unique_labels)
            unique_labels = unique_labels[:max_foreprompt]

    if max_prompt is not None:
        if len(unique_labels) + len(background_labels) > max_prompt:
            if len(unique_labels) > max_prompt:
                # unique_labels = random.sample(unique_labels, max_prompt)
                if metric_class is None:
                    prob = np.ones(len(unique_labels))
                else:
                    prob = (
                        1 - metric_class[np.array(unique_labels).astype(int)]
                        if len(label_set) == len(metric_class)
                        else 1 - metric_class[np.array(unique_labels).astype(int) - 1]
                    )
                prob = [w / sum(prob) for w in prob]
                unique_labels = np.random.choice(unique_labels, size=max_prompt, replace=False, p=prob).tolist()
                background_labels = []
            else:
                background_labels = random.sample(background_labels, max_prompt - len(unique_labels))
    _point = []
    _point_label = []
    num_p = min(max_point, int(np.abs(random.gauss(mu=0, sigma=max_point // 2))) + 1)
    num_n = min(max_point, int(np.abs(random.gauss(mu=0, sigma=max_point // 2))))
    for id in unique_labels:
        neg_id, pos_id = get_point_label(id)
        plabels = labels == int(id)
        nlabels = ~plabels
        plabelpoints = torch.nonzero(plabels)
        nlabelpoints = torch.nonzero(nlabels)
        _point.append(
            torch.stack(
                random.choices(plabelpoints, k=min(len(plabelpoints), num_p))
                + random.choices(nlabelpoints, k=min(len(nlabelpoints), num_n))
                + [torch.tensor([0, 0, 0], device=device)]
                * (num_p + num_n - min(len(plabelpoints), num_p) - min(len(nlabelpoints), num_n))
            )
        )
        _point_label.append(
            torch.tensor(
                [pos_id] * min(len(plabelpoints), num_p)
                + [neg_id] * min(len(nlabelpoints), num_n)
                + [-1] * (num_p + num_n - min(len(plabelpoints), num_p) - min(len(nlabelpoints), num_n))
            ).to(device)
        )
    for _id in background_labels:
        # pad the background labels
        _point.append(torch.zeros(num_p + num_n, 3).to(device))  # all 0
        _point_label.append(torch.zeros(num_p + num_n).to(device) - 1)  # -1 not a point
    label_prompt = torch.tensor(unique_labels + background_labels).unsqueeze(-1).to(device).long()
    point = torch.stack(_point)
    point_label = torch.stack(_point_label)
    prompt_class = copy.deepcopy(label_prompt)
    if random.uniform(0, 1) < drop_label_prob and len(unique_labels) > 0:
        label_prompt = None
        # drop out the padded
        pad = len(background_labels)
        point = point[: len(point) - pad]
        point_label = point_label[: len(point_label) - pad]
        prompt_class = prompt_class[: len(prompt_class) - pad]
    else:
        if random.uniform(0, 1) < drop_point_prob:
            point = None
            point_label = None
    if point is not None and convert_to_disc:
        point_mask = convert_points_to_disc(image_size, point, point_label, radius=radius)
    return label_prompt, point, point_label, prompt_class, point_mask


def get_gaussian_ball(image_size, radius=None):
    if radius is None:
        radius = image_size[0] // 3
    row_array = torch.arange(start=0, end=image_size[0], step=1, dtype=torch.float32)
    col_array = torch.arange(start=0, end=image_size[1], step=1, dtype=torch.float32)
    z_array = torch.arange(start=0, end=image_size[2], step=1, dtype=torch.float32)
    coord_rows, coord_cols, coord_z = torch.meshgrid(z_array, col_array, row_array, indexing="ij")
    coords = torch.stack((coord_rows, coord_cols, coord_z), dim=0)
    center = (
        torch.tensor([image_size[0] // 2, image_size[1] // 2, image_size[2] // 2])
        .to(coords.device)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .unsqueeze(-1)
    )
    ball = torch.exp(-((((coords - center) ** 2).sum(0) / (2 * radius**2)) ** 2))
    return ball


def convert_points_to_disc(image_size, point, point_label, radius=2, disc=False):
    # [b, N, 3], [b, N]
    # generate masks [b,2,h,w,d]
    if not torch.is_tensor(point):
        point = torch.from_numpy(point)
    masks = torch.zeros([point.shape[0], 2, image_size[0], image_size[1], image_size[2]], device=point.device)
    row_array = torch.arange(start=0, end=image_size[0], step=1, dtype=torch.float32, device=point.device)
    col_array = torch.arange(start=0, end=image_size[1], step=1, dtype=torch.float32, device=point.device)
    z_array = torch.arange(start=0, end=image_size[2], step=1, dtype=torch.float32, device=point.device)
    coord_rows, coord_cols, coord_z = torch.meshgrid(z_array, col_array, row_array, indexing="ij")
    # [1,3,h,w,d] -> [b, 2, 3, h,w,d]
    coords = (
        torch.stack((coord_rows, coord_cols, coord_z), dim=0)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(point.shape[0], 2, 1, 1, 1, 1)
    )
    for b in range(point.shape[0]):
        for n in range(point.shape[1]):
            if point_label[b, n] > -1:
                channel = 0 if (point_label[b, n] == 0 or point_label[b, n] == 2) else 1
                if disc:
                    masks[b, channel] += (
                        torch.pow(coords[b, channel] - point[b, n].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 2).sum(0)
                        < radius**2
                    )
                else:
                    masks[b, channel] += torch.exp(
                        -torch.pow(coords[b, channel] - point[b, n].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 2).sum(0)
                        / (2 * radius**2)
                    )
    # masks[masks>1] = 1
    return masks


def get_window_idx_c(p, roi, s):
    if p - roi // 2 < 0:
        l, r = 0, roi
    elif p + roi // 2 > s:
        l, r = s - roi, s
    else:
        l, r = int(p) - roi // 2, int(p) + roi // 2
    return l, r


def get_window_idx(p, roi, s, center_only=True, margin=5):
    l, r = get_window_idx_c(p, roi, s)
    if center_only:
        return [l], [r]
    left_most = max(0, p - roi + margin)
    right_most = min(s, p + roi - margin)
    left = [left_most, right_most - roi, l]
    right = [left_most + roi, right_most, r]
    return left, right


def pad_previous_mask(inputs, roi_size, padvalue=0):
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    if any(pad_size):
        inputs = torch.nn.functional.pad(inputs, pad=pad_size, mode="constant", value=padvalue)
    return inputs, pad_size


def point_based_window_inferer(
    inputs,
    roi_size,
    sw_batch_size,
    predictor,
    mode,
    overlap,
    sw_device,
    device,
    point_coords,
    point_labels,
    class_vector,
    prompt_class,
    prev_mask,
    point_mask=None,
    point_start=0,
    **kwargs,
):
    """
    Point based window inferer, crop a patch centered at the point, and perform inference.
    Different patches are combined with gaussian weighted weights.

    Args:
        predictor: partial(infer_wrapper, model). infer_wrapper transpose the model output.
            The model output is [B, 1, H, W, D] which needs to be transposed to [1, B, H, W, D]
        point_coords: [B, N, 3]
        point_labels: [B, N]
        class_vector: [B]
        prev_mask: [1, B, H, W, D], THE VALUE IS BEFORE SIGMOID!
    Returns:
        stitched_output: [1, B, H, W, D]. The value is before sigmoid.
    Notice: The function currently only supports SINGLE OBJECT INFERENCE with B=1.
    """
    assert point_coords.shape[0] == 1, "Only supports single object point click"
    image, pad = pad_previous_mask(copy.deepcopy(inputs), roi_size)
    point_coords = point_coords + torch.tensor([pad[-2], pad[-4], pad[-6]]).to(point_coords.device)
    prev_mask = pad_previous_mask(copy.deepcopy(prev_mask), roi_size)[0] if prev_mask is not None else None
    stitched_output = None
    center_only = True
    for p in point_coords[0][point_start:]:
        lx_, rx_ = get_window_idx(p[0], roi_size[0], image.shape[-3], center_only=center_only, margin=5)
        ly_, ry_ = get_window_idx(p[1], roi_size[1], image.shape[-2], center_only=center_only, margin=5)
        lz_, rz_ = get_window_idx(p[2], roi_size[2], image.shape[-1], center_only=center_only, margin=5)
        for i in range(len(lx_)):
            for j in range(len(ly_)):
                for k in range(len(lz_)):
                    lx, rx, ly, ry, lz, rz = lx_[i], rx_[i], ly_[j], ry_[j], lz_[k], rz_[k]
                    unravel_slice = [
                        slice(None),
                        slice(None),
                        slice(int(lx), int(rx)),
                        slice(int(ly), int(ry)),
                        slice(int(lz), int(rz)),
                    ]
                    batch_image = image[unravel_slice]
                    output = predictor(
                        batch_image,
                        point_coords=point_coords,
                        point_labels=point_labels,
                        class_vector=class_vector,
                        prompt_class=prompt_class,
                        patch_coords=unravel_slice,
                        prev_mask=prev_mask,
                        **kwargs,
                    )
                    if stitched_output is None:
                        stitched_output = torch.zeros(
                            [1, output.shape[1], image.shape[-3], image.shape[-2], image.shape[-1]], device="cpu"
                        )
                        stitched_mask = torch.zeros(
                            [1, output.shape[1], image.shape[-3], image.shape[-2], image.shape[-1]], device="cpu"
                        )
                    stitched_output[unravel_slice] += output.to("cpu")
                    stitched_mask[unravel_slice] = 1
    # if stitched_mask is 0, then NaN value
    stitched_output = stitched_output / stitched_mask
    # revert padding
    stitched_output = stitched_output[
        :, :, pad[4] : image.shape[-3] - pad[5], pad[2] : image.shape[-2] - pad[3], pad[0] : image.shape[-1] - pad[1]
    ]
    stitched_mask = stitched_mask[
        :, :, pad[4] : image.shape[-3] - pad[5], pad[2] : image.shape[-2] - pad[3], pad[0] : image.shape[-1] - pad[1]
    ]
    if prev_mask is not None:
        prev_mask = prev_mask[
            :,
            :,
            pad[4] : image.shape[-3] - pad[5],
            pad[2] : image.shape[-2] - pad[3],
            pad[0] : image.shape[-1] - pad[1],
        ]
        prev_mask = prev_mask.to("cpu")
        # for un-calculated place, use previous mask
        stitched_output[stitched_mask < 1] = prev_mask[stitched_mask < 1]

    if not hasattr(stitched_output, "meta"):
        stitched_output = monai.data.MetaTensor(stitched_output, affine=inputs.meta["affine"], meta=inputs.meta)
    return stitched_output
