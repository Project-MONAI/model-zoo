import typing as t
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tracemalloc import start

import cv2
import numpy as np
import rpack
from openslide import OpenSlide
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage import filters
from skimage.morphology import remove_small_objects

if t.TYPE_CHECKING:
    from _typeshed import StrPath

try:
    from skimage import img_as_ubyte  # type: ignore
except:
    from skimage.util import img_as_ubyte  # type: ignore


def find_contours(arr: np.ndarray, only_outer: bool = True, convex: bool = False):
    """Find contours in a binary image

    Parameters
    ----------
    arr : np.ndarray
        Binary image
    only_outer : bool
        If True, only find external contours
    convex : bool
        If True, return convex hull of contours

    Returns
    -------
    contours : list
        List of contours
    """
    mode = cv2.RETR_EXTERNAL if only_outer else cv2.RETR_LIST
    cresults = cv2.findContours(arr.astype(np.uint8), mode, cv2.CHAIN_APPROX_SIMPLE)

    contours = cresults[1] if len(cresults) == 3 else cresults[0]
    contours = list(contours) if isinstance(contours, tuple) else contours

    if convex:
        contours = [cv2.convexHull(cnt) for cnt in contours]
    return contours


def merge_overlapping_bboxes(bboxes: list):
    """Merge overlapping bounding boxes

    Parameters
    ----------
    bboxes : list
        List of bounding boxes in format (x, y, width, height)
    """
    candidate_count = 0
    while candidate_count < len(bboxes):
        candidate_count += 1
        overlap = False
        candidate_box = bboxes.pop(0)
        for index, compare_box in enumerate(bboxes):
            overlapping, new_bbox = merge_if_overlapping(candidate_box, compare_box)
            if overlapping:
                overlap = True
                candidate_count = 0
                bboxes.pop(index)
                bboxes.append(new_bbox)
                break
        if not overlap:
            bboxes.append(candidate_box)


def merge_if_overlapping(a: tuple, b: tuple):
    """Check if two bounding boxes overlap and merge them if they do

    Parameters
    ----------
    a : tuple
        First bounding box in format (x, y, width, height)
    b : tuple
        Second bounding box in format (x, y, width, height)

    Returns
    -------
    overlapping : bool
        True if boxes overlap
    new_bbox : tuple
        Merged bounding box if overlapping, empty list otherwise
    """
    bottom = np.max([a[0], b[0]])
    top = np.min([a[0] + a[2], b[0] + b[2]])
    left = np.max([a[1], b[1]])
    right = np.min([a[1] + a[3], b[1] + b[3]])

    do_intersect = bottom < top and left < right

    if do_intersect:
        x_min = np.min([a[1], b[1]])
        y_min = np.min([a[0], b[0]])
        x_max = np.max([a[1] + a[3], b[1] + b[3]])
        y_max = np.max([a[0] + a[2], b[0] + b[2]])
        new_bbox = (y_min, x_min, y_max - y_min, x_max - x_min)
        return True, new_bbox

    return False, []



def load_slide_img(
    wsi,
    level: int = 0,
) -> np.ndarray:
    """Load slide image with specific level

    Parameters
    ----------
    wsi : CuImage
        The CuImage object
    level : int
        Slide level to load

    Returns
    -------
    slide_img : np.ndarray
        Numpy array with RGB channels
    """
    slide_img = np.asarray(wsi.read_region(level=level, device="gpu", num_workers=32))
    if slide_img.shape[2] == 4:
        slide_img = slide_img[:, :, :-1]
    return slide_img


def rgb2gray(img):
    """Convert RGB image to grayscale

    Parameters
    ----------
    img : np.ndarray
        RGB image with 3 channels

    Returns
    -------
    gray : np.ndarray
        Grayscale image
    """
    return np.dot(img, [0.299, 0.587, 0.114])


def thresh_slide(gray, thresh_val, sigma=13):
    """Threshold gray image to binary image

    Parameters
    ----------
    gray : np.ndarray
        2D grayscale image
    thresh_val : float
        Thresholding value
    sigma : int
        Gaussian smoothing sigma

    Returns
    -------
    bw_img : np.ndarray
        Binary image
    """
    smooth = filters.gaussian(gray, sigma=sigma)
    smooth /= np.amax(smooth)
    bw_img = smooth < thresh_val
    return bw_img



def get_tissue_bboxes(
    mask: np.ndarray, wsi_width: int, wsi_height: int, min_tissue_size: int = 10000
):
    scale = wsi_height / mask.shape[0]

    contours = find_contours(mask)
    areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areas.append(area)

    large_contours = []
    large_areas = []
    for i, cnt in enumerate(contours):
        area_mm = areas[i]
        if area_mm >= min_tissue_size:
            large_contours.append(cnt)
            large_areas.append(area_mm)

    areas = large_areas

    boxes = [cv2.boundingRect(c) for c in large_contours]

    return (
        [cv2.boundingRect(c) for c in large_contours]
        if boxes
        else [[0, 0, wsi_width, wsi_height]]
    )


def get_tissue_positions_and_packed_size(
    boxes,
    wsi_width: int,
    wsi_height: int,
    scale: float,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    if len(boxes) > 1:
        merge_overlapping_bboxes(boxes)
    boxes = np.array(boxes, dtype=np.float32) * scale
    if len(boxes.shape) == 1:
        boxes = boxes[None]
    boxes[:, :2] = np.floor(boxes[:, :2])
    boxes[:, 0] = np.clip(boxes[:, 0], 0, wsi_width - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, wsi_height - 1)
    boxes[:, 2:] = np.ceil(boxes[:, 2:])
    boxes[:, 2] = np.clip(boxes[:, 2], 0, wsi_width - boxes[:, 0])
    boxes[:, 3] = np.clip(boxes[:, 3], 0, wsi_height - boxes[:, 1])
    boxes = boxes.astype(np.int32)

    box_sizes = [(int(box[2]), int(box[3])) for box in boxes]
    positions = rpack.pack(box_sizes)  # at processing spacing
    packed_size: tuple[int, int] = rpack.bbox_size(
        box_sizes, positions
    )  # width, height

    counter = 0
    for sdf in np.arange(0.5, 0.96, 0.05):
        # asymmetry_factor = min(packed_size)/max(packed_size)
        # if asymmetry_factor < sdf:
        rparams = {
            "max_height": int(max(packed_size) * sdf),
            "max_width": int(max(packed_size) * sdf),
        }
        try:
            positions = rpack.pack(box_sizes, **rparams)  # at processing spacing
            packed_size: tuple[int, int] = rpack.bbox_size(box_sizes, positions)
            break
        except rpack.PackingImpossibleError as ex:
            counter += 1

    return positions, (int(packed_size[0]), int(packed_size[1]))


def pack_slide(
    wsi_arr: np.ndarray,
    mask: np.ndarray,
    min_tissue_size: int = 10000,
):
    H, W = wsi_arr.shape[:2]
    boxes = get_tissue_bboxes(mask, W, H, min_tissue_size=min_tissue_size)
    if len(boxes) > 0:
        positions, packed_size = get_tissue_positions_and_packed_size(
            boxes, W, H, H / mask.shape[0]
        )
        img_out = np.full(
            (packed_size[1], packed_size[0]) + wsi_arr.shape[2:],
            255,
            dtype=wsi_arr.dtype,
        )
        mask_out = np.zeros((packed_size[1], packed_size[0]), dtype=np.bool)
        for i, pos in enumerate(positions):
            box = boxes[i]
            img_out[pos[1] : pos[1] + box[3], pos[0] : pos[0] + box[2]] = wsi_arr[
                box[1] : box[1] + box[3], box[0] : box[0] + box[2]
            ]
            mask_out[pos[1] : pos[1] + box[3], pos[0] : pos[0] + box[2]] = mask[
                box[1] : box[1] + box[3], box[0] : box[0] + box[2]
            ]
    else:
        img_out = wsi_arr
        mask_out = mask

    return img_out, mask_out


def get_level_downsamples(wsi: OpenSlide):
    level_downsamples = []
    dim_0 = wsi.level_dimensions[0]

    for downsample, dim in zip(wsi.level_downsamples, wsi.level_dimensions):
        estimated_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
        (
            level_downsamples.append(estimated_downsample)
            if estimated_downsample != (downsample, downsample)
            else level_downsamples.append((downsample, downsample))
        )

    return level_downsamples


def segment_tissue(
    wsi_path: Path,
    seg_level=-1,
    sthresh=8,
    sthresh_up=255,
    mthresh=7,
    close=4,
    filter_params={"a_t": 1, "a_h": 1, "max_n_holes": 100},
    ref_patch_size=512,
):
    """
    Segment the tissue via HSV -> Median thresholding -> Binary threshold
    """

    def _filter_contours(contours, hierarchy, filter_params):
        """
        Filter contours by: area.
        """
        filtered = []

        # find indices of foreground contours (parent == -1)
        hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
        all_holes = []

        # loop through foreground contour indices
        for cont_idx in hierarchy_1:
            # actual contour
            cont = contours[cont_idx]
            # indices of holes contained in this contour (children of parent contour)
            holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
            # take contour area (includes holes)
            a = cv2.contourArea(cont)
            # calculate the contour area of each hole
            hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
            # actual area of foreground contour region
            a = a - np.array(hole_areas).sum()
            if a == 0:
                continue
            if tuple((filter_params["a_t"],)) < tuple((a,)):
                filtered.append(cont_idx)
                all_holes.append(holes)

        foreground_contours = [contours[cont_idx] for cont_idx in filtered]

        hole_contours = []

        for hole_ids in all_holes:
            unfiltered_holes = [contours[idx] for idx in hole_ids]
            unfilered_holes = sorted(
                unfiltered_holes, key=cv2.contourArea, reverse=True
            )
            # take max_n_holes largest holes by area
            unfilered_holes = unfilered_holes[: filter_params["max_n_holes"]]
            filtered_holes = []

            # filter these holes
            for hole in unfilered_holes:
                if cv2.contourArea(hole) > filter_params["a_h"]:
                    filtered_holes.append(hole)

            hole_contours.append(filtered_holes)

        return foreground_contours, hole_contours

    def draw_white_bands(img: np.ndarray, thickness: int):
        height, width = img.shape[:2]
        white = [255, 255, 255]  # 흰색 (B, G, R)

        # cv2.copyMakeBorder 함수를 사용해 흰색 띠를 추가
        # 두께 30픽셀의 위쪽 흰색 띠 그리기
        cv2.rectangle(img, (0, 0), (width, thickness), white, -1)

        # 두께 30픽셀의 아래쪽 흰색 띠 그리기
        cv2.rectangle(img, (0, height - thickness), (width, height), white, -1)

        # 두께 30픽셀의 왼쪽 흰색 띠 그리기
        cv2.rectangle(img, (0, 0), (thickness, height), white, -1)

        # 두께 30픽셀의 오른쪽 흰색 띠 그리기
        cv2.rectangle(img, (width - thickness, 0), (width, height), white, -1)

    with OpenSlide(str(wsi_path)) as wsi:
        if seg_level < 0:
            seg_level = wsi.get_best_level_for_downsample(64)

        img = np.asarray(
            wsi.read_region(
                location=(0, 0), level=seg_level, size=wsi.level_dimensions[seg_level]
            )
        )

        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        draw_white_bands(img_rgb, thickness=20)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

        H, W = img_rgb.shape[:2]

        B_8, G_8, R_8 = cv2.split(img_rgb)
        B = B_8.astype(np.int32)
        G = G_8.astype(np.int32)
        R = R_8.astype(np.int32)

        mask = (R >= 0) & (R <= 110) & (G >= 0) & (G <= 110) & (B >= 0) & (B <= 110)

        color_difference1 = np.abs((R) - (G)) <= 15
        color_difference2 = np.abs((G) - (B)) <= 15
        color_difference3 = np.abs((R) - (B)) <= 15
        color_difference = color_difference1 & color_difference2 & color_difference3

        final_mask = mask & color_difference

        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
        laplacian_abs = cv2.convertScaleAbs(laplacian)
        mask = laplacian_abs <= 15
        img_rgb[mask] = [255, 255, 255]

        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)  # Convert to HSV space
        img_med = cv2.medianBlur(
            img_hsv[:, :, 1], mthresh
        )  # Apply median blurring #same to median filter

        # Thresholding
        _, img_thresh = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)
        # Morphological closing
        if close > 0:
            kernel = np.ones((close, close), np.uint8)
            img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)

        # before k-medicon
        scale = get_level_downsamples(wsi)[seg_level]
        scaled_ref_patch_area = int(ref_patch_size**2 / (scale[0] * scale[1]))
        filter_params = filter_params.copy()
        filter_params["a_t"] = filter_params["a_t"] * scaled_ref_patch_area
        filter_params["a_h"] = filter_params["a_h"] * scaled_ref_patch_area

        # Find and filter contours
        contours, hierarchy = cv2.findContours(
            img_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )

        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
        foreground_contours, hole_contours = _filter_contours(
            contours, hierarchy, filter_params
        )  # Necessary for filtering out artifacts

        mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
        for i, cont in enumerate(foreground_contours):
            if cont is None or len(cont) == 0:
                print(f"Warning: Empty contour at index {i}")
                continue

            if (
                cont[:, :, 0].max() >= W
                or cont[:, :, 1].max() >= H
                or cont[:, :, 0].min() < 0
                or cont[:, :, 1].min() < 0
            ):
                print(f"Warning: Contour {i} coordinates out of bounds!")
                continue

            # Fill the main tissue contour
            cv2.fillPoly(mask, [cont], 255)  # type: ignore

            # Remove holes if they exist
            if i < len(hole_contours) and hole_contours[i]:
                for hole in hole_contours[i]:  # type: ignore
                    cv2.fillPoly(mask, [hole], 0)  # type: ignore
        mask = mask.astype(np.bool)
        if not mask.any():
            mask[:, :] = True  # If no mask, return full mask

    return mask, img_rgb


def get_mask_path_by_wsi_path(wsi_path: Path, wsi_dir: Path, mask_dir: Path) -> Path:
    wsi_path, wsi_dir, mask_dir = (
        wsi_path.absolute(),
        wsi_dir.absolute(),
        mask_dir.absolute(),
    )
    rel_path = wsi_path.relative_to(wsi_dir)
    stitch_path_prefix = mask_dir / rel_path
    stitch_path_prefix = stitch_path_prefix.parent / rel_path.stem
    extensions = ["jpg", "jpeg", "png", "webp"]
    extensions += [ext.upper() for ext in extensions]
    stitch_paths = [
        stitch_path_prefix.parent / (rel_path.stem + f".{ext}") for ext in extensions
    ]
    stitch_paths += [
        stitch_path_prefix.parent / rel_path.stem / (rel_path.stem + f".{ext}")
        for ext in extensions
    ]
    ret = None
    for stitch_path in stitch_paths:
        if stitch_path.exists():
            ret = stitch_path
    if ret is None:
        raise FileNotFoundError(
            f"No mask for wsi '{wsi_path}' in mask dir '{mask_dir}' (candidates: {', '.join([str(p) for p in stitch_paths])})"
        )
    return ret


def read_mask(mask_path: Path) -> np.ndarray:
    img = Image.open(mask_path)
    w, h = img.size
    return np.asarray(img).reshape((h, w, -1)).max(-1) > 0


def read_mask_by_wsi_path(wsi_path: Path, wsi_dir: Path, mask_dir: Path) -> np.ndarray:
    wsi_path, wsi_dir, mask_dir = (
        wsi_path.absolute(),
        wsi_dir.absolute(),
        mask_dir.absolute(),
    )
    mask_path = get_mask_path_by_wsi_path(wsi_path, wsi_dir, mask_dir)
    return read_mask(mask_path)
