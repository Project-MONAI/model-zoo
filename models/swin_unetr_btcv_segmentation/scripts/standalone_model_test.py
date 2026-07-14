import os
import torch
import numpy as np
import nibabel as nib
import pydicom
from pathlib import Path
from glob import glob
import SimpleITK as sitk
from monai.transforms import (
    Compose,
    ScaleIntensityRange,
    Spacing,
    Orientation,
    EnsureChannelFirst,
    CropForeground
)

# Paths
input_dir = "input/patient1/study1/series1" ## Please supply input data.
model_path = "../models/model.ts"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Load the traced model on CPU to avoid CUDA requirements
model = torch.jit.load(model_path, map_location=torch.device('cpu'))
model.eval()

# Check file types
files = glob(os.path.join(input_dir, "*"))

# Determine file types and load accordingly
if len(files) > 0:
    # For multiple DICOM files (one per slice)
    if files[0].endswith('.dcm') or len(files) > 10:  # Assume multiple files is a DICOM series
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(input_dir)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        image_array = sitk.GetArrayFromImage(image)

        # Get spacing information from the DICOM
        spacing = image.GetSpacing()
    else:
        # For NIfTI or other formats
        image = nib.load(files[0])
        image_array = image.get_fdata()
        # NIfTI is typically (x, y, z), so transpose to (z, y, x) for MONAI
        image_array = np.transpose(image_array, (2, 1, 0))

    # Handling different dimensionality cases
    if len(image_array.shape) == 3:
        z, y, x = image_array.shape

        # Check if we have a single slice (or very few slices)
        if z == 1:
            image_array = np.repeat(image_array, 96, axis=0)  # Repeat along z to get desired depth

            # Add channel dimension for MONAI: (C, Z, Y, X)
            image_array = np.expand_dims(image_array, 0)
            image_tensor = torch.from_numpy(image_array).float()
        else:
            # Regular 3D data - add channel dimension: (C, Z, Y, X)
            image_array = np.expand_dims(image_array, 0)
            image_tensor = torch.from_numpy(image_array).float()
    else:
        # Already has channel dimension or other unusual shape
        image_tensor = torch.from_numpy(image_array).float()


    try:
        # Skip the EnsureChannelFirst transform as tensor already has channel dimension first

        # Apply Spacing
        # Doesn't work for 2D, only 3d
        if len(image_tensor.shape) >= 4:  # For tensors with at least 4 dimensions (C, Z, Y, X)
            transform = Spacing(pixdim=(1.5, 1.5, 2.0), mode="bilinear")
            image_tensor = transform(image_tensor)

        # Apply Orientation for 3d
        if len(image_tensor.shape) >= 4:
            transform = Orientation(axcodes="RAS")
            image_tensor = transform(image_tensor)

        # Scale Intensity - works for both 2d & 3d
        transform = ScaleIntensityRange(a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True)
        image_tensor = transform(image_tensor)

        # Crop Foreground - use allow_smaller=False to prevent dimension issues
        transform = CropForeground(select_fn=lambda x: x > 0, margin=0, allow_smaller=False)
        image_tensor = transform(image_tensor)

        # Add batch dimension
        if len(image_tensor.shape) == 3:  # 2d case: (C, H, W)
            image_tensor = image_tensor.unsqueeze(0)  # Add batch: (B, C, H, W)
        elif len(image_tensor.shape) == 4:  # 3d case: (C, D, H, W)
            image_tensor = image_tensor.unsqueeze(0)  # Add batch: (B, C, D, H, W)

        # Check tensor shape against model requirements
        expected_size = (96, 96, 96)

        # Center crop or pad to match expected dimensions
        def center_crop_or_pad(tensor, target_size):
            # Get current spatial dimensions (skip batch and channel)
            current_size = tensor.shape[2:]

            # Create padded tensor with target size
            if len(current_size) == 2:  # 2d case
                # For 2d, we'd need to handle differently or convert to 3d
                raise ValueError("2D input not supported for 3D model")
            elif len(current_size) == 3:  # 3d case
                d, h, w = current_size
                td, th, tw = target_size

                # Calculate start/end indices for cropping/padding
                d_start = max(0, (d - td) // 2)
                d_end = min(d, d_start + td)
                h_start = max(0, (h - th) // 2)
                h_end = min(h, h_start + th)
                w_start = max(0, (w - tw) // 2)
                w_end = min(w, w_start + tw)

                # Crop
                result = tensor[:, :, d_start:d_end, h_start:h_end, w_start:w_end]

                # Pad if necessary
                pad_d = max(0, td - (d_end - d_start))
                pad_h = max(0, th - (h_end - h_start))
                pad_w = max(0, tw - (w_end - w_start))

                if pad_d > 0 or pad_h > 0 or pad_w > 0:
                    pad_d_before = pad_d // 2
                    pad_d_after = pad_d - pad_d_before
                    pad_h_before = pad_h // 2
                    pad_h_after = pad_h - pad_h_before
                    pad_w_before = pad_w // 2
                    pad_w_after = pad_w - pad_w_before

                    padding = (pad_w_before, pad_w_after,
                               pad_h_before, pad_h_after,
                               pad_d_before, pad_d_after,
                               0, 0)

                    result = torch.nn.functional.pad(result, padding)

                return result

        # Only resize if the shape doesn't match expected
        spatial_dims = image_tensor.shape[2:]
        if spatial_dims != expected_size:
            image_tensor = center_crop_or_pad(image_tensor, expected_size)

        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)

        # Post-process
        output_array = outputs[0].argmax(dim=0).numpy().astype(np.uint8)

        # Save output
        output_nifti = nib.Nifti1Image(output_array, np.eye(4))
        output_path = os.path.join(output_dir, "segmentation.nii.gz")
        nib.save(output_nifti, output_path)

    except Exception as e:
        import traceback
        traceback.print_exc()
else:
    print(f"No files found in {input_dir}")
