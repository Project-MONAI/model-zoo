import typing as t

import torch
import torchvision.transforms.functional as TF


def tile(x: torch.Tensor, size: int, pad_value: int | float | None = None):
    C, H, W = x.shape[-3:]

    pad_h = (size - H % size) % size
    pad_w = (size - W % size) % size
    if pad_h > 0 or pad_w > 0:
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), value=pad_value)

    nh, nw = x.size(-2) // size, x.size(-1) // size
    return (
        x.view(-1, C, nh, size, nw, size)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(-1, C, size, size)
    )


def small_tiles_to_large_tiles(
    small_tiles: torch.Tensor,
    width: int,
    large_tile_size: int,
    sampled_large_tiles_idx: list | torch.Tensor | None = None,
) -> torch.Tensor:

    has_channel = small_tiles.ndim == 4
    small_tile_size = small_tiles.size(-1)
    num_small_tiles = small_tiles.size(0)

    nw = width // small_tile_size
    nh = num_small_tiles // nw

    r = large_tile_size // small_tile_size

    num_large_tiles = (nh // r) * (nw // r)
    large_tile_indices = (
        range(num_large_tiles)
        if sampled_large_tiles_idx is None
        else sampled_large_tiles_idx
    )

    tiles = []
    for k in large_tile_indices:
        start_row = (k // (nw // r)) * r
        start_col = (k % (nw // r)) * r
        for i in range(start_row, start_row + r):
            for j in range(start_col, start_col + r):
                tiles.append(small_tiles[i * nw + j])

    stacked = torch.stack(tiles, dim=0).view(-1, r, r, *small_tiles.shape[1:])
    if has_channel:
        large_tiles = stacked.permute(0, 3, 1, 4, 2, 5).reshape(
            -1, small_tiles.size(1), large_tile_size, large_tile_size
        )
    else:
        large_tiles = stacked.permute(0, 1, 3, 2, 4).reshape(
            -1, large_tile_size, large_tile_size
        )
    return large_tiles


def small_tile_flags_to_large_tile_flags(
    small_tile_flags: torch.Tensor,
    width: int,
    small_tile_size: int,
    large_tile_size: int,
    aggregation: t.Literal["any", "all"] = "any",
):
    small_tile_flags = small_tile_flags.view(-1, 1, 1)
    num_small_tiles = small_tile_flags.size(0)
    nw = width // small_tile_size
    r = large_tile_size // small_tile_size
    num_large_tiles = num_small_tiles // r**2
    large_tile_flags = small_tiles_to_large_tiles(
        small_tile_flags,
        width=nw,
        large_tile_size=r,
    ).view(num_large_tiles, -1)
    return (
        large_tile_flags.any(-1) if aggregation == "any" else large_tile_flags.all(-1)
    )


def format_first_stg_act_as_second_stg_inp(
    x: torch.Tensor,
    height: int,
    width: int,
    small_tile_size: int,
    large_tile_size: int,
):
    assert height % small_tile_size == 0 and width % small_tile_size == 0
    D = x.size(1)
    nh, nw = height // small_tile_size, width // small_tile_size
    r = large_tile_size // small_tile_size
    x = x.view(-1, nh, nw, D)
    x = x.permute(0, 3, 1, 2).reshape(-1, D, nh // r, r, nw // r, r)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(-1, D, r, r)
    return x


def format_second_stg_inp_as_first_stg_act(
    x: torch.Tensor, height: int, width: int, small_tile_size: int, large_tile_size: int
):
    D = x.size(1)
    nh, nw = height // small_tile_size, width // small_tile_size
    r = large_tile_size // small_tile_size
    x = x.view(-1, nh // r, nw // r, D, r, r)
    x = x.permute(0, 3, 1, 4, 2, 5).reshape(-1, D, nh, nw)
    x = x.permute(0, 2, 3, 1).reshape(-1, D)
    return x


def format_second_stg_act_as_third_stg_inp(
    x: torch.Tensor,
    height: int,
    width: int,
    large_tile_size: int,
):
    D = x.size(1)
    nh = height // large_tile_size
    nw = width // large_tile_size
    return x.view(-1, nh, nw, D).permute(0, 3, 1, 2).contiguous()


def forward_with_batch_size_limit(
    net,
    x: torch.Tensor,
    batch_size_on_gpu: int,
    device: str | torch.device,
    out_device: str | torch.device,
    preproc_fn: t.Callable[[torch.Tensor], torch.Tensor] | None = None,
    dtype: torch.dtype = torch.float32,
):
    features = list()
    for start_idx in range(0, x.size(0), batch_size_on_gpu):
        end_idx = min(x.size(0), start_idx + batch_size_on_gpu)
        batch = x[start_idx:end_idx].to(device=device, non_blocking=True)
        batch = preproc_fn(batch) if preproc_fn else batch
        batch = batch.to(dtype=dtype, non_blocking=True)
        actual_bs = end_idx - start_idx
        batch = pad_to_batch(batch, batch_size_on_gpu)
        batch: torch.Tensor = forward_compiled(net, batch)
        # batch = net(batch)
        features.append(batch[:actual_bs].to(device=out_device, non_blocking=True))
    if torch.device(out_device).type == "cpu" and torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    return torch.cat(features)


@t.overload
def backward_with_batch_size_limit(
    net,
    x: torch.Tensor,
    grad: torch.Tensor,
    batch_size_on_gpu: int,
    device: str | torch.device,
    out_device: str | torch.device,
    dtype: torch.dtype,
    ret_grad: t.Literal[True],
) -> torch.Tensor: ...


@t.overload
def backward_with_batch_size_limit(
    net,
    x: torch.Tensor,
    grad: torch.Tensor,
    batch_size_on_gpu: int,
    device: str | torch.device,
    out_device: str | torch.device,
    dtype: torch.dtype,
    ret_grad: t.Literal[False],
) -> None: ...


def backward_with_batch_size_limit(
    net,
    x: torch.Tensor,
    grad: torch.Tensor,
    batch_size_on_gpu: int,
    device: str | torch.device,
    out_device: str | torch.device,
    dtype: torch.dtype,
    ret_grad: bool,
):
    assert x.size(0) == grad.size(0)

    grads = []
    total = x.size(0)
    for start in range(0, total, batch_size_on_gpu):
        end = min(total, start + batch_size_on_gpu)
        actual_bs = end - start

        batch = x[start:end].to(device=device, dtype=dtype, non_blocking=True)
        batch = pad_to_batch(batch, batch_size_on_gpu)
        if ret_grad:
            batch.requires_grad_(True)

        with torch.autocast(device_type="cuda", dtype=dtype):
            out = net(batch)
            # out = forward_compiled(net, batch)

        grad_batch = grad[start:end].to(device=device, dtype=dtype, non_blocking=True)
        grad_batch = pad_to_batch(grad_batch, batch_size_on_gpu)

        with torch._dynamo.utils.maybe_enable_compiled_autograd(
            True, fullgraph=True, dynamic=False
        ):
            out.backward(grad_batch)
        # out.backward(grad_batch)

        if ret_grad:
            assert batch.grad is not None
            grads.append(batch.grad[:actual_bs].to(out_device, non_blocking=True))

    if ret_grad:
        if (
            torch.device(out_device).type == "cpu"
            and torch.device(device).type == "cuda"
        ):
            torch.cuda.synchronize()
        return torch.cat(grads)


@torch.compile(fullgraph=True, dynamic=False)
def forward_compiled(net, x: torch.Tensor) -> torch.Tensor:
    return net(x)


def pad_to_batch(t: torch.Tensor, batch_size: int) -> torch.Tensor:
    assert (
        t.size(0) <= batch_size
    ), f"'{t.shape}' size tensor cannot be padded to be batch size of '{batch_size}'"
    pad = batch_size - t.size(0)
    return torch.cat([t, t.new_zeros((pad,) + t.shape[1:])], dim=0) if pad > 0 else t


def scale_and_normalize(x: torch.Tensor, inplace: bool = False):
    x = x.clamp_(0, 255) if inplace else x.clamp(0, 255)
    x = TF.normalize(
        x / 255, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=inplace
    )
    return x


def combine_tile_list(tile_list: list[torch.Tensor], ncols: int):
    """
    Combines a flat list of tile tensors (each with shape (C, H, W)) into one output tensor,
    arranging them in a grid with the specified number of columns. The tiles in the final row
    or column may have different sizes.

    Args:
        tile_list (list of torch.Tensor): A flat list of tile tensors, each with shape
                                           (channels, tile_height, tile_width). It is assumed
                                           that the number of channels is consistent across all tiles.
        ncols (int): Number of columns to arrange the tiles in.

    Returns:
        torch.Tensor: A tensor of shape (channels, total_height, total_width), where:
            - total_height is the sum of maximum tile heights in each row.
            - total_width is the sum of maximum tile widths in each column.
    """
    if not tile_list:
        raise ValueError("tile_list is empty")

    ntiles = len(tile_list)
    nrows = (ntiles + ncols - 1) // ncols  # Ceiling division to get the number of rows

    # Convert the flat tile list into a nested list (rows of tiles)
    nested_tiles = [tile_list[i * ncols : (i + 1) * ncols] for i in range(nrows)]

    # Compute the maximum tile height for each row
    row_heights = [max(tile.shape[1] for tile in row) for row in nested_tiles]

    # Compute the maximum tile width for each column (consider only rows that have a tile in that column)
    col_widths = []
    for col in range(ncols):
        max_width = 0
        for row in nested_tiles:
            if col < len(row):
                tile_w = row[col].shape[2]
                if tile_w > max_width:
                    max_width = tile_w
        col_widths.append(max_width)

    # Calculate the total output dimensions
    total_height = sum(row_heights)
    total_width = sum(col_widths)

    # Determine the number of channels from the first tile
    channels = tile_list[0].shape[0]

    # Preallocate the output tensor (this avoids repeated concatenation and extra memory copies)
    out_tensor = torch.zeros(
        channels,
        total_height,
        total_width,
        dtype=tile_list[0].dtype,
        device=tile_list[0].device,
    )

    # Place each tile in its proper location by calculating offsets
    y_offset = 0
    for i, row in enumerate(nested_tiles):
        x_offset = 0
        for j, tile in enumerate(row):
            tile_h, tile_w = tile.shape[1], tile.shape[2]
            out_tensor[
                :, y_offset : y_offset + tile_h, x_offset : x_offset + tile_w
            ] = tile
            x_offset += col_widths[j]
        y_offset += row_heights[i]

    return out_tensor
