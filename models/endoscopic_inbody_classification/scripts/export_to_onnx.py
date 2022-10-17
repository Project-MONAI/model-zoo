import argparse
import os

import numpy as np
import onnx
import onnxruntime
import torch
from monai.networks.nets import SEResNet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_export(modelname, outname, out_channels, height, width, multigpu=False, in_channels=3):
    """
    Loading a model by name.

    Args:
        modelname: a whole path name of the model that need to be loaded.
        outname: a name for output onnx model.
        out_channels: output channels, which usually equals to 1 + class_number.
        height: input images' height.
        width: input images' width.
        multigpu: if the pre-trained model trained on a multigpu environment.
        in_channels: input images' channel number.
    """
    isopen = os.path.exists(modelname)
    if not isopen:
        raise Exception("The specified model to load does not exist!")

    model = SEResNet50(spatial_dims=2, in_channels=in_channels, num_classes=out_channels)

    if multigpu:
        model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.load_state_dict(torch.load(modelname, map_location=device))  # if the model is trained on multi gpu
    model = model.eval()

    np.random.seed(0)
    x = np.random.random((1, 3, width, height))
    x = torch.tensor(x, dtype=torch.float32)
    x = x.cuda()
    torch_out = model(x)
    input_names = ["INPUT__0"]
    output_names = ["OUTPUT__0"]
    # Export the model
    if multigpu:
        model_trans = model.module
    else:
        model_trans = model
    torch.onnx.export(
        model_trans,  # model to save
        x,  # model input
        outname,  # model save path
        export_params=True,
        verbose=True,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        opset_version=15,
        dynamic_axes={"INPUT__0": {0: "batch_size"}, "OUTPUT__0": {0: "batch_size"}},
    )
    onnx_model = onnx.load(outname)
    onnx.checker.check_model(onnx_model, full_check=True)
    ort_session = onnxruntime.InferenceSession(outname)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(["OUTPUT__0"], ort_inputs)
    numpy_torch_out = to_numpy(torch_out)
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(numpy_torch_out, ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # the original model for converting.
    parser.add_argument(
        "--model",
        type=str,
        default=r"/workspace/bundle/endoscopic_inbody_classification/models/model.pt",
        help="Input an existing model weight",
    )

    # path to save the onnx model.
    parser.add_argument(
        "--outpath",
        type=str,
        default=r"/workspace/bundle/endoscopic_inbody_classification/models/model.onnx",
        help="A path to save the onnx model.",
    )

    parser.add_argument("--width", type=int, default=256, help="Width for exporting onnx model.")

    parser.add_argument("--height", type=int, default=256, help="Height for exporting onnx model.")

    parser.add_argument(
        "--out_channels", type=int, default=2, help="Number of expected out_channels in model for exporting to onnx."
    )

    parser.add_argument("--multigpu", type=bool, default=False, help="If loading model trained with multi gpu.")

    args = parser.parse_args()
    modelname = args.model
    outname = args.outpath
    out_channels = args.out_channels
    height = args.height
    width = args.width
    multigpu = args.multigpu

    if os.path.exists(outname):
        raise Exception(
            "The specified outpath already exists! Change the outpath to avoid overwriting your saved model. "
        )
    model = load_model_and_export(modelname, outname, out_channels, height, width, multigpu)
