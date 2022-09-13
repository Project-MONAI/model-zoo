# Description
A pre-trained model for the surgical tool segmentation task.

# Model Overview
This model is trained using the flexible unet structure. The backbone is efficient-b0 [1]. The decoder uses UNet architecture [2]. The training dataset uses the 3217 samples from first batch images and 
3500 samples from second batch images. The test dataset contains 664 samples from first batch of images.
The [pytorch model](https://drive.google.com/file/d/1HJMLsswcUz2pf0LPKSGEWUQzi_PewufA/view?usp=sharing), [torchscript model](https://drive.google.com/file/d/11XlGZfd6S4jug8fCJ709lv8hLeUNIW_r/view?usp=sharing) and [onnx_model](https://drive.google.com/file/d/165ZB1-PLi17yL8KOhIkDNoSsdcsfvlAB/view?usp=sharing) are shared in google drive. 

## Data
The whole dataset is from Activ-Surgical and the share path is \\\\dc2-cdot77-corp01\Activ_Surgical.

## Training configuration
The training was performed with an at least 12GB-memory GPU.

Actual Model Input: 736 x 480 x 3

## Input and output formats
Input: 3 channel video frames

Output: 2 channels: Label 1: tools; Label 0: everything else

## Scores
This model achieves the following IoU score on the test dataset (our own split from the first batch data):

Mean IoU = 0.87

## commands example
Execute training:

```
python -m monai.bundle run training --meta_file configs/metadata.json --config_file configs/train.json --logging_file configs/logging.conf
```

Override the `train` config to execute evaluation with the trained model:

```
python -m monai.bundle run evaluating --meta_file configs/metadata.json --config_file "['configs/train.json','configs/evaluate.json']" --logging_file configs/logging.conf
```

Execute inference:

```
python -m monai.bundle run evaluating --meta_file configs/metadata.json --config_file configs/inference.json --logging_file configs/logging.conf
```

Export checkpoint to TorchScript file:

```
python -m monai.bundle ckpt_export network_def --filepath models/model.ts --ckpt_file models/model.pt --meta_file configs/metadata.json --config_file configs/inference.json
```

Export checkpoint to onnx file, which has been tested on pytorch 1.12.0:

```
python scripts/export_to_onnx.py --model models/model.pt --outpath models/model.onnx
```

Export TorchScript file to a torchscript module targeting a TensorRT engine with float16 precision. 

```
torchtrtc -p f16 models/model.ts models/model_trt.ts "[(1,3,736,480);(4,3,736,480);(8,3,736,480)]"
```
The last parameter is the dynamic input shape in which each parameter means "[(MIN_BATCH, MIN_CHANNEL, MIN_WIDTH, MIN_HEIGHT), (OPT_BATCH, .., ..., OPT_HEIGHT), (MAX_BATCH, .., ..., MAX_HEIGHT)]". Please notice if using docker, the TensorRT CUDA must match the environment CUDA and the Torch-TensorRT c++&python version must be installed. For more examples on how to use the Torch-TensorRT, you can go to this [link](https://pytorch.org/TensorRT/). The [github source code link](https://github.com/pytorch/TensorRT) here shows the detail about how to install it on your own environment.

Export TensorRT float16 model from the onnx model:

```
trtexec --onnx=models/model.onnx --saveEngine=models/model.trt --fp16 --minShapes=INPUT__0:1x3x736x480 --optShapes=INPUT__0:4x3x736x480 --maxShapes=INPUT__0:8x3x736x480 --shapes=INPUT__0:4x3x736x480
```
This command need TensorRT with correct CUDA installed in the environment. For the detail of installing TensorRT, please refer to [this link](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html). In addition, there are padding operations in this FlexibleUNet structure that not support by TensorRT. Therefore, when tried to convert the onnx model to a TensorRT engine, an extra step shown below is needed to execute. The "new_model.onnx" should be the "model.onnx" used to do converting.

```
polygraphy surgeon sanitize --fold-constants models/model.onnx -o models/new_model.onnx
```

# References
[1] Tan, M. and Le, Q. V. Efficientnet: Rethinking model scaling for convolutional neural networks. ICML, 2019a. https://arxiv.org/pdf/1905.11946.pdf

[2] O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer, 2015. https://arxiv.org/pdf/1505.04597.pdf
