# Template Classification Bundle

This bundle is meant to be an example of classification in 2D which you can copy and modify to create your own bundle.
It is only roughly trained for the synthetic data you can generate with [this notebook](./generate_data.ipynb)
so doesn't do anything useful on its own. The purpose is to demonstrate the base line for classification network bundles.

To use this bundle, copy the contents of the whole directory and change the definitions for network, data, transforms,
or whatever else you want for your own new classification bundle.

## Generating Demo Data

Run all the cells of [this notebook](./generate_data.ipynb) to generate training and test data. These will be 2D
nifti files containing volumes with randomly generated circle, triangle or rectangle. The classification task
is very easy so your network will train in minutes with the default configuration of values. A test
data directory will separately be created since the inference config is configured to apply the network to
every nifti file in a given directory with a certain pattern.

## Training

To train a new network the `train.yaml` script can be used alone with no other arguments (assume `BUNDLE` is the root
directory of the bundle):

```
python -m monai.bundle run --config_file configs/train.yaml
```

The training config includes a number of hyperparameters like `learning_rate` and `num_workers`. These control aspects
of how training operates in terms of how many processes to use, when to perform validation, when to save checkpoints,
and other things. Other aspects of the script can be modified on the command line so these aren't exhaustive but are a
guide to the kind of parameterisation that make sense for a bundle.

## Override the `train` config to execute multi-GPU training:

```
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m monai.bundle run --config_file "['configs/train.yaml','configs/multi_gpu_train.yaml']"
```

Please note that the distributed training-related options depend on the actual running environment; thus, users may need to remove `--standalone`, modify `--nnodes`, or do some other necessary changes according to the machine used. For more details, please refer to [pytorch's official tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).

## Override the `train` config to execute evaluation with the trained model:

```
python -m monai.bundle run --config_file "['configs/train.yaml','configs/evaluate.yaml']"
```

## Execute inference:

```
python -m monai.bundle run --config_file configs/inference.yaml
```

## Other Considerations

There is no `scripts` directory containing a valid Python module to be imported in your configs. This wasn't necessary
for this bundle but if you want to include custom code in a bundle please follow the bundle tutorials on how to do this.
