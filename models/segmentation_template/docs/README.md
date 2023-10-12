
# Template Segmentation Bundle

This bundle is meant to be an example of segmentation in 3D which you can copy and modify to create your own bundle.
It is only roughly trained for the synthetic data you can generate with [this notebook](./generate_data.ipynb)
so doesn't do anything useful on its own. The purpose is to demonstrate the base line for segmentation network
bundles compatible with MONAILabel amongst other things.

To use this bundle, copy the contents of the whole directory and change the definitions for network, data, transforms,
or whatever else you want for your own new segmentation bundle. Some of the names are critical for MONAILable but
otherwise you're free to change just about whatever else is defined here to suit your network.

This bundle should also demonstrate good practice and design, however there is one caveat about definitions being
copied between config files. Ideally there should be a `common.yaml` file for all the definitions used by every other
config file which is then included with that file. MONAILabel doesn't support this yet so this bundle will be updated
once it does to exemplify this better practice.

## Generating Demo Data

Run all the cells of [this notebook](./generate_data.ipynb) to generate training and test data. These will be 3D
nifti files containing volumes with randomly generated spheres of varying intensities and some noise for fun. The
segmentation task is very easy so your network will train in minutes with the default configuration of values. A test
data directory will separately be created since the test and inference configs are configured to apply the network to
every nifti file in a given directory with a certain pattern.

## Training

To train a new network the `train.yaml` script can be used alone with no other arguments (assume `BUNDLE` is the root
directory of the bundle):

```sh
python -m monai.bundle run \
    --meta_file "$BUNDLE/configs/metadata.json" \
    --config_file "$BUNDLE/configs/train.yaml" \
    --bundle_root "$BUNDLE"
```

A `train.sh` script is also provided in `docs` which implements this invocation with some helper commands. It
relies on a Conda environment called `monai` so comment or modify those lines if you're not using such an environment.
See MONAI installation information about what environment to create for the features you want.

The training config includes a number of hyperparameters like `learning_rate` and `num_workers`. These control aspects
of how training operates in terms of how many processes to use, when to perform validation, when to save checkpoints,
and other things. Other aspects of the script can be modified on the command line so these aren't exhaustive but are a
guide to the kind of parameterisation that make sense for a bundle.

## Testing and Inference

Two configs are provided (`test.yaml` and `inference.yaml`) for doing post-training inference with the model. The first
requires image and segmentation pairs which are used with network outputs to assess performance using metrics. This is
very similar to training validation but is done on separate images. This config can be set to save predicted segmentations
by setting `save_pred` to true but by default it will just run metrics and print their results.

The inference config is for generating new segmentations from images which don't have ground truths, so this is used for
actually applying the network in practice. This will apply the network to every image in an input directory matching a
pattern and save the predicted segmentations to an output directory.

Using inference on the command line is demonstrated in [this notebook](./visualise_inference.ipynb) with visualisation.
Some explanation of some command line choices are given in the notebook as well, similar command line invocations can
also be done with the included `inference.sh` script file.

## Other Considerations

There is no `scripts` directory containing a valid Python module to be imported in your configs. This wasn't necessary
for this bundle but if you want to include custom code in a bundle please follow the bundle tutorials on how to do this.

The `multi_gpu_train.yaml` config is defined as a "mixin" to implement DDP based multi-gpu training. The script
`train_multigpu.sh` illustrates an example of how to invoke these configs together with `torchrun`.

The `inference.yaml` config is compatible with MONAILabel such that you can load one of the synthetic images and perform
inference through a label server. This doesn't permit active learning however, that is a later enhancement for this
bundle. If you're changing definitions in the `inference.yaml` config file be careful about changing names and consult
the MONAILabel documentation about required definition names. An example script to start a server is given in
`run_monailabel.sh` which will download the bundle application and "install" this bundle using a symlink then start
the server. Future updates to MONAILabel will improve this process.
