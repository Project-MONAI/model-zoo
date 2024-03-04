# MONAI Model Zoo

MONAI Model Zoo hosts a collection of medical imaging models in the [MONAI Bundle](https://docs.monai.io/en/latest/bundle_intro.html) format.
All source code of models (bundles) are tracked in `models/`, and for each distinct version of a bundle, it will be archived as a `.zip` file (named in the form of `bundle_name_version.zip`) and stored in `Releases`.

## Getting Started

To browse the available models, please see https://monai.io/model-zoo.

A basic example to download and extract a bundle is:

```bash
pip install "monai[fire]"
python -m monai.bundle download "spleen_ct_segmentation" --bundle_dir "bundles/"
```

- The commands will download `spleen_ct_segmentation` to the current directory's `bundles/` subdirectory.
- For more downloading options, please run `python -m monai.bundle download -h`
- For the specific usage of a bundle, please refer to its `docs` folder, for example, `bundles/spleen_ct_segmentation/docs`.

To get started with the models, please see [the example use cases](https://github.com/Project-MONAI/tutorials/tree/main/model_zoo).

## Template Bundles

We aim to provide a number of template bundles in the zoo for you to copy and adapt to your own needs.
This should help you reduce effort in developing your own bundles and also demonstrate what we feel to be good practice and design.
We currently have the following:

 * [Segmentation Template](./models/segmentation_template)

## License

Bundles released on the MONAI Model Zoo require a license for the software itself comprising the configuration files and model weights. You are required to adhere to the license conditions included with each bundle, as well as any license conditions stated for data bundles may include or use (please check the file `docs/data_license.txt` if it is existing within the bundle directory).

The MONAI Model Zoo repository itself follows [Apache License](https://github.com/Project-MONAI/model-zoo/blob/dev/LICENSE), thus you need to follow this license if using its software, and we recommend this license for bundles.

The MONAI Model Zoo does not make any statement of the suitability of any model for a particular task, especially not for therapeutic or diagnostic use.

## Contributing

To make a contribution in MONAI Model Zoo, see the [contributing guidelines](https://github.com/Project-MONAI/model-zoo/blob/dev/CONTRIBUTING.md).

## Links
- The models are currently hosted at https://github.com/Project-MONAI/model-zoo/releases/tag/hosting_storage_v1
- MONAI Bundle API tutorials: https://github.com/Project-MONAI/tutorials/tree/main/bundle
- MONAI Bundle demo: https://github.com/Project-MONAI/tutorials/tree/main/model_zoo
- MONAI model zoo browser: https://monai.io/model-zoo.html
