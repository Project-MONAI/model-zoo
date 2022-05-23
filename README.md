# MONAI Model Zoo

MONAI Model Zoo hosts a collection of medical imaging models in the [MONAI Bundle](https://docs.monai.io/en/latest/bundle_intro.html) format.
All source code of models (bundles) are tracked in `models/`, and for each distinct version of a bundle, it will be archived as a `.zip` file (named in the form of `bundle_name_version.zip`) and stored in `Releases`.

## Model naming

The name of a bundle is suggested to contain its characteristics, such as include the task type and data type. The following are some of the examples:

```
spleen_deepedit_annotation
spleen_ct_segmentation
chest_xray_classification
pathology_metastasis_detection
```

## Model storage

Github limits the size of files allowed in the repository (see [About size limits on GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)). Therefore, MONAI Model Zoo limits each single file to be no larger than **25MB**.

### Prepare a config file for large files

If a bunlde has large files, please upload those files into a publicly accessible source, and provide a config file called `large_file.yml` (or `.yaml`, `.json`) that contains the corresponding download links. During the pull request, only the config file should be included (large files should be excluded). Please put the config file within the root directory of the bundle, and it should contain the following information:

1. `path`, relative path of the large file in the bundle.
2. `url`, URL link that can download the file.
3. `hash_val`, (**optional**) expected hash value of the file.
4. `hash_type`, (**optional**) hash type. Supprted hash type includes "md5", "sha1", "sha256" and "sha512".

The template is as follow:
```
large_files:
  - path: "models/large-file-1.pt"
    url: "url-of-large-file-1.pt"
    hash_val: ""
    hash_type: ""
  - path: "models/large-file-2.ts"
    url: "url-of-large-file-2.ts"
```

### Validate and release

As for a pull request, a CI program will try to download all large files if mentioned and do several validations. If the pull request is approved and merged, the full bundle (with all large files if exists) will be archived and send to [Releases](https://github.com/Project-MONAI/model-zoo/releases).

## Contributing

To make a contribution in MONAI Model Zoo, please follow the contribution processes.

### Preparing a bundle

Please refer to [MONAI Bundle Specification](https://docs.monai.io/en/latest/mb_specification.html#monai-bundle-specification) to prepare your bundle.

### Submitting pull requests

All code changes to the dev branch must be done via [pull requests](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/proposing-changes-to-your-work-with-pull-requests).
1. Please create a new ticket from [the issue list][monai model zoo issue list].
1. [create a new branch in your fork](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork)
of the codebase named `[ticket_id]-[task_name]`.
Ideally, the new branch should be based on the latest `dev` branch.
1. Make changes to the branch ([use detailed commit messages if possible](https://chris.beams.io/posts/git-commit/)).
1. [Create a new pull request](https://help.github.com/en/desktop/contributing-to-projects/creating-a-pull-request) from the task branch to the dev branch, with detailed descriptions of the purpose of this pull request.
1. For any large files inside the bundle, please exclude them and provide the download links instead. Please follow the instructions mentioned above to prepare the necessary `large_file.yml`.
1. Wait for reviews; if there are reviews, make point-to-point responses, make further code changes if needed.
1. If there are conflicts between the pull request branch and the dev branch, pull the changes from the dev and resolve the conflicts locally.
1. Reviewer and contributor may have discussions back and forth until all comments addressed.
1. Wait for the pull request to be merged.

[monai model zoo issue list]: https://github.com/Project-MONAI/model-zoo/issues
