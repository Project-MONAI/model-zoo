# MONAI Model Zoo

MONAI Model Zoo hosts a collection of medical imaging models in the [MONAI Bundle](https://docs.monai.io/en/latest/bundle_intro.html) format.

## Model Storage
Github limits the size of files allowed in the repository (see [About size limits on GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)). Therefore, MONAI Model Zoo suggests to use Git Large File Storage (LFS) to store large files for any single file that is larger than **25MB**.
There are multiple ways to install Git LFS. For example, download [a suitable binary package](https://github.com/git-lfs/git-lfs/releases) and run `./install.sh`.
Please refer to the [official guide](https://github.com/git-lfs/git-lfs#getting-started) for more details about how to install and use it.

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
1. For any large files inside the bundle, please use [Git LFS commands](https://github.com/git-lfs/git-lfs/edit/main/README.md#example-usage) to track them properly.
1. Wait for reviews; if there are reviews, make point-to-point responses, make further code changes if needed.
1. If there are conflicts between the pull request branch and the dev branch, pull the changes from the dev and resolve the conflicts locally.
1. Reviewer and contributor may have discussions back and forth until all comments addressed.
1. Wait for the pull request to be merged.

[monai model zoo issue list]: https://github.com/yiheng-wang-nv/model-zoo/issues