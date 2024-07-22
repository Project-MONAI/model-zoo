import os
import json
import sys
import shutil
import argparse
import subprocess
from monai.bundle import create_workflow, ConfigParser
from monai.bundle.scripts import verify_metadata
from download_latest_bundle import download_latest_bundle

def find_bundle_file(root_dir: str, file: str, suffix=("json", "yaml", "yml")):
    # find bundle file with possible suffix
    file_name = None
    for name in suffix:
        full_name = f"{file}.{name}"
        if full_name in os.listdir(root_dir):
            file_name = full_name

    return file_name

def download_properties(url, repopath):
    """Download properties from url to filepath."""
    if url is None:
        raise ValueError("URL is required to download apps repo.")
    # clean the repo path
    if os.path.exists(repopath):
        shutil.rmtree(repopath)
    os.makedirs(repopath, exist_ok=True)
    print(f"Downloading apps repo from {url} to {repopath}")

    # TODO: remove specific branch when properties.py has been merged into main branch
    try:
        subprocess.run(["git", "clone", "-b", "nvcf_dev", url, repopath], check=True)
        print(f"Successfully cloned {url} to {repopath}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone {url}: {e}")

def check_apps_bundle_properties(bundle_root):
    """Check apps bundle properties"""
    inference_file = find_bundle_file(os.path.join(bundle_root, "configs"), "inference")
    sys.path = [bundle_root] + sys.path
    override = {"bundle_root": bundle_root}

    workflow = create_workflow(
    workflow_type="infer",
    config_file=os.path.join(bundle_root, "configs", inference_file),
    meta_file=os.path.join(bundle_root, "configs/metadata.json"),
    logging_file=os.path.join(bundle_root, "configs/logging.conf"),
    **override,
    )
    # update apps bundle properties to workflow properties
    from properties import InferProperties, MetaProperties
    workflow.properties = {**MetaProperties, **InferProperties}

    ret = workflow.check_properties()
    if len(ret)>0:
        raise ValueError(f"config file does not contain {ret}")

def get_apps_url(filepath):
    """get apps properties url"""
    if not os.path.isfile(filepath):
        raise FileExistsError(f"{filepath} is not a file.")
    else:
        with open(filepath, 'r') as file:
            apps_url_data = json.load(file)
        return apps_url_data

def verify(models_path="models", bundle="", download_path="download"):
    print(f"start verifying {bundle}:")
    # add bundle path to ensure custom code can be used
    cur_path = os.path.abspath(os.path.dirname(__file__))
    model_root = os.path.dirname(cur_path)
    models_path = os.path.join(model_root, models_path)
    download_latest_bundle(bundle_name=bundle, models_path=models_path, download_path=download_path)

    bundle_root = os.path.join(download_path, bundle)

    # verify bundle schema first
    schema_path = os.path.join(bundle_root, "configs/schema.json")
    meta_path = os.path.join(bundle_root, "configs/metadata.json")
    verify_metadata(meta_file=meta_path, filepath=schema_path)

    # get bundle metadata
    metadata = ConfigParser.load_config_file(meta_path)

    # TODO: remove this line after bundle declare support_apps metadata.json
    metadata["support_apps"] = {"Test": "0.0.1"}
    support_apps = metadata.get("support_apps", {})

    # get bundle's supported apps' properties url
    # TODO: maintain apps properties url in apps.json after each team define properties
    # now just a test properties url
    apps_properties_url = get_apps_url(os.path.join(model_root, "ci/apps.json"))


    for app, version in support_apps.items():
        if app in apps_properties_url.keys():
            url = apps_properties_url[app][version]
            download_properties(url, os.path.join(bundle_root, app))
            shutil.copy(os.path.join(bundle_root, app, "properties.py"), os.path.join(bundle_root, "properties.py"))
            check_apps_bundle_properties(bundle_root)
            print(f"{bundle} has valid {app} properties.")
        else:
            print(f"{app} is not supported in this bundle.")

    shutil.rmtree(download_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='test bundle prooerties')
    parser.add_argument('--models', type=str, help='path of models', default="models")
    parser.add_argument('--bundle_name', type=str, help='bundle name', default="spleen_ct_segmentation")
    args = parser.parse_args()
    verify(args.models, args.bundle_name)