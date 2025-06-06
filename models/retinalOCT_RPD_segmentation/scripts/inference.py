import json
import logging
import os
import pickle

import pandas as pd
import progressbar
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model

from .analysis_lib import CreatePlotsRPD, EvaluateClass, OutputVis, grab_dataset
from .datasets import data
from .Ensembler import Ensembler
from .table_styles import styles

# Change directory to the script's location to ensure relative paths work correctly.
os.chdir(os.path.dirname(os.path.abspath(__file__)))


logging.basicConfig(level=logging.INFO)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dpi = 120


class MyProgressBar:
    # https://stackoverflow.com/a/53643011/3826929
    # George C
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


def create_dataset(dataset_name, extracted_path):
    """Creates a pickled dataset file from a directory of extracted images.

    This function scans the `extracted_path` for images, formats them into a
    list of dictionaries compatible with Detectron2, and saves the list as a
    pickle file.

    Args:
        dataset_name (str): The name for the dataset, used for the output .pk file.
        extracted_path (str): The directory containing the extracted image files.
    """
    stored_data = data.rpd_data(extracted_path)
    pickle.dump(stored_data, open(os.path.join(data.script_dir, f"{dataset_name}.pk"), "wb"))


def configure_model():
    """Loads and returns the model configuration from a YAML file.

    It reads a 'working.yaml' file located in the same directory as the script
    to set up the Detectron2 configuration.

    Returns:
        detectron2.config.CfgNode: The configuration object for the model.
    """
    cfg = get_cfg()
    moddir = os.path.dirname(os.path.realpath(__file__))
    name = "working.yaml"
    cfg_path = os.path.join(moddir, name)
    cfg.merge_from_file(cfg_path)
    return cfg


def register_dataset(dataset_name):
    """Registers a dataset with Detectron2's DatasetCatalog.

    This makes the dataset available to be loaded by Detectron2's data loaders.
    It sets the class metadata to 'rpd'.

    Args:
        dataset_name (str): The name under which to register the dataset.
    """
    for name in [dataset_name]:
        try:
            DatasetCatalog.register(name, grab_dataset(name))
        except AssertionError as e:
            print(f"Assertion failed: {e}. Already registered.")
        MetadataCatalog.get(name).thing_classes = ["rpd"]


def run_prediction(cfg, dataset_name, output_path):
    """Runs inference on a dataset using a cross-validation ensemble of models.

    It loads five different model weight files (fold1 to fold5), runs inference
    for each model on the specified dataset, and saves the predictions in
    separate subdirectories within `output_path`.

    Args:
        cfg (CfgNode): The model configuration object.
        dataset_name (str): The name of the registered dataset to run inference on.
        output_path (str): The base directory to save prediction outputs.
    """
    model = build_model(cfg)  # returns a torch.nn.Module
    myloader = build_detection_test_loader(cfg, dataset_name)
    myeval = COCOEvaluator(
        dataset_name, tasks={"bbox", "segm"}, output_dir=output_path
    )  # produces _coco_format.json when initialized
    for mdl in ("fold1", "fold2", "fold3", "fold4", "fold5"):
        extract_directory = "../models"
        file_name = mdl + "_model_final.pth"
        model_weights_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), extract_directory, file_name)
        print(model_weights_path)
        DetectionCheckpointer(model).load(model_weights_path)  # load a file, usually from cfg.MODEL.WEIGHTS
        model.eval()  # set model in evaluation mode
        myeval.reset()
        output_dir = os.path.join(output_path, mdl)
        myeval._output_dir = output_dir
        print("Running inference with model ", mdl)
        _ = inference_on_dataset(
            model, myloader, myeval
        )  # produces coco_instance_results.json when myeval.evaluate is called
    print("Done with predictions!")


def run_ensemble(dataset_name, output_path, iou_thresh=0.2):
    """Ensembles predictions from multiple models using NMS.

    It initializes an `Ensembler`, runs the non-maximum suppression logic, and
    saves the final combined predictions to a single COCO results file.

    Args:
        dataset_name (str): The name of the dataset.
        output_path (str): The base directory containing the individual model
            prediction subdirectories.
        iou_thresh (float, optional): The IoU threshold for ensembling. Defaults to 0.2.

    Returns:
        Ensembler: The ensembler instance after running NMS.
    """
    ens = Ensembler(output_path, dataset_name, ["fold1", "fold2", "fold3", "fold4", "fold5"], iou_thresh=iou_thresh)
    ens.mean_score_nms()
    ens.save_coco_instances()
    return ens


def evaluate_dataset(dataset_name, output_path, iou_thresh=0.2, prob_thresh=0.5):
    """Evaluates the final ensembled predictions against ground truth.

    It uses the custom `EvaluateClass` to calculate performance metrics and saves
    a summary to a JSON file.

    Args:
        dataset_name (str): The name of the dataset.
        output_path (str): The directory containing the ensembled predictions file.
        iou_thresh (float, optional): The IoU threshold for evaluation. Defaults to 0.2.
        prob_thresh (float, optional): The probability threshold for evaluation. Defaults to 0.5.

    Returns:
        EvaluateClass: The evaluation object containing detailed metrics.
    """
    myeval = EvaluateClass(dataset_name, output_path, iou_thresh=iou_thresh, prob_thresh=prob_thresh, evalsuper=False)
    myeval.evaluate()
    with open(os.path.join(output_path, "scalar_dict.json"), "w") as outfile:
        json.dump(obj=myeval.summarize_scalars(), fp=outfile)
    return myeval


def create_table(myeval):
    """Creates a DataFrame of per-image statistics from evaluation results.

    Args:
        myeval (EvaluateClass): The evaluation object containing COCO results.

    Returns:
        CreatePlotsRPD: An object containing DataFrames for image and volume stats.
    """
    dataset_table = CreatePlotsRPD.initfromcoco(myeval.mycoco, myeval.prob_thresh)
    dataset_table.dfimg.sort_index(inplace=True)
    return dataset_table
    # dataset_table.dfimg['scan'] = dataset_table.dfimg['scan'].astype('int') #depends on what we want scan field to be


def output_vol_predictions(dataset_table, vis, volid, output_path, output_mode="pred_overlay"):
    """Generates and saves visualization TIFFs for a single scan volume.

    Args:
        dataset_table (CreatePlotsRPD): Object containing the image/volume stats.
        vis (OutputVis): The visualization object.
        volid (str): The ID of the volume to visualize.
        output_path (str): The directory to save the output TIFF file.
        output_mode (str, optional): The type of visualization to create.
            Options: "pred_overlay", "pred_only", "originals", "all".
            Defaults to "pred_overlay".
    """
    dfimg = dataset_table.dfimg
    imgids = dfimg[dfimg["volID"] == volid].sort_index().index.values
    outname = os.path.join(output_path, f"{volid}.tiff")
    if output_mode == "pred_overlay":
        vis.output_pred_to_tiff(imgids, outname, pred_only=False)
    elif output_mode == "pred_only":
        vis.output_pred_to_tiff(imgids, outname, pred_only=True)
    elif output_mode == "originals":
        vis.output_ori_to_tiff(imgids, outname)
    elif output_mode == "all":
        vis.output_all_to_tiff(imgids, outname)
    else:
        print(f"Invalid mode {output_mode} for function output_vol_predictions.")


def output_dataset_predictions(dataset_table, vis, output_path, output_mode="pred_overlay", draw_mode="default"):
    """Generates and saves visualization TIFFs for all volumes in a dataset.

    Args:
        dataset_table (CreatePlotsRPD): Object containing the image/volume stats.
        vis (OutputVis): The visualization object.
        output_path (str): The base directory to save the output TIFF files.
        output_mode (str, optional): The type of visualization to create.
            Defaults to "pred_overlay".
        draw_mode (str, optional): The drawing style ("default" or "bw").
            Defaults to "default".
    """
    vis.set_draw_mode(draw_mode)
    os.makedirs(output_path, exist_ok=True)
    for volid in dataset_table.dfvol.index:
        output_vol_predictions(dataset_table, vis, volid, output_path, output_mode)


def create_dfvol(dataset_name, output_path, dataset_table):
    """Creates and saves a styled HTML table of volume-level statistics.

    Args:
        dataset_name (str): The name of the dataset.
        output_path (str): The directory to save the HTML file.
        dataset_table (CreatePlotsRPD): Object containing the volume DataFrame.
    """
    dfvol = dataset_table.dfvol.sort_values(by=["dt_instances"], ascending=False)
    with pd.option_context("styler.render.max_elements", int(dfvol.size) + 1):
        html_str = dfvol.style.format("{:.0f}").set_table_styles(styles).to_html()
    html_file = open(os.path.join(output_path, "dfvol_" + dataset_name + ".html"), "w")
    html_file.write(html_str)
    html_file.close()


def create_dfimg(dataset_name, output_path, dataset_table):
    """Creates and saves a styled HTML table of image-level statistics.

    Args:
        dataset_name (str): The name of the dataset.
        output_path (str): The directory to save the HTML file.
        dataset_table (CreatePlotsRPD): Object containing the image DataFrame.
    """
    dfimg = dataset_table.dfimg.sort_index()
    with pd.option_context("styler.render.max_elements", int(dfimg.size) + 1):
        html_str = dfimg.style.set_table_styles(styles).to_html()
    html_file = open(os.path.join(output_path, "dfimg_" + dataset_name + ".html"), "w")
    html_file.write(html_str)
    html_file.close()


def main(args):
    """Main function to orchestrate the end-to-end analysis pipeline.

    This function controls the flow from data extraction to evaluation and
    visualization based on the provided arguments.

    Args:
        args (dict): A dictionary of command-line arguments and flags
            controlling the pipeline execution.
    """
    print(f"Received arguments: {args}")

    # Unpack arguments from the dictionary with default values
    dataset_name = args.get("dataset_name")
    input_dir = args.get("input_dir")
    extracted_dir = args.get("extracted_dir")
    input_format = args.get("input_format")
    output_dir = args.get("output_dir")
    run_extract = args.get("run_extract", True)
    make_dataset = args.get("create_dataset", True)
    run_inference = args.get("run_inference", True)
    prob_thresh = args.get("prob_thresh", 0.5)
    iou_thresh = args.get("iou_thresh", 0.2)
    create_tables = args.get("create_tables", True)

    # Visualization flags
    bm = args.get("binary_mask", False)
    bmo = args.get("binary_mask_overlay", False)
    imo = args.get("instance_mask_overlay", False)
    make_visuals = bm or bmo or imo

    # --- Pipeline Steps ---
    if run_extract:
        os.makedirs(extracted_dir, exist_ok=True)
        print("Starting file extraction...")
        data.extract_files(input_dir, extracted_dir, input_format)
        print("Image extraction complete!")
    if make_dataset:
        print("Creating dataset from extracted images...")
        create_dataset(dataset_name, extracted_dir)
    if run_inference:
        print("Configuring model...")
        cfg = configure_model()
        print("Registering dataset...")
        register_dataset(dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        print("Running inference...")
        run_prediction(cfg, dataset_name, output_dir)
        print("Inference complete, running ensemble...")
        run_ensemble(dataset_name, output_dir, iou_thresh)
        print("Ensemble complete!")
    if create_tables or make_visuals:
        print("Registering dataset for evaluation...")
        register_dataset(dataset_name)
        print("Evaluating dataset...")
        eval_obj = evaluate_dataset(dataset_name, output_dir, iou_thresh, prob_thresh)
        print("Creating dataset table...")
        table = create_table(eval_obj)
        if create_tables:
            create_dfvol(dataset_name, output_dir, table)
            create_dfimg(dataset_name, output_dir, table)
            print("Dataset HTML tables complete!")
        if make_visuals:
            print("Initializing visualizer...")
            vis = OutputVis(
                dataset_name,
                prob_thresh=eval_obj.prob_thresh,
                pred_mode="file",
                pred_file=os.path.join(output_dir, "coco_instances_results.json"),
                has_annotations=False,  # Assuming we are visualizing on test data without GT
            )
            vis.scale = 1.0  # Use original scale for output visuals
            if bm:
                print("Creating binary masks TIFF (no overlay)...")
                vis.annotation_color = "w"
                output_dataset_predictions(
                    table, vis, os.path.join(output_dir, "predicted_binary_masks"), "pred_only", "bw"
                )
            if bmo:
                print("Creating binary masks TIFF (with overlay)...")
                output_dataset_predictions(
                    table, vis, os.path.join(output_dir, "predicted_binary_overlays"), "pred_overlay", "bw"
                )
            if imo:
                print("Creating instance masks TIFF (with overlay)...")
                output_dataset_predictions(
                    table, vis, os.path.join(output_dir, "predicted_instance_overlays"), "pred_overlay", "default"
                )
            print("Visualizations complete!")
