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

os.chdir(os.path.dirname(os.path.abspath(__file__)))
current_directory = os.getcwd()
print(current_directory)
from .analysis_lib import CreatePlotsRPD, EvaluateClass, OutputVis, grab_dataset
from .datasets import data
from .Ensembler import Ensembler
from .table_styles import styles


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


def create_dataset(dataset_name, extracted_path):  # Creates dataset and pk file from extracted images.
    stored_data = data.rpd_data(extracted_path)
    pickle.dump(stored_data, open(os.path.join(data.script_dir, f"{dataset_name}.pk"), "wb"))


def configure_model():
    cfg = get_cfg()
    moddir = os.path.dirname(os.path.realpath(__file__))
    name = "working.yaml"
    cfg_path = os.path.join(moddir, name)
    cfg.merge_from_file(cfg_path)
    return cfg


def register_dataset(dataset_name):
    for name in [dataset_name]:
        try:
            DatasetCatalog.register(name, grab_dataset(name))
        except AssertionError as e:
            print(f"Assertion failed: {e}. Already registered.")
        MetadataCatalog.get(name).thing_classes = ["rpd"]


def run_prediction(cfg, dataset_name, output_path):
    model = build_model(cfg)  # returns a torch.nn.Module
    myloader = build_detection_test_loader(cfg, dataset_name)
    myeval = COCOEvaluator(
        dataset_name, tasks={"bbox", "segm"}, output_dir=output_path
    )  # produces _coco_format.json when initialized
    for mdl in ("fold1", "fold2", "fold3", "fold4", "fold5"):
        extract_directory = "../model"
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
    ens = Ensembler(output_path, dataset_name, ["fold1", "fold2", "fold3", "fold4", "fold5"], iou_thresh=iou_thresh)
    ens.mean_score_nms()
    ens.save_coco_instances()
    return ens


def evaluate_dataset(dataset_name, output_path, iou_thresh=0.2, prob_thresh=0.5):
    myeval = EvaluateClass(dataset_name, output_path, iou_thresh=iou_thresh, prob_thresh=prob_thresh, evalsuper=False)
    myeval.evaluate()
    with open(os.path.join(output_path, "scalar_dict.json"), "w") as outfile:
        json.dump(obj=myeval.summarize_scalars(), fp=outfile)
    return myeval


def create_table(myeval):
    dataset_table = CreatePlotsRPD.initfromcoco(myeval.mycoco, myeval.prob_thresh)
    dataset_table.dfimg.sort_index(inplace=True)
    return dataset_table
    # dataset_table.dfimg['scan'] = dataset_table.dfimg['scan'].astype('int') #depends on what we want scan field to be


def output_vol_predictions(dataset_table, vis, volID, output_path, output_mode="pred_overlay"):
    dfimg = dataset_table.dfimg
    imgids = dfimg[dfimg.volID == volID].sort_index().index.values
    outname = os.path.join(output_path, f"{volID}.tiff")
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
    vis.set_draw_mode(draw_mode)
    os.makedirs(output_path, exist_ok=True)
    for volID in dataset_table.dfvol.index:
        output_vol_predictions(dataset_table, vis, volID, output_path, output_mode)


def create_dfvol(dataset_name, output_path, dataset_table):
    dfvol = dataset_table.dfvol.sort_values(by=["dt_instances"], ascending=False)
    with pd.option_context("styler.render.max_elements", int(dfvol.size) + 1):
        html_str = dfvol.style.format("{:.0f}").set_table_styles(styles).to_html()
    html_file = open(os.path.join(output_path, "dfvol_" + dataset_name + ".html"), "w")
    html_file.write(html_str)
    html_file.close()


def create_dfimg(dataset_name, output_path, dataset_table):
    dfimg = dataset_table.dfimg.sort_index()
    with pd.option_context("styler.render.max_elements", int(dfimg.size) + 1):
        html_str = dfimg.style.set_table_styles(styles).to_html()
    html_file = open(os.path.join(output_path, "dfimg_" + dataset_name + ".html"), "w")
    html_file.write(html_str)
    html_file.close()


def main(args):
    dataset_name = None
    input_dir = None
    extracted = None
    output = None
    run_ext = True
    run_inf = True
    prob_thresh = 0.5
    iou_thresh = 0.2
    make_table = True
    make_visuals = False
    bm = False
    bmo = False
    imo = False

    print(args)

    dataset_name = args.get("dataset_name")  # Access values from the dictionary
    input_dir = args.get("input_dir")
    extracted = args.get("extracted_dir")
    input_format = args.get("input_format")
    output = args.get("output_dir")
    run_ext = args.get("run_extract", True)  # Provide default values
    make_dataset = args.get("create_dataset", True)
    run_inf = args.get("run_inference", True)
    prob_thresh = args.get("prob_thresh", 0.5)
    make_table = args.get("create_tables", True)
    bm = args.get("binary_mask", False)
    bmo = args.get("binary_mask_overlay", False)
    imo = args.get("instance_mask_overlay", False)
    make_visuals = bm | bmo | imo

    if run_ext:
        if not os.path.isdir(extracted):
            print("Extracted dir does not exist! Making extracted dir...")
            os.mkdir(extracted)
        data.extract_files(input_dir, extracted, input_format)
        print("Image extraction complete!")
    if make_dataset:
        print("Creating dataset from extracted images...")
        create_dataset(dataset_name, extracted)
    if run_inf:
        print("Configuring model...")
        cfg = configure_model()
        print("Registering dataset...")
        register_dataset(dataset_name)
        if not os.path.isdir(output):
            print("Output dir does not exist! Making output dir...")
            os.makedirs(output)
        print("Running inference...")
        run_prediction(cfg, dataset_name, output)
        print("Inference complete, running ensemble...")
        run_ensemble(dataset_name, output)
        print("Ensemble complete!")
    if make_table or make_visuals:
        print("Registering dataset...")
        register_dataset(dataset_name)
        print("Evaluating dataset...")
        eval = evaluate_dataset(dataset_name, output, iou_thresh, prob_thresh)
        print("Creating dataset table...")
        table = create_table(eval)
        if make_table:
            create_dfvol(dataset_name, output, table)
            create_dfimg(dataset_name, output, table)
            print("Dataset htmls complete!")
        if make_visuals:
            vis = OutputVis(
                dataset_name,
                prob_thresh=eval.prob_thresh,
                pred_mode="file",
                pred_file=os.path.join(output, "coco_instances_results.json"),
                has_annotations=False,
            )
            vis.scale = 1.0
            if bm:
                print("Creating binary masks tif (no overlay)...")
                vis.annotation_color = "w"
                output_dataset_predictions(
                    table, vis, os.path.join(output, "predicted_binary_masks"), "pred_only", "bw"
                )
            if bmo:
                print("Creating binary masks tif (with overlay)...")
                output_dataset_predictions(
                    table, vis, os.path.join(output, "predicted_binary_overlays"), "pred_overlay", "bw"
                )
            if imo:
                print("Creating instances masks tif (with overlay)...")
                output_dataset_predictions(
                    table, vis, os.path.join(output, "predicted_instance_overlays"), "pred_overlay", "default"
                )
            print("Visualizations complete!")


# if __name__ == "__main__":
#     main(sys.argv[1:])
#     # main_alt()
