import argparse
import os

from monai.apps.pathology import LesionFROC


def full_path(dir: str, file: str):
    return os.path.normpath(os.path.join(dir, file))


def load_data(ground_truth_dir: str, eval_dir: str, level: int, spacing: float):
    # Get the list of probability map result files
    prob_files = os.listdir(eval_dir)

    # read the dataset and create an eval_dataset based on that.
    eval_dataset = []
    for prob_name in prob_files:
        if prob_name.endswith(".npy"):
            sample = {
                "tumor_mask": full_path(ground_truth_dir, prob_name.replace("npy", "tif")),
                "prob_map": full_path(eval_dir, prob_name),
                "level": level,
                "pixel_spacing": spacing,
            }

            eval_dataset.append(sample)

    return eval_dataset


def evaluate_froc(data, reader):
    lesion_froc = LesionFROC(data, image_reader_name=reader)
    score = lesion_froc.evaluate()
    return score


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--spacing", type=float, default=0.243, dest="spacing")
    parser.add_argument("-l", "--level", type=int, default=6, dest="level")
    parser.add_argument("-r", "--reader", type=str, default="cucim", dest="reader")
    parser.add_argument("-e", "--eval-dir", type=str, dest="eval_dir")
    parser.add_argument("-g", "--ground-truth-dir", type=str, dest="ground_truth_dir")
    args = parser.parse_args()

    # prepare FROC input data
    data = load_data(args.ground_truth_dir, args.eval_dir, args.level, args.spacing)
    if len(data) < 1:
        raise RuntimeError(f"No probability map result found in '{args.eval_dir}' with '.npy' extension.")

    # evaluate FROC
    score = evaluate_froc(data, args.reader)
    with open(full_path(args.eval_dir, "froc_score.txt"), "w") as f:
        f.write(f"FROC Score: {score}\n")
    print(f"FROC Score: {score}")
