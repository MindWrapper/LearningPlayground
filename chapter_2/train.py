from fastbook import get_image_files, Path
from fastai.vision.all import *
import argparse
import matplotlib.pyplot as plt

# Define constants
MODEL_DIR = "./models"
DEFAULT_FINE_TUNE_ITER = 4

def get_model_path(data_dir):
    model_file_name = data_dir.split("/")[-1]
    return f"{MODEL_DIR}/{model_file_name}.pkl"

def data_block():
    return DataBlock(
        blocks=(ImageBlock, CategoryBlock), 
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=RandomResizedCrop(224, min_scale=0.5),
        batch_tfms=aug_transforms(mult=2)
    )

def train_model(dls, fine_tune_iter):
    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(fine_tune_iter)
    return learn

def interpret_model(learn):
    interp = ClassificationInterpretation.from_learner(learn)
    plt.ion()
    interp.plot_confusion_matrix()
    input("Viewing confusion matrix. Press Enter to continue...")

def save_model(learn, model_file_path):
    learn.export(model_file_path)

def train(data_dir, force, fine_tune_iter):
    model_file_path = get_model_path(data_dir)

    if not force and Path(model_file_path).exists():
        print(f"Skip model training as {model_file_path} already exists. Use --force to override.")
        return model_file_path

    dls = data_block().dataloaders(data_dir)
    dls.train.show_batch(max_n=8, nrows=2, unique=True)

    print("Training model.")
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    
    learn = train_model(dls, fine_tune_iter)
    interpret_model(learn)
    
    model_file_path = get_model_path(data_dir)
    save_model(learn, model_file_path)

    return model_file_path

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process arguments.")
    parser.add_argument("--data-dir", type=str, required=True, help="Folder to train model on.")
    parser.add_argument("--force", action="store_true", help="Force learning even if model already exists.")
    parser.add_argument("--fine-tune-iter", type=int, default=DEFAULT_FINE_TUNE_ITER, help="Number of fine tuning iterations")
    return parser.parse_args()

def main():
    args = parse_arguments()
    model_saved_as = train(args.data_dir, args.force, args.fine_tune_iter)
    print(f"Done. Model saved as {model_saved_as}")

if __name__ == "__main__":
    main()
