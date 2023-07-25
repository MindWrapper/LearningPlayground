from fastbook import get_image_files, Path
from fastai.vision.all import *
import argparse
import matplotlib.pyplot as plt

DEFAULT_FINE_TUNE_ITER = 4

MODEL_NAME_TO_FUNCTION = {
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
}

def create_data_block():
    return DataBlock(
        blocks=(ImageBlock, CategoryBlock), 
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=RandomResizedCrop(224, min_scale=0.5),
        batch_tfms=aug_transforms(mult=2)
    )

def train_model(dls, fine_tune_iter, model_name):
    model_func = MODEL_NAME_TO_FUNCTION.get(model_name)
    if model_func is None:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    learn = vision_learner(dls, model_func, metrics=error_rate)
    learn.fine_tune(fine_tune_iter)
    return learn

def interpret_model(learn):
    interp = ClassificationInterpretation.from_learner(learn)
    plt.ion()
    interp.plot_confusion_matrix()
    input("Viewing confusion matrix. Press Enter to continue...")

def save_model(learn, model_file_path):
    learn.export(model_file_path)

def fine_tune_model(data_dir, save_model_as, force, fine_tune_iter, model_name):
    print(f"Data directory: {data_dir}")
    print(f"Model will be saved as: {save_model_as}")
    print(f"Force re-training: {force}")
    print(f"Fine tune iterations: {fine_tune_iter}")
    print(f"Model name: {model_name}")
    # Skip training if the model already exists and force is not specified
    if not force and Path(save_model_as).exists():
        print(f"Model {save_model_as} already exists. Use --force to retrain.")
        return save_model_as

    # Ensure data directory exists
    if not Path(data_dir).exists():
        print(f"Error: Data directory {data_dir} does not exist.")
        return

    # Load data and train model
    dls = create_data_block().dataloaders(data_dir)
    print("Training model...")
    learn = train_model(dls, fine_tune_iter, model_name)

    # Interpret and save model
    interpret_model(learn)
    save_model(learn, save_model_as)
    print("Model training completed.")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Process arguments.")
    parser.add_argument("--data-dir", type=str, required=True, help="Folder to train model on. For example: data/fruits")
    parser.add_argument("--save-model-as", type=str, required=True, help="Where to save model file.")
    parser.add_argument("--fine-tune-iter", type=int, default=DEFAULT_FINE_TUNE_ITER, help="Number of fine tuning iterations")
    parser.add_argument("--force", action="store_true", help="Force retraining even if model already exists.")
    parser.add_argument("--model-name", type=str, default="resnet152", help="Model architecture name")
    return parser.parse_args()

def main():
    """Main function to parse arguments and start model training"""
    args = parse_arguments()
    fine_tune_model(args.data_dir, args.save_model_as, args.force, args.fine_tune_iter, args.model_name)
    print(f"Done. Model saved as {args.save_model_as}")

if __name__ == "__main__":
    main()
