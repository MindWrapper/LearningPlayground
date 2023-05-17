from fastbook import get_image_files, Path
from fastai.vision.all import *
import argparse

# usage:
# --data-dir .data/guitars

# TODO:  pass fine tune iterations as a command line argument
def train(data_dir, force):
    model_file_name = data_dir.split("/")[-1]
    model_file_path =  "./models/ " + model_file_name + ".pkl"
    if not force and Path(model_file_path).exists():
         print("Skip model training as " + model_file_path + " already exists. Use --force to override.")
         return model_file_path

    bears = DataBlock(
        blocks=(ImageBlock, CategoryBlock),  # independent variable(images), dependent variable(category - types of bears)
        get_items=get_image_files, # get_image_files is a function that returns a list of all the images in the path (recursively)
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label, # get_y dependant variable if often called y. parent_label is a function that returns the parent directory of the image
       
        # We don't feed the model images one-by-one, but instead we feed them in mini-batches. 
        # Inside each mini-batch, all images must be the same size.
        # There are several resize methods
        # ResizeMethod.Crop - crops the image to the desired size
        # ResizeMethod.Pad - adds padding to the image to make it the desired size
        # ResizeMethod.Squish - squishes the image to the desired size
        # Each has it's own problematic
        # Crop - we might lose important information
        # Squish - unrealistic shapes
        # Recommend approch in is to randomly select part of the image, and crop to just that part
        # this is what RandomResizedCrop does
        item_tfms=RandomResizedCrop(224, min_scale=0.5),
        # aug_transforms which will apply set 
        # of an augmentations to the images that fast.ai authors found to be useful
        # to an entier batch_tfms, mult=2 means that we will have 2x more images
        batch_tfms=aug_transforms(mult=2)
        )
    # dls includes validation set and training set
    dls = bears.dataloaders(data_dir)

    # view results of the batch
    dls.train.show_batch(max_n=8, nrows=2, unique=True)
    print("Training model.")
    Path("./models").mkdir(parents=True, exist_ok=True)
    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(4)
    interp = ClassificationInterpretation.from_learner(learn)
    plt.ion()
    interp.plot_confusion_matrix()
    input("Viewing confusion matrix. Press Enter to continue...")
    model_file_path =  "./models/ " + model_file_name + ".pkl"
    learn.export(model_file_path)
    return model_file_path

def main():
    parser = argparse.ArgumentParser(description="Process arguments.")
    parser.add_argument("--data-dir", type=str, help="Folder with to train model on.")
    parser.add_argument("--force", action="store_true", help="Force learning even if model already exists.")
    args = parser.parse_args()
    model_saved_as = train(args.data_dir, args.force)
    print("Done. Model saved as " + model_saved_as)

if __name__ == "__main__":
    main()
